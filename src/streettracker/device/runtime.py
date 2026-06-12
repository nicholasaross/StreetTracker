"""Async live-runtime loop for the Orin deployment.

This module is the integrator: it wires together the building blocks
landed in PR 3a-3d (config, snapshotter, IR detector, dashboard) plus
the in-tree pieces (inference, sources, snap planner, track buffer,
output writers) into one cohesive asyncio loop.

Design (locked in via CLAUDE.md before code was written):

* **Asyncio throughout.** ``asyncio.run(run_session(...))`` is the
  entry point. The blocking ``cv2.VideoCapture.read()`` source lives
  on a background thread that pushes frames into an ``asyncio.Queue``;
  blocking inference runs via ``loop.run_in_executor``. HTTP -- both
  the Reolink snapshotter and the dashboard -- is ``aiohttp``.
* **Frozen config.** :class:`~streettracker.common.config.StreetTrackerConfig`
  is the single source of truth for tunables. Nothing in this module
  reads JSON; nothing here mutates the config.
* **Graceful shutdown.** ``SIGTERM`` / ``SIGINT`` flip an
  ``asyncio.Event``; the consumer notices, stops pulling frames,
  drains in-flight tracks through finalize, writes a final summary
  HTML + ``data.json`` + ``meta.json`` + ``hourly.json``, stops the
  dashboard, and exits 0. The drain phase is bounded by
  ``shutdown_grace_s`` (default 30s) so systemd's ``TimeoutStopSec``
  doesn't ``SIGKILL`` us mid-write.
* **RTSP reconnect** is the producer thread's job, not the consumer's
  -- on source exhaustion (file source: end-of-file; RTSP: connection
  drop) the producer either exits cleanly (file) or reopens with
  exponential backoff (RTSP). The consumer sees a continuous frame
  stream either way.
* **Heavy unit tests + manual Orin smoke.** The on-device end-to-end
  path (Ultralytics .engine + Reolink RTSP) cannot run in CI. Per-frame
  logic and shutdown ordering are testable here; the live frame loop
  is exercised by sshing into the Orin against the real camera.

Replaces the synchronous main loop in NanoTracker
(``nano_tracker.py:1250-1673``); behaviour is otherwise preserved so
existing session output remains schema-compatible.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import aiohttp

from streettracker.common.color import vote_color
from streettracker.common.config import StreetTrackerConfig
from streettracker.common.hourly import build_hourly_rollup
from streettracker.common.output import EventLog, IREventLog, format_wall, save_json
from streettracker.common.schema import IRPeriod, SessionMeta, TrackRecord
from streettracker.common.summary import generate_html
from streettracker.device.dashboard import Dashboard, start_dashboard
from streettracker.device.ir_detector import IRDetector
from streettracker.device.snap_planner import (
    RoadGateConfig,
    SnapPlanner,
    SnapPlannerConfig,
    TriggerDirection,
)
from streettracker.device.snapshotter import (
    ReolinkSnapshotter,
    build_snap_url,
    run_cleanup_task,
)
from streettracker.device.track_buffer import (
    BufferedTrack,
    TrackBuffer,
    compute_attributes,
    save_thumbnail,
    sharpness_score,
)

if TYPE_CHECKING:
    import numpy as np

    from streettracker.sources.base import FrameSource

logger = logging.getLogger(__name__)


# Bounded queue between the source-producer thread and the asyncio
# consumer. Size 2 keeps memory pressure low (~6 MB of 1080p frames
# at peak) and surfaces falling-behind as queue-full drops rather than
# unbounded backlog.
_FRAME_QUEUE_SIZE = 2

# Frame buffer sentinel: producer pushes None when it has nothing more
# to yield (file source EOF, or shutdown_event tripped mid-frame).
_END_OF_FRAMES = None

# 20% bbox padding for the center-sharpness probe used by the snap
# planner's blur gate. Matches NanoTracker.
_BLUR_GATE_INSET_FRAC = 0.2


# ----------------------------------------------------------------------
# Session context -- the mutable backbone of one run.


@dataclass(slots=True)
class SessionContext:
    """Mutable state shared across one session.

    Created by :func:`run_session`, passed by reference to per-frame
    helpers. The fields here are roughly what NanoTracker held as
    closure-captured locals in ``main()`` (nano_tracker.py:1269-1292);
    making them an explicit object keeps the per-frame logic testable
    without spinning up the whole loop.
    """

    config: StreetTrackerConfig
    output_dir: Path
    session_label: str
    t_start_wall: float
    t_start_mono: float

    # Pluggable subsystems. Several are Optional so tests can stub them
    # (e.g. inference is None when the test exercises only the
    # ingest -> finalize -> summary path).
    yolo: Any  # UltralyticsTracker or test stub with .track(frame) -> list[Detection]
    ir: IRDetector
    buffer: TrackBuffer
    event_log: EventLog
    is_file_source: bool

    snapshotter: ReolinkSnapshotter | None = None
    dashboard: Dashboard | None = None
    planner: SnapPlanner | None = None  # lazy on first frame (needs frame w/h)
    # IR events sidecar log -- crash-safe transition record so a SIGKILL
    # mid-IR-period doesn't lose the start timestamp. Optional so tests
    # that build a SessionContext directly can leave it ``None``; the
    # process_frame code path guards on it.
    ir_events_log: IREventLog | None = None

    # Output accumulators -- written to disk at session end.
    records: list[TrackRecord] = field(default_factory=list)
    ir_periods: list[IRPeriod] = field(default_factory=list)

    # Counters / housekeeping.
    frames_processed: int = 0
    frames_inferred: int = 0
    raw_track_count: int = 0  # pre min-duration filter, for parity with NanoTracker
    html_dirty: bool = False
    html_writes: int = 0
    last_finalize_mono: float = 0.0
    last_html_write_mono: float = 0.0
    frame_w: int = 0
    frame_h: int = 0
    # Cumulative count of frames where the snap planner returned
    # ``reason="blur_gate"`` (sharpness below ``min_sharpness``). Frames
    # are counted, not tracks -- a fast vehicle can contribute many
    # blur-skipped frames over its visible span. Surfaced in
    # SessionMeta.snap_stats at session end so post-cutover ANPR tuning
    # can correlate this against capture coverage.
    blur_skip_count: int = 0
    # Pipeline (continuous in-band) accounting. ``pipeline_fires_total``
    # is the aggregate of successful pipeline submits across all tracks.
    # ``pipeline_throttled`` counts frames where the per-track interval
    # blocked a fire; ``pipeline_budget_exhausted`` counts frames where
    # the per-track pipeline cap blocked a fire. Together they give the
    # operator visibility into whether the configured interval / cap is
    # too tight, too loose, or just right.
    pipeline_fires_total: int = 0
    pipeline_throttled: int = 0
    pipeline_budget_exhausted: int = 0
    # Per-track pipeline stats captured at finalize-time (so each entry
    # corresponds to a single track's complete visible history). At
    # session end we render p50/p90/max distributions into snap_stats.
    per_track_fires_total: list[int] = field(default_factory=list)
    per_track_time_in_band_ms: list[float] = field(default_factory=list)

    @property
    def session_t(self) -> float:
        """Seconds since session start. Use only for housekeeping --
        per-frame ``session_t`` is computed from the source's own ``t``
        (file) or wall clock (live RTSP), see :func:`_compute_session_t`."""
        return time.monotonic() - self.t_start_mono


# ----------------------------------------------------------------------
# Pure helpers.


def make_session_label(prefix: str, t_start_wall: float) -> str:
    """``{prefix}_YYYYMMDD_HHMMSS`` local-time label.

    Matches NanoTracker's session label format so the output directory
    naming convention stays stable.
    """
    import datetime

    stamp = datetime.datetime.fromtimestamp(t_start_wall).astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def session_paths(output_root: Path, session_label: str) -> dict[str, Path]:
    """The file paths the runtime writes during / after a session."""
    base = output_root / session_label
    return {
        "dir": base,
        "events_jsonl": base / f"{session_label}_events.jsonl",
        "ir_events_jsonl": base / f"{session_label}_ir_events.jsonl",
        "data_json": base / f"{session_label}_data.json",
        "meta_json": base / f"{session_label}_meta.json",
        "hourly_json": base / f"{session_label}_hourly.json",
        "summary_html": base / f"{session_label}_summary.html",
        "heartbeat": base / ".heartbeat",
    }


def _percentile_summary(
    xs: list[float], *, ms_unit: bool = False, round_ndigits: int = 1
) -> dict[str, float | int]:
    """Nearest-rank percentile summary used in snap_stats distributions.

    Same convention as ``ReolinkSnapshotter.latency_summary`` -- no
    interpolation, rounded to ``round_ndigits``. ``ms_unit=True`` emits
    ``p50_ms`` style keys for time-unit histograms; otherwise plain
    ``p50``.
    """
    sorted_xs = sorted(xs)
    n = len(sorted_xs)

    def pct(p: float) -> float:
        return sorted_xs[min(n - 1, int(n * p))]

    sfx = "_ms" if ms_unit else ""
    return {
        "n": n,
        f"p50{sfx}": round(pct(0.5), round_ndigits),
        f"p90{sfx}": round(pct(0.9), round_ndigits),
        f"max{sfx}": round(sorted_xs[-1], round_ndigits),
        f"mean{sfx}": round(sum(sorted_xs) / n, round_ndigits),
    }


def build_session_meta(ctx: SessionContext) -> SessionMeta:
    """Snapshot of session metadata, written at session end."""
    elapsed = ctx.session_t
    avg_infer = float(getattr(ctx.yolo, "avg_infer_ms", 0.0))
    snap_stats: dict[str, Any] | None = None
    if ctx.snapshotter is not None:
        s = ctx.snapshotter.stats
        snap_stats = {
            "attempts": s.attempts,
            "successes": s.successes,
            "failures": s.failures,
            "dropped": s.dropped,
            "blur_skipped_frames": ctx.blur_skip_count,
            # Pipeline-mode accounting. Aggregated from the planner's
            # per-frame decisions, not the snapshotter (which only sees
            # the union of fire submits). Present even when pipelining
            # is disabled -- the values are just zero, which makes
            # before/after comparisons in soak _meta.json files easier.
            "pipeline_fires": ctx.pipeline_fires_total,
            "pipeline_throttled": ctx.pipeline_throttled,
            "pipeline_budget_exhausted": ctx.pipeline_budget_exhausted,
            "latency": s.latency_summary(),
        }
        # Per-track distributions: only emitted when we actually have
        # finalized tracks with pipeline state. Lets us tune
        # pipeline_interval_ms after a soak by reading the achieved
        # fires-per-track distribution against time-in-band.
        if ctx.per_track_fires_total:
            snap_stats["fires_per_track"] = {
                **_percentile_summary(
                    [float(x) for x in ctx.per_track_fires_total],
                    round_ndigits=2,
                ),
                "zero": sum(1 for x in ctx.per_track_fires_total if x == 0),
            }
        if ctx.per_track_time_in_band_ms:
            snap_stats["time_in_band_ms_per_track"] = _percentile_summary(
                ctx.per_track_time_in_band_ms, ms_unit=True
            )
    return SessionMeta(
        session_label=ctx.session_label,
        session_start_unix=ctx.t_start_wall,
        frames_processed=ctx.frames_processed,
        pipe_fps=round(ctx.frames_processed / elapsed, 2) if elapsed > 0 else 0.0,
        avg_infer_ms=round(avg_infer, 2),
        ir_periods=list(ctx.ir_periods),
        snap_stats=snap_stats,
        # Sub-stream frame size captured from the first frame -- used by
        # analysis-side ALPR to scale per-snap bboxes (which live in
        # sub-stream coords on the TrackRecord) into the 4K snap's coord
        # system. ``[0, 0]`` only if the session never saw a frame.
        frame_size=[ctx.frame_w, ctx.frame_h] if ctx.frame_w and ctx.frame_h else None,
    )


def _compute_session_t(ctx: SessionContext, source_t: float) -> float:
    """Per-frame session time.

    For file sources, video-time (``frame_idx / fps``) makes durations
    meaningful even when inference runs faster or slower than realtime.
    For live RTSP, wall-clock seconds since session start keeps a
    single contiguous timeline across reconnects.
    """
    return source_t if ctx.is_file_source else (time.time() - ctx.t_start_wall)


def _maybe_lazy_init_planner(ctx: SessionContext) -> None:
    """Build the SnapPlanner once the first frame has given us w/h.

    Pulled out so tests can preload ``ctx.planner`` directly without
    needing a real frame.
    """
    if ctx.planner is not None or ctx.snapshotter is None:
        return
    if ctx.frame_w <= 0 or ctx.frame_h <= 0:
        return
    cfg = ctx.config.snapshot
    road_gate: RoadGateConfig | None = None
    if cfg.snap_gate is not None:
        s = cfg.snap_gate
        directions: list[TriggerDirection] | None = None
        if s.trigger_directions is not None:
            directions = list(s.trigger_directions)  # type: ignore[arg-type]
        road_gate = RoadGateConfig(
            polygon_frac=list(s.polygon_frac),
            trigger_t_prime=list(s.trigger_t_prime),
            t_usable_frac=s.t_usable_frac,
            trigger_directions=directions,
            pipeline_interval_ms=s.pipeline_interval_ms,
            pipeline_max_per_track=s.pipeline_max_per_track,
            pipeline_interval_ms_by_direction=(
                dict(s.pipeline_interval_ms_by_direction)
                if s.pipeline_interval_ms_by_direction is not None
                else None
            ),
            # The strict loader passes dict values through untyped (JSON
            # lists); RoadGate.from_config validates keys + ranges.
            pipeline_t_usable_by_direction=(
                {k: (v[0], v[1]) for k, v in s.pipeline_t_usable_by_direction.items()}
                if s.pipeline_t_usable_by_direction is not None
                else None
            ),
        )
    ctx.planner = SnapPlanner(
        frame_width=ctx.frame_w,
        frame_height=ctx.frame_h,
        config=SnapPlannerConfig(
            area_threshold_frac=cfg.area_threshold_frac,
            max_per_track=cfg.max_snaps_per_track,
            min_sharpness=cfg.min_sharpness,
            decay_ratio=cfg.decay_ratio,
            plateau_frames=cfg.plateau_frames,
            max_wait_frames=cfg.max_wait_frames,
            exit_margin_frac=cfg.exit_margin_frac,
            post_fire_cooldown_frames=cfg.post_fire_cooldown_frames,
            road_gate=road_gate,
        ),
    )
    logger.info(
        "[runtime] SnapPlanner initialized (%dx%d, mode=%s)",
        ctx.frame_w,
        ctx.frame_h,
        "road_gate" if road_gate is not None else "right_half",
    )


def _bbox_center_sharpness(
    frame: np.ndarray, bbox: tuple[float, float, float, float]
) -> float | None:
    """Sharpness of the bbox's center 60% region (matches NanoTracker)."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bx1 = max(0, int(x1))
    by1 = max(0, int(y1))
    bx2 = min(fw, int(x2))
    by2 = min(fh, int(y2))
    bw = bx2 - bx1
    bh = by2 - by1
    if bw <= 0 or bh <= 0:
        return None
    mx = int(bw * _BLUR_GATE_INSET_FRAC)
    my = int(bh * _BLUR_GATE_INSET_FRAC)
    cx1 = bx1 + mx
    cx2 = bx2 - mx
    cy1 = by1 + my
    cy2 = by2 - my
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return sharpness_score(frame[cy1:cy2, cx1:cx2])


def _main_snap_path(output_dir: Path, prefix: str, track_id: int, snap_index: int) -> Path:
    """``{output_dir}/{prefix}_{id}_main_{N}.jpg`` path for a fire."""
    return output_dir / f"{prefix}_{track_id}_main_{snap_index}.jpg"


def _rename_snap_file(
    output_dir: Path,
    track_id: int,
    snap_index: int,
    old_prefix: str,
    new_prefix: str,
) -> bool:
    """Rename a saved ``_main_N.jpg`` from ``old_prefix`` to ``new_prefix``.

    Called when a track's class flips between fire and finalize (BotSORT
    can reassign a track's class as evidence accumulates). Best-effort:
    a missing source file or an existing destination is logged and
    swallowed -- the alternative is a crashing finalize for a cosmetic
    issue.
    """
    old = _main_snap_path(output_dir, old_prefix, track_id, snap_index)
    new = _main_snap_path(output_dir, new_prefix, track_id, snap_index)
    try:
        if not old.exists():
            return False
        if new.exists():
            logger.warning(
                "[snap] cannot rename %s -> %s: destination exists",
                old.name,
                new.name,
            )
            return False
        old.rename(new)
        logger.info("[snap] renamed %s -> %s (class-flip)", old.name, new.name)
        return True
    except OSError as exc:
        logger.warning("[snap] rename failed %s -> %s: %s", old.name, new.name, exc)
        return False


def _fire_snap(ctx: SessionContext, track: BufferedTrack, snap_index: int) -> None:
    """Submit a snapshotter fire and mark the index on the track."""
    if ctx.snapshotter is None:
        return
    from streettracker.common.schema import asset_prefix_for_class

    fire_prefix = asset_prefix_for_class(track.class_id)
    out_path = _main_snap_path(ctx.output_dir, fire_prefix, track.id, snap_index)
    task = ctx.snapshotter.submit(out_path)
    if task is None:
        return  # snapshotter dropped (concurrency cap)
    track.snap_fire_prefixes[snap_index] = fire_prefix
    # Capture the BotSORT-tracked sub-stream bbox at the moment of
    # fire so analysis-side ALPR can pre-crop the 4K snap to the
    # tracked vehicle (vs largest-in-frame). Stored as integer pixel
    # coords; ``last_bbox`` is None only for brand-new tracks with
    # zero motion points, which can't fire a snap, so the guard is
    # defensive.
    last_bbox = track.last_bbox
    if last_bbox is not None:
        track.snap_fire_bboxes[snap_index] = (
            int(last_bbox[0]), int(last_bbox[1]),
            int(last_bbox[2]), int(last_bbox[3]),
        )
    output_dir = ctx.output_dir

    def _on_done(t: asyncio.Task[bool]) -> None:
        with contextlib.suppress(Exception):
            if not t.result():
                return
            track.snap_saved_indexes.add(snap_index)
            # Re-capture the track's bbox at snap COMPLETION. The 4K
            # HTTP snap lands ~0.7-1.3s after the fire decision (p50
            # 710 ms) and the image is exposed near response time, so
            # the completion-time bbox is where the car actually IS in
            # the saved image -- the Step 16 triage showed 96 % of R→L
            # read failures were the car exiting its fire-time bbox in
            # that gap. Offline hint resolution prefers this when
            # present. If the track expired before the snap landed,
            # ``last_bbox`` is its final observed position -- still
            # closer to the truth than the fire-time box.
            done_bbox = track.last_bbox
            if done_bbox is not None:
                track.snap_done_bboxes[snap_index] = (
                    int(done_bbox[0]), int(done_bbox[1]),
                    int(done_bbox[2]), int(done_bbox[3]),
                )
            # If finalize has already locked a different prefix (the
            # snap landed after the track expired), rename now. The
            # synchronous sweep in finalize_track skipped this index
            # because it wasn't in snap_saved_indexes yet.
            final_prefix = track.final_prefix
            if final_prefix is not None and final_prefix != fire_prefix:
                _rename_snap_file(
                    output_dir, track.id, snap_index, fire_prefix, final_prefix
                )

    task.add_done_callback(_on_done)
    track.snap_count = snap_index


def finalize_track(ctx: SessionContext, track: BufferedTrack) -> None:
    """Convert a buffered track to a :class:`TrackRecord` and persist.

    Steps:
      1. Last-chance snap rescue via the planner.
      2. Color vote over buffered crops (non-edge first).
      3. Save thumbnails (dashboard tile + HQ best).
      4. Build TrackRecord; if filtered (too short / parked), drop.
      5. Append to EventLog + in-memory record list; mark HTML dirty.
    """
    ctx.raw_track_count += 1

    # 1. Last-chance snap rescue.
    if ctx.planner is not None and ctx.snapshotter is not None:
        rescue = ctx.planner.on_track_finalize(track.id)
        if rescue.should_fire and rescue.snap_index is not None:
            logger.info(
                "[snap] track %d fire %d reason=%s (finalize rescue)",
                track.id,
                rescue.snap_index,
                rescue.reason,
            )
            _fire_snap(ctx, track, rescue.snap_index)
        # Capture per-track pipeline stats BEFORE forget(). Only emit
        # into the session distributions when road_gate mode is active
        # (i.e., the planner actually has road-gate state to report);
        # in non-road-gate mode time_in_band_ms is meaningless.
        if ctx.planner.road_gate is not None:
            stats = ctx.planner.track_pipeline_stats(track.id, time.monotonic() * 1000.0)
            if stats is not None:
                ctx.per_track_fires_total.append(int(stats["fires_total"]))
                ctx.per_track_time_in_band_ms.append(float(stats["time_in_band_ms"]))
        ctx.planner.forget(track.id)

    # 1b. Lock the asset prefix and reconcile already-saved 4K snaps
    # whose fire-time prefix differs from the final class. BotSORT can
    # flip a track's class mid-life (person <-> vehicle), so a track may
    # have fired its first snaps under one prefix and finalize under
    # the other. Without this rename the on-disk filenames disagree
    # with ``record.asset_prefix`` and downstream consumers
    # (summary HTML, ALPR runner) treat the files as orphans.
    from streettracker.common.schema import asset_prefix_for_class

    final_prefix = asset_prefix_for_class(track.class_id)
    track.final_prefix = final_prefix
    for snap_index in sorted(track.snap_saved_indexes):
        fire_prefix = track.snap_fire_prefixes.get(snap_index)
        if fire_prefix is not None and fire_prefix != final_prefix:
            _rename_snap_file(
                ctx.output_dir, track.id, snap_index, fire_prefix, final_prefix
            )

    # 2. Color vote.
    color = "unknown"
    if track.crops:
        non_edge = [c for c in track.crops if not c.on_edge] or track.crops
        # Pick the largest non-edge crop -- biggest sample beats sharpest
        # when color is uniform across the vehicle.
        best = max(
            non_edge,
            key=lambda c: int(c.crop.shape[0]) * int(c.crop.shape[1]),
        )
        color = vote_color(best.crop)

    # 3. Thumbnails (reuse the prefix locked in step 1b).
    if ctx.config.output.save_thumbnails and track.crops:
        midpoint_t = 0.5 * (track.points[0].t + track.points[-1].t)
        mid_crop = min(track.crops, key=lambda c: abs(c.t - midpoint_t)).crop
        save_thumbnail(mid_crop, ctx.output_dir / f"{final_prefix}_{track.id}.jpg")
        if track.hq_best_crop is not None:
            save_thumbnail(
                track.hq_best_crop,
                ctx.output_dir / f"{final_prefix}_{track.id}_hq.jpg",
                quality=95,
            )

    # 4. Compute attributes (filters short / parked tracks).
    record = compute_attributes(
        track,
        frame_h=ctx.frame_h or 1080,
        min_duration_s=ctx.config.tracking.min_track_duration_s,
        parked_disp_px=ctx.config.tracking.parked_displacement_px,
        color=color,
        t_start_wall=ctx.t_start_wall,
        main_snaps=sorted(track.snap_saved_indexes),
    )
    if record is None:
        return

    # 5. Persist.
    ctx.event_log.append(record)
    ctx.records.append(record)
    ctx.last_finalize_mono = time.monotonic()
    ctx.html_dirty = True


def write_html(ctx: SessionContext) -> None:
    """Regenerate the session's summary HTML + ``vehicles.json`` row payload."""
    paths = session_paths(ctx.output_dir.parent, ctx.session_label)
    meta = build_session_meta(ctx)
    generate_html(
        cast("list[TrackRecord | dict[str, Any]]", list(ctx.records)),
        ctx.output_dir,
        paths["summary_html"],
        ctx.session_label,
        meta.to_json_dict(),
        refresh_seconds=ctx.config.output.html_refresh_seconds,
    )
    ctx.html_writes += 1
    ctx.html_dirty = False
    ctx.last_html_write_mono = time.monotonic()


def write_session_outputs(ctx: SessionContext, *, no_html: bool, no_json: bool) -> None:
    """Write the consolidated final outputs at session end.

    Idempotent: callable from a ``finally:`` block even if the loop
    body never completed a full frame -- empty lists serialize cleanly.
    """
    paths = session_paths(ctx.output_dir.parent, ctx.session_label)
    if not no_html:
        write_html(ctx)
    if not no_json:
        records_widened = cast("list[TrackRecord | dict[str, Any]]", list(ctx.records))
        save_json(
            records_widened,
            build_session_meta(ctx),
            paths["data_json"],
            paths["meta_json"],
        )
        ir_widened = cast("list[IRPeriod | dict[str, Any]]", list(ctx.ir_periods))
        hourly = build_hourly_rollup(records_widened, ir_widened)
        paths["hourly_json"].write_text(json.dumps(hourly, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# Per-frame body.


async def process_frame(
    ctx: SessionContext, frame_idx: int, source_t: float, frame: np.ndarray
) -> None:
    """One frame's worth of work; mutates ``ctx`` in place.

    Awaits the inference call (``run_in_executor``) so the event loop
    can service other tasks (snapshotter HTTP, dashboard, cleanup)
    during the ~10-50 ms inference window. Everything else is fast and
    happens inline.
    """
    ctx.frames_processed += 1
    if ctx.frame_h == 0:
        ctx.frame_h = int(frame.shape[0])
        ctx.frame_w = int(frame.shape[1])

    session_t = _compute_session_t(ctx, source_t)

    # IR check + day/night transition handling. The same wall_time is
    # used for the detector update and the sidecar IREventLog so the
    # transition timestamps round-trip exactly even if a SIGKILL lands
    # right after one is recorded.
    wall_time = time.time()
    was_in_ir = ctx.ir.in_ir_mode
    closed = ctx.ir.update(frame, wall_time)
    if closed is not None:
        ctx.ir_periods.append(closed)
        if ctx.ir_events_log is not None:
            ctx.ir_events_log.append_end(wall_time)
        logger.info("[runtime] returning to day at %s -- resuming inference", closed.end)
    if not was_in_ir and ctx.ir.in_ir_mode:
        if ctx.ir_events_log is not None:
            ctx.ir_events_log.append_start(wall_time)
        logger.info(
            "[runtime] entering IR/night at %s -- skipping inference",
            format_wall(wall_time),
        )
        # Cut active tracks cleanly across the day/night boundary.
        for tr in ctx.buffer.flush():
            finalize_track(ctx, tr)
    if ctx.ir.in_ir_mode:
        return

    # Inference off the event loop.
    loop = asyncio.get_running_loop()
    dets = await loop.run_in_executor(None, ctx.yolo.track, frame)
    ctx.frames_inferred += 1

    # TrackBuffer ingest + expirations.
    expired = ctx.buffer.ingest(dets, frame_idx, session_t, frame)
    for tr in expired:
        finalize_track(ctx, tr)

    # SnapPlanner walk.
    _maybe_lazy_init_planner(ctx)
    if ctx.planner is not None and ctx.snapshotter is not None:
        cfg = ctx.config.snapshot
        for tr in ctx.buffer.active():
            if not tr.points:
                continue
            p = tr.points[-1]
            bbox = (p.x1, p.y1, p.x2, p.y2)
            sharpness: float | None = None
            if cfg.min_sharpness > 0.0:
                sharpness = _bbox_center_sharpness(frame, bbox)
            decision = ctx.planner.consider(
                track_id=tr.id,
                bbox=bbox,
                frame_idx=frame_idx,
                sharpness=sharpness,
            )
            if not decision.should_fire or decision.snap_index is None:
                # Surface blur-gate rejections with per-track throttle so
                # the journal carries enough signal to correlate against
                # capture coverage without flooding on a fast vehicle
                # (which can stay blurred for its whole visible span).
                # Mirrors NanoTracker (nano_tracker.py:1514-1519); my PR
                # 3e port silently dropped these.
                if decision.reason == "blur_gate":
                    ctx.blur_skip_count += 1
                    now_mono = time.monotonic()
                    if now_mono - tr.last_blur_skip_logged_mono >= 5.0:
                        logger.info(
                            "[snap] track %d blur skip (sharpness=%.0f < min=%.0f)",
                            tr.id,
                            sharpness if sharpness is not None else 0.0,
                            cfg.min_sharpness,
                        )
                        tr.last_blur_skip_logged_mono = now_mono
                continue
            logger.info(
                "[snap] track %d fire %d reason=%s score=%.3f",
                tr.id,
                decision.snap_index,
                decision.reason,
                decision.score,
            )
            _fire_snap(ctx, tr, decision.snap_index)

        # Pipeline (continuous in-band) decision path -- evaluated
        # separately so trigger crossings retain priority on the
        # snap_index ordering, and so a disabled pipeline (interval=0)
        # is a near-zero-cost no-op (consider_pipeline returns
        # immediately on the interval check).
        wall_ms = time.monotonic() * 1000.0
        for tr in ctx.buffer.active():
            if not tr.points:
                continue
            p = tr.points[-1]
            bbox = (p.x1, p.y1, p.x2, p.y2)
            sharpness = None
            if cfg.min_sharpness > 0.0:
                sharpness = _bbox_center_sharpness(frame, bbox)
            pipe_dec = ctx.planner.consider_pipeline(
                track_id=tr.id,
                bbox=bbox,
                frame_idx=frame_idx,
                wall_ms=wall_ms,
                sharpness=sharpness,
            )
            if not pipe_dec.should_fire or pipe_dec.snap_index is None:
                if pipe_dec.reason == "pipeline_throttled":
                    ctx.pipeline_throttled += 1
                elif pipe_dec.reason == "pipeline_budget_exhausted":
                    ctx.pipeline_budget_exhausted += 1
                continue
            ctx.pipeline_fires_total += 1
            logger.info(
                "[snap] track %d pipeline fire %d",
                tr.id,
                pipe_dec.snap_index,
            )
            _fire_snap(ctx, tr, pipe_dec.snap_index)

    # Idle HTML regen.
    now = time.monotonic()
    idle = ctx.config.output.html_idle_seconds
    if ctx.html_dirty and (now - ctx.last_finalize_mono) >= idle:
        write_html(ctx)


# ----------------------------------------------------------------------
# Frame producer (blocking source -> asyncio.Queue).


def _produce_frames(
    src: FrameSource,
    queue: asyncio.Queue[tuple[int, float, np.ndarray] | None],
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
    *,
    is_file: bool,
    max_backoff_s: float = 30.0,
) -> None:
    """Background-thread frame producer.

    Pushes ``(frame_idx, source_t, frame)`` onto ``queue`` until the
    source ends (file EOF) or ``stop_event`` is set. For RTSP sources
    the generator returning is treated as a disconnect and we reopen
    with exponential backoff. Pushes ``None`` as a sentinel before
    returning so the consumer breaks out of ``await queue.get()``.
    """
    backoff = 1.0

    def _enqueue(item: tuple[int, float, np.ndarray] | None) -> None:
        # Drop oldest frame if the consumer can't keep up. Live RTSP
        # would otherwise back up unbounded under inference slowdown.
        def _put() -> None:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(item)

        loop.call_soon_threadsafe(_put)

    try:
        while not stop_event.is_set():
            try:
                src.open()
                backoff = 1.0
            except Exception as exc:
                if is_file:
                    logger.error("[runtime] source open failed: %s", exc)
                    break
                logger.warning(
                    "[runtime] source open failed: %s; retrying in %.1fs",
                    exc,
                    backoff,
                )
                if stop_event.wait(backoff):
                    break
                backoff = min(backoff * 2.0, max_backoff_s)
                continue

            try:
                for item in src.frames():
                    if stop_event.is_set():
                        break
                    _enqueue(item)
            except Exception as exc:
                logger.warning("[runtime] source error: %s", exc)
            finally:
                with contextlib.suppress(Exception):
                    src.close()

            if is_file or stop_event.is_set():
                break
            logger.warning("[runtime] RTSP source ended; reconnecting in %.1fs", backoff)
            if stop_event.wait(backoff):
                break
            backoff = min(backoff * 2.0, max_backoff_s)
    finally:
        _enqueue(_END_OF_FRAMES)


# ----------------------------------------------------------------------
# Signal handlers.


def install_signal_handlers(loop: asyncio.AbstractEventLoop, shutdown_event: asyncio.Event) -> None:
    """Wire SIGTERM/SIGINT to ``shutdown_event.set()``.

    POSIX uses ``loop.add_signal_handler``. Windows raises
    ``NotImplementedError`` for that, so we fall back to ``signal.signal``
    with a thread-safe set via ``call_soon_threadsafe``.
    """

    def _set() -> None:
        shutdown_event.set()

    try:
        loop.add_signal_handler(signal.SIGTERM, _set)
        loop.add_signal_handler(signal.SIGINT, _set)
    except NotImplementedError:

        def _windows_handler(signum: int, frame: object) -> None:
            loop.call_soon_threadsafe(_set)

        signal.signal(signal.SIGTERM, _windows_handler)
        signal.signal(signal.SIGINT, _windows_handler)


# ----------------------------------------------------------------------
# Top-level orchestrator.


async def run_session(
    config: StreetTrackerConfig,
    source: FrameSource,
    *,
    output_root: Path | None = None,
    no_html: bool = False,
    no_json: bool = False,
    duration_s: float | None = None,
    shutdown_grace_s: float = 30.0,
    yolo: Any = None,
    ir_detector: IRDetector | None = None,
    enable_snapshotter: bool = True,
    enable_dashboard: bool = True,
) -> int:
    """Run the live session until source exhaustion, signal, or timeout.

    Returns the process exit code (0 on a clean stop, non-zero on
    unrecoverable error).

    Args:
        config: validated runtime config.
        source: opened/closeable :class:`FrameSource`.
        output_root: parent dir for session output. Defaults to
            ``config.output.dir``. A subdir named after the session
            label is created under this root.
        no_html: skip both idle and final HTML regen (useful for
            headless tests).
        no_json: skip ``data.json`` + ``meta.json`` + ``hourly.json``.
        duration_s: hard cap on session length, in seconds. ``None``
            means run until source EOF or signal.
        shutdown_grace_s: drain budget after shutdown_event fires.
        yolo: override the default ``UltralyticsTracker`` -- mostly
            for tests (the default constructs an Ultralytics instance,
            which requires torch + the engine file).
        ir_detector: override the default :class:`IRDetector`.
        enable_snapshotter: if False, skip the Reolink HTTP
            snapshotter (and its periodic cleanup task). Used by
            ``streettracker batch`` -- there's no live camera to fetch
            4K stills from. Combined with the config flag via AND, so
            ``config.snapshot.enabled=False`` still wins.
        enable_dashboard: if False, skip the built-in aiohttp
            dashboard. Used by ``streettracker batch`` -- nothing
            watches a one-off file run. Combined with the config flag
            via AND.
    """
    output_root = output_root or config.output.dir
    output_root.mkdir(parents=True, exist_ok=True)
    t_start_wall = time.time()
    session_label = make_session_label(config.output.session_label_prefix, t_start_wall)
    paths = session_paths(output_root, session_label)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    logger.info("[runtime] session %s -> %s", session_label, paths["dir"])

    if yolo is None:
        from streettracker.inference.ultralytics_runner import UltralyticsTracker

        yolo = UltralyticsTracker(
            config.inference.engine_path,
            conf=config.inference.conf_threshold,
            iou=config.inference.iou_threshold,
            class_filter=config.inference.vehicle_classes,
            input_size=config.inference.input_size,
        )
    if ir_detector is None:
        ir_detector = IRDetector()

    ctx = SessionContext(
        config=config,
        output_dir=paths["dir"],
        session_label=session_label,
        t_start_wall=t_start_wall,
        t_start_mono=time.monotonic(),
        yolo=yolo,
        ir=ir_detector,
        buffer=TrackBuffer(max_misses=config.tracking.max_misses),
        event_log=EventLog(paths["events_jsonl"]),
        is_file_source=bool(getattr(source, "is_file", False)),
        ir_events_log=IREventLog(paths["ir_events_jsonl"]),
    )

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    install_signal_handlers(loop, shutdown_event)

    # aiohttp session shared between the snapshotter and any future
    # subsystem that needs outbound HTTP. Owned by run_session.
    async with aiohttp.ClientSession() as http:
        if config.snapshot.enabled and enable_snapshotter:
            snap_url = build_snap_url(
                config.camera.ip,
                config.ports.http,
                config.camera.username,
                config.camera.password,
            )
            ctx.snapshotter = ReolinkSnapshotter(
                url=snap_url,
                session=http,
                timeout_s=config.snapshot.timeout_s,
                max_concurrent=config.snapshot.max_concurrent,
            )

        if config.http.enabled and enable_dashboard:
            ctx.dashboard = await start_dashboard(
                paths["dir"],
                host=config.http.host,
                port=config.http.port,
            )

        cleanup_task: asyncio.Task[None] | None = None
        if config.snapshot.keep_days > 0 and enable_snapshotter:
            cleanup_task = asyncio.create_task(
                run_cleanup_task(
                    output_root,
                    interval_s=config.snapshot.cleanup_interval_s,
                    keep_days=config.snapshot.keep_days,
                ),
                name="snap-cleanup",
            )

        producer_stop = threading.Event()
        queue: asyncio.Queue[tuple[int, float, np.ndarray] | None] = asyncio.Queue(
            maxsize=_FRAME_QUEUE_SIZE
        )
        producer_thread = threading.Thread(
            target=_produce_frames,
            args=(source, queue, loop, producer_stop),
            kwargs={"is_file": ctx.is_file_source},
            name="frame-producer",
            daemon=True,
        )
        producer_thread.start()

        deadline_mono = time.monotonic() + duration_s if duration_s is not None else None

        try:
            await _consumer_loop(ctx, queue, shutdown_event, deadline_mono)
        finally:
            # Stop the producer first so no new frames arrive during drain.
            producer_stop.set()
            # Drain any still-active tracks, bounded by shutdown_grace_s.
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(_drain_active, ctx), timeout=shutdown_grace_s
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[runtime] drain exceeded %.0fs; forcing shutdown",
                    shutdown_grace_s,
                )
            # Close any open IR period so meta reflects a complete history.
            _close_open_ir_period(ctx)
            # Final outputs (idempotent; safe even with empty record list).
            write_session_outputs(ctx, no_html=no_html, no_json=no_json)
            ctx.event_log.close()
            if ctx.ir_events_log is not None:
                ctx.ir_events_log.close()
            if cleanup_task is not None:
                cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await cleanup_task
            if ctx.dashboard is not None:
                await ctx.dashboard.stop()
            producer_thread.join(timeout=2.0)

    logger.info(
        "[runtime] session ended: %d frames, %d records kept (of %d raw), "
        "%d html writes, %d IR periods",
        ctx.frames_processed,
        len(ctx.records),
        ctx.raw_track_count,
        ctx.html_writes,
        len(ctx.ir_periods),
    )
    return 0


async def _consumer_loop(
    ctx: SessionContext,
    queue: asyncio.Queue[tuple[int, float, np.ndarray] | None],
    shutdown_event: asyncio.Event,
    deadline_mono: float | None,
) -> None:
    """Pull frames from the queue and run :func:`process_frame` until
    EOF, shutdown, or deadline. Pulled out so tests can drive it
    directly without spinning up the producer thread."""
    while True:
        if shutdown_event.is_set():
            logger.info("[runtime] shutdown event received; stopping consumer")
            return
        if deadline_mono is not None and time.monotonic() >= deadline_mono:
            logger.info("[runtime] duration cap reached; stopping consumer")
            return

        # Bound the get() so we periodically re-check shutdown_event /
        # deadline even when the producer stalls.
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        if item is _END_OF_FRAMES:
            logger.info("[runtime] source ended; stopping consumer")
            return

        frame_idx, source_t, frame = item
        try:
            await process_frame(ctx, frame_idx, source_t, frame)
        except Exception:
            logger.exception("[runtime] frame %d processing failed", frame_idx)


def _drain_active(ctx: SessionContext) -> None:
    """Finalize all in-flight tracks at shutdown. Sync, off-thread."""
    for tr in ctx.buffer.flush():
        finalize_track(ctx, tr)


def _close_open_ir_period(ctx: SessionContext) -> None:
    """If the IR detector is mid-period at shutdown, synthesize a
    closing entry so the meta JSON reflects a complete history.

    The detector itself only emits an :class:`IRPeriod` on the IR->day
    edge, but a session shutdown mid-night should still record what
    happened. We probe the detector's private start timestamp via a
    one-off update with a wall-time frame that should NOT flip the
    state (so it doesn't pollute the period).

    Also records the synthetic close on the IR events sidecar so a
    subsequent SIGKILL during ``write_session_outputs`` still leaves
    a complete on-disk transition history.
    """
    if not ctx.ir.in_ir_mode:
        return
    # The detector's history may not all be IR; we just need the
    # period start. Reach into the dataclass: this is the same module
    # family, not an external API, so the coupling is OK.
    start_unix = getattr(ctx.ir, "_ir_period_start_unix", None)
    if start_unix is None:
        return
    end_unix = time.time()
    ctx.ir_periods.append(
        IRPeriod(
            start=format_wall(start_unix),
            end=format_wall(end_unix),
            duration_s=round(end_unix - start_unix, 1),
        )
    )
    if ctx.ir_events_log is not None:
        ctx.ir_events_log.append_end(end_unix)


__all__ = [
    "SessionContext",
    "build_session_meta",
    "finalize_track",
    "install_signal_handlers",
    "make_session_label",
    "process_frame",
    "run_session",
    "session_paths",
    "write_html",
    "write_session_outputs",
]
