"""Per-track Reolink 4K snapshot scheduler.

Decides, on each sub-stream frame, whether to fire a full-resolution
HTTP snapshot for a tracked vehicle. The caller does the dispatch; this
module is pure logic and contains no I/O so the algorithm can be
exercised against synthetic trajectories in unit tests.

Algorithm
---------

Three firing modes are supported, in increasing order of spatial
precision. A planner uses whichever is configured; the others are
ignored.

**Road polygon + axis triggers (`SnapPlannerConfig.road_gate`).** The
operator-validated mode. Configuration carries:

* a polygon traced over the visible road surface (fractional coords);
* a list of trigger positions along the road's principal axis, each
  expressed as ``t'`` in ``[0, 1]`` over the usable t-range;
* optionally, a parallel list of per-trigger direction filters --
  ``"forward"``, ``"reverse"``, or ``"both"`` (default). A forward
  trigger only fires when ``t'`` is *increasing* through it (approach
  side, front-plate visible); a reverse trigger only when ``t'`` is
  *decreasing* (departure side, rear-plate visible). This lets the
  operator place asymmetric triggers -- e.g. an early-approach
  forward trigger and an early-departure reverse trigger -- without
  one direction's snaps consuming the other direction's budget.

On every frame the bbox centre is tested against the polygon; if
inside, its projection along the polygon's PCA axis is compared with
the last frame's projection. If any not-yet-fired trigger lies between
the two values *and the per-trigger direction filter allows the
motion sign*, the trigger fires (and is marked fired so a return
crossing later in the track cannot retrigger it). This produces N
spatially distinct snaps per vehicle independent of speed; with
asymmetric triggers, snap counts can also differ per direction.

**Right-half spatial gate (`right_half_only=True`).** Legacy gate
used before the polygon mode landed. Splits the right half of the
frame into ``max_per_track`` equal-width zones; each zone fires once.
Kept for backwards compatibility and for installations that don't
warrant a hand-traced polygon.

**Peak/decay scoring (`right_half_only=False`).** Original area *
sharpness quality score with peak-decay / plateau / exit-imminent /
wait-timeout triggers, plus a post-fire cooldown. Retained for
benchmarking.

The finalize-time fallback (a "last-chance" fire for a track that
ended without a live snap) is gated to the active mode: it fires only
if the track visited the active gate (right half, or the road
polygon's usable t range, respectively) at some point.

This is the canonical Py3.12 implementation. ``NanoTracker/snap_planner.py``
mirrors the same algorithm in Py3.6-compatible syntax for the on-device
deployment and must stay behaviour-equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ----------------------------------------------------------------------
# Trigger reason strings -- one Literal alias for type narrowing on the
# decision record.

SnapReason = Literal[
    "ineligible",
    "blur_gate",
    "budget_exhausted",
    "tracking",
    "peak_decay",
    "exit_imminent",
    "plateau",
    "wait_timeout",
    "finalize_last_chance",
    "right_half_entry",
    "out_of_gate",
    "trigger_crossing",
    "outside_road_polygon",
    # Pipeline (continuous in-band) firing path. Distinguishing pipeline
    # outcomes from trigger outcomes keeps the journal + snap_stats
    # legible -- the operator can see whether captures came from
    # operator-traced trigger crossings or from the opportunistic
    # in-band stream.
    "pipeline",
    "pipeline_throttled",
    "pipeline_budget_exhausted",
]


# Per-trigger direction filter for road-gate mode.
#
# "forward"  -- fires only when t' is INCREASING through the trigger
#               (i.e., motion in the direction of the principal axis;
#               in our camera setup this is the camera-approach direction
#               and is the side where the front plate is visible).
# "reverse"  -- fires only when t' is DECREASING through the trigger
#               (motion away from the camera; rear-plate side).
# "both"     -- default; fires on either direction of motion.
TriggerDirection = Literal["forward", "reverse", "both"]
_VALID_DIRECTIONS = ("forward", "reverse", "both")


# ----------------------------------------------------------------------
# Config + decision value objects.


@dataclass(slots=True)
class RoadGateConfig:
    """Operator-traced road polygon plus a list of trigger positions.

    The polygon is given in fractional frame coordinates so it is
    resolution-independent. ``trigger_t_prime`` is a list of values in
    ``[0, 1]`` along the *usable* portion of the polygon's principal
    axis -- the segment between ``t_usable_frac[0]`` and
    ``t_usable_frac[1]`` of the raw t-range. Anything outside that
    band is treated as "off-road" and cannot fire even if technically
    inside the polygon.

    ``trigger_directions`` is an optional parallel array of per-trigger
    direction filters (see ``TriggerDirection``). When omitted, every
    trigger defaults to ``"both"`` -- backwards-compatible with configs
    written before the asymmetric-trigger feature. With asymmetric
    triggers you can place a forward-only trigger at a small t' value
    (fires only on R-to-L approach, catches the front plate early) and
    a separate reverse-only trigger at a large t' value (fires only on
    L-to-R departure, catches the rear plate early), without one
    direction's snaps consuming the other direction's budget.
    """

    polygon_frac: list[tuple[float, float]]
    trigger_t_prime: list[float]
    t_usable_frac: tuple[float, float] = (0.0, 1.0)
    trigger_directions: list[TriggerDirection] | None = None

    # Pipeline (continuous in-band) firing knobs. Set to 0 to disable
    # (preserves the pre-pipeline trigger-only behaviour). When > 0,
    # the planner fires a snap every ``pipeline_interval_ms`` of
    # wall-clock time that a track spends inside the polygon's usable
    # t-band, capped at ``pipeline_max_per_track`` per track. The
    # snapshotter's ``max_concurrent`` is still the cross-track
    # throughput cap; pipeline fires that arrive while the snapshotter
    # is saturated are dropped, not queued.
    pipeline_interval_ms: int = 0
    pipeline_max_per_track: int = 10


@dataclass(slots=True)
class SnapPlannerConfig:
    """Tunable knobs. Defaults match the live Nano deployment."""

    # Minimum bbox area as a fraction of frame area before a frame
    # counts as a fire candidate.
    area_threshold_frac: float = 0.05

    # Hard cap on snaps committed per track. Each fire is an HTTP
    # round-trip that queues on the snapshotter. In road_gate mode the
    # number of triggers IS the per-track cap; this knob is only used
    # for the legacy and right_half_only modes.
    max_per_track: int = 1

    # Minimum variance-of-Laplacian on the bbox-centre crop. 0 disables.
    # Blurred frames are skipped, not counted against budget.
    min_sharpness: float = 0.0

    # Legacy-mode triggers (ignored when right_half_only=True).
    decay_ratio: float = 0.92
    plateau_frames: int = 20
    max_wait_frames: int = 90
    exit_margin_frac: float = 0.04

    # After a fire, suppress new triggers for this many frames so the
    # decay tail doesn't re-peak.
    post_fire_cooldown_frames: int = 30

    # Right-half spatial gate (used only when road_gate is None).
    right_half_only: bool = True

    # Road polygon + axis triggers. When set, supersedes both
    # right_half_only and the legacy peak/decay path.
    road_gate: RoadGateConfig | None = None


@dataclass(slots=True)
class SnapDecision:
    """Per-frame decision. ``snap_index`` is the 1-based fire ordinal
    (matches the ``vehicle_<id>_main_<N>.jpg`` filename convention) and
    is only populated when ``should_fire`` is True."""

    should_fire: bool
    reason: SnapReason
    score: float
    snap_index: int | None = None


@dataclass(slots=True)
class _TrackState:
    """Per-track scheduler state. Internal."""

    fires_committed: int = 0
    peak_score: float = 0.0
    frame_idx_at_peak: int = -1
    frames_eligible: int = 0
    last_bbox: tuple[float, float, float, float] | None = None
    cooldown_until_frame: int = -1
    # Sticky once True: needed so finalize-fallback only fires for
    # tracks that visited the plate-readable zone at some point.
    ever_in_right_half: bool = False
    # Zone indices already fired (right_half_only mode only). The
    # right half is split into ``max_per_track`` equal-width zones;
    # each fires at most once.
    zones_fired: set[int] = field(default_factory=set)
    # Road-gate mode: trigger indices that have already fired for this
    # track, and the previous frame's t' value (None on first frame
    # inside the polygon -- crossings need two consecutive samples).
    triggers_fired: set[int] = field(default_factory=set)
    prev_t_prime: float | None = None
    ever_in_road_gate: bool = False
    # Pipeline-mode (continuous in-band) state. ``pipeline_fires`` is
    # incremented per opportunistic fire; ``last_pipeline_submit_wall_ms``
    # is the wall-clock timestamp of the most recent pipeline fire (0
    # before the first). ``in_band_since_wall_ms`` tracks the entry
    # timestamp for the current contiguous span in the usable t-band;
    # ``time_in_band_ms`` is the cumulative time across all spans (only
    # closed spans contribute -- the open span is added at finalize).
    pipeline_fires: int = 0
    last_pipeline_submit_wall_ms: float = 0.0
    in_band_since_wall_ms: float | None = None
    time_in_band_ms: float = 0.0


# ----------------------------------------------------------------------
# Pure scoring helpers (used in legacy mode and by tests).


def compute_quality_score(
    bbox: tuple[float, float, float, float],
    frame_size: tuple[int, int],
    sharpness: float | None = None,
) -> float:
    """Quality score in ``[0, 1]`` for a single bbox observation.

    Combines area fraction (saturating at 25 % frame coverage) with
    variance-of-Laplacian sharpness (saturating at 100). Position is
    deliberately not in the score -- an edge-clipped bbox already loses
    area, so the bias is captured by the area term alone.
    """
    x1, y1, x2, y2 = bbox
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw == 0 or bh == 0:
        return 0.0
    fw, fh = frame_size
    frame_area = max(1.0, float(fw) * float(fh))
    area_frac = (bw * bh) / frame_area
    area_score = min(1.0, area_frac / 0.25)
    sharp_score = 1.0 if sharpness is None else min(1.0, max(0.0, sharpness) / 100.0)
    return area_score * sharp_score


def is_exit_imminent(
    bbox: tuple[float, float, float, float],
    prev_bbox: tuple[float, float, float, float] | None,
    frame_size: tuple[int, int],
    exit_margin_frac: float,
) -> bool:
    """True iff the bbox is hugging the frame edge in its motion direction."""
    fw, fh = frame_size
    margin_x = exit_margin_frac * fw
    margin_y = exit_margin_frac * fh
    x1, y1, x2, y2 = bbox

    if prev_bbox is None:
        return (
            x1 < margin_x * 2
            or y1 < margin_y * 2
            or x2 > fw - margin_x * 2
            or y2 > fh - margin_y * 2
        )

    px1, py1, px2, py2 = prev_bbox
    dx = (x1 + x2) * 0.5 - (px1 + px2) * 0.5
    dy = (y1 + y2) * 0.5 - (py1 + py2) * 0.5

    if dx > 0 and x2 > fw - margin_x:
        return True
    if dx < 0 and x1 < margin_x:
        return True
    if dy > 0 and y2 > fh - margin_y:
        return True
    if dy < 0 and y1 < margin_y:  # noqa: SIM103
        return True
    return False


def _bbox_area_frac(
    bbox: tuple[float, float, float, float],
    frame_size: tuple[int, int],
) -> float:
    x1, y1, x2, y2 = bbox
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    fw, fh = frame_size
    frame_area = max(1.0, float(fw) * float(fh))
    return (bw * bh) / frame_area


def right_half_zone_index(bbox_center_x: float, frame_w: int, num_zones: int) -> int:
    """Zone index in ``[0, num_zones-1]`` for a bbox centre inside the
    right half. Zone 0 is adjacent to the midline; ``num_zones - 1`` is
    adjacent to the right edge. Values just inside the midline map to
    zone 0; values past the right edge clamp to the last zone."""
    if num_zones <= 1:
        return 0
    half_w = frame_w * 0.5
    if half_w <= 0:
        return 0
    rel = (bbox_center_x - half_w) / half_w
    if rel < 0.0:
        rel = 0.0
    if rel > 0.9999999:
        rel = 0.9999999
    return int(rel * num_zones)


def _point_in_polygon(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test. Polygon vertices
    are in the same coordinate system as ``(px, py)``."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


@dataclass(slots=True)
class RoadGate:
    """Geometry helper for road-polygon + axis-trigger mode.

    Constructed once per session from a ``RoadGateConfig`` plus the
    frame dimensions. Caches the polygon in pixel coords, the polygon
    centroid, the PCA principal axis, and the t-range conversion
    needed to map an image-space point to a normalised ``t'`` value
    along the usable portion of the road.

    The PCA is computed in pixel coords (not fractional) because the
    eigenvectors depend on the metric and the planner reasons about
    points in pixel space. The polygon vertices in the config are
    fractional so the same config works at any source resolution.
    """

    polygon_px: list[tuple[float, float]]
    axis_x: float
    axis_y: float  # unit vector along the road, pointing toward "near"
    centroid_x: float
    centroid_y: float
    t_min_raw: float
    t_max_raw: float
    t_usable_lo: float
    t_usable_hi: float
    triggers_t_prime: tuple[float, ...]
    trigger_directions: tuple[TriggerDirection, ...]

    @classmethod
    def from_config(cls, cfg: RoadGateConfig, frame_w: int, frame_h: int) -> RoadGate:
        if len(cfg.polygon_frac) < 3:
            raise ValueError("road polygon needs at least 3 vertices")
        if not cfg.trigger_t_prime:
            raise ValueError("road gate has no triggers")
        if cfg.trigger_directions is None:
            directions: tuple[TriggerDirection, ...] = tuple("both" for _ in cfg.trigger_t_prime)
        else:
            if len(cfg.trigger_directions) != len(cfg.trigger_t_prime):
                raise ValueError(
                    "trigger_directions length must match trigger_t_prime length "
                    f"({len(cfg.trigger_directions)} vs {len(cfg.trigger_t_prime)})"
                )
            for d in cfg.trigger_directions:
                if d not in _VALID_DIRECTIONS:
                    raise ValueError(
                        f"invalid trigger direction {d!r}; expected one of {_VALID_DIRECTIONS}"
                    )
            directions = tuple(cfg.trigger_directions)
        # Polygon in pixel coords for the active frame size.
        poly_px = [(fx * frame_w, fy * frame_h) for fx, fy in cfg.polygon_frac]
        # Centroid + PCA principal axis on pixel coords.
        n = len(poly_px)
        cx = sum(p[0] for p in poly_px) / n
        cy = sum(p[1] for p in poly_px) / n
        # 2x2 covariance over the vertex set.
        sxx = syy = sxy = 0.0
        for x, y in poly_px:
            dx = x - cx
            dy = y - cy
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy
        sxx /= n
        syy /= n
        sxy /= n
        # Largest eigenvalue of [[sxx, sxy], [sxy, syy]] and its eigenvector.
        tr = sxx + syy
        det = sxx * syy - sxy * sxy
        disc = max(0.0, (tr * tr) * 0.25 - det)
        lam = tr * 0.5 + disc**0.5
        # Eigenvector: solve (sxx - lam) ex + sxy ey = 0.
        if abs(sxy) > 1e-9:
            ex, ey = lam - syy, sxy
        else:
            # Diagonal covariance: principal axis is whichever axis has
            # the larger variance.
            ex, ey = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
        norm = (ex * ex + ey * ey) ** 0.5
        ex, ey = ex / norm, ey / norm
        # Orient axis so it points toward "near" (image y increases
        # downward; in our camera setup the closest road segment has
        # the largest y). Flip if the axis points the wrong way.
        if ey < 0:
            ex, ey = -ex, -ey
        # Project each polygon vertex onto the axis to find t-range.
        ts = [(x - cx) * ex + (y - cy) * ey for x, y in poly_px]
        t_min = min(ts)
        t_max = max(ts)
        t_usable_lo, t_usable_hi = cfg.t_usable_frac
        if not 0.0 <= t_usable_lo < t_usable_hi <= 1.0:
            raise ValueError(
                f"t_usable_frac must satisfy 0 <= lo < hi <= 1, got {cfg.t_usable_frac}"
            )
        return cls(
            polygon_px=poly_px,
            axis_x=ex,
            axis_y=ey,
            centroid_x=cx,
            centroid_y=cy,
            t_min_raw=t_min,
            t_max_raw=t_max,
            t_usable_lo=t_usable_lo,
            t_usable_hi=t_usable_hi,
            triggers_t_prime=tuple(cfg.trigger_t_prime),
            trigger_directions=directions,
        )

    def contains(self, px: float, py: float) -> bool:
        return _point_in_polygon(px, py, self.polygon_px)

    def t_prime(self, px: float, py: float) -> float | None:
        """Map an image-space point to a normalised ``t'`` value in
        ``[0, 1]`` over the usable t-range. Returns None if the point's
        projection lies outside the usable band."""
        t_raw = (px - self.centroid_x) * self.axis_x + (py - self.centroid_y) * self.axis_y
        # Normalise to [0, 1] over the full polygon's t-range first.
        span = self.t_max_raw - self.t_min_raw
        if span <= 0:
            return None
        t_norm = (t_raw - self.t_min_raw) / span
        if t_norm < self.t_usable_lo or t_norm > self.t_usable_hi:
            return None
        usable_span = self.t_usable_hi - self.t_usable_lo
        return (t_norm - self.t_usable_lo) / usable_span

    def crossings(self, prev_tp: float | None, cur_tp: float, already_fired: set[int]) -> list[int]:
        """Return trigger indices whose t' value lies strictly between
        ``prev_tp`` and ``cur_tp``, filtered by each trigger's direction
        tag and excluding any already in ``already_fired``. Order is the
        trigger crossing order along the trajectory (closest-to-prev
        first).

        A ``"forward"`` trigger only fires when ``cur_tp > prev_tp``;
        a ``"reverse"`` trigger only when ``cur_tp < prev_tp``;
        ``"both"`` fires regardless. Stationary frames
        (``cur_tp == prev_tp``) never fire any trigger."""
        if prev_tp is None:
            return []
        if cur_tp > prev_tp:
            motion: TriggerDirection = "forward"
            lo, hi = prev_tp, cur_tp
        elif cur_tp < prev_tp:
            motion = "reverse"
            lo, hi = cur_tp, prev_tp
        else:
            return []
        hits: list[tuple[float, int]] = []
        for idx, t in enumerate(self.triggers_t_prime):
            if idx in already_fired:
                continue
            direction = self.trigger_directions[idx]
            if direction != "both" and direction != motion:
                continue
            if lo < t <= hi:
                # Distance from the previous sample, so ordering matches
                # actual crossing time even when moving in reverse.
                hits.append((abs(t - prev_tp), idx))
        hits.sort()
        return [idx for _, idx in hits]

    @property
    def max_per_track(self) -> int:
        return len(self.triggers_t_prime)


# ----------------------------------------------------------------------
# Planner: per-session decision engine.


@dataclass(slots=True)
class SnapPlanner:
    """Per-frame decision engine. One instance per session.

    Usage::

        planner = SnapPlanner(frame_w, frame_h)
        for tr in active_tracks:
            decision = planner.consider(tr.id, tr.bbox, frame_idx,
                                        sharpness=measure_sharpness(...))
            if decision.should_fire:
                fire_snapshot(tr, decision.snap_index)
        for expired in expired_this_frame:
            decision = planner.on_track_finalize(expired.id)
            if decision.should_fire:
                fire_snapshot(expired, decision.snap_index)
            planner.forget(expired.id)
    """

    frame_width: int
    frame_height: int
    config: SnapPlannerConfig = field(default_factory=SnapPlannerConfig)
    _state: dict[int, _TrackState] = field(default_factory=dict, init=False, repr=False)
    _road_gate: RoadGate | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.road_gate is not None:
            self._road_gate = RoadGate.from_config(
                self.config.road_gate, self.frame_width, self.frame_height
            )

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.frame_width, self.frame_height)

    @property
    def road_gate(self) -> RoadGate | None:
        return self._road_gate

    def consider(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        frame_idx: int,
        sharpness: float | None = None,
    ) -> SnapDecision:
        """Decide whether to fire a snap for ``track_id`` on this frame."""
        cfg = self.config
        st = self._state.get(track_id)
        if st is None:
            st = _TrackState()
            self._state[track_id] = st

        prev_bbox = st.last_bbox
        st.last_bbox = bbox

        score = compute_quality_score(bbox, self.frame_size, sharpness)
        area_frac = _bbox_area_frac(bbox, self.frame_size)

        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) * 0.5
        bbox_center_y = (y1 + y2) * 0.5
        in_right_half = bbox_center_x >= (self.frame_width * 0.5)
        if in_right_half:
            st.ever_in_right_half = True

        # Road-gate mode supersedes the other paths entirely. Area
        # threshold + cooldown still apply as basic sanity checks.
        if self._road_gate is not None:
            return self._consider_road_gate(
                st,
                frame_idx,
                sharpness,
                score,
                area_frac,
                bbox_center_x,
                bbox_center_y,
            )

        if area_frac < cfg.area_threshold_frac:
            # Transition-rescue: a track that was eligible last frame,
            # never fired live, and is now dropping below threshold
            # would otherwise wait for finalize -- by which time the
            # vehicle has left the frame. Fire once on the eligible to
            # ineligible transition instead. In right_half_only mode
            # the rescue is itself gated on the track having been
            # observed in the right half.
            should_rescue = (
                st.frames_eligible > 0
                and st.fires_committed == 0
                and st.peak_score > 0
                and frame_idx >= st.cooldown_until_frame
                and (not cfg.right_half_only or st.ever_in_right_half)
            )
            st.frames_eligible = 0
            if should_rescue:
                st.fires_committed += 1
                st.cooldown_until_frame = frame_idx + cfg.post_fire_cooldown_frames
                return SnapDecision(
                    True,
                    "exit_imminent",
                    score,
                    st.fires_committed + st.pipeline_fires,
                )
            return SnapDecision(False, "ineligible", score)

        st.frames_eligible += 1
        in_cooldown = frame_idx < st.cooldown_until_frame

        if score > st.peak_score:
            st.peak_score = score
            st.frame_idx_at_peak = frame_idx

        frames_since_peak = frame_idx - st.frame_idx_at_peak

        if in_cooldown:
            return SnapDecision(False, "tracking", score)

        # Spatial-gate fast path: split the right half into
        # ``max_per_track`` equal-width zones and fire ASAP on the
        # first eligible frame in each zone the track enters. Each
        # zone fires at most once, so an oscillating tracker can't
        # re-fire the same one. The peak/decay wait is skipped --
        # the spatial gate is the "wait for the right moment" signal.
        if cfg.right_half_only:
            if not in_right_half:
                return SnapDecision(False, "out_of_gate", score)
            if st.fires_committed >= cfg.max_per_track:
                return SnapDecision(False, "budget_exhausted", score)
            zone = right_half_zone_index(bbox_center_x, self.frame_width, cfg.max_per_track)
            if zone in st.zones_fired:
                return SnapDecision(False, "tracking", score)
            if cfg.min_sharpness > 0 and (sharpness is None or sharpness < cfg.min_sharpness):
                return SnapDecision(False, "blur_gate", score)
            st.fires_committed += 1
            st.zones_fired.add(zone)
            st.cooldown_until_frame = frame_idx + cfg.post_fire_cooldown_frames
            st.peak_score = 0.0
            st.frame_idx_at_peak = frame_idx
            return SnapDecision(
                True,
                "right_half_entry",
                score,
                st.fires_committed + st.pipeline_fires,
            )

        # Legacy peak/decay triggers.
        reason: SnapReason | None = None
        if is_exit_imminent(bbox, prev_bbox, self.frame_size, cfg.exit_margin_frac):
            reason = "exit_imminent"
        elif st.peak_score > 0 and score < st.peak_score * cfg.decay_ratio:
            reason = "peak_decay"
        elif frames_since_peak >= cfg.plateau_frames:
            reason = "plateau"
        elif st.frames_eligible >= cfg.max_wait_frames:
            reason = "wait_timeout"

        if reason is None:
            return SnapDecision(False, "tracking", score)
        if st.fires_committed >= cfg.max_per_track:
            return SnapDecision(False, "budget_exhausted", score)
        if cfg.min_sharpness > 0 and (sharpness is None or sharpness < cfg.min_sharpness):
            return SnapDecision(False, "blur_gate", score)

        st.fires_committed += 1
        st.cooldown_until_frame = frame_idx + cfg.post_fire_cooldown_frames
        st.peak_score = 0.0
        st.frame_idx_at_peak = frame_idx
        return SnapDecision(True, reason, score, st.fires_committed + st.pipeline_fires)

    def _consider_road_gate(
        self,
        st: _TrackState,
        frame_idx: int,
        sharpness: float | None,
        score: float,
        area_frac: float,
        cx: float,
        cy: float,
    ) -> SnapDecision:
        """Road-polygon + axis-crossing trigger evaluation.

        Outside the polygon or outside the usable t-band, reset
        ``prev_t_prime`` so a re-entry starts fresh (we can't claim a
        "crossing" across a gap of unseen frames). Inside, update
        ``prev_t_prime`` and check whether any not-yet-fired trigger
        sits between the previous and current t' values. At most one
        trigger fires per frame; if multiple are crossed in a single
        large step, ``prev_t_prime`` is advanced only to the fired
        trigger's position so the next frame's crossing test still
        finds the remaining triggers.
        """
        assert self._road_gate is not None
        cfg = self.config
        gate = self._road_gate

        if not gate.contains(cx, cy):
            st.prev_t_prime = None
            return SnapDecision(False, "outside_road_polygon", score)

        cur_tp = gate.t_prime(cx, cy)
        if cur_tp is None:
            # Inside the polygon but outside the usable t-band (e.g.
            # the distant tip we trimmed).
            st.prev_t_prime = None
            return SnapDecision(False, "out_of_gate", score)

        st.ever_in_road_gate = True

        if area_frac < cfg.area_threshold_frac:
            # Track is in the gate but the bbox is too small to be
            # useful. Hold prev_t_prime so the next sufficiently-large
            # frame still produces a meaningful crossing test.
            return SnapDecision(False, "ineligible", score)

        prev_tp = st.prev_t_prime
        if prev_tp is None:
            st.prev_t_prime = cur_tp
            return SnapDecision(False, "tracking", score)

        crossings = gate.crossings(prev_tp, cur_tp, st.triggers_fired)
        if not crossings:
            st.prev_t_prime = cur_tp
            return SnapDecision(False, "tracking", score)

        if frame_idx < st.cooldown_until_frame:
            # Hold prev_t_prime; the trigger stays detectable next frame.
            return SnapDecision(False, "tracking", score)
        if st.fires_committed >= gate.max_per_track:
            st.prev_t_prime = cur_tp
            return SnapDecision(False, "budget_exhausted", score)
        if cfg.min_sharpness > 0 and (sharpness is None or sharpness < cfg.min_sharpness):
            return SnapDecision(False, "blur_gate", score)

        trig_idx = crossings[0]
        st.fires_committed += 1
        st.triggers_fired.add(trig_idx)
        st.cooldown_until_frame = frame_idx + cfg.post_fire_cooldown_frames
        # Advance prev to the fired trigger position (not cur_tp) so
        # any further triggers in the same forward motion are still
        # detected on subsequent frames.
        st.prev_t_prime = gate.triggers_t_prime[trig_idx]
        return SnapDecision(True, "trigger_crossing", score, st.fires_committed + st.pipeline_fires)

    def consider_pipeline(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        frame_idx: int,
        wall_ms: float,
        sharpness: float | None = None,
    ) -> SnapDecision:
        """Pipeline-mode opportunistic fire while in the usable t-band.

        Independent decision path from :meth:`consider`. Designed to be
        called *after* ``consider()`` on the same frame so trigger
        crossings retain priority on the snap_index ordering. Returns
        a ``SnapDecision`` with reason ``"pipeline"`` (fire),
        ``"pipeline_throttled"`` (interval not yet elapsed since last
        pipeline fire for this track), ``"pipeline_budget_exhausted"``
        (per-track pipeline cap reached), ``"blur_gate"`` (sharpness
        below ``min_sharpness``), or one of ``"out_of_gate"`` /
        ``"outside_road_polygon"`` when geometry fails. Unlike
        :meth:`consider`, this path:

        * Ignores ``area_threshold_frac`` -- a small sub-stream bbox
          still produces a full 4K main snap that ANPR can read.
        * Ignores ``post_fire_cooldown_frames`` -- the per-track
          interval is the throttle.
        * Doesn't share the trigger budget -- ``pipeline_fires`` is
          counted separately from ``fires_committed``.

        Disabled when ``road_gate is None`` or
        ``road_gate.pipeline_interval_ms <= 0``.
        """
        del frame_idx  # accepted for symmetry with consider(); not used
        cfg = self.config
        gate_cfg = cfg.road_gate
        if self._road_gate is None or gate_cfg is None:
            return SnapDecision(False, "out_of_gate", 0.0)
        interval = gate_cfg.pipeline_interval_ms
        if interval <= 0:
            return SnapDecision(False, "out_of_gate", 0.0)

        st = self._state.get(track_id)
        if st is None:
            st = _TrackState()
            self._state[track_id] = st

        x1, _y1, x2, _y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5

        gate = self._road_gate
        if not gate.contains(cx, cy):
            self._close_in_band_span(st, wall_ms)
            return SnapDecision(False, "outside_road_polygon", 0.0)
        cur_tp = gate.t_prime(cx, cy)
        if cur_tp is None:
            self._close_in_band_span(st, wall_ms)
            return SnapDecision(False, "out_of_gate", 0.0)

        if st.in_band_since_wall_ms is None:
            st.in_band_since_wall_ms = wall_ms

        if (
            st.last_pipeline_submit_wall_ms > 0.0
            and (wall_ms - st.last_pipeline_submit_wall_ms) < interval
        ):
            return SnapDecision(False, "pipeline_throttled", 0.0)

        if st.pipeline_fires >= gate_cfg.pipeline_max_per_track:
            return SnapDecision(False, "pipeline_budget_exhausted", 0.0)

        if cfg.min_sharpness > 0 and (sharpness is None or sharpness < cfg.min_sharpness):
            return SnapDecision(False, "blur_gate", 0.0)

        st.pipeline_fires += 1
        st.last_pipeline_submit_wall_ms = wall_ms
        # Filename ordinal spans both pools so trigger and pipeline
        # captures never collide on a single track's _main_N.jpg path.
        snap_index = st.fires_committed + st.pipeline_fires
        return SnapDecision(True, "pipeline", 0.0, snap_index)

    @staticmethod
    def _close_in_band_span(st: _TrackState, wall_ms: float) -> None:
        """Accumulate a completed in-band span into ``time_in_band_ms``."""
        if st.in_band_since_wall_ms is not None:
            st.time_in_band_ms += max(0.0, wall_ms - st.in_band_since_wall_ms)
            st.in_band_since_wall_ms = None

    def track_pipeline_stats(self, track_id: int, wall_ms: float) -> dict[str, float | int] | None:
        """Per-track pipeline counters, for session-end distributions.

        Must be called BEFORE :meth:`forget`. Closes any still-open
        in-band span using ``wall_ms`` so the returned ``time_in_band_ms``
        reflects the track's full visible history. Returns ``None`` if
        no state exists for ``track_id``.
        """
        st = self._state.get(track_id)
        if st is None:
            return None
        self._close_in_band_span(st, wall_ms)
        return {
            "fires_total": st.fires_committed + st.pipeline_fires,
            "trigger_fires": st.fires_committed,
            "pipeline_fires": st.pipeline_fires,
            "time_in_band_ms": st.time_in_band_ms,
        }

    def on_track_finalize(self, track_id: int) -> SnapDecision:
        """Last-chance fire for a track that ended without any snap.

        Behaviour depends on the active firing mode:
        * Road-gate mode has no fallback: a track that never crossed
          a trigger line doesn't earn a snap (firing at an arbitrary
          mid-zone t-value would defeat the operator-traced design).
        * ``right_half_only`` mode requires the track to have visited
          the right half at some point -- otherwise the fallback would
          emit exactly the left-half snap the spatial gate exists to
          prevent.
        * Legacy mode (no spatial gate) fires the fallback whenever
          the track was eligible at some point.
        """
        st = self._state.get(track_id)
        if st is None:
            return SnapDecision(False, "ineligible", 0.0)
        if st.fires_committed > 0:
            return SnapDecision(False, "budget_exhausted", st.peak_score)
        if self._road_gate is not None:
            return SnapDecision(False, "out_of_gate", st.peak_score)
        if st.peak_score == 0:
            return SnapDecision(False, "ineligible", 0.0)
        if self.config.right_half_only and not st.ever_in_right_half:
            return SnapDecision(False, "out_of_gate", st.peak_score)
        st.fires_committed += 1
        return SnapDecision(
            True,
            "finalize_last_chance",
            st.peak_score,
            st.fires_committed + st.pipeline_fires,
        )

    def forget(self, track_id: int) -> None:
        """Drop state for ``track_id``. Call after a track is finalized."""
        self._state.pop(track_id, None)

    def state_snapshot(self) -> dict[int, dict[str, float | int | bool]]:
        """Read-only dict view of per-track state -- for live logging."""
        return {
            tid: {
                "fires": s.fires_committed,
                "peak": s.peak_score,
                "frames_eligible": s.frames_eligible,
                "ever_in_right_half": s.ever_in_right_half,
                "ever_in_road_gate": s.ever_in_road_gate,
                "triggers_fired_count": len(s.triggers_fired),
            }
            for tid, s in self._state.items()
        }
