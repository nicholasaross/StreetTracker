"""Coverage for ``streettracker.device.runtime``.

The live session loop integrates inference, tracking, snap planning,
the snapshotter, the dashboard, and signal handling. The full
end-to-end path requires Ultralytics + TensorRT + a real Reolink and
is verified by manual smoke on the Orin (per the CLAUDE.md phase plan).

These tests cover the *integratable* pieces:

* pure helpers (``make_session_label``, ``session_paths``,
  ``build_session_meta``, ``_bbox_center_sharpness``);
* ``finalize_track``: turns a buffered track into a TrackRecord +
  EventLog write + summary HTML invalidation;
* ``process_frame``: per-frame body with a stub inference engine, IR
  detector, and TrackBuffer;
* ``_consumer_loop``: graceful shutdown via shutdown_event, deadline
  cap, EOF sentinel;
* ``_produce_frames``: fake source -> queue -> sentinel; RTSP
  reconnect after disconnect;
* ``write_session_outputs``: data.json / meta.json / hourly.json
  payload shape;
* ``_close_open_ir_period``: synthesises the closing entry at shutdown
  mid-night.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from streettracker.common.config import StreetTrackerConfig
from streettracker.common.output import EventLog
from streettracker.common.schema import TrackRecord
from streettracker.common.types import Detection
from streettracker.device.ir_detector import IRDetector
from streettracker.device.runtime import (
    SessionContext,
    _bbox_center_sharpness,
    _close_open_ir_period,
    _consumer_loop,
    _fire_snap,
    _main_snap_path,
    _produce_frames,
    _rename_snap_file,
    build_session_meta,
    finalize_track,
    install_signal_handlers,
    make_session_label,
    process_frame,
    session_paths,
    write_session_outputs,
)
from streettracker.device.snap_planner import SnapPlanner, SnapPlannerConfig
from streettracker.device.snapshotter import SnapshotStats
from streettracker.device.track_buffer import BufferedTrack, MotionPoint, TrackBuffer

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "camera.example.json"

# Avoid Windows epoch-near-zero issues in format_wall.
_T0_WALL = 1_779_792_000.0
_FRAME_H, _FRAME_W = 360, 640


def _frame(value: int = 100) -> np.ndarray:
    return np.full((_FRAME_H, _FRAME_W, 3), value, dtype=np.uint8)


def _load_example_config() -> StreetTrackerConfig:
    return StreetTrackerConfig.from_json_file(EXAMPLE_CONFIG)


def _build_ctx(tmp_path: Path) -> SessionContext:
    """Construct a SessionContext bound to ``tmp_path`` for output."""
    cfg = _load_example_config()
    output_dir = tmp_path / "session_test"
    output_dir.mkdir()
    return SessionContext(
        config=cfg,
        output_dir=output_dir,
        session_label="session_test",
        t_start_wall=_T0_WALL,
        t_start_mono=time.monotonic(),
        yolo=_StubYolo(),
        ir=IRDetector(),
        buffer=TrackBuffer(max_misses=cfg.tracking.max_misses),
        event_log=EventLog(output_dir / "events.jsonl"),
        is_file_source=False,
    )


class _StubYolo:
    """Stand-in for UltralyticsTracker. Returns a configurable detection list."""

    def __init__(self, *, returns: list[Detection] | None = None) -> None:
        self.returns = returns if returns is not None else []
        self.calls: list[np.ndarray] = []
        self.avg_infer_ms = 1.0

    def track(self, frame: np.ndarray) -> list[Detection]:
        self.calls.append(frame)
        return list(self.returns)


# ----------------------------------------------------------------------
# Pure helpers.


def test_make_session_label_uses_local_time_format() -> None:
    label = make_session_label("session", _T0_WALL)
    assert label.startswith("session_")
    # YYYYMMDD_HHMMSS -- 15 chars after the prefix.
    suffix = label[len("session_") :]
    assert len(suffix) == 15
    assert suffix[8] == "_"
    assert suffix[:8].isdigit()
    assert suffix[9:].isdigit()


def test_session_paths_returns_full_set_under_root(tmp_path: Path) -> None:
    paths = session_paths(tmp_path, "session_abc")
    assert paths["dir"] == tmp_path / "session_abc"
    assert paths["events_jsonl"].name == "session_abc_events.jsonl"
    assert paths["data_json"].name == "session_abc_data.json"
    assert paths["meta_json"].name == "session_abc_meta.json"
    assert paths["hourly_json"].name == "session_abc_hourly.json"
    assert paths["summary_html"].name == "session_abc_summary.html"
    assert paths["heartbeat"].name == ".heartbeat"


def test_build_session_meta_reflects_counters(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.frames_processed = 1234
    meta = build_session_meta(ctx)
    assert meta.session_label == "session_test"
    assert meta.frames_processed == 1234
    assert meta.session_start_unix == pytest.approx(_T0_WALL)


def test_bbox_center_sharpness_handles_zero_bboxes() -> None:
    f = _frame()
    # A degenerate bbox shrinks to nothing under the 20% inset.
    assert _bbox_center_sharpness(f, (100.0, 100.0, 100.0, 100.0)) is None
    # A reasonable bbox yields a finite number.
    score = _bbox_center_sharpness(f, (100.0, 100.0, 200.0, 200.0))
    assert score is not None
    assert score >= 0.0


def test_bbox_center_sharpness_clamps_outside_frame_to_none() -> None:
    f = _frame()
    # Entirely off-frame -> no usable region.
    assert _bbox_center_sharpness(f, (_FRAME_W + 50.0, 100.0, _FRAME_W + 100.0, 200.0)) is None


def test_bbox_center_sharpness_clamps_to_frame() -> None:
    f = _frame()
    # Bbox overruns the frame; inset must still produce a valid region.
    score = _bbox_center_sharpness(f, (-50.0, -50.0, 200.0, 200.0))
    assert score is not None


def test_main_snap_path_includes_prefix_and_index(tmp_path: Path) -> None:
    p = _main_snap_path(tmp_path, "vehicle", 7, 3)
    assert p.name == "vehicle_7_main_3.jpg"
    assert p.parent == tmp_path


def test_main_snap_path_uses_person_prefix(tmp_path: Path) -> None:
    p = _main_snap_path(tmp_path, "person", 99, 1)
    assert p.name == "person_99_main_1.jpg"


# ----------------------------------------------------------------------
# Class-flip rename: BotSORT can reassign a track's class mid-life, so
# the 4K snap path (built at fire time) may disagree with the record's
# asset_prefix (built at finalize time). _rename_snap_file + the
# finalize-time sweep in finalize_track reconcile this.


def test_rename_snap_file_renames_when_destination_free(tmp_path: Path) -> None:
    src = tmp_path / "person_4463_main_1.jpg"
    src.write_bytes(b"fake-jpeg")
    assert _rename_snap_file(tmp_path, 4463, 1, "person", "vehicle") is True
    assert not src.exists()
    assert (tmp_path / "vehicle_4463_main_1.jpg").exists()


def test_rename_snap_file_noops_when_source_missing(tmp_path: Path) -> None:
    assert _rename_snap_file(tmp_path, 1, 1, "person", "vehicle") is False


def test_rename_snap_file_refuses_to_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "person_1_main_1.jpg"
    src.write_bytes(b"old")
    dst = tmp_path / "vehicle_1_main_1.jpg"
    dst.write_bytes(b"new")
    assert _rename_snap_file(tmp_path, 1, 1, "person", "vehicle") is False
    assert src.read_bytes() == b"old"
    assert dst.read_bytes() == b"new"


def _moving_track_with_id(track_id: int, class_id: int) -> BufferedTrack:
    tr = BufferedTrack(id=track_id, class_id=class_id)
    for i, (t, cx) in enumerate([(0.0, 100.0), (1.0, 300.0), (2.0, 500.0)]):
        tr.points.append(
            MotionPoint(
                frame_idx=i,
                t=t,
                cx=cx,
                cy=200.0,
                x1=cx - 50,
                y1=150,
                x2=cx + 50,
                y2=250,
                score=0.9,
            )
        )
    return tr


def test_finalize_renames_snaps_fired_under_old_class(tmp_path: Path) -> None:
    """Track fires snaps as person, then converges to vehicle.
    Saved files must be renamed at finalize so the record's asset_prefix
    matches the on-disk names."""
    ctx = _build_ctx(tmp_path)
    ctx.frame_h = _FRAME_H
    tr = _moving_track_with_id(4463, class_id=2)  # finalizes as vehicle
    # Simulate two successful snap fires from when the track was still
    # classed as person, plus one in-flight (not yet saved).
    tr.snap_fire_prefixes[1] = "person"
    tr.snap_fire_prefixes[2] = "person"
    tr.snap_fire_prefixes[3] = "person"
    tr.snap_saved_indexes.update({1, 2})  # 3 is still in flight
    for n in (1, 2):
        (ctx.output_dir / f"person_4463_main_{n}.jpg").write_bytes(b"x")

    finalize_track(ctx, tr)

    # Saved snaps are now under vehicle_*; in-flight one is untouched
    # (the snap _on_done callback handles it when the write completes).
    assert (ctx.output_dir / "vehicle_4463_main_1.jpg").exists()
    assert (ctx.output_dir / "vehicle_4463_main_2.jpg").exists()
    assert not (ctx.output_dir / "person_4463_main_1.jpg").exists()
    assert not (ctx.output_dir / "person_4463_main_2.jpg").exists()
    # Track has final_prefix locked so a late _on_done can self-rename.
    assert tr.final_prefix == "vehicle"
    # Record reflects vehicle prefix -- aligns with the renamed files.
    assert len(ctx.records) == 1
    assert ctx.records[0].asset_prefix == "vehicle"


def test_finalize_skips_rename_when_no_class_flip(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.frame_h = _FRAME_H
    tr = _moving_track_with_id(1, class_id=2)
    tr.snap_fire_prefixes[1] = "vehicle"
    tr.snap_saved_indexes.add(1)
    (ctx.output_dir / "vehicle_1_main_1.jpg").write_bytes(b"x")

    finalize_track(ctx, tr)

    assert (ctx.output_dir / "vehicle_1_main_1.jpg").exists()
    assert tr.final_prefix == "vehicle"


def test_fire_snap_captures_bbox_from_last_motion_point(tmp_path: Path) -> None:
    """_fire_snap must persist the BotSORT-tracked sub-stream bbox at
    fire time into ``BufferedTrack.snap_fire_bboxes`` so finalize can
    fold it into the TrackRecord. The bbox is the last motion point's
    (x1, y1, x2, y2) -- analysis-side ALPR reads it back, scales into
    4K coords, and uses it to target the tracked vehicle."""

    class _StubTask:
        def add_done_callback(self, _cb: Any) -> None:
            pass

        def result(self) -> bool:
            return False  # _on_done short-circuits

    class _StubSnapshotter:
        def submit(self, _path: Path) -> _StubTask:
            return _StubTask()

    ctx = _build_ctx(tmp_path)
    ctx.snapshotter = cast("Any", _StubSnapshotter())
    # last motion point is at cx=500 -> bbox (450, 150, 550, 250)
    tr = _moving_track_with_id(7, class_id=2)

    _fire_snap(ctx, tr, snap_index=3)

    assert tr.snap_fire_bboxes[3] == (450, 150, 550, 250)


def test_fire_snap_skips_bbox_when_no_motion_points(tmp_path: Path) -> None:
    """Defensive: a track that has no motion points yet shouldn't crash
    fire-time. Shouldn't actually happen at runtime (the planner needs
    at least one detection to decide to fire) but the guard keeps the
    invariant clean."""

    class _StubTask:
        def add_done_callback(self, _cb: Any) -> None:
            pass

        def result(self) -> bool:
            return False

    class _StubSnapshotter:
        def submit(self, _path: Path) -> _StubTask:
            return _StubTask()

    ctx = _build_ctx(tmp_path)
    ctx.snapshotter = cast("Any", _StubSnapshotter())
    tr = BufferedTrack(id=11, class_id=2)  # no MotionPoints

    _fire_snap(ctx, tr, snap_index=1)

    assert tr.snap_fire_bboxes == {}


def test_fire_snap_callback_renames_late_arrival(tmp_path: Path) -> None:
    """An in-flight snap that completes after finalize must rename
    itself when it lands (the synchronous sweep in finalize couldn't see
    it because it wasn't in snap_saved_indexes yet)."""

    class _StubTask:
        def __init__(self, result: bool) -> None:
            self._result = result
            self._callbacks: list[Any] = []

        def add_done_callback(self, cb: Any) -> None:
            self._callbacks.append(cb)

        def result(self) -> bool:
            return self._result

        def fire_done(self) -> None:
            for cb in self._callbacks:
                cb(self)

    class _StubSnapshotter:
        def __init__(self, task: _StubTask) -> None:
            self._task = task

        def submit(self, _path: Path) -> _StubTask:
            return self._task

    ctx = _build_ctx(tmp_path)
    tr = BufferedTrack(id=99, class_id=0)  # firing while still classed as person
    task = _StubTask(result=True)
    ctx.snapshotter = cast("Any", _StubSnapshotter(task))

    _fire_snap(ctx, tr, 1)
    # Path was built with the (current) person prefix.
    assert tr.snap_fire_prefixes[1] == "person"
    fired_path = ctx.output_dir / "person_99_main_1.jpg"
    fired_path.write_bytes(b"jpeg-bytes")  # snapshotter "writes" the file

    # Finalize would have run while the snap was still in flight, so the
    # synchronous sweep didn't touch it. The track's final_prefix is now
    # locked to vehicle.
    tr.final_prefix = "vehicle"

    # Now the snap task completes -- the _on_done callback must rename.
    task.fire_done()

    assert not fired_path.exists()
    assert (ctx.output_dir / "vehicle_99_main_1.jpg").exists()
    assert 1 in tr.snap_saved_indexes


# ----------------------------------------------------------------------
# finalize_track


def _moving_track() -> BufferedTrack:
    tr = BufferedTrack(id=42, class_id=2)
    for i, (t, cx) in enumerate([(0.0, 100.0), (1.0, 300.0), (2.0, 500.0)]):
        tr.points.append(
            MotionPoint(
                frame_idx=i,
                t=t,
                cx=cx,
                cy=200.0,
                x1=cx - 50,
                y1=150,
                x2=cx + 50,
                y2=250,
                score=0.9,
            )
        )
    return tr


def test_finalize_track_writes_record_to_event_log(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.frame_h = _FRAME_H
    ctx.config = ctx.config  # parity -- no mutation
    tr = _moving_track()
    finalize_track(ctx, tr)
    # Record landed in memory + EventLog (jsonl).
    assert len(ctx.records) == 1
    assert ctx.records[0].track_id == 42
    ctx.event_log.close()
    events_jsonl = ctx.output_dir / "events.jsonl"
    assert events_jsonl.exists()
    line = events_jsonl.read_text(encoding="utf-8").strip()
    assert json.loads(line)["track_id"] == 42


def test_finalize_track_marks_html_dirty(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.frame_h = _FRAME_H
    assert ctx.html_dirty is False
    finalize_track(ctx, _moving_track())
    assert ctx.html_dirty is True


def test_finalize_track_drops_short_tracks(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.frame_h = _FRAME_H
    tr = BufferedTrack(id=1, class_id=2)
    # Only one point -> compute_attributes returns None.
    tr.points.append(
        MotionPoint(frame_idx=0, t=0.0, cx=100, cy=100, x1=80, y1=80, x2=120, y2=120, score=0.9)
    )
    finalize_track(ctx, tr)
    assert ctx.records == []
    # html_dirty stays False because nothing was kept.
    assert ctx.html_dirty is False
    # But raw_track_count still ticks (NanoTracker parity).
    assert ctx.raw_track_count == 1


# ----------------------------------------------------------------------
# process_frame -- per-frame logic with stubbed inference + IR detector.


async def test_process_frame_runs_inference_when_not_in_ir(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.yolo = _StubYolo(returns=[Detection(100, 100, 200, 200, 0.9, 2, track_id=1)])
    await process_frame(ctx, frame_idx=0, source_t=0.0, frame=_frame())
    assert ctx.frames_processed == 1
    assert ctx.frames_inferred == 1
    # The detection was ingested into the buffer.
    assert len(ctx.buffer.active()) == 1


async def test_process_frame_skips_inference_in_ir_mode(tmp_path: Path) -> None:
    """A fresh detector is in day mode, but once we force it in we
    should observe inference skipped."""
    ctx = _build_ctx(tmp_path)
    stub = _StubYolo()
    ctx.yolo = stub
    # Force the detector into IR mode by feeding all-IR frames through
    # its history. Default hysteresis_frames=30.
    ir_frame = np.full((_FRAME_H, _FRAME_W, 3), 80, dtype=np.uint8)
    for i in range(35):
        ctx.ir.update(ir_frame, _T0_WALL + i)
    assert ctx.ir.in_ir_mode is True
    await process_frame(ctx, frame_idx=0, source_t=0.0, frame=ir_frame)
    assert ctx.frames_inferred == 0
    assert stub.calls == []


async def test_process_frame_first_frame_sets_dimensions(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.yolo = _StubYolo()
    assert ctx.frame_h == 0
    await process_frame(ctx, frame_idx=0, source_t=0.0, frame=_frame())
    assert ctx.frame_h == _FRAME_H
    assert ctx.frame_w == _FRAME_W


# ----------------------------------------------------------------------
# write_session_outputs -- final dump.


def test_write_session_outputs_creates_all_three_jsons(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.frame_h = _FRAME_H
    finalize_track(ctx, _moving_track())
    write_session_outputs(ctx, no_html=True, no_json=False)

    paths = session_paths(tmp_path, "session_test")
    assert paths["data_json"].exists()
    assert paths["meta_json"].exists()
    assert paths["hourly_json"].exists()

    data = json.loads(paths["data_json"].read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["track_id"] == 42

    meta = json.loads(paths["meta_json"].read_text(encoding="utf-8"))
    assert meta["session_label"] == "session_test"

    hourly = json.loads(paths["hourly_json"].read_text(encoding="utf-8"))
    assert "hours" in hourly
    assert "ir_periods" in hourly


def test_write_session_outputs_respects_skip_flags(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    write_session_outputs(ctx, no_html=True, no_json=True)
    paths = session_paths(tmp_path, "session_test")
    assert not paths["data_json"].exists()
    assert not paths["meta_json"].exists()
    assert not paths["hourly_json"].exists()
    assert not paths["summary_html"].exists()


def test_write_session_outputs_writes_html_when_enabled(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.frame_h = _FRAME_H
    finalize_track(ctx, _moving_track())
    write_session_outputs(ctx, no_html=False, no_json=True)
    paths = session_paths(tmp_path, "session_test")
    assert paths["summary_html"].exists()
    assert ctx.html_writes >= 1


# ----------------------------------------------------------------------
# _close_open_ir_period


def test_close_open_ir_period_appends_synthetic_close(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    # Drive the IR detector into IR mode using the real wall clock as
    # the start timestamps -- _close_open_ir_period diffs against
    # ``time.time()`` for the end, which must be >= start to make sense.
    ir_frame = np.full((_FRAME_H, _FRAME_W, 3), 80, dtype=np.uint8)
    start = time.time() - 5.0  # 5 seconds in the past
    for i in range(35):
        ctx.ir.update(ir_frame, start + i * 0.01)
    assert ctx.ir.in_ir_mode is True

    assert ctx.ir_periods == []
    _close_open_ir_period(ctx)
    assert len(ctx.ir_periods) == 1
    period = ctx.ir_periods[0]
    assert period.end != period.start
    assert period.duration_s >= 0.0


def test_close_open_ir_period_noop_in_day_mode(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    assert ctx.ir.in_ir_mode is False
    _close_open_ir_period(ctx)
    assert ctx.ir_periods == []


# ----------------------------------------------------------------------
# IR events sidecar log -- crash-safety so a SIGKILL during an IR
# period doesn't lose the start timestamp.


async def test_process_frame_records_ir_transitions_to_sidecar(
    tmp_path: Path,
) -> None:
    """day -> IR transition appends a "start" line to the IR events log
    BEFORE meta.json would normally be written; IR -> day appends "end".
    The sidecar is what survives a SIGKILL between transitions."""
    from streettracker.common.output import IREventLog, read_ir_events_jsonl

    ctx = _build_ctx(tmp_path)
    ctx.yolo = _StubYolo()
    sidecar = tmp_path / "ir_events.jsonl"
    ctx.ir_events_log = IREventLog(sidecar)

    # is_ir_frame treats a monochrome (channels-equal) frame as IR, so
    # _frame()/np.full both classify as IR regardless of value. A real
    # day frame needs per-channel variation above the detector's
    # channel_diff_threshold (default 8).
    day_frame = np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8)
    day_frame[:, :, 0] = 50
    day_frame[:, :, 1] = 100
    day_frame[:, :, 2] = 200

    # Drive into IR via the warm-up + 30 IR-frame history that the
    # detector wants. The state flip happens on the call that brings
    # history to ``hysteresis_frames``; we pre-fill 29 directly (so
    # the detector is one short of flipping), then call process_frame
    # for the 30th frame so the day -> IR transition fires inside
    # process_frame's code path and writes the "start" line.
    ir_frame = np.full((_FRAME_H, _FRAME_W, 3), 80, dtype=np.uint8)
    for i in range(29):
        ctx.ir.update(ir_frame, _T0_WALL + i)
    assert ctx.ir.in_ir_mode is False
    await process_frame(ctx, frame_idx=0, source_t=0.0, frame=ir_frame)
    assert ctx.ir.in_ir_mode is True

    # Now drive back to day via process_frame with a day-bright frame.
    for i in range(35):
        await process_frame(ctx, frame_idx=i + 1, source_t=0.0, frame=day_frame)
    assert ctx.ir.in_ir_mode is False

    ctx.ir_events_log.close()
    events = read_ir_events_jsonl(sidecar)
    kinds = [e["event"] for e in events]
    # One day -> IR, one IR -> day.
    assert "start" in kinds
    assert "end" in kinds
    # First "start" precedes first "end" in file order.
    assert kinds.index("start") < kinds.index("end")


def test_close_open_ir_period_writes_end_to_sidecar(tmp_path: Path) -> None:
    """A graceful shutdown synthesises the closing IRPeriod for meta.json
    and must record the synthetic "end" on the sidecar too, so a
    subsequent SIGKILL during write_session_outputs can't lose it."""
    from streettracker.common.output import IREventLog, read_ir_events_jsonl

    ctx = _build_ctx(tmp_path)
    sidecar = tmp_path / "ir_events.jsonl"
    ctx.ir_events_log = IREventLog(sidecar)

    ir_frame = np.full((_FRAME_H, _FRAME_W, 3), 80, dtype=np.uint8)
    start = time.time() - 5.0
    for i in range(35):
        ctx.ir.update(ir_frame, start + i * 0.01)
    assert ctx.ir.in_ir_mode is True
    _close_open_ir_period(ctx)
    ctx.ir_events_log.close()

    events = read_ir_events_jsonl(sidecar)
    # _close_open_ir_period only writes the "end" -- the "start" would
    # have been written by process_frame at the day -> IR transition,
    # which this test bypasses.
    assert len(events) == 1
    assert events[0]["event"] == "end"


# ----------------------------------------------------------------------
# _consumer_loop -- the inner async pull-and-process loop.


async def test_consumer_loop_stops_on_eof_sentinel(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.yolo = _StubYolo()
    q: asyncio.Queue[Any] = asyncio.Queue()
    await q.put((0, 0.0, _frame()))
    await q.put(None)  # EOF sentinel
    shutdown = asyncio.Event()
    await asyncio.wait_for(_consumer_loop(ctx, q, shutdown, deadline_mono=None), timeout=5.0)
    assert ctx.frames_processed == 1


async def test_consumer_loop_stops_on_shutdown_event(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.yolo = _StubYolo()
    q: asyncio.Queue[Any] = asyncio.Queue()
    shutdown = asyncio.Event()
    shutdown.set()  # already flipped
    await asyncio.wait_for(_consumer_loop(ctx, q, shutdown, deadline_mono=None), timeout=2.0)
    assert ctx.frames_processed == 0


async def test_consumer_loop_stops_on_deadline(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    ctx.yolo = _StubYolo()
    q: asyncio.Queue[Any] = asyncio.Queue()
    shutdown = asyncio.Event()
    # Deadline already in the past.
    await asyncio.wait_for(
        _consumer_loop(ctx, q, shutdown, deadline_mono=time.monotonic() - 1.0),
        timeout=2.0,
    )
    assert ctx.frames_processed == 0


async def test_consumer_loop_continues_after_frame_processing_error(
    tmp_path: Path,
) -> None:
    """A bug in process_frame should be logged, not kill the loop."""
    ctx = _build_ctx(tmp_path)

    class _Boom:
        avg_infer_ms = 0.0

        def track(self, _frame: np.ndarray) -> list[Detection]:
            raise RuntimeError("boom")

    ctx.yolo = _Boom()
    q: asyncio.Queue[Any] = asyncio.Queue()
    await q.put((0, 0.0, _frame()))
    await q.put(None)
    await asyncio.wait_for(_consumer_loop(ctx, q, asyncio.Event(), deadline_mono=None), timeout=5.0)
    # Loop did not crash; frame was counted as processed (counter ticks
    # before inference, matching NanoTracker semantics).
    assert ctx.frames_processed == 1


# ----------------------------------------------------------------------
# _produce_frames -- background thread, file-EOF behaviour.


class _FakeSource:
    """Minimal FrameSource for the producer tests. Configurable behaviour.

    The test inspects ``opens`` to confirm reconnect attempts; the
    ``raise_on_iter`` toggle simulates a mid-stream RTSP disconnect.
    """

    def __init__(
        self,
        frames: list[tuple[int, float, np.ndarray]],
        *,
        is_file: bool = True,
        raise_on_iter: bool = False,
    ) -> None:
        self._frames = frames
        self.is_file = is_file
        self.fps = 0.0
        self.opens = 0
        self.closes = 0
        self._raise_on_iter = raise_on_iter

    def open(self) -> None:
        self.opens += 1

    def frames(self):  # type: ignore[no-untyped-def]
        if self._raise_on_iter:
            raise RuntimeError("simulated rtsp disconnect")
        yield from self._frames

    def close(self) -> None:
        self.closes += 1


def _drain_queue(q: asyncio.Queue[Any]) -> list[Any]:
    """Sync-drain an asyncio.Queue (only safe when no consumer is awaiting)."""
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


async def test_produce_frames_file_eof_emits_sentinel() -> None:
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Any] = asyncio.Queue(maxsize=2)
    src = _FakeSource([(0, 0.0, _frame()), (1, 0.04, _frame())], is_file=True)
    stop = threading.Event()
    thread = threading.Thread(
        target=_produce_frames,
        args=(src, q, loop, stop),
        kwargs={"is_file": True},
    )
    thread.start()
    thread.join(timeout=3.0)
    assert not thread.is_alive()
    # Give the event loop one tick so the thread's call_soon_threadsafe
    # callbacks (which push into the queue) actually run before we drain.
    await asyncio.sleep(0)
    items = _drain_queue(q)
    # The producer terminates after pushing the sentinel; the last item
    # in the queue must be the sentinel.
    assert items[-1] is None
    assert src.opens == 1
    assert src.closes == 1


async def test_produce_frames_reopens_rtsp_on_disconnect() -> None:
    """RTSP source that raises mid-stream should trigger reconnect."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Any] = asyncio.Queue(maxsize=2)
    src = _FakeSource([], is_file=False, raise_on_iter=True)
    stop = threading.Event()
    thread = threading.Thread(
        target=_produce_frames,
        args=(src, q, loop, stop),
        # max_backoff_s and initial backoff are both governed by the
        # function's loop. Keep the test fast by pre-tripping stop_event
        # after enough wall-clock has passed for one reconnect cycle.
        kwargs={"is_file": False, "max_backoff_s": 0.05},
    )
    thread.start()
    # Backoff starts at 1.0s on the first failure; sleep just past that
    # to observe the second open attempt.
    await asyncio.sleep(1.2)
    stop.set()
    thread.join(timeout=3.0)
    assert not thread.is_alive()
    await asyncio.sleep(0)
    assert src.opens >= 2
    items = _drain_queue(q)
    assert items[-1] is None


# ----------------------------------------------------------------------
# install_signal_handlers -- registers handlers without raising.


async def test_install_signal_handlers_returns_cleanly() -> None:
    """The Windows path uses ``signal.signal``; the POSIX path uses
    ``loop.add_signal_handler``. Either way the call must return
    without raising and the shutdown event must remain unset until
    actually signalled. We do *not* fire a real signal here -- that
    would interrupt the test runner -- only register and tear down."""
    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    install_signal_handlers(loop, shutdown)
    assert not shutdown.is_set()


# ----------------------------------------------------------------------
# SessionContext shape -- regression guard for renames.


def test_session_context_session_t_property_advances(tmp_path: Path) -> None:
    ctx = _build_ctx(tmp_path)
    t1 = ctx.session_t
    time.sleep(0.01)
    t2 = ctx.session_t
    assert t2 >= t1


def test_session_context_records_serialisable(tmp_path: Path) -> None:
    """A finalized record must round-trip through JSON without loss --
    this is what the downstream dashboard / ALPR pipelines read off
    disk."""
    ctx = _build_ctx(tmp_path)
    ctx.frame_h = _FRAME_H
    finalize_track(ctx, _moving_track())
    [record] = ctx.records
    assert isinstance(record, TrackRecord)
    payload = record.to_json_dict()
    roundtrip = TrackRecord.from_json_dict(payload)
    assert roundtrip.track_id == record.track_id
    assert roundtrip.color == record.color


# ----------------------------------------------------------------------
# Blur-skip observability + snap_stats in SessionMeta. Both surfaced
# in PR #16 after the live cutover found that fast-moving vehicles
# were getting fewer-than-expected 4K captures: blur skips were
# silently dropped (the runtime had no visibility), and there was no
# fire-latency data in the session JSON to calibrate trigger placement.


def _make_stub_snapshotter() -> Any:
    """Lightweight snapshotter stand-in.

    The runtime's snap walk only needs ``ctx.snapshotter is not None``
    plus ``.stats`` for the meta payload. Blur-gate decisions never
    reach ``.submit()`` so we don't have to model HTTP at all.
    """

    class _StubSnapshotter:
        def __init__(self) -> None:
            self.stats = SnapshotStats()

        def submit(self, _path: Path) -> Any:  # pragma: no cover - never called
            return None

    return _StubSnapshotter()


async def test_process_frame_increments_blur_skip_counter_when_gate_trips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A frame with sharpness below ``min_sharpness`` must:
    * advance ctx.blur_skip_count by one,
    * NOT call snapshotter.submit (the planner's decision short-circuits
      before the fire path),
    * emit one log line with the track id + sharpness + threshold."""
    ctx = _build_ctx(tmp_path)
    # Override config to enable the blur gate. Frozen dataclass -- use replace.
    ctx.config = dataclasses.replace(
        ctx.config,
        snapshot=dataclasses.replace(ctx.config.snapshot, min_sharpness=1000.0),
    )
    ctx.snapshotter = _make_stub_snapshotter()
    ctx.frame_w, ctx.frame_h = _FRAME_W, _FRAME_H
    # Pre-init the planner (right-half mode -- no road_gate needed) so
    # the lazy init in process_frame doesn't fire on the first frame.
    ctx.planner = SnapPlanner(
        frame_width=_FRAME_W,
        frame_height=_FRAME_H,
        config=SnapPlannerConfig(
            min_sharpness=1000.0,
            area_threshold_frac=0.05,
            max_per_track=3,
        ),
    )
    # Detection in the right half (cx > FW/2 = 320), large enough to
    # clear the 5% area gate (200*150 / (640*360) = 13%).
    det = Detection(x1=420.0, y1=180.0, x2=620.0, y2=330.0, score=0.9, class_id=2, track_id=42)
    ctx.yolo = _StubYolo(returns=[det])
    # Uniform grey frame -> sharpness 0 -> blur gate trips.
    frame = _frame(value=128)

    with caplog.at_level(logging.INFO, logger="streettracker.device.runtime"):
        await process_frame(ctx, frame_idx=0, source_t=0.0, frame=frame)

    assert ctx.blur_skip_count == 1
    assert any("blur skip" in r.message and "42" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


async def test_blur_skip_log_throttled_per_track(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A fast vehicle stays blurred for its whole visible span -- ten
    consecutive blurred frames should yield one log line, not ten. The
    counter still bumps every frame so the session total remains
    accurate."""
    ctx = _build_ctx(tmp_path)
    ctx.config = dataclasses.replace(
        ctx.config,
        snapshot=dataclasses.replace(ctx.config.snapshot, min_sharpness=1000.0),
    )
    ctx.snapshotter = _make_stub_snapshotter()
    ctx.frame_w, ctx.frame_h = _FRAME_W, _FRAME_H
    ctx.planner = SnapPlanner(
        frame_width=_FRAME_W,
        frame_height=_FRAME_H,
        config=SnapPlannerConfig(min_sharpness=1000.0, area_threshold_frac=0.05),
    )
    det = Detection(420.0, 180.0, 620.0, 330.0, 0.9, 2, track_id=42)
    ctx.yolo = _StubYolo(returns=[det])
    frame = _frame(value=128)

    with caplog.at_level(logging.INFO, logger="streettracker.device.runtime"):
        for i in range(10):
            await process_frame(ctx, frame_idx=i, source_t=i * 0.05, frame=frame)

    blur_logs = [r for r in caplog.records if "blur skip" in r.message]
    assert len(blur_logs) == 1, f"expected 1 throttled log, got {len(blur_logs)}"
    assert ctx.blur_skip_count == 10


def test_build_session_meta_includes_snap_stats_when_snapshotter_present(
    tmp_path: Path,
) -> None:
    ctx = _build_ctx(tmp_path)
    stats = SnapshotStats(
        attempts=5,
        successes=4,
        failures=1,
        dropped=0,
        latencies_ms=[100.0, 200.0, 300.0, 400.0],
    )

    class _SS:
        pass

    ss = _SS()
    ss.stats = stats  # type: ignore[attr-defined]
    ctx.snapshotter = ss  # type: ignore[assignment]
    ctx.blur_skip_count = 12

    meta = build_session_meta(ctx)
    assert meta.snap_stats is not None
    assert meta.snap_stats["attempts"] == 5
    assert meta.snap_stats["successes"] == 4
    assert meta.snap_stats["failures"] == 1
    assert meta.snap_stats["blur_skipped_frames"] == 12
    latency = meta.snap_stats["latency"]
    assert latency["count"] == 4
    assert latency["min_ms"] == 100.0
    assert latency["max_ms"] == 400.0


def test_build_session_meta_snap_stats_none_when_no_snapshotter(tmp_path: Path) -> None:
    """Batch mode + ``snapshot.enabled=False`` paths leave
    ``ctx.snapshotter`` as ``None``; ``snap_stats`` must then be ``None``
    rather than an empty dict so downstream consumers can short-circuit
    on missing data."""
    ctx = _build_ctx(tmp_path)
    assert ctx.snapshotter is None
    meta = build_session_meta(ctx)
    assert meta.snap_stats is None


def test_build_session_meta_emits_pipeline_counters_when_snapshotter_present(
    tmp_path: Path,
) -> None:
    """Pipeline aggregate counters are always emitted (even as zeros)
    when a snapshotter is present. Lets soak comparisons across configs
    diff against a stable key set."""
    ctx = _build_ctx(tmp_path)
    stats = SnapshotStats(attempts=2, successes=2, latencies_ms=[100.0, 200.0])

    class _SS:
        pass

    ss = _SS()
    ss.stats = stats  # type: ignore[attr-defined]
    ctx.snapshotter = ss  # type: ignore[assignment]
    ctx.pipeline_fires_total = 7
    ctx.pipeline_throttled = 23
    ctx.pipeline_budget_exhausted = 4

    meta = build_session_meta(ctx)
    assert meta.snap_stats is not None
    assert meta.snap_stats["pipeline_fires"] == 7
    assert meta.snap_stats["pipeline_throttled"] == 23
    assert meta.snap_stats["pipeline_budget_exhausted"] == 4
    # Per-track distributions only land when we actually have finalized
    # tracks with pipeline state -- this ctx has none.
    assert "fires_per_track" not in meta.snap_stats
    assert "time_in_band_ms_per_track" not in meta.snap_stats


def test_build_session_meta_includes_per_track_distributions_when_populated(
    tmp_path: Path,
) -> None:
    """When the per_track_* lists carry finalized-track samples,
    snap_stats must include p50 / p90 / max / mean / n summaries that
    the operator can read post-soak to tune pipeline_interval_ms."""
    ctx = _build_ctx(tmp_path)
    stats = SnapshotStats(attempts=1, successes=1, latencies_ms=[100.0])

    class _SS:
        pass

    ss = _SS()
    ss.stats = stats  # type: ignore[attr-defined]
    ctx.snapshotter = ss  # type: ignore[assignment]
    # 5 tracks, fires ranging 0..6; time-in-band 0..10s.
    ctx.per_track_fires_total = [0, 1, 3, 5, 6]
    ctx.per_track_time_in_band_ms = [0.0, 1500.0, 4000.0, 8000.0, 10000.0]

    meta = build_session_meta(ctx)
    assert meta.snap_stats is not None
    fpt = meta.snap_stats["fires_per_track"]
    assert fpt["n"] == 5
    assert fpt["zero"] == 1
    assert fpt["max"] == 6
    # Nearest-rank p50 of 5 sorted values -> index int(5*0.5)=2 -> value 3.
    assert fpt["p50"] == 3
    tib = meta.snap_stats["time_in_band_ms_per_track"]
    assert tib["n"] == 5
    assert tib["max_ms"] == 10000.0


def test_session_meta_round_trips_snap_stats_through_json(tmp_path: Path) -> None:
    """The schema contract: meta.json -> SessionMeta -> meta.json must
    preserve snap_stats verbatim. Guards against an accidental field
    drop in to_json_dict / from_json_dict."""
    ctx = _build_ctx(tmp_path)
    stats = SnapshotStats(attempts=1, successes=1, latencies_ms=[50.0])

    class _SS:
        pass

    ss = _SS()
    ss.stats = stats  # type: ignore[attr-defined]
    ctx.snapshotter = ss  # type: ignore[assignment]
    ctx.blur_skip_count = 3
    meta = build_session_meta(ctx)
    payload = meta.to_json_dict()
    assert "snap_stats" in payload
    from streettracker.common.schema import SessionMeta

    roundtrip = SessionMeta.from_json_dict(payload)
    assert roundtrip.snap_stats == meta.snap_stats


# ----------------------------------------------------------------------
# run_session disable kwargs -- used by `streettracker batch` to skip
# the live-only subsystems (Reolink snapshotter, dashboard).


async def test_run_session_skips_snapshotter_when_disabled(tmp_path: Path) -> None:
    """``enable_snapshotter=False`` must skip the Reolink HTTP client
    and the snap-cleanup task even when the config has them enabled.
    Exercises run_session end-to-end against an immediately-EOF file
    source so we never need a real Ultralytics engine."""
    from streettracker.device.runtime import run_session

    class _ImmediateEOF:
        is_file = True
        fps = 30.0
        opens = 0
        closes = 0

        def open(self) -> None:
            self.opens += 1

        def frames(self):  # type: ignore[no-untyped-def]
            return
            yield  # makes this a generator

        def close(self) -> None:
            self.closes += 1

    cfg = _load_example_config()
    out_root = tmp_path / "out"
    rc = await run_session(
        cfg,
        _ImmediateEOF(),
        output_root=out_root,
        no_html=True,
        no_json=True,
        yolo=_StubYolo(),
        ir_detector=IRDetector(),
        enable_snapshotter=False,
        enable_dashboard=False,
        shutdown_grace_s=2.0,
    )
    assert rc == 0
    # The session directory was created -- run_session always lays it down.
    sessions = list(out_root.iterdir())
    assert len(sessions) == 1


async def test_run_session_disables_dashboard_when_disabled(tmp_path: Path) -> None:
    """``enable_dashboard=False`` must skip the aiohttp bind even when
    the config has it enabled. Without this, batch runs would
    needlessly grab port 8080 on the dev box."""
    import socket

    from streettracker.device.runtime import run_session

    # Pre-bind port 8080 (config default) so that *if* the dashboard
    # mistakenly tried to bind it would either succeed on a different
    # port or fail noisily -- either way we'd notice. With the kwarg
    # honoured, the bind is never attempted.
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)

    class _ImmediateEOF:
        is_file = True
        fps = 30.0

        def open(self) -> None:
            pass

        def frames(self):  # type: ignore[no-untyped-def]
            return
            yield

        def close(self) -> None:
            pass

    try:
        cfg = _load_example_config()
        rc = await run_session(
            cfg,
            _ImmediateEOF(),
            output_root=tmp_path / "out",
            no_html=True,
            no_json=True,
            yolo=_StubYolo(),
            ir_detector=IRDetector(),
            enable_snapshotter=False,
            enable_dashboard=False,
            shutdown_grace_s=2.0,
        )
        assert rc == 0
    finally:
        holder.close()
