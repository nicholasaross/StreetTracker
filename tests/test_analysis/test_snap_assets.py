"""Shared session-asset helpers: snap discovery, bbox index, scaling.

No torch dependency (these are pure JSON / PIL-header helpers), so they
run on CI. The bbox→snap-coord scaling is the load-bearing bit ALPR and
make/model both rely on to crop the *tracked* vehicle.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from streettracker.analysis.snap_assets import (
    discover_vehicle_snaps,
    load_bbox_index,
    load_done_bbox_index,
    resolve_bbox_hint,
    resolve_bbox_hint_window,
)


def _session(tmp_path: Path) -> Path:
    sess = tmp_path / "session_x"
    sess.mkdir()
    for name in ("vehicle_1_main_1.jpg", "vehicle_1_main_2.jpg", "person_2_main_1.jpg"):
        Image.new("RGB", (400, 300), (10, 10, 10)).save(sess / name)
    sess.joinpath("session_x_data.json").write_text(
        json.dumps(
            [{"track_id": 1, "main_snaps": [1, 2], "main_snap_bboxes": [[64, 36, 128, 108], None]}]
        )
    )
    sess.joinpath("session_x_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    return sess


def test_discover_filters_to_vehicle_snaps(tmp_path: Path) -> None:
    snaps = discover_vehicle_snaps(_session(tmp_path))
    assert [(p.name, t, n, c) for p, t, n, c in snaps] == [
        ("vehicle_1_main_1.jpg", 1, 1, "vehicle"),
        ("vehicle_1_main_2.jpg", 1, 2, "vehicle"),
    ]


def test_load_bbox_index_skips_none_and_reads_frame_size(tmp_path: Path) -> None:
    idx, sub = load_bbox_index(_session(tmp_path))
    assert sub == (640, 360)
    assert idx[(1, 1)] == (64, 36, 128, 108)
    assert (1, 2) not in idx  # the None-bbox snap is skipped


def test_resolve_scales_substream_bbox_to_snap_pixels(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    idx, sub = load_bbox_index(sess)
    # snap is 400x300, sub-stream 640x360 -> sx=0.625, sy=0.8333
    hint = resolve_bbox_hint(sess / "vehicle_1_main_1.jpg", 1, 1, idx, sub)
    assert hint == (40, 30, 80, 90)


def test_resolve_returns_none_without_bbox_or_size(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    idx, sub = load_bbox_index(sess)
    # snap 2 has a None bbox -> no hint
    assert resolve_bbox_hint(sess / "vehicle_1_main_2.jpg", 1, 2, idx, sub) is None
    # missing sub-stream size -> can't scale safely
    assert resolve_bbox_hint(sess / "vehicle_1_main_1.jpg", 1, 1, idx, None) is None


def test_load_bbox_index_empty_for_older_session(tmp_path: Path) -> None:
    """A session with no data.json (pre-bbox-capture) yields an empty
    index + None size rather than erroring."""
    sess = tmp_path / "session_old"
    sess.mkdir()
    idx, sub = load_bbox_index(sess)
    assert idx == {}
    assert sub is None


# ----------------------------------------------------------------------
# Motion-aware hint window (snap latency vs fire-time bbox).


def _window_session(tmp_path: Path, bboxes: list) -> tuple[Path, dict, tuple]:
    """Session with one R->L-ish track whose per-snap bboxes march left.
    Snap image is 400x300 against a 400x300 'sub-stream' so scaling is
    identity and the window arithmetic is easy to assert."""
    sess = tmp_path / "session_w"
    sess.mkdir()
    Image.new("RGB", (400, 300), (10, 10, 10)).save(sess / "vehicle_7_main_2.jpg")
    sess.joinpath("session_w_data.json").write_text(
        json.dumps(
            [
                {
                    "track_id": 7,
                    "main_snaps": list(range(1, len(bboxes) + 1)),
                    "main_snap_bboxes": bboxes,
                }
            ]
        )
    )
    sess.joinpath("session_w_meta.json").write_text(json.dumps({"frame_size": [400, 300]}))
    idx, sub = load_bbox_index(sess)
    return sess, idx, sub


def test_window_unions_lookahead_bboxes(tmp_path: Path) -> None:
    """Mid-track: the hint covers everywhere the car was actually
    observed over the next ~1.2s of fires, not just the stale fire-time
    box."""
    # Car moving left, 40px per fire: snaps 1..5.
    bboxes = [[200 - 40 * i, 100, 280 - 40 * i, 140] for i in range(5)]
    sess, idx, sub = _window_session(tmp_path, bboxes)

    hint = resolve_bbox_hint_window(sess / "vehicle_7_main_2.jpg", 7, 2, idx, sub, lookahead=3)

    # Union of snaps 2,3,4,5: x from (200-160)=40 .. (280-40)=240.
    assert hint == (40, 100, 240, 140)


def test_window_lookahead_zero_matches_legacy(tmp_path: Path) -> None:
    bboxes = [[200 - 40 * i, 100, 280 - 40 * i, 140] for i in range(5)]
    sess, idx, sub = _window_session(tmp_path, bboxes)
    img = sess / "vehicle_7_main_2.jpg"

    assert resolve_bbox_hint_window(img, 7, 2, idx, sub, lookahead=0) == resolve_bbox_hint(
        img, 7, 2, idx, sub
    )


def test_window_extrapolates_at_end_of_track(tmp_path: Path) -> None:
    """Last snap of the track (the worst case: car leaving the band at
    speed, no later fires exist) -- extend along the last observed step."""
    bboxes = [[240, 100, 320, 140], [200, 100, 280, 140]]  # snap 1 -> 2: dx=-40
    sess, idx, sub = _window_session(tmp_path, bboxes)

    hint = resolve_bbox_hint_window(sess / "vehicle_7_main_2.jpg", 7, 2, idx, sub, lookahead=3)

    # base (200..280) unioned with itself shifted by dx*3 = -120.
    assert hint == (80, 100, 280, 140)


def test_window_extrapolation_shift_is_capped(tmp_path: Path) -> None:
    """A BotSORT glitch step (huge bbox jump) must not blow the window
    up to most of the frame -- the shift caps at 2.5 bbox-dims/axis."""
    bboxes = [[3200, 100, 3280, 140], [200, 100, 280, 140]]  # absurd dx=-3000
    sess, idx, sub = _window_session(tmp_path, bboxes)

    hint = resolve_bbox_hint_window(sess / "vehicle_7_main_2.jpg", 7, 2, idx, sub, lookahead=3)

    # bbox width 80 -> max shift 200: window x1 = 200-200 = 0 (not -9000).
    assert hint == (0, 100, 280, 140)


def test_window_none_without_base_bbox_or_size(tmp_path: Path) -> None:
    bboxes = [[200, 100, 280, 140]]
    sess, idx, sub = _window_session(tmp_path, bboxes)
    img = sess / "vehicle_7_main_2.jpg"

    assert resolve_bbox_hint_window(img, 7, 9, idx, sub, lookahead=3) is None
    assert resolve_bbox_hint_window(img, 7, 1, idx, None, lookahead=3) is None


def test_window_prefers_completion_bbox_when_present(tmp_path: Path) -> None:
    """A completion-time bbox replaces lookahead/extrapolation entirely:
    the window is exactly union(fire bbox, done bbox) -- the car's true
    position in the saved image, no guessing."""
    bboxes = [[200 - 40 * i, 100, 280 - 40 * i, 140] for i in range(5)]
    sess, idx, sub = _window_session(tmp_path, bboxes)
    done = {(7, 2): (40, 110, 120, 150)}

    hint = resolve_bbox_hint_window(
        sess / "vehicle_7_main_2.jpg", 7, 2, idx, sub, lookahead=3, done_index=done
    )

    # snap 2 fire bbox = (160, 100, 240, 140); union with the done bbox
    # -- NOT the lookahead union (which would reach x=40..240 anyway but
    # y stays 100..140; the done bbox extends y to 150).
    assert hint == (40, 100, 240, 150)


def test_window_ignores_done_index_without_entry(tmp_path: Path) -> None:
    """No completion bbox for this snap -> normal lookahead union."""
    bboxes = [[200 - 40 * i, 100, 280 - 40 * i, 140] for i in range(5)]
    sess, idx, sub = _window_session(tmp_path, bboxes)

    hint = resolve_bbox_hint_window(
        sess / "vehicle_7_main_2.jpg", 7, 2, idx, sub, lookahead=3,
        done_index={(7, 99): (0, 0, 10, 10)},
    )

    assert hint == (40, 100, 240, 140)  # same as the no-done-index case


def test_load_done_bbox_index_reads_parallel_field(tmp_path: Path) -> None:
    sess = tmp_path / "session_d"
    sess.mkdir()
    sess.joinpath("session_d_data.json").write_text(
        json.dumps(
            [
                {
                    "track_id": 9,
                    "main_snaps": [1, 2],
                    "main_snap_bboxes": [[10, 10, 30, 30], [50, 10, 70, 30]],
                    "main_snap_bboxes_done": [[12, 11, 32, 31], None],
                }
            ]
        )
    )

    done = load_done_bbox_index(sess)

    assert done == {(9, 1): (12, 11, 32, 31)}


def test_load_done_bbox_index_empty_for_older_sessions(tmp_path: Path) -> None:
    """Sessions written before the field existed yield an empty map --
    callers fall back to fire-time windows."""
    sess, idx, sub = _window_session(
        tmp_path, [[200, 100, 280, 140], [160, 100, 240, 140]]
    )
    assert load_done_bbox_index(sess) == {}


def test_window_scales_to_snap_pixels(tmp_path: Path) -> None:
    """Same independent x/y scaling as the single-bbox resolver."""
    sess = tmp_path / "session_s"
    sess.mkdir()
    Image.new("RGB", (400, 300), (10, 10, 10)).save(sess / "vehicle_1_main_1.jpg")
    sess.joinpath("session_s_data.json").write_text(
        json.dumps(
            [
                {
                    "track_id": 1,
                    "main_snaps": [1, 2],
                    "main_snap_bboxes": [[64, 36, 128, 108], [128, 36, 192, 108]],
                }
            ]
        )
    )
    sess.joinpath("session_s_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    idx, sub = load_bbox_index(sess)

    hint = resolve_bbox_hint_window(sess / "vehicle_1_main_1.jpg", 1, 1, idx, sub, lookahead=3)

    # Union in sub coords: (64,36,192,108); sx=0.625, sy=0.8333.
    assert hint == (40, 30, 120, 90)
