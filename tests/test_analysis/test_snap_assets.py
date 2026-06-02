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
    resolve_bbox_hint,
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
