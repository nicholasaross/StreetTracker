"""aggregate_by_track for the body-type classifier: confidence-weighted
per-track voting + the conf gate."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from streettracker.analysis.makemodel.bodytype_infer import aggregate_by_track  # noqa: E402


def _row(tid: int, body_type: str | None, conf: float) -> dict:
    return {"track_id": tid, "body_type": body_type, "conf": conf}


def test_confidence_weighted_vote() -> None:
    # Track 1: two suv reads (0.6 + 0.55 = 1.15) beat one van (0.9).
    rows = [_row(1, "suv", 0.6), _row(1, "van", 0.9), _row(1, "suv", 0.55)]
    [t] = aggregate_by_track(rows, conf_threshold=0.5)
    assert t["track_id"] == 1
    assert t["body_type"] == "suv"
    assert t["conf"] == 0.6  # winner's best single read
    assert t["n_reads"] == 3
    assert t["n_high_conf_reads"] == 3


def test_below_threshold_emits_none() -> None:
    rows = [_row(7, "hatchback", 0.3), _row(7, "hatchback", 0.4)]
    [t] = aggregate_by_track(rows, conf_threshold=0.5)
    assert t["body_type"] is None  # never cleared the gate
    assert t["conf"] == 0.4  # kept for debugging
    assert t["n_high_conf_reads"] == 0


def test_rows_without_bbox_are_dropped() -> None:
    # body_type=None rows (no usable bbox) don't seed a track.
    rows = [_row(1, None, 0.0), _row(2, "hatchback", 0.8)]
    tracks = aggregate_by_track(rows, conf_threshold=0.5)
    assert [t["track_id"] for t in tracks] == [2]
