"""aggregate_by_track for the colour classifier: confidence-weighted
per-track voting + the conf gate + the colour_source tag."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from streettracker.analysis.makemodel.colour_infer import aggregate_by_track  # noqa: E402


def _row(tid: int, colour: str | None, conf: float) -> dict:
    return {"track_id": tid, "colour": colour, "conf": conf}


def test_confidence_weighted_vote() -> None:
    # Track 1: two silver reads (0.6 + 0.55 = 1.15) beat one white (0.9).
    rows = [_row(1, "silver", 0.6), _row(1, "white", 0.9), _row(1, "silver", 0.55)]
    [t] = aggregate_by_track(rows, conf_threshold=0.5)
    assert t["track_id"] == 1
    assert t["colour"] == "silver"
    assert t["colour_source"] == "cnn"
    assert t["conf"] == 0.6  # winner's best single read
    assert t["n_reads"] == 3
    assert t["n_high_conf_reads"] == 3


def test_below_threshold_emits_none() -> None:
    rows = [_row(7, "blue", 0.3), _row(7, "blue", 0.4)]
    [t] = aggregate_by_track(rows, conf_threshold=0.5)
    assert t["colour"] is None  # never cleared the gate
    assert t["conf"] == 0.4  # kept for debugging
    assert t["n_high_conf_reads"] == 0


def test_rows_without_bbox_are_dropped() -> None:
    # colour=None rows (no usable bbox) don't seed a track.
    rows = [_row(1, None, 0.0), _row(2, "black", 0.8)]
    tracks = aggregate_by_track(rows, conf_threshold=0.5)
    assert [t["track_id"] for t in tracks] == [2]
