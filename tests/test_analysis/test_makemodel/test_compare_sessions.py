"""makemodel-compare --eval-session: held-out cars from fresh sessions' DVSA
labels, minus every car a compared model trained on. Pure (no torch), so it
runs on CI; the end-to-end CLI test lives in test_compare.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streettracker.analysis.makemodel.compare import session_cars, target_label


def _labels(output_root: Path, sess: str, labels: dict) -> None:
    d = output_root / sess
    d.mkdir(parents=True)
    (d / f"{sess}_dvsa_labels.json").write_text(json.dumps({"labels": labels}))


def _row(make: str, track_ids: list[int], **extra: str) -> dict:
    return {"make": make, "track_ids": track_ids, **extra}


def test_session_cars_skip_trained_and_unlabelled_cars(tmp_path: Path) -> None:
    _labels(
        tmp_path,
        "session_a",
        {
            "AB12CDE": _row("FORD", [1, 2]),
            "TRAINED1": _row("AUDI", [3]),  # in a training corpus: excluded
            "NOTRACK1": _row("KIA", []),  # cleared by the plate gate
            "NOMAKE1": _row("", [4]),
        },
    )
    cars = session_cars(tmp_path, ["session_a"], exclude_cars={"TRAINED1"})
    assert [(c.car, c.label, c.tracks) for c in cars] == [
        ("AB12CDE", "FORD", [("session_a", 1), ("session_a", 2)])
    ]


def test_session_cars_merge_sessions_and_cap_tracks(tmp_path: Path) -> None:
    _labels(tmp_path, "session_a", {"AB12CDE": _row("FORD", [1, 2])})
    _labels(tmp_path, "session_b", {"AB12CDE": _row("FORD", [7, 8])})
    cars = session_cars(tmp_path, ["session_a", "session_b"], max_tracks_per_car=0)
    assert cars[0].tracks == [
        ("session_a", 1),
        ("session_a", 2),
        ("session_b", 7),
        ("session_b", 8),
    ]
    capped = session_cars(tmp_path, ["session_a", "session_b"], max_tracks_per_car=3)
    assert len(capped[0].tracks) == 3
    # Seeded: every model compared on these cars gets the same tracks.
    again = session_cars(tmp_path, ["session_a", "session_b"], max_tracks_per_car=3)
    assert again[0].tracks == capped[0].tracks


def test_session_cars_label_each_target_like_the_corpus_builder(tmp_path: Path) -> None:
    _labels(
        tmp_path,
        "session_a",
        {"AB12CDE": _row("FORD", [1], model="FIESTA", primary_colour="Silver")},
    )
    row = {"make": "FORD", "model": "FIESTA", "primary_colour": "Silver"}
    for target in ("make", "colour", "body_type"):
        cars = session_cars(tmp_path, ["session_a"], target=target)
        assert cars[0].label == target_label(row, target) != ""


def test_session_cars_unreadable_labels_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "session_a").mkdir()
    with pytest.raises(ValueError, match="no readable DVSA labels"):
        session_cars(tmp_path, ["session_a"])


def test_target_label_rejects_an_unknown_target() -> None:
    with pytest.raises(ValueError, match="target must be one of"):
        target_label({"make": "FORD"}, "wheels")
