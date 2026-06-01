"""apply_dvsa_labels() folds the DVSA harvest onto per-track records."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from streettracker.analysis.dvsa_apply import apply_dvsa_labels
from streettracker.analysis.dvsa_apply import main as dvsa_apply_main
from streettracker.common.schema import TrackRecord


def _write_session(
    tmp_path: Path,
    tracks: list[TrackRecord],
    dvsa_labels: dict | None = None,
) -> Path:
    session = tmp_path / "session_test"
    session.mkdir()
    recs = [asdict(t) for t in tracks]
    (session / "session_test_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8"
    )
    (session / "session_test_data.json").write_text(
        json.dumps(recs), encoding="utf-8"
    )
    if dvsa_labels is not None:
        (session / "session_test_dvsa_labels.json").write_text(
            json.dumps(dvsa_labels), encoding="utf-8"
        )
    return session


def _labels(track_ids: list[int]) -> dict:
    return {
        "labels": {
            "AB12CDE": {
                "plate": "AB12CDE", "make": "FORD", "model": "FOCUS",
                "year": 2017, "track_ids": track_ids,
            },
        },
        "unknown": [], "skipped_non_canonical": [],
    }


def _load_data(session: Path) -> dict[int, dict]:
    arr = json.loads((session / "session_test_data.json").read_text())
    return {r["track_id"]: r for r in arr}


def test_apply_writes_make_model_to_data_and_events(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    session = _write_session(tmp_path, [sample_track], _labels([42]))
    stats = apply_dvsa_labels(session)
    assert stats.cars_labelled == 1

    rec = _load_data(session)[42]
    assert rec["make"] == "FORD"
    assert rec["model"] == "FOCUS"
    assert rec["year"] == 2017
    assert rec["make_model_source"] == "dvsa"

    # events.jsonl carries it too, so a later `recolor` re-derive (which
    # rebuilds data.json FROM events.jsonl) preserves the enrichment.
    line = (session / "session_test_events.jsonl").read_text().splitlines()[0]
    assert json.loads(line)["make"] == "FORD"

    # round-trips through the schema
    back = TrackRecord.from_json_dict(rec)
    assert back.make == "FORD"
    assert back.make_model_source == "dvsa"


def test_apply_leaves_unlabelled_tracks_none(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    t2 = replace(sample_track, track_id=99)
    session = _write_session(tmp_path, [sample_track, t2], _labels([42]))
    apply_dvsa_labels(session)
    data = _load_data(session)
    assert data[42]["make"] == "FORD"
    assert data[99]["make"] is None
    assert data[99]["make_model_source"] is None


def test_apply_skips_person_tracks(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """A label whose track_ids (wrongly) include a person track must not
    write make/model onto a person record -- cars only."""
    person = replace(sample_track, track_id=7, class_name="person",
                     class_id=0, asset_prefix="person")
    session = _write_session(tmp_path, [person], _labels([7]))
    stats = apply_dvsa_labels(session)
    assert stats.cars_labelled == 0
    assert _load_data(session)[7]["make"] is None


def test_apply_is_idempotent(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    session = _write_session(tmp_path, [sample_track], _labels([42]))
    apply_dvsa_labels(session)
    first = (session / "session_test_data.json").read_text()
    apply_dvsa_labels(session)
    assert (session / "session_test_data.json").read_text() == first


def test_main_without_harvest_returns_2(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    session = _write_session(tmp_path, [sample_track], dvsa_labels=None)
    assert dvsa_apply_main([str(session)]) == 2
