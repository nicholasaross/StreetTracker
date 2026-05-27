"""build_vehicles() against synthetic closed sessions.

Covers the three main grouping paths:
1. Single-visit plated vehicles (1 track, 1 plate-anchored Vehicle).
2. Recurring plated vehicles (multiple tracks sharing a plate).
3. Unread tracks (no high-conf plate) -> plate=None Vehicles.

Also covers conf_threshold + include_unread filters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from streettracker.analysis.vehicles import build_vehicles
from streettracker.common.schema import TrackRecord


def _write_session(
    tmp_path: Path,
    tracks: list[TrackRecord],
    alpr_by_track: dict | None = None,
) -> Path:
    session = tmp_path / "session_test"
    session.mkdir()
    data_path = session / "session_test_data.json"
    data_path.write_text(json.dumps([asdict(t) for t in tracks]))
    if alpr_by_track is not None:
        (session / "session_test_alpr_by_track.json").write_text(
            json.dumps(alpr_by_track)
        )
    return session


def test_build_vehicles_single_plated_visit(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """One track with a high-conf plate -> one Vehicle with n_visits=1."""
    alpr = {
        "tracks": [{
            "track_id": 42,
            "best_preferred": {
                "track_id": 42, "snap_index": 1,
                "image": "vehicle_42_main_1.jpg",
                "ocr_text": "AB12CDE", "ocr_conf": 0.98,
                "det_conf": 0.85,
            },
        }],
    }
    session = _write_session(tmp_path, [sample_track], alpr)

    vehicles = build_vehicles(session)

    assert len(vehicles) == 1
    v = vehicles[0]
    assert v.plate == "AB12CDE"
    assert v.plate_conf == pytest.approx(0.98)
    assert v.n_visits == 1
    assert v.track_ids == [42]
    assert v.gap_minutes_max == 0.0  # single visit, no gap
    assert v.directions == {"left to right": 1}
    assert v.colors == {"blue": 1}
    assert len(v.visits) == 1
    assert v.visits[0].best_image == "vehicle_42_main_1.jpg"


def test_build_vehicles_recurring_plate_groups_visits(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """Two tracks with the same high-conf plate -> one Vehicle with
    n_visits=2 and a gap_minutes value computed from time_end of the
    first to time_start of the second."""
    # First visit: 14:32:05 -> 14:32:11. Second visit: 14:42:11 (10
    # minutes after the first ended).
    t1 = sample_track
    t2 = replace(
        sample_track,
        track_id=99,
        time_start="2026-05-17T14:42:11+01:00",
        time_end="2026-05-17T14:42:18+01:00",
        time_start_unix=sample_track.time_end_unix + 600,
        time_end_unix=sample_track.time_end_unix + 607,
        direction="right to left",
        color="red",
    )
    alpr = {
        "tracks": [
            {
                "track_id": 42,
                "best_preferred": {
                    "track_id": 42, "snap_index": 1,
                    "image": "vehicle_42_main_1.jpg",
                    "ocr_text": "AB12CDE", "ocr_conf": 0.98,
                    "det_conf": 0.85,
                },
            },
            {
                "track_id": 99,
                "best_preferred": {
                    "track_id": 99, "snap_index": 2,
                    "image": "vehicle_99_main_2.jpg",
                    "ocr_text": "AB12CDE", "ocr_conf": 0.93,
                    "det_conf": 0.81,
                },
            },
        ],
    }
    session = _write_session(tmp_path, [t1, t2], alpr)

    vehicles = build_vehicles(session)

    assert len(vehicles) == 1
    v = vehicles[0]
    assert v.plate == "AB12CDE"
    assert v.plate_conf == pytest.approx(0.98)  # max across visits
    assert v.n_visits == 2
    assert v.track_ids == [42, 99]  # chronological
    assert v.gap_minutes_max == pytest.approx(10.0)
    assert v.gap_minutes_min == pytest.approx(10.0)
    assert v.directions == {"left to right": 1, "right to left": 1}
    assert v.colors == {"blue": 1, "red": 1}


def test_build_vehicles_unread_tracks_emit_anonymous_vehicles(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """Tracks below conf threshold + tracks absent from alpr rollup
    both fall under plate=None and emit one Vehicle each."""
    t1 = sample_track  # in rollup at high conf
    t2 = replace(sample_track, track_id=51,
                 time_start_unix=sample_track.time_start_unix + 30)
    # t2 is in the rollup but at LOW conf -> treated as unread.
    # t3 is not in the rollup at all.
    t3 = replace(sample_track, track_id=52,
                 time_start_unix=sample_track.time_start_unix + 60)
    alpr = {
        "tracks": [
            {
                "track_id": 42,
                "best_preferred": {
                    "track_id": 42, "snap_index": 1,
                    "image": "vehicle_42_main_1.jpg",
                    "ocr_text": "GOOD123", "ocr_conf": 0.95,
                    "det_conf": 0.85,
                },
            },
            {
                "track_id": 51,
                "best_preferred": {
                    "track_id": 51, "snap_index": 1,
                    "image": "vehicle_51_main_1.jpg",
                    "ocr_text": "MAYBE", "ocr_conf": 0.42,  # below 0.9
                    "det_conf": 0.40,
                },
            },
        ],
    }
    session = _write_session(tmp_path, [t1, t2, t3], alpr)

    vehicles = build_vehicles(session)
    plates = [v.plate for v in vehicles]
    assert plates.count("GOOD123") == 1
    assert plates.count(None) == 2  # t2 (low conf) + t3 (absent)


def test_build_vehicles_no_unread_filter(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """include_unread=False drops anonymous-vehicle tracks."""
    t1 = sample_track
    t2 = replace(sample_track, track_id=51,
                 time_start_unix=sample_track.time_start_unix + 30)
    alpr = {
        "tracks": [{
            "track_id": 42,
            "best_preferred": {
                "track_id": 42, "snap_index": 1,
                "image": "vehicle_42_main_1.jpg",
                "ocr_text": "GOOD123", "ocr_conf": 0.95,
                "det_conf": 0.85,
            },
        }],
    }
    session = _write_session(tmp_path, [t1, t2], alpr)

    vehicles = build_vehicles(session, include_unread=False)
    assert len(vehicles) == 1
    assert vehicles[0].plate == "GOOD123"


def test_build_vehicles_persons_are_skipped(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """class_name == 'person' tracks are excluded from the vehicle
    aggregation -- plate snaps would be anatomically wrong anyway."""
    person = replace(sample_track, track_id=200, class_name="person",
                     class_id=0, asset_prefix="person")
    car = sample_track
    session = _write_session(tmp_path, [person, car], alpr_by_track={"tracks": []})

    vehicles = build_vehicles(session)
    assert all(v.track_ids != [200] for v in vehicles)
    assert any(v.track_ids == [42] for v in vehicles)


def test_build_vehicles_missing_alpr_rollup_is_tolerated(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """A session that never had alpr-run -> all tracks fall under
    plate=None."""
    session = _write_session(tmp_path, [sample_track], alpr_by_track=None)
    vehicles = build_vehicles(session)
    assert len(vehicles) == 1
    assert vehicles[0].plate is None


def test_build_vehicles_sorted_by_first_seen(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """Output is chronological by first_seen."""
    t_late = sample_track
    t_early = replace(
        sample_track,
        track_id=10,
        time_start="2026-05-17T14:00:00+01:00",
        time_start_unix=sample_track.time_start_unix - 1925,
        time_end_unix=sample_track.time_end_unix - 1925,
    )
    session = _write_session(tmp_path, [t_late, t_early],
                             alpr_by_track={"tracks": []})
    vehicles = build_vehicles(session)
    assert [v.track_ids for v in vehicles] == [[10], [42]]
