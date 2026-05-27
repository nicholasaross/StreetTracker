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


def test_fuzzy_clustering_merges_one_char_variants(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """Two tracks whose plate strings differ by exactly one character
    (same length) should collapse into a single Vehicle. Canonical is
    the higher-confidence plate; lower-conf one lands in
    plate_variants. Documented in Step 12 of CLAUDE.md as the
    known limitation that this feature addresses."""
    t1 = sample_track  # AB12CDE @ 0.98
    t2 = replace(
        sample_track,
        track_id=99,
        time_start_unix=sample_track.time_end_unix + 120,
        time_end_unix=sample_track.time_end_unix + 127,
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
                    "track_id": 99, "snap_index": 1,
                    "image": "vehicle_99_main_1.jpg",
                    "ocr_text": "AB12CXE",  # one-char diff at position 5
                    "ocr_conf": 0.93,
                    "det_conf": 0.81,
                },
            },
        ],
    }
    session = _write_session(tmp_path, [t1, t2], alpr)

    vehicles = build_vehicles(session)
    plated = [v for v in vehicles if v.plate is not None]
    assert len(plated) == 1
    v = plated[0]
    assert v.plate == "AB12CDE"  # higher conf wins canonical
    assert v.n_visits == 2
    assert v.track_ids == [42, 99]
    assert v.plate_variants == [("AB12CXE", pytest.approx(0.93))]


def test_fuzzy_clustering_does_not_merge_different_lengths(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """Plates of different lengths must NOT cluster -- a length
    difference suggests an OCR truncation / dropped char that is
    visually too risky to collapse without manual review. Documented
    as a deliberate conservative choice in vehicles.py."""
    t1 = sample_track
    t2 = replace(sample_track, track_id=99,
                 time_start_unix=sample_track.time_end_unix + 60)
    alpr = {
        "tracks": [
            {
                "track_id": 42,
                "best_preferred": {
                    "track_id": 42, "snap_index": 1,
                    "image": "vehicle_42_main_1.jpg",
                    "ocr_text": "AB12CDE", "ocr_conf": 0.98,  # len 7
                    "det_conf": 0.85,
                },
            },
            {
                "track_id": 99,
                "best_preferred": {
                    "track_id": 99, "snap_index": 1,
                    "image": "vehicle_99_main_1.jpg",
                    "ocr_text": "B12CDE",  # len 6, ratio is high but
                                            # length differs -- skip
                    "ocr_conf": 0.93,
                    "det_conf": 0.81,
                },
            },
        ],
    }
    session = _write_session(tmp_path, [t1, t2], alpr)

    vehicles = build_vehicles(session)
    plates = sorted([v.plate for v in vehicles if v.plate is not None])
    assert plates == ["AB12CDE", "B12CDE"]


def test_fuzzy_clustering_disabled_keeps_strict_string_equality(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """fuzzy_ratio=None preserves the strict-string grouping (legacy
    behaviour). The two near-variants stay separate."""
    t1 = sample_track
    t2 = replace(sample_track, track_id=99,
                 time_start_unix=sample_track.time_end_unix + 60)
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
                    "track_id": 99, "snap_index": 1,
                    "image": "vehicle_99_main_1.jpg",
                    "ocr_text": "AB12CXE", "ocr_conf": 0.93,
                    "det_conf": 0.81,
                },
            },
        ],
    }
    session = _write_session(tmp_path, [t1, t2], alpr)

    vehicles = build_vehicles(session, fuzzy_ratio=None)
    plated = sorted([v.plate for v in vehicles if v.plate is not None])
    assert plated == ["AB12CDE", "AB12CXE"]


def test_fuzzy_clustering_allows_temporally_overlapping_merges(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """Two plates that fuzzy-match by string AND have overlapping
    tracks should merge anyway. BotSORT ID-switches routinely produce
    two overlapping tracks for ONE physical vehicle (brief occlusion
    spawns a new track ID before the old one finalises). A real
    example surfaced on session_20260526_124704 tracks 1516 (LD22BMG,
    R→L) and 1517 (LD22BWG, L→R) -- both 4K snaps show the same
    silver hatchback one second apart, but BotSORT had ID-switched
    mid-transit and even mislabelled the new track's direction.
    Earlier code REJECTED overlap-conflicting merges; that produced
    more false negatives than the overlap heuristic caught true
    positives, so the rejection is gone."""
    t1 = sample_track  # 14:32:05 -> 14:32:11
    t2 = replace(
        sample_track,
        track_id=99,
        time_start="2026-05-17T14:32:08+01:00",
        time_end="2026-05-17T14:32:15+01:00",
        time_start_unix=sample_track.time_start_unix + 3,
        time_end_unix=sample_track.time_end_unix + 4,
        direction="right to left",  # opposite of t1 -- mirrors the
                                    # real BotSORT ID-switch case
    )
    alpr = {
        "tracks": [
            {
                "track_id": 42,
                "best_preferred": {
                    "track_id": 42, "snap_index": 1,
                    "image": "vehicle_42_main_1.jpg",
                    "ocr_text": "LD22BWG", "ocr_conf": 0.98,
                    "det_conf": 0.85,
                },
            },
            {
                "track_id": 99,
                "best_preferred": {
                    "track_id": 99, "snap_index": 1,
                    "image": "vehicle_99_main_1.jpg",
                    "ocr_text": "LD22BMG",  # 1-char diff, ratio 85.7
                    "ocr_conf": 0.93,
                    "det_conf": 0.81,
                },
            },
        ],
    }
    session = _write_session(tmp_path, [t1, t2], alpr)

    vehicles = build_vehicles(session)
    plated = [v for v in vehicles if v.plate is not None]
    assert len(plated) == 1
    v = plated[0]
    # Higher-conf plate becomes canonical, other lands in variants.
    assert v.plate == "LD22BWG"
    assert v.n_visits == 2
    assert v.plate_variants == [("LD22BMG", pytest.approx(0.93))]


def test_fuzzy_clustering_high_threshold_does_not_merge_distant_plates(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """Two plates differing by 3+ chars on a 7-char plate fall below
    the default 85 threshold and stay separate, even though same
    length. Guards against false positives on visually-similar but
    distinct plates."""
    t1 = sample_track  # AB12CDE
    t2 = replace(sample_track, track_id=99,
                 time_start_unix=sample_track.time_end_unix + 60)
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
                    "track_id": 99, "snap_index": 1,
                    "image": "vehicle_99_main_1.jpg",
                    "ocr_text": "XY99CFG",  # 4-char diff, ratio < 50
                    "ocr_conf": 0.93,
                    "det_conf": 0.81,
                },
            },
        ],
    }
    session = _write_session(tmp_path, [t1, t2], alpr)

    vehicles = build_vehicles(session)  # default fuzzy_ratio=85
    plated = sorted([v.plate for v in vehicles if v.plate is not None])
    assert plated == ["AB12CDE", "XY99CFG"]


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
