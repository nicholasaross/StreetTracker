"""Door-origin walk classification (analysis/walks.py)."""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.analysis.walks import (
    ORIGINATED,
    PASSING,
    RETURNED,
    ROUND_TRIP,
    UNKNOWN,
    DoorZone,
    classify_walk_origin,
    door_origin_for_records,
    is_own_trip,
)

# A door zone in the bottom-left corner of the frame.
_ZONE = DoorZone(polygon_frac=[[0.0, 0.7], [0.2, 0.7], [0.2, 1.0], [0.0, 1.0]])


def test_point_in_and_out_of_zone() -> None:
    assert _ZONE.contains([0.1, 0.85]) is True
    assert _ZONE.contains([0.5, 0.5]) is False
    assert _ZONE.contains(None) is False
    assert _ZONE.contains([0.1]) is False


def test_originated_returned_round_trip_passing() -> None:
    # entry in zone, exit out -> left the door
    assert classify_walk_origin([0.1, 0.85], [0.6, 0.2], _ZONE) == ORIGINATED
    # entry out, exit in zone -> came back to the door
    assert classify_walk_origin([0.6, 0.2], [0.1, 0.85], _ZONE) == RETURNED
    # both in zone -> there and back within one track
    assert classify_walk_origin([0.05, 0.8], [0.15, 0.95], _ZONE) == ROUND_TRIP
    # neither -> through-traffic
    assert classify_walk_origin([0.6, 0.2], [0.3, 0.25], _ZONE) == PASSING


def test_unknown_when_no_points() -> None:
    assert classify_walk_origin(None, None, _ZONE) == UNKNOWN
    # A single recorded point is still classifiable (not unknown).
    assert classify_walk_origin([0.1, 0.85], None, _ZONE) == ORIGINATED


def test_is_own_trip() -> None:
    assert is_own_trip(ORIGINATED)
    assert is_own_trip(RETURNED)
    assert is_own_trip(ROUND_TRIP)
    assert not is_own_trip(PASSING)
    assert not is_own_trip(UNKNOWN)


def test_door_origin_for_records_persons_only() -> None:
    records = [
        {
            "track_id": 1,
            "class_name": "person",
            "entry_point_frac": [0.1, 0.85],
            "exit_point_frac": [0.6, 0.2],
        },
        {
            "track_id": 2,
            "class_name": "person",
            "entry_point_frac": [0.6, 0.2],
            "exit_point_frac": [0.3, 0.25],
        },
        {
            "track_id": 3,
            "class_name": "car",
            "entry_point_frac": [0.1, 0.85],
            "exit_point_frac": [0.1, 0.9],
        },  # not a person -> ignored
        {
            "track_id": 4,
            "class_name": "person",
            "class_suspect": True,
            "entry_point_frac": [0.1, 0.85],
            "exit_point_frac": [0.1, 0.9],
        },  # suspect
        {"track_id": 5, "class_name": "person"},  # no points -> unknown
    ]
    out = door_origin_for_records(records, _ZONE)
    assert out == {1: ORIGINATED, 2: PASSING, 5: UNKNOWN}


def test_load_missing_and_invalid(tmp_path: Path) -> None:
    assert DoorZone.load(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    assert DoorZone.load(bad) is None
    too_few = tmp_path / "few.json"
    too_few.write_text(json.dumps({"polygon_frac": [[0.0, 0.0], [1.0, 1.0]]}))
    assert DoorZone.load(too_few) is None


def test_load_valid(tmp_path: Path) -> None:
    p = tmp_path / "door_zone.json"
    p.write_text(json.dumps({"polygon_frac": [[0.0, 0.7], [0.2, 0.7], [0.1, 1.0]]}))
    zone = DoorZone.load(p)
    assert zone is not None
    assert len(zone.polygon_frac) == 3
    assert zone.contains([0.1, 0.8]) is True
