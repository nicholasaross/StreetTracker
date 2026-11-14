"""The stationary-vehicle rule: a box that the nearest tracks' snaps show both
before and after a track is not the car that track followed, and nor is a
box hidden behind it."""

from __future__ import annotations

import pytest

from streettracker.analysis.stationary import (
    SnapSighting,
    covered_frac,
    find_stationary_vehicles,
    group_refused,
    hidden_behind,
    iou,
    same_vehicle,
    sits_on,
    taken_by,
)

# A van stopped at the far end, and the same van as the detector sees it a
# minute later (a pixel or two of jitter).
_VAN = (2650.0, 230.0, 3150.0, 560.0)
_VAN_AGAIN = (2652.0, 231.0, 3149.0, 562.0)


def _car(x: float) -> tuple[float, float, float, float]:
    """A moving car, somewhere along the road."""
    return (x, 600.0, x + 700.0, 1100.0)


def _snap(track: int, time: float, *boxes: tuple[float, float, float, float]) -> SnapSighting:
    return SnapSighting(track, 1, time, boxes)


def test_iou_and_covered_frac() -> None:
    assert iou(_VAN, _VAN) == 1.0
    assert iou(_VAN, _VAN_AGAIN) > 0.97
    assert iou(_VAN, _car(100.0)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 1 / 3
    assert covered_frac((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5
    assert covered_frac((0, 0, 10, 10), (0, 0, 100, 100)) == 1.0
    assert covered_frac((0, 0, 100, 100), (0, 0, 10, 10)) == 0.01
    assert covered_frac(_VAN, _car(100.0)) == 0.0


# ----------------------------------------------------------------------
# Which vehicles are standing still around a snap.


def test_a_box_seen_before_and_after_by_other_tracks_is_stationary() -> None:
    snaps = [
        _snap(1, 0.0, _VAN, _car(100.0)),
        _snap(2, 60.0, _VAN_AGAIN, _car(400.0)),
        _snap(3, 120.0, _VAN, _car(900.0)),
    ]
    # Only the middle track has the van on both sides of it. The vehicle is
    # given as that snap's own box for it.
    assert find_stationary_vehicles(snaps) == {(2, 1): [_VAN_AGAIN]}


def test_a_car_that_arrives_and_stays_is_not_stationary_in_its_own_snap() -> None:
    # Track 1 drives in and parks: nobody saw that box before it.
    snaps = [
        _snap(9, -60.0, _car(100.0)),
        _snap(1, 0.0, _VAN),
        _snap(2, 60.0, _VAN, _car(400.0)),
        _snap(3, 120.0, _VAN, _car(900.0)),
        _snap(4, 180.0, _VAN, _car(200.0)),
    ]
    found = find_stationary_vehicles(snaps)
    assert (1, 1) not in found
    assert set(found) == {(2, 1), (3, 1)}


def test_a_car_that_pulls_away_is_not_stationary_in_its_own_snap() -> None:
    snaps = [
        _snap(1, 0.0, _VAN, _car(100.0)),
        _snap(2, 60.0, _VAN, _car(400.0)),
        _snap(3, 120.0, _VAN),  # the van's own track, as it leaves
        _snap(4, 180.0, _car(900.0)),
    ]
    assert set(find_stationary_vehicles(snaps)) == {(2, 1)}


def test_moving_cars_are_never_stationary() -> None:
    snaps = [_snap(t, 30.0 * t, _car(100.0 + 37.0 * t)) for t in range(1, 9)]
    assert find_stationary_vehicles(snaps) == {}


def test_two_sightings_are_not_enough() -> None:
    # Two moving cars snapped at the same place, one before track 2 and one
    # after it. Track 2's own car is somewhere else.
    snaps = [_snap(1, 0.0, _VAN), _snap(2, 60.0, _car(400.0)), _snap(3, 120.0, _VAN)]
    assert find_stationary_vehicles(snaps) == {}
    assert set(find_stationary_vehicles(snaps, min_tracks=2)) == {(2, 1)}


def test_a_vehicle_the_snap_does_not_show_is_found_from_three_neighbours() -> None:
    # Track 3's own detection of the van is off (someone standing beside it).
    nudged = (2650.0, 230.0, 3230.0, 560.0)
    assert iou(nudged, _VAN) < 0.9
    snaps = [
        _snap(1, 0.0, _VAN),
        _snap(2, 60.0, _VAN_AGAIN),
        _snap(3, 120.0, nudged, _car(300.0)),
        _snap(4, 180.0, _VAN),
    ]
    found = find_stationary_vehicles(snaps)
    # Given as the box nearest in time before the snap.
    assert found[(3, 1)] == [_VAN_AGAIN]
    assert taken_by(nudged, found[(3, 1)]) == _VAN_AGAIN


def test_snaps_of_the_same_track_are_not_evidence() -> None:
    # One track with three snaps of a car that hardly moves.
    snaps = [SnapSighting(1, n, 20.0 * n, (_VAN,)) for n in (1, 2, 3)]
    assert find_stationary_vehicles(snaps) == {}


def test_snaps_within_the_gap_are_not_evidence() -> None:
    # One car as two tracks a few seconds apart (an ID switch) looks the same.
    snaps = [_snap(1, 0.0, _VAN), _snap(2, 4.0, _VAN), _snap(3, 8.0, _VAN)]
    assert find_stationary_vehicles(snaps) == {}
    assert set(find_stationary_vehicles(snaps, min_gap_s=2.0)) == {(2, 1)}


def test_only_the_nearest_tracks_on_each_side_count() -> None:
    # The van left long before track 5: three other tracks have passed since.
    snaps = [
        _snap(1, 0.0, _VAN),
        *[_snap(t, 60.0 * (t - 1), _car(100.0 * t)) for t in (2, 3, 4)],
        _snap(5, 240.0, _VAN),
        _snap(6, 300.0, _VAN),
    ]
    assert (5, 1) not in find_stationary_vehicles(snaps)
    # Looking one track further back reaches it.
    assert (5, 1) in find_stationary_vehicles(snaps, neighbour_tracks=4)


def test_evidence_too_far_away_in_time_does_not_count() -> None:
    snaps = [_snap(1, 0.0, _VAN), _snap(2, 2000.0, _VAN), _snap(3, 2060.0, _VAN)]
    assert find_stationary_vehicles(snaps) == {}
    assert set(find_stationary_vehicles(snaps, max_gap_s=3000.0)) == {(2, 1)}


def test_every_snap_of_a_neighbouring_track_is_looked_at() -> None:
    # A passing car hides the van in the neighbour's first snap only.
    snaps = [
        SnapSighting(1, 1, 0.0, (_car(100.0),)),
        SnapSighting(1, 2, 0.4, (_VAN,)),
        _snap(2, 60.0, _VAN),
        SnapSighting(3, 1, 120.0, (_car(500.0),)),
        SnapSighting(3, 2, 120.4, (_VAN,)),
    ]
    assert set(find_stationary_vehicles(snaps)) == {(2, 1)}


def test_a_vehicle_is_listed_once_and_snaps_without_boxes_are_fine() -> None:
    snaps = [
        SnapSighting(1, 1, 0.0, (_VAN,)),
        SnapSighting(1, 2, 0.4, (_VAN_AGAIN,)),
        _snap(2, 60.0),  # nothing detected in this snap
        _snap(3, 120.0, _VAN),
        _snap(4, 180.0, _VAN),
    ]
    found = find_stationary_vehicles(snaps)
    assert found[(2, 1)] == [_VAN_AGAIN]  # nearest in time before, once
    assert find_stationary_vehicles([_snap(1, 0.0), _snap(2, 60.0), _snap(3, 120.0)]) == {}


# ----------------------------------------------------------------------
# Which crops a stationary vehicle rules out.


def test_sits_on_needs_one_of_the_tracks_own_boxes_on_the_vehicle() -> None:
    passing = (1200.0, 700.0, 1900.0, 1200.0)
    assert sits_on(_VAN, [_VAN_AGAIN])
    assert sits_on(_VAN, [passing, _VAN_AGAIN])  # arrived: on it when the snap landed
    assert not sits_on(_VAN, [passing, None])
    assert not sits_on(_VAN, [])
    # Clipping the vehicle's edge is not sitting on it.
    edge = (2950.0, 230.0, 3450.0, 560.0)
    assert 0.0 < iou(_VAN, edge) < 0.5
    assert not sits_on(_VAN, [edge])


def test_a_box_that_is_the_vehicle_is_taken() -> None:
    assert taken_by(_VAN_AGAIN, [_VAN]) == _VAN
    # The detector's box for it, stretched by someone standing beside it.
    nudged = (2650.0, 230.0, 3230.0, 560.0)
    assert 0.7 <= iou(nudged, _VAN) < 0.9
    assert taken_by(nudged, [_VAN]) == _VAN
    assert taken_by(_car(100.0), [_VAN]) is None
    assert taken_by(_VAN, []) is None


@pytest.mark.parametrize(
    ("box", "taken"),
    [
        # Further up the road (its box ends higher), a third of it behind the van.
        ((2400.0, 150.0, 2900.0, 480.0), True),
        # The same, but only clipping the van: the crop is still this car.
        ((2150.0, 150.0, 2700.0, 480.0), False),
        # Nearer the camera (its box ends lower): it hides the van, not the
        # other way round, however much they overlap.
        ((2500.0, 300.0, 3300.0, 800.0), False),
    ],
    ids=["behind it", "clipping it", "in front of it"],
)
def test_a_box_hidden_behind_the_vehicle_is_taken(
    box: tuple[float, float, float, float], taken: bool
) -> None:
    assert iou(box, _VAN) < 0.7
    assert (taken_by(box, [_VAN]) == _VAN) is taken


def test_the_two_reasons_can_be_asked_separately() -> None:
    behind = (2400.0, 150.0, 2900.0, 480.0)
    assert 0.25 <= covered_frac(behind, _VAN) < 0.5
    # A box behind the vehicle is not the vehicle; the vehicle is not behind itself.
    assert same_vehicle(behind, [_VAN]) is None
    assert hidden_behind(behind, [_VAN]) == _VAN
    assert same_vehicle(_VAN_AGAIN, [_VAN]) == _VAN
    assert hidden_behind(_VAN, [_VAN]) is None
    # Their thresholds.
    assert hidden_behind(behind, [_VAN], min_frac=0.5) is None
    assert same_vehicle(_VAN_AGAIN, [_VAN], min_iou=0.99) is None


# ----------------------------------------------------------------------
# The summary of what was refused.


def test_refused_crops_are_grouped_by_vehicle() -> None:
    bus = (10.0, 1200.0, 1600.0, 2500.0)
    refused = [
        (60.0, 2, _VAN_AGAIN),
        (120.0, 3, _VAN),
        (120.4, 3, _VAN),
        (180.0, 4, _VAN),
        # Hours later, another vehicle somewhere else.
        (9060.0, 6, bus),
    ]
    assert group_refused(reversed(refused)) == [
        {
            "box": list(_VAN_AGAIN),
            "first_unix": 60.0,
            "last_unix": 180.0,
            "n_crops": 4,
            "n_tracks": 3,
        },
        {"box": list(bus), "first_unix": 9060.0, "last_unix": 9060.0, "n_crops": 1, "n_tracks": 1},
    ]
    assert group_refused([]) == []


def test_the_same_place_much_later_is_another_entry() -> None:
    refused = [(60.0, 1, _VAN), (120.0, 2, _VAN), (9000.0, 3, _VAN), (9060.0, 4, _VAN)]
    groups = group_refused(refused)
    assert [(g["first_unix"], g["n_tracks"]) for g in groups] == [(60.0, 2), (9000.0, 2)]
