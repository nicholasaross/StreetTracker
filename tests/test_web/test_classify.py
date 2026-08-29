"""Unit tests for the behavioural vehicle classifier (web/classify.py).

The classifier is pure, so these drive :func:`classify_vehicle` directly with
synthetic trip sequences -- one per bucket plus the edge cases (single pass,
one-way-only, near-zero dwell, owner/parked priors, the chance control).
"""

from __future__ import annotations

from datetime import datetime

from streettracker.web import classify
from streettracker.web.classify import (
    BRIEF,
    RESIDENT,
    UNCLASSIFIED,
    VISITOR,
    YMCA_STAFF,
    _chance_pairs,
    _to_trips,
    classify_vehicle,
)

_IN = classify._IN  # "right to left" — into the cul-de-sac
_OUT = classify._OUT  # "left to right" — out to the main road


def _trip(date: str, hhmm: str, direction: str, dur_s: float = 8.0):
    """One pass as the (start_unix, end_unix, direction, iso) tuple the
    classifier consumes. unix is derived from the ISO string so gap math and
    day/minute-of-day math agree."""
    iso = f"{date}T{hhmm}:00+01:00"
    start = datetime.fromisoformat(iso).timestamp()
    return (start, start + dur_s, direction, iso)


def test_resident_leaves_then_returns() -> None:
    # A genuine resident: many days of OUT in the morning, IN in the evening
    # (rests inside, leaves-and-returns well above coincidence).
    trips = []
    for d in range(1, 25, 2):  # 12 distinct days across three weeks
        day = f"2026-06-{d:02d}"
        trips.append(_trip(day, "08:30", _OUT))
        trips.append(_trip(day, "17:45", _IN))
    c = classify_vehicle(trips)
    assert c.bucket == RESIDENT
    assert c.certainty in ("medium", "high")


def test_occasional_out_and_back_is_not_resident() -> None:
    # Just a couple of leave-then-return days do NOT make a resident -- the
    # round trips must exceed the coincidence floor and/or span many days.
    trips = []
    for day in ("2026-06-01", "2026-06-03"):
        trips.append(_trip(day, "08:30", _OUT))
        trips.append(_trip(day, "17:45", _IN))
    c = classify_vehicle(trips)
    assert c.bucket != RESIDENT


def test_ymca_staff_long_weekday_stay() -> None:
    # Mon-Fri: IN ~08:00, OUT ~17:00 (a full working day), tight arrival.
    trips = []
    for day in ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"):
        trips.append(_trip(day, "08:05", _IN))
        trips.append(_trip(day, "17:10", _OUT))
    c = classify_vehicle(trips)
    assert c.bucket == YMCA_STAFF
    assert c.evidence["median_dwell_min"] > classify.VISITOR_MAX_MIN


def test_visitor_short_class_stay() -> None:
    # Arrives ~10:45, leaves ~11:55 (~70 min gym class), a couple of dates.
    trips = []
    for day in ("2026-06-01", "2026-06-04", "2026-06-11"):
        trips.append(_trip(day, "10:45", _IN))
        trips.append(_trip(day, "11:55", _OUT))
    c = classify_vehicle(trips)
    assert c.bucket == VISITOR
    assert 20 < c.evidence["median_dwell_min"] < classify.VISITOR_MAX_MIN


def test_brief_drop_off() -> None:
    # In and straight back out within a few minutes, repeatedly.
    trips = []
    for day in ("2026-06-01", "2026-06-02", "2026-06-03"):
        trips.append(_trip(day, "09:00", _IN))
        trips.append(_trip(day, "09:03", _OUT))
    c = classify_vehicle(trips)
    assert c.bucket == BRIEF


def test_single_sighting_is_unclassified() -> None:
    c = classify_vehicle([_trip("2026-06-01", "12:00", _IN)])
    assert c.bucket == UNCLASSIFIED
    assert c.certainty == "low"


def test_one_way_only_is_unclassified() -> None:
    # Seen several times but always the same direction -> can't tell order.
    trips = [_trip(d, "12:00", _IN) for d in ("2026-06-01", "2026-06-02", "2026-06-03")]
    c = classify_vehicle(trips)
    assert c.bucket == UNCLASSIFIED


def test_in_first_visitor_is_not_resident() -> None:
    # Regression: a car that consistently arrives-then-leaves (IN-first) is a
    # visitor, NOT a resident -- even if the operator has tagged its owner. An
    # owner/name tag means "known car", not "resident", so it must not flip the
    # bucket (this was the WJ60VDR "Granny & Pop" bug).
    trips = []
    for day in ("2026-06-01", "2026-06-04", "2026-06-11", "2026-06-18"):
        trips.append(_trip(day, "11:30", _IN))
        trips.append(_trip(day, "13:00", _OUT))  # ~90 min visit
    c = classify_vehicle(trips)
    assert c.bucket == VISITOR


def test_variable_dwell_visitor_is_not_brief() -> None:
    # A car with some short pops AND some multi-hour stays is a visitor with
    # variable dwell, not a "brief" drop-off vehicle.
    trips = [
        _trip("2026-06-01", "10:00", _IN),
        _trip("2026-06-01", "10:08", _OUT),  # 8 min
        _trip("2026-06-04", "12:00", _IN),
        _trip("2026-06-04", "12:06", _OUT),  # 6 min
        _trip("2026-06-08", "11:00", _IN),
        _trip("2026-06-08", "14:30", _OUT),  # 3.5 h — a real visit
    ]
    c = classify_vehicle(trips)
    assert c.bucket != BRIEF


def test_arrive_bias_is_not_resident() -> None:
    # Seen arriving (IN) far more than leaving -- a frequent VISITOR pattern, not
    # a resident. A mere arrive-bias must not tip it to resident (it lacks the
    # genuine leave-then-return of someone based inside).
    trips = []
    for d in range(1, 21):
        day = f"2026-06-{d:02d}"
        trips.append(_trip(day, "14:00", _IN))
        trips.append(_trip(day, "15:30", _IN))
    trips.append(_trip("2026-06-05", "09:00", _OUT))
    trips.append(_trip("2026-06-12", "09:00", _OUT))
    c = classify_vehicle(trips)
    assert c.bucket != RESIDENT


def test_parked_episodes_alone_do_not_force_resident() -> None:
    # A parked stint is no longer a resident shortcut (it over-fired on
    # visitors/vans that briefly sit in view); residence needs leave-then-return.
    trips = []
    for day in ("2026-06-01", "2026-06-04", "2026-06-11"):
        trips.append(_trip(day, "10:45", _IN))
        trips.append(_trip(day, "11:55", _OUT))  # arrive-then-leave = visitor
    c = classify_vehicle(trips, parked_episodes=[{"first_seen": "x"}])
    assert c.bucket != RESIDENT


def test_chance_control_none_on_short_span() -> None:
    # A single-day record is far shorter than 2x the shift window.
    trips = _to_trips([_trip("2026-06-01", "08:00", _IN), _trip("2026-06-01", "09:00", _OUT)])
    assert _chance_pairs(trips) is None


def test_chance_control_runs_on_long_span() -> None:
    # A multi-week record is long enough for the time-shift null model.
    trips = _to_trips(
        [_trip(d, "10:00", _IN) for d in ("2026-06-01", "2026-06-15", "2026-07-01")]
        + [_trip(d, "11:00", _OUT) for d in ("2026-06-01", "2026-06-15", "2026-07-01")]
    )
    assert isinstance(_chance_pairs(trips), int)


def test_evidence_is_serialisable_and_scored() -> None:
    c = classify_vehicle([_trip("2026-06-01", "08:30", _OUT), _trip("2026-06-01", "17:45", _IN)])
    d = c.to_json_dict()
    assert d["bucket"] in classify.BUCKETS
    assert 0.0 <= d["score"] <= 1.0
    assert isinstance(d["evidence"], dict)
