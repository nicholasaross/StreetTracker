"""build_people() / pair_companions() / the ``people`` CLI against
synthetic sessions.

Covers:
1. Dog pairing: same-direction overlap pairs, direction mismatch and
   short overlap don't, and a dog picks its best-overlap person.
2. Kind classification: cyclist (bicycle-paired) beats the speed split,
   jogger needs calibration + enough detections, walker is the floor.
3. Uncalibrated sessions: no jogger split, speed_m_s stays None.
4. CLI end-to-end: writes {session}_people.json with the summary block.
"""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.analysis.people import (
    build_people,
    classify_kind,
    main,
    pair_companions,
    resolve_m_per_px,
)
from streettracker.analysis.walks import DoorZone
from streettracker.common.schema import TrackRecord

_L2R = "left to right"
_R2L = "right to left"
_BASE = 1_780_000_000.0
_CLASS_IDS = {"person": 0, "bicycle": 1, "car": 2, "dog": 16}


def _rec(
    tid: int,
    cls: str = "person",
    *,
    start: float = 0.0,
    end: float = 20.0,
    direction: str = _L2R,
    speed_px_s: float = 30.0,
    num_detections: int = 58,
    color: str = "unknown",
    entry: list[float] | None = None,
    exit: list[float] | None = None,
) -> dict:
    return TrackRecord(
        track_id=tid,
        class_id=_CLASS_IDS[cls],
        class_name=cls,
        time_start="2026-07-01T10:00:00+01:00",
        time_end="2026-07-01T10:00:20+01:00",
        time_start_unix=_BASE + start,
        time_end_unix=_BASE + end,
        time_start_s=start,
        time_end_s=end,
        duration_visible=end - start,
        direction=direction,
        speed_px_s=speed_px_s,
        color=color,
        lane="middle",
        avg_confidence=0.8,
        displacement_px=500.0,
        net_displacement_px=480.0,
        num_detections=num_detections,
        asset_prefix="person" if cls == "person" else "vehicle",
        entry_point_frac=entry,
        exit_point_frac=exit,
    ).to_json_dict()


# ----------------------------------------------------------------------
# pair_companions
# ----------------------------------------------------------------------


def test_dog_pairs_with_overlapping_same_direction_person() -> None:
    persons = [_rec(1)]
    dogs = [_rec(50, "dog", start=5.0, end=15.0)]
    assert pair_companions(persons, dogs) == {1: [50]}


def test_direction_mismatch_does_not_pair() -> None:
    persons = [_rec(1, direction=_L2R)]
    dogs = [_rec(50, "dog", direction=_R2L)]
    assert pair_companions(persons, dogs) == {}


def test_short_overlap_does_not_pair() -> None:
    persons = [_rec(1, start=0.0, end=20.0)]
    dogs = [_rec(50, "dog", start=19.0, end=30.0)]  # 1 s < default 3 s
    assert pair_companions(persons, dogs) == {}


def test_dog_picks_best_overlap_person() -> None:
    a = _rec(1, start=0.0, end=10.0)  # 4 s overlap with the dog
    b = _rec(2, start=6.0, end=30.0)  # 12 s overlap
    dogs = [_rec(50, "dog", start=6.0, end=18.0)]
    assert pair_companions([a, b], dogs) == {2: [50]}


def test_two_dogs_one_walker() -> None:
    persons = [_rec(1)]
    dogs = [_rec(50, "dog", start=2.0, end=18.0), _rec(51, "dog", start=4.0, end=16.0)]
    assert pair_companions(persons, dogs) == {1: [50, 51]}


def test_short_dog_pairs_on_duration_relative_floor() -> None:
    # A 2 s dog visible next to a long-dwelling person for 1.5 s pairs:
    # the flat 3 s floor would reject it (longer than the whole dog track),
    # but the duration-relative floor is min(3.0, 0.6*2) = 1.2 s.
    person = _rec(1, start=0.0, end=20.0)
    dog = _rec(50, "dog", start=5.0, end=7.0)  # 2 s dwell, 2 s overlap
    assert pair_companions([person], [dog]) == {1: [50]}


def test_short_dog_needs_majority_of_its_life_overlapping() -> None:
    # Same 2 s dog but only 0.5 s of it overlaps the person (< 1.2 s
    # required) -> no pair. Guards against a brief incidental crossing.
    person = _rec(1, start=0.0, end=5.5)
    dog = _rec(50, "dog", start=5.0, end=7.0)  # overlap 5.0..5.5 = 0.5 s
    assert pair_companions([person], [dog]) == {}


def test_lower_overlap_frac_is_more_permissive() -> None:
    # A 2 s dog overlapping a person for only 0.4 s: rejected at the
    # default 0.6 (needs 1.2 s) but paired at 0.15 (needs 0.3 s). The
    # knob lowers the bar directionally.
    person = _rec(1, start=0.0, end=5.4)
    dog = _rec(50, "dog", start=5.0, end=7.0)  # overlap 5.0..5.4 = 0.4 s
    assert pair_companions([person], [dog]) == {}
    assert pair_companions([person], [dog], overlap_frac=0.15) == {1: [50]}


def test_long_companion_still_needs_the_full_cap() -> None:
    # Two long tracks overlapping only 2 s (< 3 s cap): the relative term
    # (0.6*20 = 12 s) is far above the cap, so the cap binds and they
    # don't pair -- long-track behaviour is unchanged by the new floor.
    person = _rec(1, start=0.0, end=20.0)
    dog = _rec(50, "dog", start=18.0, end=40.0)  # 22 s dwell, 2 s overlap
    assert pair_companions([person], [dog]) == {}


# ----------------------------------------------------------------------
# classify_kind
# ----------------------------------------------------------------------


def test_cyclist_beats_speed_split() -> None:
    kind = classify_kind(speed_m_s=3.4, num_detections=58, has_bicycle=True)
    assert kind == "cyclist"


def test_jogger_above_threshold() -> None:
    assert classify_kind(speed_m_s=3.0, num_detections=58, has_bicycle=False) == "jogger"


def test_walker_below_threshold() -> None:
    assert classify_kind(speed_m_s=1.5, num_detections=58, has_bicycle=False) == "walker"


def test_glitch_track_is_not_a_jogger() -> None:
    # 3-frame BotSORT glitches produce huge speeds; the detection guard
    # keeps them out of the jogger bucket.
    assert classify_kind(speed_m_s=8.0, num_detections=3, has_bicycle=False) == "walker"


def test_uncalibrated_speed_is_walker() -> None:
    assert classify_kind(speed_m_s=None, num_detections=58, has_bicycle=False) == "walker"


# ----------------------------------------------------------------------
# build_people
# ----------------------------------------------------------------------


def test_build_people_kinds_flags_and_summary() -> None:
    records = [
        _rec(1, speed_px_s=30.0),  # walker + dog
        _rec(2, speed_px_s=60.0),  # jogger (3.0 m/s @ 0.05)
        _rec(3, speed_px_s=90.0),  # cyclist (bicycle-paired)
        _rec(50, "dog", start=2.0, end=18.0),  # pairs with 1 (only walker-speed
        _rec(60, "bicycle", start=2.0, end=18.0),  #   ... see below)
        _rec(99, "car"),  # ignored
    ]
    # Aim the dog at person 1 and the bicycle at person 3 by direction.
    records[3]["direction"] = _L2R
    for r in (records[1], records[4]):
        r["direction"] = _R2L
    records[2]["direction"] = _R2L
    # person 2 now R->L alongside the bicycle -- give person 3 the longer
    # overlap so the bicycle picks it.
    records[1]["time_end_unix"] = _BASE + 8.0

    people, summary = build_people(records, m_per_px=0.05)
    by_id = {p.track_id: p for p in people}

    assert set(by_id) == {1, 2, 3}
    assert by_id[1].kind == "walker" and by_id[1].dog_walker
    assert by_id[1].dog_track_ids == [50]
    assert by_id[2].kind == "jogger" and not by_id[2].dog_walker
    assert by_id[3].kind == "cyclist"
    assert by_id[3].bicycle_track_ids == [60]
    assert by_id[1].speed_m_s == 1.5

    assert summary == {
        "n_person_tracks": 3,
        "walkers": 1,
        "joggers": 1,
        "cyclists": 1,
        "dog_walkers": 1,
        "n_dog_tracks": 1,
        "n_dogs_paired": 1,
        "n_bicycle_tracks": 1,
        "n_bicycles_paired": 1,
        "n_suspect_excluded": 0,
        "walks": 3,
        "n_split_merged": 0,
        "round_trips": 0,
        "round_trips_chance": None,  # session far too short for the control
        "away_minutes_median": None,
        "own_trips": None,  # no door zone configured
    }


def test_build_people_uncalibrated() -> None:
    people, summary = build_people([_rec(1, speed_px_s=500.0)], m_per_px=None)
    assert people[0].speed_m_s is None
    assert people[0].kind == "walker"
    assert summary["joggers"] == 0


def test_group_walks_merges_split_fragments() -> None:
    from streettracker.analysis.people import group_walks

    a = _rec(1, start=0.0, end=10.0)
    b = _rec(2, start=12.0, end=20.0)  # 2 s gap: same walk
    c = _rec(3, start=22.5, end=30.0)  # 2.5 s after b: chains on
    assert group_walks([a, b, c]) == {1: 1, 2: 1, 3: 1}


def test_group_walks_respects_direction_and_gap() -> None:
    from streettracker.analysis.people import group_walks

    a = _rec(1, start=0.0, end=10.0)
    other_dir = _rec(2, start=11.0, end=20.0, direction=_R2L)
    too_late = _rec(3, start=14.0, end=25.0)  # 4 s after a: new walk
    walks = group_walks([a, other_dir, too_late])
    assert walks == {1: 1, 2: 2, 3: 3}


def test_group_walks_colour_rules() -> None:
    from streettracker.analysis.people import group_walks

    a = _rec(1, start=0.0, end=10.0)
    a["color"] = "red"
    diff = _rec(2, start=11.0, end=20.0)
    diff["color"] = "blue"  # known-different: two people
    unk = _rec(3, start=11.0, end=20.0)
    unk["color"] = "unknown"  # unknown: compatible with a
    assert group_walks([a, diff]) == {1: 1, 2: 2}
    assert group_walks([a, unk]) == {1: 1, 3: 1}


def test_group_walks_leaves_companions_apart() -> None:
    """Two people walking together overlap for most of their tracks --
    far beyond the 2 s overlap bound -- and must stay separate walks."""
    from streettracker.analysis.people import group_walks

    a = _rec(1, start=0.0, end=12.0)
    b = _rec(2, start=1.0, end=13.0)
    assert group_walks([a, b]) == {1: 1, 2: 2}


def test_build_people_reports_walks_and_walk_ids() -> None:
    a = _rec(1, start=0.0, end=10.0)
    b = _rec(2, start=12.0, end=20.0)  # fragment of a's walk
    c = _rec(3, start=100.0, end=110.0)
    people, summary = build_people([a, b, c], m_per_px=0.05)
    by_id = {p.track_id: p for p in people}
    assert by_id[1].walk_id == 1 and by_id[2].walk_id == 1
    assert by_id[3].walk_id == 3
    assert summary["walks"] == 2
    assert summary["n_split_merged"] == 1


def test_build_people_excludes_class_suspect_tracks() -> None:
    suspect = _rec(1)
    suspect["class_suspect"] = True
    people, summary = build_people([suspect, _rec(2)], m_per_px=0.05)

    assert [p.track_id for p in people] == [2]
    assert summary["n_person_tracks"] == 1
    assert summary["n_suspect_excluded"] == 1


# ----------------------------------------------------------------------
# round trips (out-and-back pairing)
# ----------------------------------------------------------------------


def test_round_trip_pairs_opposite_directions() -> None:
    out_ = _rec(1, start=0.0, end=20.0, direction=_L2R)
    back = _rec(2, start=620.0, end=640.0, direction=_R2L)
    people, summary = build_people([out_, back], m_per_px=0.05)
    by_id = {p.track_id: p for p in people}
    assert by_id[1].round_trip_id == 1
    assert by_id[2].round_trip_id == 1
    assert summary["round_trips"] == 1
    assert summary["away_minutes_median"] == 10.0  # (620-20)/60


def test_round_trip_gap_floor_and_ceiling() -> None:
    out_ = _rec(1, start=0.0, end=20.0, direction=_L2R)
    too_soon = _rec(2, start=80.0, end=100.0, direction=_R2L)  # gap 60 < 180
    people, summary = build_people([out_, too_soon], m_per_px=0.05)
    assert all(p.round_trip_id is None for p in people)
    assert summary["round_trips"] == 0

    too_late = _rec(2, start=20.0 + 7300.0, end=20.0 + 7320.0, direction=_R2L)
    people, summary = build_people([out_, too_late], m_per_px=0.05)
    assert all(p.round_trip_id is None for p in people)


def test_round_trip_known_different_colours_block() -> None:
    out_ = _rec(1, start=0.0, end=20.0, direction=_L2R, color="red")
    back = _rec(2, start=620.0, end=640.0, direction=_R2L, color="blue")
    people, summary = build_people([out_, back], m_per_px=0.05)
    assert summary["round_trips"] == 0

    # An unknown on either side is compatible (mirrors the walk-merge rule).
    back_unknown = _rec(2, start=620.0, end=640.0, direction=_R2L)
    _, summary = build_people([out_, back_unknown], m_per_px=0.05)
    assert summary["round_trips"] == 1


def test_round_trip_dog_flag_must_match() -> None:
    out_ = _rec(1, start=0.0, end=20.0, direction=_L2R)
    dog = _rec(50, "dog", start=2.0, end=18.0, direction=_L2R)
    back = _rec(2, start=620.0, end=640.0, direction=_R2L)  # no dog
    people, summary = build_people([out_, dog, back], m_per_px=0.05)
    assert summary["dog_walkers"] == 1
    assert summary["round_trips"] == 0

    dog_back = _rec(51, "dog", start=622.0, end=638.0, direction=_R2L)
    _, summary = build_people([out_, dog, back, dog_back], m_per_px=0.05)
    assert summary["round_trips"] == 1


def test_round_trip_kind_must_match() -> None:
    # 0.1 m/px: 30 px/s -> 3.0 m/s jogger out; 15 px/s -> 1.5 m/s walker back.
    out_ = _rec(1, start=0.0, end=20.0, direction=_L2R, speed_px_s=30.0)
    back = _rec(2, start=620.0, end=640.0, direction=_R2L, speed_px_s=15.0)
    _, summary = build_people([out_, back], m_per_px=0.1)
    assert summary["round_trips"] == 0


def test_round_trip_prefers_most_recent_outbound() -> None:
    early = _rec(1, start=0.0, end=20.0, direction=_L2R)
    late = _rec(2, start=300.0, end=320.0, direction=_L2R)
    back = _rec(3, start=920.0, end=940.0, direction=_R2L)
    people, summary = build_people([early, late, back], m_per_px=0.05)
    by_id = {p.track_id: p for p in people}
    assert by_id[3].round_trip_id == 2
    assert by_id[2].round_trip_id == 2
    assert by_id[1].round_trip_id is None
    assert summary["round_trips"] == 1


def test_round_trip_chance_control() -> None:
    out_ = _rec(1, start=0.0, end=20.0, direction=_L2R)
    back = _rec(2, start=620.0, end=640.0, direction=_R2L)
    _, summary = build_people([out_, back], m_per_px=0.05)
    # ~11 min of data: far too short for the time-shift control.
    assert summary["round_trips_chance"] is None

    # Stretch the session past 2x the shift window; the shifted return
    # lands mid-gap with no partner in range, so chance = 0 while the
    # real pairing still finds the trip.
    filler = _rec(3, start=17000.0, end=17020.0, direction=_L2R)
    _, summary = build_people([out_, back, filler], m_per_px=0.05)
    assert summary["round_trips"] == 1
    assert summary["round_trips_chance"] == 0


def test_round_trip_spans_split_fragments() -> None:
    a1 = _rec(1, start=0.0, end=10.0, direction=_L2R)
    a2 = _rec(2, start=12.0, end=20.0, direction=_L2R)  # fragment of 1's walk
    back = _rec(3, start=620.0, end=640.0, direction=_R2L)
    people, summary = build_people([a1, a2, back], m_per_px=0.05)
    by_id = {p.track_id: p for p in people}
    assert by_id[1].round_trip_id == by_id[2].round_trip_id == by_id[3].round_trip_id == 1
    assert summary["round_trips"] == 1


# ----------------------------------------------------------------------
# door-origin ("my walks")
# ----------------------------------------------------------------------

# Door zone in the bottom-left corner.
_DOOR = DoorZone(polygon_frac=[[0.0, 0.7], [0.2, 0.7], [0.2, 1.0], [0.0, 1.0]])


def test_door_origin_tags_rows_and_counts_own_trips() -> None:
    records = [
        _rec(1, entry=[0.1, 0.85], exit=[0.6, 0.2]),  # left the door
        _rec(2, entry=[0.6, 0.2], exit=[0.1, 0.85]),  # back to the door
        _rec(3, entry=[0.6, 0.2], exit=[0.4, 0.25]),  # passing
        _rec(4),  # no entry/exit points -> unknown
    ]
    people, summary = build_people(records, m_per_px=0.05, door_zone=_DOOR)
    by_id = {p.track_id: p for p in people}
    assert by_id[1].door_origin == "originated"
    assert by_id[2].door_origin == "returned"
    assert by_id[3].door_origin == "passing"
    assert by_id[4].door_origin == "unknown"
    assert summary["own_trips"] == 2  # rows 1 and 2 touch the door


def test_own_trips_none_without_door_zone() -> None:
    _, summary = build_people([_rec(1, entry=[0.1, 0.85], exit=[0.6, 0.2])], m_per_px=0.05)
    assert summary["own_trips"] is None
    people, _ = build_people([_rec(1, entry=[0.1, 0.85])], m_per_px=0.05)
    assert people[0].door_origin == ""  # unset when no zone configured


# ----------------------------------------------------------------------
# resolve_m_per_px
# ----------------------------------------------------------------------


def test_resolve_explicit_wins(tmp_path: Path) -> None:
    cfg = tmp_path / "showcase.json"
    cfg.write_text(json.dumps({"m_per_px": 0.9}))
    assert resolve_m_per_px(0.05, 100.0, config_path=cfg) == 0.05


def test_resolve_road_length(tmp_path: Path) -> None:
    got = resolve_m_per_px(None, 801.0, config_path=tmp_path / "missing.json")
    assert got == 1.0  # 801 m over the 801 px axis


def test_resolve_from_config(tmp_path: Path) -> None:
    cfg = tmp_path / "showcase.json"
    cfg.write_text(json.dumps({"road_length_m": 801.0}))
    assert resolve_m_per_px(None, None, config_path=cfg) == 1.0


def test_resolve_uncalibrated(tmp_path: Path) -> None:
    assert resolve_m_per_px(None, None, config_path=tmp_path / "missing.json") is None


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def test_cli_writes_people_json(tmp_path: Path) -> None:
    session = tmp_path / "session_test"
    session.mkdir()
    records = [
        _rec(1),
        _rec(50, "dog", start=2.0, end=18.0),
        _rec(99, "car"),
    ]
    (session / "session_test_data.json").write_text(json.dumps(records))

    assert main([str(session), "--m-per-px", "0.05"]) == 0

    out = json.loads((session / "session_test_people.json").read_text())
    assert out["session"] == "session_test"
    assert out["params"]["m_per_px"] == 0.05
    assert out["summary"]["n_person_tracks"] == 1
    assert out["summary"]["dog_walkers"] == 1
    assert out["people"][0]["track_id"] == 1
    assert out["people"][0]["dog_track_ids"] == [50]


def test_cli_missing_dir(tmp_path: Path) -> None:
    assert main([str(tmp_path / "nope")]) == 2
