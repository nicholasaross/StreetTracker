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
        color="unknown",
        lane="middle",
        avg_confidence=0.8,
        displacement_px=500.0,
        net_displacement_px=480.0,
        num_detections=num_detections,
        asset_prefix="person" if cls == "person" else "vehicle",
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
