"""Per-person-track activity enrichment.

Folds a session's person tracks (``data.json``) into
``{session}_people.json``: each person track gets an activity ``kind``
(``walker`` / ``jogger`` / ``cyclist``) plus companion pairings — dog
tracks set the ``dog_walker`` flag, bicycle tracks produce the
``cyclist`` kind.

Track records carry no trajectory, so pairing is temporal +
directional: a companion track (dog / bicycle) is attributed to the
person track with the largest time overlap in the same recorded
direction whose overlap clears a *duration-relative* floor —
``min(--min-overlap-s, --overlap-frac * min(companion_dwell,
person_dwell))``. Dogs and bikes often cross in under ``--min-overlap-s``;
the fraction lets a short companion pair on most of its own brief life
while long tracks keep the absolute cap. Each companion pairs with at
most one person; a person can accumulate several companions (two dogs,
one walker).

Speed: ``speed_px_s`` is inference-frame pixels. With a calibration
(``--m-per-px``, ``--road-length-m``, or ``configs/showcase.json`` —
same resolution scheme as the showcase) speeds convert to m/s and the
walker/jogger split happens at ``--jogger-min-ms`` (default 2.5 m/s:
the valley between the walking mass and the jogging bump in the
five-soak person-coverage measurement of 2026-07-07,
``.claude/person_coverage.py``). Bicycle-paired tracks are classified
``cyclist`` FIRST — riders detect as person, and the soak's >4 m/s
person-speed tail was riders, so a speed-only jogger split would
mislabel them. Uncalibrated sessions classify every non-cyclist as a
walker (surfaced in the output's ``params``).

Round trips (out-and-back): a walk pairs with a later opposite-direction
walk — same kind, same dog_walker flag, colours not known-different —
starting within ``[--rt-min-gap-s, --rt-max-gap-s]`` of its end. Both
walks' rows share a ``round_trip_id``. This is a same-day, within-session
ESTIMATE: on this pavement ~75-80 % of raw pairs are coincidence (two
different people satisfying the rule), so the summary also carries
``round_trips_chance`` -- the pair count a time-shifted control produces
-- and the honest headline is the excess of real over chance. Nothing is
ever matched across days.

Output schema::

    {
      "session": "session_20260628_073521",
      "params": {"m_per_px": 0.0421, "jogger_min_m_s": 2.5,
                 "min_overlap_s": 3.0, "overlap_frac": 0.6,
                 "walk_gap_s": 3.0, "rt_min_gap_s": 180.0,
                 "rt_max_gap_s": 7200.0},
      "summary": {"n_person_tracks": 4211, "walkers": ..., "joggers": ...,
                  "cyclists": ..., "dog_walkers": ...,
                  "n_dog_tracks": ..., "n_dogs_paired": ...,
                  "n_bicycle_tracks": ..., "n_bicycles_paired": ...,
                  "n_suspect_excluded": ..., "walks": ...,
                  "n_split_merged": ..., "round_trips": ...,
                  "round_trips_chance": ..., "away_minutes_median": ...,
                  "own_trips": ...},
      "people": [
        {"track_id": 42, "time_start": "...", "time_end": "...",
         "direction": "left to right", "duration_visible": 12.8,
         "speed_px_s": 37.1, "speed_m_s": 1.56, "num_detections": 61,
         "kind": "walker", "dog_walker": true, "walk_id": 42,
         "dog_track_ids": [43], "bicycle_track_ids": [],
         "color": "red", "round_trip_id": 42, "door_origin": "originated"},
        ...
      ]
    }

Door-origin ("my walks"): when a door zone is configured
(``--door-zone``, default ``configs/door_zone.json``), each row's
``door_origin`` marks whether the walk started/ended in the door zone
(``originated`` / ``returned`` / ``round_trip`` / ``passing`` /
``unknown``); ``summary.own_trips`` counts the household's own
door-touching trips. Needs the per-track entry/exit points recorded by
the post-2026-07-19 runtime, so only sessions captured after that deploy
contribute -- see ``analysis/walks.py``.

One person track ~= one pass, with the same BotSORT-split caveat as
cars. Dog and bicycle tracks only exist where the deployment's
``inference.vehicle_classes`` includes COCO classes 16 / 1.

Run via the CLI::

    streettracker people output/session_20260628_073521
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from streettracker.analysis.walks import (
    DEFAULT_DOOR_ZONE_PATH,
    DoorZone,
    door_origin_for_records,
    is_own_trip,
)

# Walker/jogger boundary, m/s. Empirical: the pooled five-soak person
# speed histogram (13,350 tracks) shows the walking mass ending ~2.0-2.5
# and a distinct jogging bump at 2.5-4.0.
JOGGER_MIN_M_S_DEFAULT = 2.5

# A companion (dog/bicycle) must co-occur with a person to pair. The
# required overlap is ``min(MIN_OVERLAP_S_DEFAULT, OVERLAP_FRAC_DEFAULT *
# min(companion_dwell, person_dwell))`` -- an absolute cap for long
# tracks, but a *duration-relative* floor for short ones. Dogs and bikes
# are small/fast-crossing: on the 2026-07-07 dog first-light soak 44 % of
# dog tracks and 49 % of bike tracks were visible < 3 s, so a flat 3 s
# floor structurally rejected them even when the owner/rider was present
# (16 of 17 unpaired dogs failed the overlap rule, only 1 on direction).
# frac 0.6 lifted dog-pair recall 47 %->81 % and bike 43 %->90 % on that
# soak with dog ambiguity unchanged (see .claude/dog_pair_sweep note in
# the 2026-07-09 review). The cap keeps long tracks needing solid
# co-presence, so a brief passer-by near a long-dwelling person still
# won't pair.
MIN_OVERLAP_S_DEFAULT = 3.0
OVERLAP_FRAC_DEFAULT = 0.6

# Speed classification needs this many detections — same guard as the
# stats page's fastest board, against 2-3 frame BotSORT glitch tracks
# whose displacement/duration spikes.
MIN_DETECTIONS_FOR_SPEED = 6

# Traced-road travel-axis length in the 896x512 inference frame (px).
# Mirrors web/stats.py DEFAULT_ROAD_AXIS_PX (kept local: analysis/ does
# not import from web/). Measured from .claude/triggers_proposal.json.
DEFAULT_ROAD_AXIS_PX = 801.0

# Walk dedup: BotSORT can split one pass into consecutive tracks. Track
# B joins A's walk when it starts within [-WALK_OVERLAP_MAX_S,
# +walk_gap_s] of A's end, same direction, and their clothing-colour
# votes aren't known-different. Six-soak sweep (15,913 person tracks):
# gap 3 s + this colour rule merges 7.6 % of tracks (strict colour
# equality 3.3 % -- too strict, 31 % of tracks vote colour "unknown"
# and split fragments are short/small-crop, exactly the unknown-voters;
# no colour check 10.9 % -- merges distinct same-direction people).
# Companions walking together are naturally excluded: their tracks
# overlap for most of their life, far beyond the -2 s bound.
WALK_GAP_S_DEFAULT = 3.0
_WALK_OVERLAP_MAX_S = 2.0
_WALK_PRUNE_S = 30.0  # drop open walks this stale (bounds the scan)

# Round-trip (out-and-back) pairing: a walk pairs with a LATER walk in
# the opposite direction -- same kind, same dog_walker flag, colours not
# known-different -- whose start falls within [rt_min_gap_s,
# rt_max_gap_s] of the first walk's end. The gap floor keeps
# BotSORT direction-flip artifacts (which surface as an immediate
# "return") out; the ceiling bounds it to an errand, not a workday.
# Pairing is same-day by construction (the ceiling) and greedy
# most-recent-first; each walk joins at most one round trip. This is an
# ESTIMATE: on a busy pavement two different people can satisfy the
# rule, so downstream consumers should label it as such.
RT_MIN_GAP_S_DEFAULT = 180.0
RT_MAX_GAP_S_DEFAULT = 7200.0

_L2R = "left to right"
_R2L = "right to left"
_OPPOSITE = {_L2R: _R2L, _R2L: _L2R}


@dataclass(slots=True)
class PersonTrack:
    """One person track's enrichment row (``people`` array element)."""

    track_id: int
    time_start: str
    time_end: str
    time_start_unix: float
    time_end_unix: float
    direction: str
    duration_visible: float
    speed_px_s: float
    speed_m_s: float | None  # None when uncalibrated
    num_detections: int
    kind: str  # "walker" | "jogger" | "cyclist"
    dog_walker: bool
    # Walk group: BotSORT-split fragments of one pass share a walk_id
    # (the first fragment's track_id). Distinct-walk counting uses this.
    walk_id: int = 0
    dog_track_ids: list[int] = field(default_factory=list)
    bicycle_track_ids: list[int] = field(default_factory=list)
    color: str = "unknown"  # clothing-colour vote, from the track record
    # Out-and-back: both walks of a paired round trip share this id (the
    # outbound walk's walk_id). None = unpaired.
    round_trip_id: int | None = None
    # Door relationship (analysis/walks.py): originated / returned /
    # round_trip / passing / unknown. Only set when a door zone is
    # configured AND the track carries entry/exit points; "" otherwise.
    # "originated"/"returned"/"round_trip" mark a household ("my") trip.
    door_origin: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _overlap_s(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Seconds of wall-clock overlap between two track records."""
    start = max(float(a["time_start_unix"]), float(b["time_start_unix"]))
    end = min(float(a["time_end_unix"]), float(b["time_end_unix"]))
    return end - start


def _dwell_s(r: dict[str, Any]) -> float:
    return float(r.get("time_end_unix") or 0.0) - float(r.get("time_start_unix") or 0.0)


def pair_companions(
    persons: list[dict[str, Any]],
    companions: list[dict[str, Any]],
    *,
    min_overlap_s: float = MIN_OVERLAP_S_DEFAULT,
    overlap_frac: float = OVERLAP_FRAC_DEFAULT,
) -> dict[int, list[int]]:
    """Attribute companion tracks (dogs / bicycles) to person tracks.

    Each companion pairs with the person track it overlaps longest, in
    the same recorded direction, whose overlap meets a duration-relative
    floor: ``min(min_overlap_s, overlap_frac * min(companion_dwell,
    person_dwell))``. Long tracks keep the absolute ``min_overlap_s`` cap;
    short companions (dogs/bikes crossing in < min_overlap_s) only need to
    overlap a fraction of their own brief life -- otherwise they could
    never reach the flat floor. Returns ``{person_track_id:
    [companion_track_id, ...]}`` (only persons that gained a companion
    appear).
    """
    paired: dict[int, list[int]] = {}
    for comp in companions:
        comp_dwell = _dwell_s(comp)
        best_person: int | None = None
        best_overlap = 0.0
        for person in persons:
            if person.get("direction") != comp.get("direction"):
                continue
            ov = _overlap_s(person, comp)
            if ov <= 0.0:
                continue
            required = min(min_overlap_s, overlap_frac * min(comp_dwell, _dwell_s(person)))
            if ov >= required and ov > best_overlap:
                best_overlap = ov
                best_person = person["track_id"]
        if best_person is not None:
            paired.setdefault(best_person, []).append(comp["track_id"])
    return paired


def group_walks(
    persons: list[dict[str, Any]],
    *,
    gap_s: float = WALK_GAP_S_DEFAULT,
) -> dict[int, int]:
    """Group BotSORT-split person tracks into walks.

    Greedy chain over tracks sorted by start time: a track joins an
    open walk when it starts within ``[-2 s, +gap_s]`` of that walk's
    end, moves the same direction, and the colour votes aren't
    known-different (an ``unknown`` on either side is compatible --
    split fragments are short and often vote unknown). Returns
    ``{track_id: walk_id}`` where walk_id is the walk's first track_id.
    """
    walks: dict[int, int] = {}
    # open walk: [end_unix, direction, colour, walk_id]
    open_walks: list[list[Any]] = []
    for r in sorted(persons, key=lambda r: float(r.get("time_start_unix") or 0.0)):
        start = float(r.get("time_start_unix") or 0.0)
        end = float(r.get("time_end_unix") or 0.0)
        d, c = r.get("direction"), r.get("color") or "unknown"
        joined: list[Any] | None = None
        for w in open_walks:
            if d != w[1]:
                continue
            if not (-_WALK_OVERLAP_MAX_S <= start - w[0] <= gap_s):
                continue
            if c != "unknown" and w[2] != "unknown" and c != w[2]:
                continue
            joined = w
            break
        if joined is None:
            open_walks.append([end, d, c, r["track_id"]])
            walks[r["track_id"]] = r["track_id"]
        else:
            joined[0] = max(joined[0], end)
            if joined[2] == "unknown":
                joined[2] = c
            walks[r["track_id"]] = joined[3]
        open_walks = [w for w in open_walks if w[0] >= start - _WALK_PRUNE_S]
    return walks


@dataclass(slots=True)
class _Walk:
    """Per-walk aggregate over its split fragments, for round-trip pairing."""

    walk_id: int
    start_unix: float
    end_unix: float
    direction: str
    kind: str
    dog_walker: bool
    color: str


def _aggregate_walks(rows: list[PersonTrack]) -> list[_Walk]:
    """Fold enriched person rows into one aggregate per walk_id. Kind
    comes from the fragment with the most detections (the best-observed
    one); colour is the first known vote; dog_walker if any fragment
    paired a dog. Fragments share a direction by group_walks' rule."""
    by_walk: dict[int, list[PersonTrack]] = {}
    for p in rows:
        by_walk.setdefault(p.walk_id, []).append(p)
    walks: list[_Walk] = []
    for wid, frags in by_walk.items():
        lead = max(frags, key=lambda p: p.num_detections)
        known = [p.color for p in frags if p.color != "unknown"]
        walks.append(
            _Walk(
                walk_id=wid,
                start_unix=min(p.time_start_unix for p in frags),
                end_unix=max(p.time_end_unix for p in frags),
                direction=frags[0].direction,
                kind=lead.kind,
                dog_walker=any(p.dog_walker for p in frags),
                color=known[0] if known else "unknown",
            )
        )
    return walks


def chance_round_trips(
    walks: list[_Walk],
    *,
    min_gap_s: float = RT_MIN_GAP_S_DEFAULT,
    max_gap_s: float = RT_MAX_GAP_S_DEFAULT,
) -> int | None:
    """How many round-trip pairs pure coincidence produces.

    Re-runs the pairing with one direction's walks circularly
    time-shifted (well past the pairing window) inside the session's
    span -- genuine out-and-backs are destroyed, walk density and the
    time-of-day profile are preserved, so surviving pairs measure the
    chance-match floor. Measured 2026-07-19 on three multi-day soaks:
    ~75-80 % of raw pairs are chance on this pavement, so the honest
    headline is the EXCESS of real over chance, not the raw count.
    Returns ``None`` when the session is too short (< 2x the shift) for
    the control to mean anything.
    """
    if not walks:
        return None
    t0 = min(w.start_unix for w in walks)
    span = max(w.end_unix for w in walks) - t0
    shift = max_gap_s + 600.0
    if span < 2.0 * shift:
        return None
    shifted: list[_Walk] = []
    for w in walks:
        if w.direction == _R2L:
            new_start = t0 + (w.start_unix - t0 + shift) % span
            w = _Walk(
                walk_id=w.walk_id,
                start_unix=new_start,
                end_unix=new_start + (w.end_unix - w.start_unix),
                direction=w.direction,
                kind=w.kind,
                dog_walker=w.dog_walker,
                color=w.color,
            )
        shifted.append(w)
    pairs = pair_round_trips(shifted, min_gap_s=min_gap_s, max_gap_s=max_gap_s)
    return len(set(pairs.values()))


def pair_round_trips(
    walks: list[_Walk],
    *,
    min_gap_s: float = RT_MIN_GAP_S_DEFAULT,
    max_gap_s: float = RT_MAX_GAP_S_DEFAULT,
) -> dict[int, int]:
    """Pair walks into out-and-back round trips.

    Greedy over walks in start order: each walk tries to be the RETURN
    of the most recently ended unmatched walk in the opposite direction
    with the same kind + dog_walker flag, colours not known-different,
    and a gap (this start - that end) inside ``[min_gap_s, max_gap_s]``.
    Each walk joins at most one round trip. Returns ``{walk_id:
    round_trip_id}`` for both members of every pair (the outbound
    walk's id); unpaired walks are absent.
    """
    result: dict[int, int] = {}
    candidates: list[_Walk] = []  # ended, so far unmatched
    for w in sorted(walks, key=lambda w: w.start_unix):
        want_dir = _OPPOSITE.get(w.direction)
        best: _Walk | None = None
        best_gap = 0.0
        if want_dir is not None:
            for a in candidates:
                if a.direction != want_dir or a.kind != w.kind or a.dog_walker != w.dog_walker:
                    continue
                gap = w.start_unix - a.end_unix
                if not (min_gap_s <= gap <= max_gap_s):
                    continue
                if a.color != "unknown" and w.color != "unknown" and a.color != w.color:
                    continue
                if best is None or gap < best_gap:
                    best, best_gap = a, gap
        if best is not None:
            result[best.walk_id] = best.walk_id
            result[w.walk_id] = best.walk_id
            candidates.remove(best)
        else:
            candidates.append(w)
        candidates = [a for a in candidates if w.start_unix - a.end_unix <= max_gap_s]
    return result


def classify_kind(
    *,
    speed_m_s: float | None,
    num_detections: int,
    has_bicycle: bool,
    jogger_min_m_s: float = JOGGER_MIN_M_S_DEFAULT,
) -> str:
    """Activity kind for one person track. Cyclist wins over speed —
    riders detect as person and would otherwise land in the jogger
    bucket. Jogger needs a calibrated speed and enough detections to
    trust it; everything else is a walker."""
    if has_bicycle:
        return "cyclist"
    if (
        speed_m_s is not None
        and num_detections >= MIN_DETECTIONS_FOR_SPEED
        and speed_m_s >= jogger_min_m_s
    ):
        return "jogger"
    return "walker"


def build_people(
    records: list[dict[str, Any]],
    *,
    m_per_px: float | None = None,
    jogger_min_m_s: float = JOGGER_MIN_M_S_DEFAULT,
    min_overlap_s: float = MIN_OVERLAP_S_DEFAULT,
    overlap_frac: float = OVERLAP_FRAC_DEFAULT,
    walk_gap_s: float = WALK_GAP_S_DEFAULT,
    rt_min_gap_s: float = RT_MIN_GAP_S_DEFAULT,
    rt_max_gap_s: float = RT_MAX_GAP_S_DEFAULT,
    door_zone: DoorZone | None = None,
) -> tuple[list[PersonTrack], dict[str, Any]]:
    """Fold a session's raw track records into person enrichment rows
    plus the summary block. ``records`` is the parsed ``data.json``
    array; non-person/dog/bicycle records are ignored.

    When ``door_zone`` is given, each person row gets a ``door_origin``
    classification (see ``analysis/walks.py``) and the summary counts the
    household's own door-touching trips."""
    # class_suspect = the runtime's kinematics guardrail (car-shaped
    # "person" bboxes, i.e. persistently misclassified parked cars);
    # excluded from people analytics but counted in the summary.
    persons = [r for r in records if r.get("class_name") == "person" and not r.get("class_suspect")]
    n_suspect = sum(
        1 for r in records if r.get("class_name") == "person" and r.get("class_suspect")
    )
    dogs = [r for r in records if r.get("class_name") == "dog"]
    bikes = [r for r in records if r.get("class_name") == "bicycle"]

    dog_by_person = pair_companions(
        persons, dogs, min_overlap_s=min_overlap_s, overlap_frac=overlap_frac
    )
    bike_by_person = pair_companions(
        persons, bikes, min_overlap_s=min_overlap_s, overlap_frac=overlap_frac
    )
    walk_by_person = group_walks(persons, gap_s=walk_gap_s)
    door_by_person = door_origin_for_records(records, door_zone) if door_zone else {}

    out: list[PersonTrack] = []
    for r in persons:
        tid = r["track_id"]
        speed_px_s = float(r.get("speed_px_s") or 0.0)
        speed_m_s = round(speed_px_s * m_per_px, 2) if m_per_px is not None else None
        num_detections = int(r.get("num_detections") or 0)
        dog_ids = sorted(dog_by_person.get(tid, []))
        bike_ids = sorted(bike_by_person.get(tid, []))
        out.append(
            PersonTrack(
                track_id=tid,
                time_start=r.get("time_start") or "",
                time_end=r.get("time_end") or "",
                time_start_unix=float(r.get("time_start_unix") or 0.0),
                time_end_unix=float(r.get("time_end_unix") or 0.0),
                direction=r.get("direction") or "",
                duration_visible=float(r.get("duration_visible") or 0.0),
                speed_px_s=speed_px_s,
                speed_m_s=speed_m_s,
                num_detections=num_detections,
                kind=classify_kind(
                    speed_m_s=speed_m_s,
                    num_detections=num_detections,
                    has_bicycle=bool(bike_ids),
                    jogger_min_m_s=jogger_min_m_s,
                ),
                dog_walker=bool(dog_ids),
                walk_id=walk_by_person.get(tid, tid),
                dog_track_ids=dog_ids,
                bicycle_track_ids=bike_ids,
                color=r.get("color") or "unknown",
                door_origin=door_by_person.get(tid, ""),
            )
        )

    # Out-and-back pairing over walk aggregates; stamp both members'
    # fragments with the shared round_trip_id.
    walk_aggs = _aggregate_walks(out)
    rt_by_walk = pair_round_trips(walk_aggs, min_gap_s=rt_min_gap_s, max_gap_s=rt_max_gap_s)
    for p in out:
        p.round_trip_id = rt_by_walk.get(p.walk_id)
    agg_by_id = {w.walk_id: w for w in walk_aggs}
    away_minutes: list[float] = []
    for rt_id in set(rt_by_walk.values()):
        pair = sorted(
            (agg_by_id[wid] for wid, r in rt_by_walk.items() if r == rt_id),
            key=lambda w: w.start_unix,
        )
        if len(pair) == 2:
            away_minutes.append((pair[1].start_unix - pair[0].end_unix) / 60.0)
    away_minutes.sort()
    n_chance = chance_round_trips(walk_aggs, min_gap_s=rt_min_gap_s, max_gap_s=rt_max_gap_s)

    summary = {
        "n_person_tracks": len(out),
        "walkers": sum(1 for p in out if p.kind == "walker"),
        "joggers": sum(1 for p in out if p.kind == "jogger"),
        "cyclists": sum(1 for p in out if p.kind == "cyclist"),
        "dog_walkers": sum(1 for p in out if p.dog_walker),
        "n_dog_tracks": len(dogs),
        "n_dogs_paired": sum(len(v) for v in dog_by_person.values()),
        "n_bicycle_tracks": len(bikes),
        "n_bicycles_paired": sum(len(v) for v in bike_by_person.values()),
        "n_suspect_excluded": n_suspect,
        # Distinct walks after BotSORT-split merging; the difference to
        # n_person_tracks is how many fragments were folded in.
        "walks": len(set(walk_by_person.values())),
        "n_split_merged": len(out) - len(set(walk_by_person.values())),
        # Out-and-back estimate: pairs of opposite-direction walks that
        # look like the same person going and coming back (same day by
        # construction -- the gap ceiling). round_trips is the RAW pair
        # count; round_trips_chance is what a time-shifted control pairs
        # (~75-80 % of raw on this pavement), so the likely-genuine
        # number is the excess -- see chance_round_trips.
        "round_trips": len(away_minutes),
        "round_trips_chance": n_chance,
        "away_minutes_median": (
            round(away_minutes[len(away_minutes) // 2], 1) if away_minutes else None
        ),
        # Door-origin (household "my") trips: walks that started or ended
        # in the operator-marked door zone. None when no door zone was
        # configured; 0+ once one is (needs entry/exit points, so only
        # sessions captured after that runtime change contribute).
        "own_trips": (sum(1 for p in out if is_own_trip(p.door_origin)) if door_zone else None),
    }
    return out, summary


def resolve_m_per_px(
    m_per_px: float | None,
    road_length_m: float | None,
    config_path: Path = Path("configs/showcase.json"),
) -> float | None:
    """Calibration resolution, mirroring the showcase's scheme: explicit
    ``--m-per-px`` wins, then ``--road-length-m`` / road-axis pixels,
    then ``configs/showcase.json`` (either key), else uncalibrated."""
    if m_per_px is not None:
        return m_per_px
    if road_length_m is not None:
        return road_length_m / DEFAULT_ROAD_AXIS_PX
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if cfg.get("m_per_px") is not None:
            return float(cfg["m_per_px"])
        if cfg.get("road_length_m") is not None:
            return float(cfg["road_length_m"]) / DEFAULT_ROAD_AXIS_PX
    return None


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="streettracker people",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_dir", type=Path)
    ap.add_argument(
        "--jogger-min-ms",
        type=float,
        default=JOGGER_MIN_M_S_DEFAULT,
        help=f"walker/jogger boundary in m/s (default {JOGGER_MIN_M_S_DEFAULT})",
    )
    ap.add_argument(
        "--min-overlap-s",
        type=float,
        default=MIN_OVERLAP_S_DEFAULT,
        help="absolute cap on the person/companion overlap needed to pair, "
        f"seconds (default {MIN_OVERLAP_S_DEFAULT})",
    )
    ap.add_argument(
        "--overlap-frac",
        type=float,
        default=OVERLAP_FRAC_DEFAULT,
        help="duration-relative overlap floor: a short companion needs to "
        "overlap this fraction of the shorter track's dwell (default "
        f"{OVERLAP_FRAC_DEFAULT}); the cap above still applies to long tracks",
    )
    ap.add_argument(
        "--walk-gap-s",
        type=float,
        default=WALK_GAP_S_DEFAULT,
        help=f"max gap between split fragments of one walk, seconds (default {WALK_GAP_S_DEFAULT})",
    )
    ap.add_argument(
        "--rt-min-gap-s",
        type=float,
        default=RT_MIN_GAP_S_DEFAULT,
        help="round-trip pairing: minimum time away before an opposite-direction "
        f"walk can be a return, seconds (default {RT_MIN_GAP_S_DEFAULT:.0f})",
    )
    ap.add_argument(
        "--rt-max-gap-s",
        type=float,
        default=RT_MAX_GAP_S_DEFAULT,
        help=f"round-trip pairing: maximum time away, seconds (default {RT_MAX_GAP_S_DEFAULT:.0f})",
    )
    ap.add_argument(
        "--m-per-px",
        type=float,
        default=None,
        help="speed calibration; overrides --road-length-m and configs/showcase.json",
    )
    ap.add_argument(
        "--road-length-m",
        type=float,
        default=None,
        help="visible road length in metres (converted via the traced "
        f"road axis, {DEFAULT_ROAD_AXIS_PX:.0f} px)",
    )
    ap.add_argument(
        "--door-zone",
        type=Path,
        default=DEFAULT_DOOR_ZONE_PATH,
        help="operator-traced door polygon for door-origin ('my walks') "
        f"detection (default {DEFAULT_DOOR_ZONE_PATH}; skipped if absent)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default <session>/<session>_people.json)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2
    session = session_dir.name
    data_path = session_dir / f"{session}_data.json"
    if not data_path.exists():
        print(f"missing {data_path}", file=sys.stderr)
        return 2
    records = json.loads(data_path.read_text(encoding="utf-8"))

    m_per_px = resolve_m_per_px(args.m_per_px, args.road_length_m)
    door_zone = DoorZone.load(args.door_zone)
    people, summary = build_people(
        records,
        m_per_px=m_per_px,
        jogger_min_m_s=args.jogger_min_ms,
        min_overlap_s=args.min_overlap_s,
        overlap_frac=args.overlap_frac,
        walk_gap_s=args.walk_gap_s,
        rt_min_gap_s=args.rt_min_gap_s,
        rt_max_gap_s=args.rt_max_gap_s,
        door_zone=door_zone,
    )

    out_path = args.out or (session_dir / f"{session}_people.json")
    out_path.write_text(
        json.dumps(
            {
                "session": session,
                "params": {
                    "m_per_px": m_per_px,
                    "jogger_min_m_s": args.jogger_min_ms,
                    "min_overlap_s": args.min_overlap_s,
                    "overlap_frac": args.overlap_frac,
                    "walk_gap_s": args.walk_gap_s,
                    "rt_min_gap_s": args.rt_min_gap_s,
                    "rt_max_gap_s": args.rt_max_gap_s,
                    "door_zone": str(args.door_zone) if door_zone else None,
                },
                "summary": summary,
                "people": [p.to_json_dict() for p in people],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[people] wrote {out_path}  ({summary['n_person_tracks']} person tracks, "
        f"{summary['walks']} walks after split-merge)"
    )
    print(
        f"  walkers: {summary['walkers']}   joggers: {summary['joggers']}   "
        f"cyclists: {summary['cyclists']}   dog walkers: {summary['dog_walkers']}"
    )
    if summary["round_trips"]:
        chance = summary["round_trips_chance"]
        extra = (
            f", ~{max(0, summary['round_trips'] - chance)} beyond chance"
            if chance is not None
            else ""
        )
        print(
            f"  round trips: {summary['round_trips']} "
            f"(median away {summary['away_minutes_median']} min{extra})"
        )
    if summary["n_dog_tracks"]:
        loose = summary["n_dog_tracks"] - summary["n_dogs_paired"]
        print(
            f"  dog tracks: {summary['n_dog_tracks']}  "
            f"(paired {summary['n_dogs_paired']}, unpaired {loose})"
        )
    if door_zone is not None:
        classified = sum(1 for p in people if p.door_origin and p.door_origin != "unknown")
        print(
            f"  door-origin (your) trips: {summary['own_trips']} "
            f"({classified}/{summary['n_person_tracks']} tracks had entry/exit points)"
        )
        if classified == 0:
            print(
                "  (no entry/exit points -- this session predates the capture "
                "change; door-origin needs sessions recorded after it deploys)"
            )
    if m_per_px is None:
        print("  (uncalibrated -- no jogger split; set configs/showcase.json or --road-length-m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
