"""Traffic summary statistics for the showcase site.

Computes volume + temporal patterns over **all vehicle tracks** (every
``class_name == "car"`` record), across every session that has a ``data.json``
-- a wider net than the gallery, which only shows the plated subset. (Pruned
JSON-only sessions still carry ``data.json``, so the stats reach further back
than the images do.)

The same pass also aggregates **person tracks** (``class_name == "person"``)
into the ``people`` block: walk counts, direction split, weekday x hour
heatmap, and a time-in-view (dwell) distribution. One person track ~= one
walk past, with the same BotSORT-split caveat as cars; a bicycle rider
detects as a person. Where a session carries ``{session}_people.json``
(``streettracker people``), its summary rolls into ``people["kinds"]`` —
walker/jogger/cyclist counts + dog walks. Cyclist classification needs
bicycle tracks (live since 2026-06-13) and dog pairing needs dog tracks
(live since 2026-07-07); earlier sessions contribute zeros there, and
their jogger counts include unfiltered riders.

Sidecars carrying per-walk rows also feed two aggregate features:
**round trips** (out-and-back pairs computed by ``analysis/people.py``,
pooled here into a count + median time away) and **recurring walk
patterns** (``_habitual_slots``: time-of-day windows where dog walks /
joggers / cyclists recur across >= 3 distinct dates). Both are
statistics about the street, not per-person records -- pairing is
same-day only and patterns carry no identity.

One car ``TrackRecord`` is treated as one "journey"/pass. BotSORT can split a
single pass into two tracks, so counts are approximate -- surfaced as a caveat
in the UI.

Everything is bucketed by ``time_start`` (the ISO-local string with the
camera's tz offset), **not** ``time_start_unix`` + the dev box's tz, so dates /
day-of-week / time-of-day reflect the camera's local clock regardless of where
this runs. A session that spans several calendar dates (e.g. a multi-day live
run) therefore distributes correctly across those dates.

Speed is recorded only in ``px/s`` (inference-frame pixels, see
``track_buffer.compute_attributes``). With a calibration (``m_per_px``) we
convert to mph: ``mph = px/s * m_per_px * 2.2369``. Without one we report px/s
labelled "relative".
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from streettracker.analysis.dvsa import is_canonical_uk_plate
from streettracker.analysis.makemodel.bodytype import body_type_for
from streettracker.analysis.makemodel.colour import colour_class_for
from streettracker.web.aggregate import discover_sessions, resolve_image_urls

# m/s -> mph.
MPH_PER_M_S = 2.2369362920544

# Traced-road travel-axis length in the 896x512 inference frame (px). Used to
# turn a one-number road length (metres) into m_per_px. Measured from the live
# polygon in .claude/triggers_proposal.json; override via road_length_m only if
# the polygon/frame changes.
DEFAULT_ROAD_AXIS_PX = 801.0

_DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
N_FASTEST = 8
N_TOP_MAKES = 12
N_TOP_COLOURS = 10
# Dwell (time-in-view) histogram: 10s buckets up to 60s, then one open
# "60s+" bucket -- median walks sit ~12s on this scene, so six closed
# buckets cover the bulk without flattening the chart.
_DWELL_BUCKET_S = 10.0
_DWELL_MAX_BUCKETS = 6
# A track needs this many detections to be eligible for the "fastest" board --
# guards against 2-3 frame BotSORT ID-switch glitches that produce a huge
# net_disp / duration spike.
_FASTEST_MIN_DETECTIONS = 6
_L2R = "left to right"
_R2L = "right to left"

# Recurring walk patterns: time-of-day clustering of walks per
# (category, direction). Only the sparse, distinctive categories are
# mined -- dog walks / joggers / cyclists; plain walkers are far too
# dense for a time slot to mean anything. A slot is a
# _HABITUAL_WINDOW_MIN-wide window hit on >= _HABITUAL_MIN_DAYS distinct
# dates AND averaging <= _HABITUAL_MAX_WALKS_PER_DAY walks per day seen
# -- the density guard separates "a single recurring walk" from a
# rush-hour wave (first render without it: all 8 slots were jogger
# waves of 150-300 walks hit ~50/55 days, drowning the dog walks).
# These are aggregate statistics about the street ("dog walks recur
# ~07:50"), NOT per-person records -- nothing links a walk on one day
# to a walk on another.
_HABITUAL_WINDOW_MIN = 40
_HABITUAL_MIN_DAYS = 3
_HABITUAL_MAX_SLOTS = 8
_HABITUAL_MAX_WALKS_PER_DAY = 1.6


@dataclass(slots=True)
class Stats:
    """The traffic-stats read model rendered by the ``/stats`` page."""

    overall: dict[str, Any]
    daily: list[dict[str, Any]]
    dow: dict[str, dict[str, int]]
    by_day_15min: dict[str, list[dict[str, int]]]
    heatmap: list[list[int]]  # [weekday 0-6][hour 0-23]
    speed: dict[str, Any]
    makes: list[list[Any]]  # [[make, n_distinct_cars], ...]
    colours: list[list[Any]]  # [[colour, n_journeys], ...]
    bodytypes: list[list[Any]]  # [[body_type, n_journeys], ...]; DVSA-derived + CNN fallback
    people: dict[str, Any]  # person-track aggregates; see _empty_people()
    speed_unit: str  # "mph" | "px/s"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def m_per_px_from_road_length(road_length_m: float, axis_px: float = DEFAULT_ROAD_AXIS_PX) -> float:
    """Metres-per-pixel from the visible road's real length / its axis pixels."""
    return road_length_m / axis_px


def _speed_factor(m_per_px: float | None) -> float | None:
    """mph per (px/s), or ``None`` when uncalibrated."""
    return None if m_per_px is None else m_per_px * MPH_PER_M_S


def _disp(px_s: float, factor: float | None) -> float:
    """px/s -> display unit (mph if calibrated, else px/s), rounded."""
    return round(px_s * factor, 1) if factor is not None else round(px_s)


def _empty_people() -> dict[str, Any]:
    return {
        "total": 0,
        "n_days": 0,
        "per_day_mean": 0,
        "pct_l2r": 0,
        "pct_r2l": 0,
        "busiest_hour": None,
        "dow": {d: {"l2r": 0, "r2l": 0, "total": 0} for d in _DOW},
        "heatmap": [[0] * 24 for _ in range(7)],
        "dwell": {"hist": [], "median_s": 0, "p90_s": 0},
        "kinds": _empty_kinds(),
        "habitual": [],  # recurring time-of-day walk slots; see _habitual_slots
    }


def _empty_kinds() -> dict[str, Any]:
    return {
        "walkers": 0,
        "joggers": 0,
        "cyclists": 0,
        "dog_walks": 0,
        "classified": 0,  # person tracks covered by a _people.json
        "walks": 0,  # distinct walks after BotSORT-split merging
        "pct_walkers": 0,
        "pct_joggers": 0,
        "pct_cyclists": 0,
        "round_trips": 0,  # raw out-and-back pairs (same-day candidates)
        "round_trips_est": None,  # chance-corrected likely-genuine count
        "away_min_median": None,  # median minutes away across those pairs
    }


def _dwell_stats(durations: list[float]) -> dict[str, Any]:
    """Time-in-view distribution: 10s buckets, open-ended past 60s."""
    if not durations:
        return {"hist": [], "median_s": 0, "p90_s": 0}
    ds = sorted(durations)
    median = ds[len(ds) // 2]
    p90 = ds[min(len(ds) - 1, int(0.9 * len(ds)))]
    counts = [0] * (_DWELL_MAX_BUCKETS + 1)
    for v in durations:
        counts[min(int(v // _DWELL_BUCKET_S), _DWELL_MAX_BUCKETS)] += 1
    hist = [
        {
            "lo": i * _DWELL_BUCKET_S,
            "hi": None if i == _DWELL_MAX_BUCKETS else (i + 1) * _DWELL_BUCKET_S,
            "n": c,
        }
        for i, c in enumerate(counts)
    ]
    return {"hist": hist, "median_s": round(median, 1), "p90_s": round(p90, 1)}


def _walks_from_people_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold one session's ``people`` rows into per-walk dicts (walk_id
    groups the BotSORT-split fragments). Category is dog walk > cyclist >
    jogger > walker (dog wins: a jogging dog owner is a dog walk)."""
    by_walk: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        wid_raw = r.get("walk_id") or r.get("track_id")
        if wid_raw is None:
            continue
        try:
            by_walk[int(wid_raw)].append(r)
        except (TypeError, ValueError):
            continue
    walks: list[dict[str, Any]] = []
    for wid, frags in by_walk.items():
        first = min(frags, key=lambda r: float(r.get("time_start_unix") or 0.0))
        ts = first.get("time_start")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        lead = max(frags, key=lambda r: int(r.get("num_detections") or 0))
        kind = lead.get("kind")
        if any(r.get("dog_walker") for r in frags):
            cat = "dog walk"
        elif kind in ("cyclist", "jogger"):
            cat = kind
        else:
            cat = None  # plain walker: not mined for patterns
        direction = first.get("direction")
        rt = first.get("round_trip_id")
        walks.append(
            {
                "walk_id": wid,
                "date": dt.date().isoformat(),
                "minute": dt.hour * 60 + dt.minute,
                "cat": cat,
                "dkey": "l2r" if direction == _L2R else "r2l" if direction == _R2L else None,
                "start_unix": float(first.get("time_start_unix") or 0.0),
                "end_unix": max(float(r.get("time_end_unix") or 0.0) for r in frags),
                "round_trip_id": int(rt) if rt is not None else None,
            }
        )
    return walks


def _habitual_slots(
    points: dict[tuple[str, str], list[tuple[str, int, float | None]]],
    *,
    window_min: int = _HABITUAL_WINDOW_MIN,
    min_days: int = _HABITUAL_MIN_DAYS,
    max_slots: int = _HABITUAL_MAX_SLOTS,
) -> list[dict[str, Any]]:
    """Mine recurring time-of-day walk slots.

    ``points`` maps ``(category, dkey)`` to ``(date, minute-of-day,
    away_minutes|None)`` tuples, one per walk. Greedy per key: slide a
    ``window_min``-wide window over the minute-sorted points (O(n) with
    a date counter), take the window covering the most distinct dates,
    keep it as a slot if that's >= ``min_days`` AND it averages <=
    ``_HABITUAL_MAX_WALKS_PER_DAY`` walks per day seen (denser windows
    are traffic waves, not an individual's habit -- consumed but not
    reported), remove its points and repeat. Slots report the median
    minute, days seen vs the category's total distinct dates, and the
    median away-time of any round trips that started in the slot.
    """
    cat_days: dict[str, set[str]] = defaultdict(set)
    for (cat, _), pts in points.items():
        cat_days[cat].update(d for d, _, _ in pts)

    slots: list[dict[str, Any]] = []
    for (cat, dkey), pts in points.items():
        remaining = sorted(pts, key=lambda p: p[1])
        while remaining:
            cnt: Counter[str] = Counter()
            j = 0
            best_days, best_ij = 0, (0, 0)
            for i in range(len(remaining)):
                if i > 0:
                    d0 = remaining[i - 1][0]
                    cnt[d0] -= 1
                    if not cnt[d0]:
                        del cnt[d0]
                while j < len(remaining) and remaining[j][1] < remaining[i][1] + window_min:
                    cnt[remaining[j][0]] += 1
                    j += 1
                if (len(cnt), j - i) > (best_days, best_ij[1] - best_ij[0]):
                    best_days, best_ij = len(cnt), (i, j)
            if best_days < min_days:
                break
            window = remaining[best_ij[0] : best_ij[1]]
            remaining = remaining[: best_ij[0]] + remaining[best_ij[1] :]
            if len(window) / best_days > _HABITUAL_MAX_WALKS_PER_DAY:
                continue  # a wave, not a habit -- consume and move on
            minutes = sorted(m for _, m, _ in window)
            med = minutes[len(minutes) // 2]
            aways = sorted(a for _, _, a in window if a is not None)
            slots.append(
                {
                    "kind": cat,
                    "dkey": dkey,
                    "time": f"{med // 60:02d}:{med % 60:02d}",
                    "days_seen": best_days,
                    "days_covered": len(cat_days[cat]),
                    "n_walks": len(window),
                    "away_min_median": round(aways[len(aways) // 2], 1) if aways else None,
                }
            )
    # Rank by regularity (fraction of the category's covered days), not raw
    # day count -- dog walks only have data since the dog class went live,
    # so absolute days_seen would bury them under longer-covered categories.
    slots.sort(
        key=lambda s: (
            -s["days_seen"] / max(1, s["days_covered"]),
            -s["days_seen"],
            s["time"],
        )
    )
    return slots[:max_slots]


def _empty_stats(unit: str) -> Stats:
    return Stats(
        overall={
            "total_journeys": 0,
            "n_sessions": 0,
            "n_days": 0,
            "date_from": None,
            "date_to": None,
            "pct_l2r": 0,
            "pct_r2l": 0,
            "busiest_date": None,
            "busiest_hour": None,
            "peak_quarter": None,
            "per_day_mean": 0,
        },
        daily=[],
        dow={d: {"l2r": 0, "r2l": 0, "total": 0} for d in _DOW},
        by_day_15min={},
        heatmap=[[0] * 24 for _ in range(7)],
        speed={"hist": [], "avg_all": 0, "avg_l2r": 0, "avg_r2l": 0, "unit": unit, "fastest": []},
        makes=[],
        colours=[],
        bodytypes=[],
        people=_empty_people(),
        speed_unit=unit,
    )


def build_stats(output_root: Path, *, m_per_px: float | None = None) -> Stats:
    """Aggregate traffic stats across all sessions under ``output_root``."""
    factor = _speed_factor(m_per_px)
    unit = "mph" if factor is not None else "px/s"

    sessions = discover_sessions(output_root)
    if not sessions:
        return _empty_stats(unit)

    daily: dict[str, dict[str, int]] = defaultdict(lambda: {"l2r": 0, "r2l": 0, "total": 0})
    dow = {d: {"l2r": 0, "r2l": 0, "total": 0} for d in _DOW}
    by_day_15: dict[str, list[dict[str, int]]] = defaultdict(
        lambda: [{"l2r": 0, "r2l": 0} for _ in range(96)]
    )
    heatmap = [[0] * 24 for _ in range(7)]
    quarter_profile = [0] * 96  # time-of-day 15-min totals, summed over all days
    hour_totals = [0] * 24

    speeds_l2r: list[float] = []
    speeds_r2l: list[float] = []
    all_speeds: list[float] = []
    colours: Counter[str] = Counter()
    bodytypes: Counter[str] = Counter()
    makes_by_plate: dict[str, str] = {}
    fastest_raw: list[tuple[float, str, dict[str, Any]]] = []
    total = 0
    dates: set[str] = set()

    # People accumulators (person tracks; separate date set so the car
    # per-day mean keeps its existing denominator).
    p_total = 0
    p_dates: set[str] = set()
    p_dow = {d: {"l2r": 0, "r2l": 0, "total": 0} for d in _DOW}
    p_heatmap = [[0] * 24 for _ in range(7)]
    p_hour = [0] * 24
    p_dir = {"l2r": 0, "r2l": 0}
    p_durations: list[float] = []
    p_kinds = _empty_kinds()
    # (category, dkey) -> (date, minute-of-day, away_minutes|None) per walk,
    # for the recurring-pattern miner; away gaps pooled for the tile median.
    p_habit: dict[tuple[str, str], list[tuple[str, int, float | None]]] = defaultdict(list)
    p_away_gaps: list[float] = []
    p_round_trips = 0
    # Chance-match floor, summed over sessions whose sidecar carries the
    # time-shift control (analysis/people.chance_round_trips); the
    # likely-genuine trip count is raw minus this.
    p_rt_chance = 0
    p_rt_chance_known = False

    for d in sessions:
        name = d.name
        try:
            data = json.loads((d / f"{name}_data.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = []

        # Per-track body type for the mix chart: prefer the DVSA-model-derived
        # class (reliable, plated cars) and fall back to the confident CNN
        # prediction (the unreadable majority). Built here so the car loop can
        # count it. Also harvests makes_by_plate from the same DVSA read.
        # Per-track colour for the mix chart follows the same DVSA-first
        # precedence: DVSA register colour (ground truth, plated cars) ->
        # confident colour CNN -> the low-res HSV vote (`color` field) as a
        # last resort. The CNN reads the 4K snap and scores ~87% vs DVSA
        # (grouped) where HSV managed ~40%, so it's strongly preferred over
        # HSV for the unplated majority.
        track_body: dict[int, str] = {}
        track_colour: dict[int, str] = {}
        dl = d / f"{name}_dvsa_labels.json"
        if dl.exists():
            try:
                labels = json.loads(dl.read_text(encoding="utf-8")).get("labels", {})
            except (OSError, json.JSONDecodeError):
                labels = {}
            for plate, row in labels.items():
                make = (row.get("make") or "").strip()
                if make and plate not in makes_by_plate:
                    makes_by_plate[plate] = make
                bt = body_type_for(row.get("make"), row.get("model"))
                col = colour_class_for(row.get("primary_colour"))
                for tid in row.get("track_ids", []):
                    if bt:
                        track_body[int(tid)] = bt
                    if col:
                        track_colour[int(tid)] = col
        bp = d / f"{name}_bodytype_by_track.json"
        if bp.exists():
            try:
                bt_tracks = json.loads(bp.read_text(encoding="utf-8")).get("tracks", [])
            except (OSError, json.JSONDecodeError):
                bt_tracks = []
            for t in bt_tracks:
                tid, bt = t.get("track_id"), t.get("body_type")
                if bt and tid is not None and tid not in track_body:  # DVSA wins
                    track_body[int(tid)] = bt
        cp = d / f"{name}_colour_by_track.json"
        if cp.exists():
            try:
                col_tracks = json.loads(cp.read_text(encoding="utf-8")).get("tracks", [])
            except (OSError, json.JSONDecodeError):
                col_tracks = []
            for t in col_tracks:
                tid, col = t.get("track_id"), t.get("colour")
                if col and tid is not None and int(tid) not in track_colour:  # DVSA wins
                    track_colour[int(tid)] = col

        for r in data:
            cls = r.get("class_name")
            if cls not in ("car", "person"):
                continue
            # Kinematics guardrail: tracks whose voted class is
            # contradicted by bbox geometry (today: car-shaped
            # "persons" -- persistently misclassified parked cars)
            # count as neither people nor cars.
            if r.get("class_suspect"):
                continue
            ts = r.get("time_start")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            date = dt.date().isoformat()
            wd = dt.weekday()
            q = (dt.hour * 60 + dt.minute) // 15  # 0..95
            direction = r.get("direction")
            dkey = "l2r" if direction == _L2R else "r2l" if direction == _R2L else None

            if cls == "person":
                p_total += 1
                p_dates.add(date)
                p_dow[_DOW[wd]]["total"] += 1
                p_heatmap[wd][dt.hour] += 1
                p_hour[dt.hour] += 1
                if dkey:
                    p_dow[_DOW[wd]][dkey] += 1
                    p_dir[dkey] += 1
                dur = float(r.get("duration_visible") or 0.0)
                if dur > 0:
                    p_durations.append(dur)
                continue

            total += 1
            dates.add(date)
            daily[date]["total"] += 1
            dow[_DOW[wd]]["total"] += 1
            heatmap[wd][dt.hour] += 1
            hour_totals[dt.hour] += 1
            tid = r.get("track_id")
            cnn_col = track_colour.get(int(tid)) if tid is not None else None
            colours[cnn_col or r.get("color") or "unknown"] += 1
            bt = track_body.get(r.get("track_id"))
            if bt:
                bodytypes[bt] += 1
            if dkey:
                daily[date][dkey] += 1
                dow[_DOW[wd]][dkey] += 1
                by_day_15[date][q][dkey] += 1
                quarter_profile[q] += 1

            sp = float(r.get("speed_px_s") or 0.0)
            all_speeds.append(sp)
            if direction == _L2R:
                speeds_l2r.append(sp)
            elif direction == _R2L:
                speeds_r2l.append(sp)
            if (r.get("num_detections") or 0) >= _FASTEST_MIN_DETECTIONS:
                fastest_raw.append((sp, name, r))

        pp = d / f"{name}_people.json"
        if pp.exists():
            try:
                pdoc = json.loads(pp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pdoc = {}
            psum = pdoc.get("summary", {})
            p_kinds["walkers"] += int(psum.get("walkers") or 0)
            p_kinds["joggers"] += int(psum.get("joggers") or 0)
            p_kinds["cyclists"] += int(psum.get("cyclists") or 0)
            p_kinds["dog_walks"] += int(psum.get("dog_walkers") or 0)
            p_kinds["classified"] += int(psum.get("n_person_tracks") or 0)
            # Sidecars written before walk dedup lack "walks" -- count
            # their tracks 1:1 so the walks total stays comparable.
            p_kinds["walks"] += int(psum.get("walks") or psum.get("n_person_tracks") or 0)
            if psum.get("round_trips_chance") is not None:
                p_rt_chance += int(psum["round_trips_chance"])
                p_rt_chance_known = True

            # Per-walk rows feed round-trip pooling + the pattern miner.
            # Sidecars written before round trips lack these fields --
            # they just contribute no pairs / no away times.
            walks = _walks_from_people_rows(pdoc.get("people") or [])
            away_by_walk: dict[int, float] = {}
            rt_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for w in walks:
                if w["round_trip_id"] is not None:
                    rt_groups[w["round_trip_id"]].append(w)
            for pair in rt_groups.values():
                if len(pair) != 2:
                    continue
                pair.sort(key=lambda w: w["start_unix"])
                gap_min = (pair[1]["start_unix"] - pair[0]["end_unix"]) / 60.0
                p_round_trips += 1
                p_away_gaps.append(gap_min)
                away_by_walk[pair[0]["walk_id"]] = gap_min  # outbound leg only
            for w in walks:
                if w["cat"] and w["dkey"]:
                    p_habit[(w["cat"], w["dkey"])].append(
                        (w["date"], w["minute"], away_by_walk.get(w["walk_id"]))
                    )

    daily_list = [{"date": dt, **counts} for dt, counts in sorted(daily.items())]
    n_days = len(dates)
    sum_l2r = sum(c["l2r"] for c in daily.values())
    sum_r2l = sum(c["r2l"] for c in daily.values())
    directed = sum_l2r + sum_r2l

    busiest = max(daily.items(), key=lambda kv: kv[1]["total"], default=None)
    bh = max(range(24), key=lambda h: hour_totals[h]) if total else None
    pq = max(range(96), key=lambda q: quarter_profile[q]) if directed else None

    overall = {
        "total_journeys": total,
        "n_sessions": len(sessions),
        "n_days": n_days,
        "date_from": min(dates) if dates else None,
        "date_to": max(dates) if dates else None,
        "pct_l2r": round(100 * sum_l2r / directed) if directed else 0,
        "pct_r2l": round(100 * sum_r2l / directed) if directed else 0,
        "busiest_date": ({"date": busiest[0], "total": busiest[1]["total"]} if busiest else None),
        "busiest_hour": ({"hour": bh, "total": hour_totals[bh]} if bh is not None else None),
        "peak_quarter": (
            {"start": _quarter_label(pq), "total": quarter_profile[pq]} if pq is not None else None
        ),
        "per_day_mean": round(total / n_days, 1) if n_days else 0,
    }

    speed = {
        "hist": _speed_hist(all_speeds, factor),
        "avg_all": _disp(sum(all_speeds) / len(all_speeds), factor) if all_speeds else 0,
        "avg_l2r": _disp(sum(speeds_l2r) / len(speeds_l2r), factor) if speeds_l2r else 0,
        "avg_r2l": _disp(sum(speeds_r2l) / len(speeds_r2l), factor) if speeds_r2l else 0,
        "unit": unit,
        "fastest": _build_fastest(output_root, fastest_raw, factor, unit),
    }

    makes = [[m, n] for m, n in Counter(makes_by_plate.values()).most_common(N_TOP_MAKES)]
    colours_out = [[c, n] for c, n in colours.most_common(N_TOP_COLOURS)]
    bodytypes_out = [[b, n] for b, n in bodytypes.most_common()]

    p_directed = p_dir["l2r"] + p_dir["r2l"]
    p_bh = max(range(24), key=lambda h: p_hour[h]) if p_total else None
    if p_kinds["classified"]:
        for k in ("walkers", "joggers", "cyclists"):
            p_kinds[f"pct_{k}"] = round(100 * p_kinds[k] / p_kinds["classified"])
    p_kinds["round_trips"] = p_round_trips
    p_kinds["round_trips_est"] = max(0, p_round_trips - p_rt_chance) if p_rt_chance_known else None
    p_away_gaps.sort()
    p_kinds["away_min_median"] = (
        round(p_away_gaps[len(p_away_gaps) // 2], 1) if p_away_gaps else None
    )
    people = {
        "total": p_total,
        "n_days": len(p_dates),
        "per_day_mean": round(p_total / len(p_dates), 1) if p_dates else 0,
        "pct_l2r": round(100 * p_dir["l2r"] / p_directed) if p_directed else 0,
        "pct_r2l": round(100 * p_dir["r2l"] / p_directed) if p_directed else 0,
        "busiest_hour": ({"hour": p_bh, "total": p_hour[p_bh]} if p_bh is not None else None),
        "dow": p_dow,
        "heatmap": p_heatmap,
        "dwell": _dwell_stats(p_durations),
        "kinds": p_kinds,
        "habitual": _habitual_slots(p_habit),
    }

    return Stats(
        overall=overall,
        daily=daily_list,
        dow=dow,
        by_day_15min=dict(by_day_15),
        heatmap=heatmap,
        speed=speed,
        makes=makes,
        colours=colours_out,
        bodytypes=bodytypes_out,
        people=people,
        speed_unit=unit,
    )


def _quarter_label(q: int) -> str:
    """Bucket index 0..95 -> 'HH:MM' start of the 15-min window."""
    minutes = q * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _speed_hist(speeds: list[float], factor: float | None) -> list[dict[str, Any]]:
    """Histogram of speeds in the display unit. mph -> 5-wide buckets;
    px/s -> 25-wide buckets. Empty input -> []."""
    if not speeds:
        return []
    width = 5.0 if factor is not None else 25.0
    disp = [s * factor if factor is not None else s for s in speeds]
    top = max(disp)
    n_buckets = int(top // width) + 1
    counts = [0] * n_buckets
    for v in disp:
        counts[min(int(v // width), n_buckets - 1)] += 1
    return [
        {"lo": round(i * width, 1), "hi": round((i + 1) * width, 1), "n": c}
        for i, c in enumerate(counts)
    ]


def _build_fastest(
    output_root: Path,
    fastest_raw: list[tuple[float, str, dict[str, Any]]],
    factor: float | None,
    unit: str,
) -> list[dict[str, Any]]:
    """Top-N fastest tracks, with existence-checked thumbnails and a plate link
    when the track resolved to a confident canonical plate."""
    top = sorted(fastest_raw, key=lambda x: x[0], reverse=True)[:N_FASTEST]
    if not top:
        return []

    # Resolve plate (+ best snap image) for the top tracks from each session's
    # alpr_by_track, loaded once per session involved.
    plate_img: dict[tuple[str, int], tuple[str | None, str | None]] = {}
    for sess in {s for _sp, s, _r in top}:
        p = output_root / sess / f"{sess}_alpr_by_track.json"
        if not p.exists():
            continue
        try:
            tracks = json.loads(p.read_text(encoding="utf-8")).get("tracks", [])
        except (OSError, json.JSONDecodeError):
            continue
        for t in tracks:
            best = t.get("best_preferred")
            if best and (best.get("ocr_conf") or 0) >= 0.9:
                plate_img[(sess, t["track_id"])] = (best.get("ocr_text"), best.get("image"))

    out: list[dict[str, Any]] = []
    for sp, sess, r in top:
        tid = r["track_id"]
        plate_raw, best_image = plate_img.get((sess, tid), (None, None))
        plate = None
        if plate_raw:
            norm = plate_raw.strip().upper().replace(" ", "")
            plate = norm if is_canonical_uk_plate(norm) else None
        thumb, full, _small = resolve_image_urls(
            output_root, sess, tid, prefix="vehicle", best_image=best_image
        )
        dt = datetime.fromisoformat(r["time_start"])
        out.append(
            {
                "plate": plate,
                "speed_px_s": round(sp, 1),
                "speed": _disp(sp, factor),
                "unit": unit,
                "date": dt.date().isoformat(),
                "time": dt.strftime("%H:%M"),
                "direction": r.get("direction"),
                "session": sess,
                "track_id": tid,
                "thumb": thumb,
                "full": full,
            }
        )
    return out
