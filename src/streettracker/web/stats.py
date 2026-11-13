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
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from streettracker.analysis.alpr.gate import PlateGate, load_plate_support, plate_gate, read_passes
from streettracker.analysis.dvsa import is_canonical_uk_plate
from streettracker.analysis.makemodel.bodytype import body_type_for
from streettracker.analysis.makemodel.colour import colour_class_for
from streettracker.common.filememo import FileMemo
from streettracker.common.session_files import SessionFiles
from streettracker.web.aggregate import discover_sessions, resolve_image_urls
from streettracker.web.uk_makes import make_benchmark

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
# "Fastest by make / by colour" leaderboards: rank groups by *mean* speed.
# On this residential street the whole speed distribution is compressed
# (~6-9 mph), so most groups sit inside each other's confidence intervals --
# a naive top-N is dominated by tiny, noisy groups. Guard both ends: only
# rank a group with at least this many tracks, and carry n + a 95% margin on
# each row so the UI can show how uncertain the ranking is.
N_SPEED_RANK = 5
_SPEED_RANK_MIN_N = 40
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

# YMCA class-schedule inference (identity-free). A gym class/session starting
# ~T shows up as a recurring wave of ARRIVALS (R->L, toward the gym) clustered
# around T, followed by a wave of DEPARTURES (L->R) after the typical stay. We
# mine, per weekday, arrival waves that (a) rise well above that weekday's
# baseline flow and (b) recur across many same-weekday dates -- then read the
# typical stay from the departure lag with the strongest excess. A street-level
# pattern, not per-vehicle: works for the unread majority. Residents/staff are a
# tiny fraction of the thousands of daily arrivals, so they don't move the waves.
_SCHED_BIN_MIN = 5  # histogram resolution
_SCHED_ACTIVE = (72, 252)  # 06:00-21:00, in 5-min bins
# A 15-min arrival window must exceed this many times the weekday's median
# 5-min arrival count to count as a wave (~1.7x the expected 3-bin baseline).
_SCHED_ARR_FACTOR = 5.0
_SCHED_MIN_DATES = 4  # must recur on this many distinct same-weekday dates
_SCHED_MERGE_BINS = 6  # collapse waves within 30 min, keep the strongest
_SCHED_MAX_PER_DAY = 8
_SCHED_LAGS = (45, 60, 75, 90, 120)  # candidate stay/class lengths (min)


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
    # Top makes vs the UK (DfT licensed cars, age-matched to the street's
    # registration years); see web/uk_makes.make_benchmark. None without data.
    makes_benchmark: dict[str, Any] | None
    colours: list[list[Any]]  # [[colour, n_journeys], ...]
    bodytypes: list[list[Any]]  # [[body_type, n_journeys], ...]; DVSA-derived + CNN fallback
    fastest_makes: list[dict[str, Any]]  # makes ranked by mean speed; see _speed_ranking
    fastest_colours: list[dict[str, Any]]  # colours ranked by mean speed
    people: dict[str, Any]  # person-track aggregates; see _empty_people()
    schedules: list[dict[str, Any]]  # inferred YMCA class/session waves; _infer_schedules
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


def _bin5(points: list[tuple[str, int]]) -> tuple[list[int], list[set[str]]]:
    """5-min histogram of ``(date, minute-of-day)`` points: per-bin counts and
    the set of distinct dates contributing to each bin."""
    nb = 1440 // _SCHED_BIN_MIN
    counts = [0] * nb
    dates: list[set[str]] = [set() for _ in range(nb)]
    for d, m in points:
        b = m // _SCHED_BIN_MIN
        counts[b] += 1
        dates[b].add(d)
    return counts, dates


def _infer_schedules(
    arrivals: dict[int, list[tuple[str, int]]],
    departures: dict[int, list[tuple[str, int]]],
) -> list[dict[str, Any]]:
    """Infer the YMCA class/session schedule from aggregate traffic waves.

    ``arrivals`` / ``departures`` map a weekday (0=Mon) to ``(date,
    minute-of-day)`` points -- arrivals are R->L (toward the gym), departures
    L->R. Per weekday we find arrival waves that rise above that weekday's
    baseline and recur across ``_SCHED_MIN_DATES`` distinct dates, then read the
    typical stay from the departure lag (in ``_SCHED_LAGS``) with the strongest
    excess over baseline. Rows carry the arrival time, an estimated stay +
    leave time, the typical wave size, and how many of the weekday's dates it
    appeared on -- ranked by size. See the module constants for thresholds.
    """
    lo, hi = _SCHED_ACTIVE
    out: list[dict[str, Any]] = []
    for wd in range(7):
        apts = arrivals.get(wd, [])
        n_dates_wd = len({d for d, _ in apts})
        if n_dates_wd < _SCHED_MIN_DATES:
            continue
        ah, ad = _bin5(apts)
        dh, _dd = _bin5(departures.get(wd, []))
        active_counts = [ah[b] for b in range(lo, hi)]
        abase = statistics.median(active_counts) or 1.0
        dbase = statistics.median([dh[b] for b in range(lo, hi)]) or 1.0

        cand: list[tuple[int, int, int, int | None, int]] = []
        for b in range(lo + 1, hi - 1):
            awin = ah[b - 1] + ah[b] + ah[b + 1]  # 15-min arrivals
            if awin < abase * _SCHED_ARR_FACTOR:
                continue
            if ah[b] < max(ah[b - 2 : b + 3]):  # must be the local peak
                continue
            wave_dates = ad[b - 1] | ad[b] | ad[b + 1]
            if len(wave_dates) < _SCHED_MIN_DATES:
                continue
            # Typical stay: the departure lag with the strongest 15-min excess.
            best_lag: int | None = None
            best_exc = 0
            for lag in _SCHED_LAGS:
                c = b + lag // _SCHED_BIN_MIN
                if c + 1 >= len(dh):
                    continue
                dwin = dh[c - 1] + dh[c] + dh[c + 1]
                exc = int(dwin - dbase * 3)
                if best_lag is None or exc > best_exc:
                    best_lag, best_exc = lag, exc
            cand.append((b, awin, len(wave_dates), best_lag if best_exc > 0 else None, best_exc))

        # Collapse waves within _SCHED_MERGE_BINS, keeping the largest.
        cand.sort(key=lambda s: -s[1])
        kept: list[tuple[int, int, int, int | None, int]] = []
        for s in cand:
            if all(abs(s[0] - k[0]) > _SCHED_MERGE_BINS for k in kept):
                kept.append(s)

        for b, awin, n_dates, stay, _dep_exc in kept[:_SCHED_MAX_PER_DAY]:
            t = b * _SCHED_BIN_MIN
            per_occurrence = round(awin / max(1, n_dates), 1)
            regularity = n_dates / n_dates_wd
            if awin >= abase * 8 and regularity >= 0.7 and stay is not None:
                conf = "high"
            elif regularity >= 0.5 and stay is not None:
                conf = "medium"
            else:
                conf = "low"
            leave = (t + stay) if stay is not None else None
            out.append(
                {
                    "wd": wd,
                    "weekday": _DOW[wd],
                    "arrive": f"{t // 60:02d}:{t % 60:02d}",
                    "stay_min": stay,
                    "leave": (f"{leave // 60:02d}:{leave % 60:02d}") if leave is not None else None,
                    "cars_per_time": per_occurrence,
                    "days_seen": n_dates,
                    "days_covered": n_dates_wd,
                    "confidence": conf,
                }
            )
    out.sort(key=lambda s: (s["wd"], s["arrive"]))
    return out


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
        makes_benchmark=None,
        colours=[],
        bodytypes=[],
        fastest_makes=[],
        fastest_colours=[],
        people=_empty_people(),
        schedules=[],
        speed_unit=unit,
    )


def _dir_counts() -> dict[str, int]:
    return {"l2r": 0, "r2l": 0, "total": 0}


def _dow_counts() -> dict[str, dict[str, int]]:
    return {d: _dir_counts() for d in _DOW}


def _week_grid() -> list[list[int]]:
    return [[0] * 24 for _ in range(7)]


def _kind_counts() -> dict[str, int]:
    return dict.fromkeys(("walkers", "joggers", "cyclists", "dog_walks", "classified", "walks"), 0)


@dataclass(slots=True)
class _Tally:
    """The running totals behind the stats page, for one session or for all.

    Everything here adds up: :meth:`merge` folds another tally in, and folding
    every session's tally in session order gives exactly what one pass over
    every record in that order would. That is what lets ``build_stats``
    remember each session's tally and redo only the sessions whose files
    changed.

    Order is part of the result, so ``merge`` keeps it. Ties in a chart go to
    whichever value was seen first, a mean depends on the order its floats were
    summed in, and the dicts' own order reaches the page. Lists are therefore
    joined in session order and dict keys keep first-seen order.
    """

    # Cars. One record ~= one pass.
    total: int = 0
    dates: set[str] = field(default_factory=set)
    daily: dict[str, dict[str, int]] = field(default_factory=dict)
    dow: dict[str, dict[str, int]] = field(default_factory=_dow_counts)
    by_day_15: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    heatmap: list[list[int]] = field(default_factory=_week_grid)
    quarter_profile: list[int] = field(default_factory=lambda: [0] * 96)
    hour_totals: list[int] = field(default_factory=lambda: [0] * 24)
    speeds_all: list[float] = field(default_factory=list)
    speeds_l2r: list[float] = field(default_factory=list)
    speeds_r2l: list[float] = field(default_factory=list)
    colours: Counter[str] = field(default_factory=Counter)
    bodytypes: Counter[str] = field(default_factory=Counter)
    # plate -> DVSA make / year of first registration. The first session to
    # carry a plate decides both.
    makes_by_plate: dict[str, str] = field(default_factory=dict)
    years_by_plate: dict[str, int | None] = field(default_factory=dict)
    # Per-track speeds for the "fastest by make / colour" boards.
    speeds_by_make: dict[str, list[float]] = field(default_factory=dict)
    speeds_by_colour: dict[str, list[float]] = field(default_factory=dict)
    # (speed, session name, record) candidates for the fastest-cars board. A
    # session's tally keeps only its own top N_FASTEST: no other record of
    # that session can reach the overall top N_FASTEST.
    fastest: list[tuple[float, str, dict[str, Any]]] = field(default_factory=list)
    # weekday -> (date, minute-of-day) of each R->L arrival / L->R departure,
    # for the class-schedule miner (_infer_schedules).
    sched_arr: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    sched_dep: dict[int, list[tuple[str, int]]] = field(default_factory=dict)

    # People. A separate date set, so the cars' per-day mean keeps its own
    # denominator.
    p_total: int = 0
    p_dates: set[str] = field(default_factory=set)
    p_dow: dict[str, dict[str, int]] = field(default_factory=_dow_counts)
    p_heatmap: list[list[int]] = field(default_factory=_week_grid)
    p_hour: list[int] = field(default_factory=lambda: [0] * 24)
    p_dir: dict[str, int] = field(default_factory=lambda: {"l2r": 0, "r2l": 0})
    p_durations: list[float] = field(default_factory=list)
    p_kinds: dict[str, int] = field(default_factory=_kind_counts)
    # (category, dkey) -> (date, minute-of-day, away_minutes|None) per walk,
    # for the recurring-pattern miner (_habitual_slots).
    p_habit: dict[tuple[str, str], list[tuple[str, int, float | None]]] = field(
        default_factory=dict
    )
    p_away_gaps: list[float] = field(default_factory=list)
    p_round_trips: int = 0
    # Chance-match floor, summed over the sessions whose sidecar carries the
    # time-shift control (analysis/people.chance_round_trips).
    p_rt_chance: int = 0
    p_rt_chance_known: bool = False

    def merge(self, other: _Tally) -> None:
        """Add ``other`` to this tally. ``other`` is left untouched and nothing
        of it is shared afterwards (a remembered tally must never change)."""
        self.total += other.total
        self.dates |= other.dates
        for date, counts in other.daily.items():
            _add_counts(self.daily.setdefault(date, _dir_counts()), counts)
        _add_nested(self.dow, other.dow)
        for date, quarters in other.by_day_15.items():
            mine = self.by_day_15.setdefault(date, [{"l2r": 0, "r2l": 0} for _ in range(96)])
            for into, add in zip(mine, quarters, strict=True):
                _add_counts(into, add)
        _add_grid(self.heatmap, other.heatmap)
        _add_row(self.quarter_profile, other.quarter_profile)
        _add_row(self.hour_totals, other.hour_totals)
        self.speeds_all.extend(other.speeds_all)
        self.speeds_l2r.extend(other.speeds_l2r)
        self.speeds_r2l.extend(other.speeds_r2l)
        self.colours.update(other.colours)
        self.bodytypes.update(other.bodytypes)
        for plate, make in other.makes_by_plate.items():
            if plate not in self.makes_by_plate:
                self.makes_by_plate[plate] = make
                self.years_by_plate[plate] = other.years_by_plate[plate]
        _extend_lists(self.speeds_by_make, other.speeds_by_make)
        _extend_lists(self.speeds_by_colour, other.speeds_by_colour)
        self.fastest.extend(other.fastest)
        _extend_lists(self.sched_arr, other.sched_arr)
        _extend_lists(self.sched_dep, other.sched_dep)

        self.p_total += other.p_total
        self.p_dates |= other.p_dates
        _add_nested(self.p_dow, other.p_dow)
        _add_grid(self.p_heatmap, other.p_heatmap)
        _add_row(self.p_hour, other.p_hour)
        _add_counts(self.p_dir, other.p_dir)
        self.p_durations.extend(other.p_durations)
        _add_counts(self.p_kinds, other.p_kinds)
        _extend_lists(self.p_habit, other.p_habit)
        self.p_away_gaps.extend(other.p_away_gaps)
        self.p_round_trips += other.p_round_trips
        self.p_rt_chance += other.p_rt_chance
        self.p_rt_chance_known = self.p_rt_chance_known or other.p_rt_chance_known


def _add_counts(into: dict[str, int], add: dict[str, int]) -> None:
    for key, n in add.items():
        into[key] = into.get(key, 0) + n


def _add_nested(into: dict[str, dict[str, int]], add: dict[str, dict[str, int]]) -> None:
    for key, counts in add.items():
        _add_counts(into[key], counts)


def _add_row(into: list[int], add: list[int]) -> None:
    for i, n in enumerate(add):
        into[i] += n


def _add_grid(into: list[list[int]], add: list[list[int]]) -> None:
    for row, add_row in zip(into, add, strict=True):
        _add_row(row, add_row)


_K = TypeVar("_K")
_V = TypeVar("_V")


def _extend_lists(into: dict[_K, list[_V]], add: dict[_K, list[_V]]) -> None:
    for key, values in add.items():
        into.setdefault(key, []).extend(values)


@dataclass(slots=True)
class _TrackAttributes:
    """What a session's sidecars say about its tracks, by track id.

    Body type and colour take the DVSA register's value where the track has a
    gated plate read (ground truth for a correctly read plate) and fall back
    to the confident CNN prediction for the unread majority. The low-res HSV
    vote (the record's ``color`` field) is never used: it scores about 40 %
    grouped against DVSA and drifts light cars to black or blue (review
    R11/E0.8).
    """

    body: dict[int, str] = field(default_factory=dict)
    colour: dict[int, str] = field(default_factory=dict)
    make: dict[int, str] = field(default_factory=dict)


def _read_dvsa_attributes(path: Path, tally: _Tally, attrs: _TrackAttributes) -> None:
    """Fold a session's DVSA labels into ``attrs`` and the tally's makes."""
    if not path.exists():
        return
    try:
        labels = json.loads(path.read_text(encoding="utf-8")).get("labels", {})
    except (OSError, json.JSONDecodeError):
        labels = {}
    for plate, row in labels.items():
        make = (row.get("make") or "").strip()
        # Only plates still attributed to a track count as cars: labels
        # are cached forever, and the plate gate / parked-beacon
        # suppression / re-enrichment orphan them by clearing
        # track_ids (review R13/E0.2).
        if make and row.get("track_ids") and plate not in tally.makes_by_plate:
            tally.makes_by_plate[plate] = make
            year = row.get("year")
            tally.years_by_plate[plate] = year if isinstance(year, int) else None
        bt: str | None = body_type_for(row.get("make"), row.get("model"))
        col = colour_class_for(row.get("primary_colour"))
        for tid in row.get("track_ids", []):
            if bt:
                attrs.body[int(tid)] = bt
            if col:
                attrs.colour[int(tid)] = col
            if make:
                attrs.make[int(tid)] = make.upper()


def _read_cnn_body_types(path: Path, attrs: _TrackAttributes) -> None:
    if not path.exists():
        return
    try:
        tracks = json.loads(path.read_text(encoding="utf-8")).get("tracks", [])
    except (OSError, json.JSONDecodeError):
        tracks = []
    for t in tracks:
        tid, bt = t.get("track_id"), t.get("body_type")
        if bt and tid is not None and tid not in attrs.body:  # DVSA wins
            attrs.body[int(tid)] = bt


def _read_cnn_colours(path: Path, attrs: _TrackAttributes) -> None:
    if not path.exists():
        return
    try:
        tracks = json.loads(path.read_text(encoding="utf-8")).get("tracks", [])
    except (OSError, json.JSONDecodeError):
        tracks = []
    for t in tracks:
        tid, col = t.get("track_id"), t.get("colour")
        if col and tid is not None and int(tid) not in attrs.colour:  # DVSA wins
            attrs.colour[int(tid)] = col


def _tally_person(tally: _Tally, r: dict[str, Any], dt: datetime, dkey: str | None) -> None:
    wd = dt.weekday()
    tally.p_total += 1
    tally.p_dates.add(dt.date().isoformat())
    tally.p_dow[_DOW[wd]]["total"] += 1
    tally.p_heatmap[wd][dt.hour] += 1
    tally.p_hour[dt.hour] += 1
    if dkey:
        tally.p_dow[_DOW[wd]][dkey] += 1
        tally.p_dir[dkey] += 1
    dur = float(r.get("duration_visible") or 0.0)
    if dur > 0:
        tally.p_durations.append(dur)


def _tally_car(
    tally: _Tally,
    session_name: str,
    r: dict[str, Any],
    dt: datetime,
    dkey: str | None,
    attrs: _TrackAttributes,
) -> None:
    date = dt.date().isoformat()
    wd = dt.weekday()
    tally.total += 1
    tally.dates.add(date)
    day = tally.daily.setdefault(date, _dir_counts())
    day["total"] += 1
    tally.dow[_DOW[wd]]["total"] += 1
    tally.heatmap[wd][dt.hour] += 1
    tally.hour_totals[dt.hour] += 1
    tid = r.get("track_id")
    colour = attrs.colour.get(int(tid)) if tid is not None else None
    tally.colours[colour or "unknown"] += 1
    bt = attrs.body.get(tid) if tid is not None else None
    if bt:
        tally.bodytypes[bt] += 1
    if dkey:
        day[dkey] += 1
        tally.dow[_DOW[wd]][dkey] += 1
        minute = dt.hour * 60 + dt.minute
        quarters = tally.by_day_15.setdefault(date, [{"l2r": 0, "r2l": 0} for _ in range(96)])
        quarters[minute // 15][dkey] += 1
        tally.quarter_profile[minute // 15] += 1
        sched = tally.sched_arr if dkey == "r2l" else tally.sched_dep
        sched.setdefault(wd, []).append((date, minute))

    sp = float(r.get("speed_px_s") or 0.0)
    tally.speeds_all.append(sp)
    if dkey == "l2r":
        tally.speeds_l2r.append(sp)
    elif dkey == "r2l":
        tally.speeds_r2l.append(sp)
    # Same detection guard for the fastest-cars board and the by-make /
    # by-colour boards: see _FASTEST_MIN_DETECTIONS.
    if (r.get("num_detections") or 0) >= _FASTEST_MIN_DETECTIONS:
        tally.fastest.append((sp, session_name, r))
        if sp > 0:
            make = attrs.make.get(int(tid)) if tid is not None else None
            if make:
                tally.speeds_by_make.setdefault(make, []).append(sp)
            if colour:
                tally.speeds_by_colour.setdefault(colour, []).append(sp)


def _tally_records(
    tally: _Tally, session_name: str, data: list[dict[str, Any]], attrs: _TrackAttributes
) -> None:
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
        direction = r.get("direction")
        dkey = "l2r" if direction == _L2R else "r2l" if direction == _R2L else None
        if cls == "person":
            _tally_person(tally, r, dt, dkey)
        else:
            _tally_car(tally, session_name, r, dt, dkey, attrs)


def _tally_people_sidecar(tally: _Tally, path: Path) -> None:
    """Fold a session's ``_people.json`` into the tally: the kind counts, the
    round-trip pairs and the points the pattern miner works from."""
    if not path.exists():
        return
    try:
        pdoc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pdoc = {}
    psum = pdoc.get("summary", {})
    kinds = tally.p_kinds
    kinds["walkers"] += int(psum.get("walkers") or 0)
    kinds["joggers"] += int(psum.get("joggers") or 0)
    kinds["cyclists"] += int(psum.get("cyclists") or 0)
    kinds["dog_walks"] += int(psum.get("dog_walkers") or 0)
    kinds["classified"] += int(psum.get("n_person_tracks") or 0)
    # Sidecars written before walk dedup lack "walks" -- count
    # their tracks 1:1 so the walks total stays comparable.
    kinds["walks"] += int(psum.get("walks") or psum.get("n_person_tracks") or 0)
    if psum.get("round_trips_chance") is not None:
        tally.p_rt_chance += int(psum["round_trips_chance"])
        tally.p_rt_chance_known = True

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
        tally.p_round_trips += 1
        tally.p_away_gaps.append(gap_min)
        away_by_walk[pair[0]["walk_id"]] = gap_min  # outbound leg only
    for w in walks:
        if w["cat"] and w["dkey"]:
            tally.p_habit.setdefault((w["cat"], w["dkey"]), []).append(
                (w["date"], w["minute"], away_by_walk.get(w["walk_id"]))
            )


def _stats_files(session_dir: Path) -> list[Path]:
    """The files a session's tally is computed from."""
    files = SessionFiles(session_dir)
    return [
        files.data,
        files.dvsa_labels,
        files.bodytype_by_track,
        files.colour_by_track,
        files.people,
    ]


def _tally_session(session_dir: Path) -> _Tally:
    """Everything one session adds to the stats.

    Reads only that session's files (:func:`_stats_files`) and depends on
    nothing else: not the other sessions, not the speed calibration, not the
    plate gate. That is what makes it safe to remember.
    """
    files = SessionFiles(session_dir)
    tally = _Tally()
    try:
        data = json.loads(files.data.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = []
    attrs = _TrackAttributes()
    _read_dvsa_attributes(files.dvsa_labels, tally, attrs)
    _read_cnn_body_types(files.bodytype_by_track, attrs)
    _read_cnn_colours(files.colour_by_track, attrs)
    _tally_records(tally, session_dir.name, data, attrs)
    _tally_people_sidecar(tally, files.people)
    # Stable, so equal speeds keep record order, as they would in one sort
    # over every session's records.
    tally.fastest = sorted(tally.fastest, key=lambda x: x[0], reverse=True)[:N_FASTEST]
    return tally


# A session's tally, remembered until one of its files changes. Sessions
# closed weeks ago never change, so a refresh re-reads only what did.
_SESSION_TALLIES: FileMemo[_Tally] = FileMemo()


def _session_tally(session_dir: Path) -> _Tally:
    return _SESSION_TALLIES.get(
        str(session_dir), _stats_files(session_dir), lambda: _tally_session(session_dir)
    )


def _overall(tally: _Tally, n_sessions: int) -> dict[str, Any]:
    daily, dates = tally.daily, tally.dates
    n_days = len(dates)
    sum_l2r = sum(c["l2r"] for c in daily.values())
    sum_r2l = sum(c["r2l"] for c in daily.values())
    directed = sum_l2r + sum_r2l

    busiest = max(daily.items(), key=lambda kv: kv[1]["total"], default=None)
    hour_totals, quarter_profile = tally.hour_totals, tally.quarter_profile
    bh = max(range(24), key=lambda h: hour_totals[h]) if tally.total else None
    pq = max(range(96), key=lambda q: quarter_profile[q]) if directed else None
    return {
        "total_journeys": tally.total,
        "n_sessions": n_sessions,
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
        "per_day_mean": round(tally.total / n_days, 1) if n_days else 0,
    }


def _people_block(tally: _Tally) -> dict[str, Any]:
    kinds = {**_empty_kinds(), **tally.p_kinds}
    if kinds["classified"]:
        for k in ("walkers", "joggers", "cyclists"):
            kinds[f"pct_{k}"] = round(100 * kinds[k] / kinds["classified"])
    kinds["round_trips"] = tally.p_round_trips
    kinds["round_trips_est"] = (
        max(0, tally.p_round_trips - tally.p_rt_chance) if tally.p_rt_chance_known else None
    )
    away_gaps = sorted(tally.p_away_gaps)
    kinds["away_min_median"] = round(away_gaps[len(away_gaps) // 2], 1) if away_gaps else None

    p_dir, p_hour, p_total, p_dates = tally.p_dir, tally.p_hour, tally.p_total, tally.p_dates
    p_directed = p_dir["l2r"] + p_dir["r2l"]
    p_bh = max(range(24), key=lambda h: p_hour[h]) if p_total else None
    return {
        "total": p_total,
        "n_days": len(p_dates),
        "per_day_mean": round(p_total / len(p_dates), 1) if p_dates else 0,
        "pct_l2r": round(100 * p_dir["l2r"] / p_directed) if p_directed else 0,
        "pct_r2l": round(100 * p_dir["r2l"] / p_directed) if p_directed else 0,
        "busiest_hour": ({"hour": p_bh, "total": p_hour[p_bh]} if p_bh is not None else None),
        "dow": tally.p_dow,
        "heatmap": tally.p_heatmap,
        "dwell": _dwell_stats(tally.p_durations),
        "kinds": kinds,
        "habitual": _habitual_slots(tally.p_habit),
    }


def build_stats(output_root: Path, *, m_per_px: float | None = None) -> Stats:
    """Aggregate traffic stats across all sessions under ``output_root``.

    Each session's contribution is computed once (:func:`_tally_session`) and
    remembered until one of that session's files changes, so calling this
    again costs the changed sessions plus the merge. The first call in a
    process reads every session.
    """
    factor = _speed_factor(m_per_px)
    unit = "mph" if factor is not None else "px/s"

    sessions = discover_sessions(output_root)
    if not sessions:
        return _empty_stats(unit)

    # A fresh tally every call: the remembered ones are only ever read.
    tally = _Tally()
    for d in sessions:
        tally.merge(_session_tally(d))

    all_speeds, speeds_l2r, speeds_r2l = tally.speeds_all, tally.speeds_l2r, tally.speeds_r2l
    speed = {
        "hist": _speed_hist(all_speeds, factor),
        "avg_all": _disp(sum(all_speeds) / len(all_speeds), factor) if all_speeds else 0,
        "avg_l2r": _disp(sum(speeds_l2r) / len(speeds_l2r), factor) if speeds_l2r else 0,
        "avg_r2l": _disp(sum(speeds_r2l) / len(speeds_r2l), factor) if speeds_r2l else 0,
        "unit": unit,
        "fastest": _build_fastest(output_root, tally.fastest, factor, unit, gate=plate_gate()),
    }

    makes_by_plate = tally.makes_by_plate
    makes = [[m, n] for m, n in Counter(makes_by_plate.values()).most_common(N_TOP_MAKES)]
    return Stats(
        overall=_overall(tally, len(sessions)),
        daily=[{"date": dt, **counts} for dt, counts in sorted(tally.daily.items())],
        dow=tally.dow,
        by_day_15min=tally.by_day_15,
        heatmap=tally.heatmap,
        speed=speed,
        makes=makes,
        makes_benchmark=make_benchmark(makes_by_plate, tally.years_by_plate, makes),
        colours=[[c, n] for c, n in tally.colours.most_common(N_TOP_COLOURS)],
        bodytypes=[[b, n] for b, n in tally.bodytypes.most_common()],
        fastest_makes=_speed_ranking(tally.speeds_by_make, factor, unit),
        fastest_colours=_speed_ranking(tally.speeds_by_colour, factor, unit),
        people=_people_block(tally),
        schedules=_infer_schedules(tally.sched_arr, tally.sched_dep),
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


def _speed_ranking(
    groups: dict[str, list[float]],
    factor: float | None,
    unit: str,
    *,
    min_n: int = _SPEED_RANK_MIN_N,
    top: int = N_SPEED_RANK,
) -> list[dict[str, Any]]:
    """Rank groups (make / colour) by mean speed, fastest first.

    Only groups with >= ``min_n`` tracks qualify (a mean over a handful of
    tracks is meaningless on this compressed distribution). Each row carries
    the sample size and a 95% margin (1.96 * standard error of the mean) in
    the display unit, so the UI can show the ranking's uncertainty rather than
    implying a hard order. Input speeds are px/s; ``factor`` converts to mph.
    """
    rows: list[dict[str, Any]] = []
    for name, speeds in groups.items():
        if len(speeds) < min_n:
            continue
        disp = [s * factor if factor is not None else s for s in speeds]
        n = len(disp)
        mean = sum(disp) / n
        var = sum((x - mean) ** 2 for x in disp) / (n - 1) if n > 1 else 0.0
        margin = 1.96 * (var**0.5) / (n**0.5)
        rows.append(
            {
                "name": name,
                "n": n,
                "mean": round(mean, 1),
                "margin": round(margin, 1),
                "unit": unit,
            }
        )
    rows.sort(key=lambda r: r["mean"], reverse=True)
    return rows[:top]


def _track_snap(
    session_dir: Path,
    prefix: str,
    track_id: int,
    record: dict[str, Any],
    *,
    preferred: str | None = None,
) -> str | None:
    """A 4K snap of the track that exists on disk: ``preferred`` when it
    does, else the one where the car's box was largest (closest to the
    camera; completion-time boxes over fire-time ones), else ``None``."""
    if preferred and (session_dir / preferred).is_file():
        return preferred
    snaps = record.get("main_snaps") or []
    boxes = record.get("main_snap_bboxes_done") or record.get("main_snap_bboxes") or []

    def area(i: int) -> int:
        b = boxes[i] if i < len(boxes) else None
        return (b[2] - b[0]) * (b[3] - b[1]) if b else -1

    for i in sorted(range(len(snaps)), key=area, reverse=True):
        name = f"{prefix}_{track_id}_main_{snaps[i]}.jpg"
        if (session_dir / name).is_file():
            return name
    return None


def _best_reads_by_track(path: Path) -> dict[int, dict[str, Any]]:
    """``track_id -> best preferred read`` from one ``_alpr_by_track.json``
    (empty when the file is missing or unreadable)."""
    if not path.exists():
        return {}
    try:
        tracks = json.loads(path.read_text(encoding="utf-8")).get("tracks", [])
    except (OSError, json.JSONDecodeError):
        return {}
    return {t["track_id"]: t["best_preferred"] for t in tracks if t.get("best_preferred")}


# A session's best reads, remembered until its rollup changes. Only the
# sessions that hold one of the fastest tracks are ever loaded.
_BEST_READS: FileMemo[dict[int, dict[str, Any]]] = FileMemo(max_entries=64)


def _build_fastest(
    output_root: Path,
    fastest_raw: list[tuple[float, str, dict[str, Any]]],
    factor: float | None,
    unit: str,
    *,
    gate: PlateGate,
) -> list[dict[str, Any]]:
    """Top-N fastest tracks, with existence-checked thumbnails and a plate link
    when the track resolved to a canonical plate that passes ``gate`` (the
    shared plate gate). The gate decides only the plate.

    The thumbnail is the tracker's own crop of the car (the HQ crop, else the
    tile): a fast car has often left the frame, or shrunk to a speck, by the
    time its 4K snap lands ~0.7 s after the fire, and a box picked from the
    4K frame can be the wrong vehicle. Without a crop on disk (an
    ``--only-main`` session pulled before 2026-10-04 and not backfilled with
    ``pull --crops-only``) it falls back to the best read's snap whatever its
    confidence, else the track's closest 4K snap on disk. ``full`` is the 4K
    snap."""
    top = sorted(fastest_raw, key=lambda x: x[0], reverse=True)[:N_FASTEST]
    if not top:
        return []
    support = load_plate_support(output_root) if gate.needs_support else None

    # Best reads of each session that holds one of the top tracks.
    best_reads: dict[str, dict[int, dict[str, Any]]] = {}
    for sess in {s for _sp, s, _r in top}:
        path = SessionFiles(output_root / sess).alpr_by_track
        best_reads[sess] = _BEST_READS.get(str(path), [path], partial(_best_reads_by_track, path))

    out: list[dict[str, Any]] = []
    for sp, sess, r in top:
        tid = r["track_id"]
        best = best_reads[sess].get(tid, {})
        plate = None
        plate_raw = best.get("ocr_text")
        if plate_raw and read_passes(gate, best, support):
            norm = plate_raw.strip().upper().replace(" ", "")
            plate = norm if is_canonical_uk_plate(norm) else None
        prefix = r.get("asset_prefix") or "vehicle"
        snap = _track_snap(output_root / sess, prefix, tid, r, preferred=best.get("image"))
        thumb, full, _small = resolve_image_urls(
            output_root, sess, tid, prefix=prefix, best_image=snap, prefer_hq=True
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
