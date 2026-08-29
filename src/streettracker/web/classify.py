"""Behavioural classification of showcase cars into traffic buckets.

The camera watches a **closed cul-de-sac**: the only road in/out passes it,
homes are up the road, and a YMCA gym sits at the dead end. So there is one
travel axis with two directions:

* **IN** (into the pocket, toward both the homes *and* the gym) =
  ``"right to left"``.
* **OUT** (back to the main road) = ``"left to right"``.

Residents and YMCA people use the *same* two directions -- what separates them
is the **order** of a paired trip and where they rest overnight:

===============  ================  =======================================
Bucket           Rests overnight   Signature
===============  ================  =======================================
``resident``     inside (home)     OUT-first: leaves, later returns
``ymca_staff``   outside           IN-first + long stay (most of a day)
``visitor``      outside           IN-first + short stay (~1 h class)
``brief``        --                in-and-out with near-zero dwell
                                   (drop-off / courier / turnaround)
``unclassified`` --                too little evidence to call
===============  ================  =======================================

The single strongest, most missed-read-robust signal is the **direction of a
car's first crossing each day**: a resident starts the day inside so their
first move is OUT; a YMCA person starts outside so their first move is IN. Same-
day round-trip pairing then supplies the **dwell** that splits IN-first cars
into visitor vs staff. Certainty is kept honest by a **time-shift chance
control** (ported from :mod:`streettracker.analysis.people`) so coincidental
pairings on this quiet street don't inflate it.

Pure and deterministic: no I/O, no wall clock. :func:`classify_vehicle` takes
one car's pooled trips and returns a :class:`VehicleClassification`.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

# Direction strings (must match TrackRecord.direction / analysis.people).
_L2R = "left to right"
_R2L = "right to left"
_OPPOSITE = {_L2R: _R2L, _R2L: _L2R}
# Scene semantics: IN = toward the homes + gym, OUT = toward the main road.
_OUT = _L2R
_IN = _R2L

# Buckets.
RESIDENT = "resident"
YMCA_STAFF = "ymca_staff"
VISITOR = "visitor"
BRIEF = "brief"
UNCLASSIFIED = "unclassified"
BUCKETS = (RESIDENT, YMCA_STAFF, VISITOR, BRIEF, UNCLASSIFIED)

# Human labels, reused by the classifier's justification + the eval script.
# (The templates carry their own emoji-prefixed labels.)
BUCKET_LABEL = {
    RESIDENT: "Resident",
    YMCA_STAFF: "YMCA staff",
    VISITOR: "Visitor",
    BRIEF: "Brief / drop-off",
    UNCLASSIFIED: "Unclassified",
}

# --- pairing / dwell thresholds -------------------------------------------
# A round-trip pair joins an outbound leg to a later opposite-direction leg
# whose gap is in [min, max]. The floor drops BotSORT direction-flip artifacts
# (which surface as a near-instant "return", often overlapping); the ceiling
# bounds a stay to a long working day, beyond which it is a separate trip.
PAIR_MIN_GAP_S = 45.0
PAIR_MAX_GAP_S = 12.0 * 3600.0

# Dwell bands (minutes) for an IN-first stay.
BRIEF_MAX_MIN = 12.0  # drop-off / courier / turnaround
VISITOR_MAX_MIN = 150.0  # a gym class (+ a coffee); longer reads as staff
# A car with even a modest share of genuinely long stays isn't a "brief"
# drop-off vehicle -- it's a visitor with variable dwell.
BRIEF_LONG_MIN = 30.0  # a stay past this is not a drop-off
BRIEF_LONG_SHARE_MAX = 0.25  # brief only if fewer than this fraction are long

# Day boundary anchored at 04:00 so a late-evening return and the same
# morning's departure land on one "activity day" (and a rare post-midnight
# return doesn't spawn a spurious IN-first day).
_DAY_ANCHOR_HOURS = 4

# Regularity gates for the staff signal (a work rota: most weekdays, same
# clock time each morning).
STAFF_MIN_DAYS = 4
STAFF_WEEKDAY_FRAC = 0.8
STAFF_ARRIVAL_STD_MAX = 75.0  # minutes

# Evidence gate: below this many sightings we can't say anything.
MIN_SIGHTINGS = 2

# Resident test (deliberately strict -- bias toward calling a car a visitor).
# A resident is based INSIDE, so it genuinely leaves-then-returns. But on a
# busy trajectory apparent out-and-backs happen by coincidence, so we require
# the OUT-first round trips to EXCEED the time-shift chance floor by this much
# (the same null model people.py uses). ``RESIDENT_RT_EXCESS`` net round trips
# is the primary signal. A commuter whose same-day return we often miss can
# still qualify via a strongly leaves-first daily pattern sustained over many
# days (``RESIDENT_OUT_RATIO`` / ``RESIDENT_MIN_DAYS``). Everything else -- a
# mere arrive-bias, a single parked stint, a few coincidental pairs -- is NOT
# enough for resident and falls through to visitor.
RESIDENT_RT_EXCESS = 3
RESIDENT_OUT_RATIO = 3.0
RESIDENT_MIN_OUT_DAYS = 3
RESIDENT_MIN_DAYS = 10

# Certainty bands from the 0..1 score.
_HIGH = 0.66
_MEDIUM = 0.4


@dataclass(slots=True)
class VehicleClassification:
    """One car's inferred bucket with a certainty band and justification."""

    bucket: str
    certainty: str  # "low" | "medium" | "high"
    score: float  # 0..1
    reason: str  # human-readable justification
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


# A pooled trip: (start_unix, end_unix, direction, time_start_iso).
Trip = tuple[float, float, str, str]


@dataclass(slots=True)
class _Trip:
    start: float
    end: float
    direction: str
    iso: str


@dataclass(slots=True)
class _Pair:
    polarity: str  # "out_first" (resident) | "in_first" (ymca)
    dwell_min: float


def _to_trips(trips: list[Trip]) -> list[_Trip]:
    out = [_Trip(float(s), float(e), d, iso) for (s, e, d, iso) in trips if d in _OPPOSITE]
    out.sort(key=lambda t: t.start)
    return out


def _activity_date(iso: str) -> str | None:
    """The 04:00-anchored local date of an ISO-local timestamp, or ``None``."""
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return (dt - timedelta(hours=_DAY_ANCHOR_HOURS)).date().isoformat()


def _minute_of_day(iso: str) -> int | None:
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return dt.hour * 60 + dt.minute


def _weekday(iso: str) -> int | None:
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return dt.weekday()


def _pair_trips(
    trips: list[_Trip],
    *,
    min_gap_s: float = PAIR_MIN_GAP_S,
    max_gap_s: float = PAIR_MAX_GAP_S,
) -> list[_Pair]:
    """Greedily pair each trip as the RETURN of the most-recently-ended
    unmatched opposite-direction trip within ``[min_gap_s, max_gap_s]``.

    Ported from :func:`streettracker.analysis.people.pair_round_trips`. The
    earlier leg's direction sets the polarity: OUT-first (left then returned)
    is resident-like; IN-first (arrived then left) is YMCA-like. ``dwell`` is
    the gap between the outbound end and the return start, in minutes.
    """
    pairs: list[_Pair] = []
    candidates: list[_Trip] = []  # ended, so far unmatched
    for w in trips:  # already start-sorted
        want = _OPPOSITE.get(w.direction)
        best: _Trip | None = None
        best_gap = 0.0
        if want is not None:
            for a in candidates:
                if a.direction != want:
                    continue
                gap = w.start - a.end
                if not (min_gap_s <= gap <= max_gap_s):
                    continue
                if best is None or gap < best_gap:
                    best, best_gap = a, gap
        if best is not None:
            polarity = "out_first" if best.direction == _OUT else "in_first"
            pairs.append(_Pair(polarity=polarity, dwell_min=best_gap / 60.0))
            candidates.remove(best)
        else:
            candidates.append(w)
        candidates = [a for a in candidates if w.start - a.end <= max_gap_s]
    return pairs


def _chance_pairs(
    trips: list[_Trip],
    *,
    min_gap_s: float = PAIR_MIN_GAP_S,
    max_gap_s: float = PAIR_MAX_GAP_S,
) -> int | None:
    """How many pairs pure coincidence yields (time-shift null model).

    Circularly shifts the IN-direction trips well past the pairing window,
    preserving density and time-of-day profile while destroying genuine
    out-and-backs. Returns ``None`` when the record is too short (< 2x the
    shift) for the control to mean anything. Mirrors
    :func:`streettracker.analysis.people.chance_round_trips`.
    """
    if not trips:
        return None
    t0 = min(t.start for t in trips)
    span = max(t.end for t in trips) - t0
    shift = max_gap_s + 600.0
    if span < 2.0 * shift:
        return None
    shifted: list[_Trip] = []
    for t in trips:
        if t.direction == _IN:
            new_start = t0 + (t.start - t0 + shift) % span
            t = _Trip(new_start, new_start + (t.end - t.start), t.direction, t.iso)
        shifted.append(t)
    shifted.sort(key=lambda t: t.start)
    return len(_pair_trips(shifted, min_gap_s=min_gap_s, max_gap_s=max_gap_s))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _band(score: float) -> str:
    if score >= _HIGH:
        return "high"
    if score >= _MEDIUM:
        return "medium"
    return "low"


def classify_vehicle(
    trips: list[Trip],
    *,
    parked_episodes: list[dict[str, Any]] | tuple[Any, ...] = (),
) -> VehicleClassification:
    """Classify one car from its pooled trips (across every session).

    ``trips`` is ``(start_unix, end_unix, direction, time_start_iso)`` per
    pass. ``parked_episodes`` (a car that repeatedly sits parked in view) is the
    one non-behavioural resident prior -- a physically stationary car on this
    street lives here. NOTE: an operator ``owner``/``name`` tag is deliberately
    NOT a prior: it means "I know whose car this is" (which includes visiting
    family), not "resident". The operator asserts a bucket via the explicit
    ``classification_override``, kept orthogonal to identity. See the module
    docstring for the model.
    """
    ts = _to_trips(list(trips))
    n = len(ts)
    ev: dict[str, Any] = {"n_sightings": n}

    if parked_episodes:
        ev["parked_episodes"] = len(parked_episodes)

    if n < MIN_SIGHTINGS:
        return VehicleClassification(UNCLASSIFIED, "low", 0.0, "Too few sightings to classify.", ev)

    dirs = Counter(t.direction for t in ts)
    ev["in_passes"] = dirs.get(_IN, 0)
    ev["out_passes"] = dirs.get(_OUT, 0)
    # A car we only ever caught going one way carries no order information --
    # in a closed pocket it must travel both ways, so a one-direction record is
    # a read-rate artifact (ANPR is direction-asymmetric), not behaviour.
    has_both = ev["in_passes"] > 0 and ev["out_passes"] > 0

    # Activity-day grouping + the day's-first-crossing polarity vote.
    by_day: dict[str, list[_Trip]] = defaultdict(list)
    for t in ts:
        d = _activity_date(t.iso)
        if d is not None:
            by_day[d].append(t)
    ev["n_days"] = len(by_day)
    days_out_first = days_in_first = 0
    for day_trips in by_day.values():
        first = min(day_trips, key=lambda t: t.start).direction
        if first == _OUT:
            days_out_first += 1
        elif first == _IN:
            days_in_first += 1
    ev["days_out_first"] = days_out_first
    ev["days_in_first"] = days_in_first

    # Same-day round-trip pairing -> polarity + dwell, with a chance floor.
    pairs = _pair_trips(ts)
    chance = _chance_pairs(ts)
    in_first = [p for p in pairs if p.polarity == "in_first"]
    out_first = [p for p in pairs if p.polarity == "out_first"]
    ev["round_trips"] = len(pairs)
    ev["round_trips_chance"] = chance
    ev["round_trips_in_first"] = len(in_first)
    ev["round_trips_out_first"] = len(out_first)
    net_pairs = len(pairs) - chance if chance is not None else len(pairs)
    net_pairs = max(0, net_pairs)
    # Dwell for the visitor/staff split comes from IN-first stays (falling
    # back to whatever pairs exist).
    dwell_src = in_first or pairs
    median_dwell = statistics.median(p.dwell_min for p in dwell_src) if dwell_src else None
    ev["median_dwell_min"] = round(median_dwell, 1) if median_dwell is not None else None

    # Arrival regularity (IN legs) for the staff rota signal.
    in_minutes = [m for t in ts if t.direction == _IN and (m := _minute_of_day(t.iso)) is not None]
    weekdays = [wd for t in ts if (wd := _weekday(t.iso)) is not None]
    weekday_frac = (sum(1 for wd in weekdays if wd < 5) / len(weekdays)) if weekdays else 0.0
    arrival_std = statistics.pstdev(in_minutes) if len(in_minutes) >= 2 else None
    ev["arrival_weekday_frac"] = round(weekday_frac, 2)
    if in_minutes:
        med_arr = int(statistics.median(in_minutes))
        ev["median_arrival"] = f"{med_arr // 60:02d}:{med_arr % 60:02d}"
    if arrival_std is not None:
        ev["arrival_std_min"] = round(arrival_std, 1)
    staffish = (
        len(by_day) >= STAFF_MIN_DAYS
        and weekday_frac >= STAFF_WEEKDAY_FRAC
        and (arrival_std is None or arrival_std <= STAFF_ARRIVAL_STD_MAX)
    )

    reason: list[str] = []
    reason.append(f"seen {n}× over {len(by_day)} day{'s' if len(by_day) != 1 else ''}")

    # --- decision cascade --------------------------------------------------
    # Resident = genuine leave-then-return ABOVE the coincidence floor (or a
    # strongly leaves-first daily pattern sustained over many days). This is
    # deliberately strict: a mere arrive-bias, a single parked stint, or a few
    # coincidental out-and-backs are NOT enough -- such cars fall through to
    # visitor. See the RESIDENT_* constants.
    brief_pairs = [p for p in pairs if p.dwell_min <= BRIEF_MAX_MIN]
    long_pairs = [p for p in pairs if p.dwell_min > BRIEF_LONG_MIN]
    # A true drop-off/courier NEVER stays long; a car with even a modest share
    # of multi-hour stays is a visitor with variable dwell, not "brief".
    is_brief = (
        bool(pairs)
        and len(brief_pairs) / len(pairs) >= 0.6
        and len(long_pairs) / len(pairs) < BRIEF_LONG_SHARE_MAX
    )
    net_out_rt = len(out_first) - (chance or 0)
    ev["net_out_round_trips"] = net_out_rt
    resident_by_rt = net_out_rt >= RESIDENT_RT_EXCESS and days_out_first >= days_in_first
    resident_by_daily = (
        days_out_first >= RESIDENT_OUT_RATIO * days_in_first
        and days_out_first >= RESIDENT_MIN_OUT_DAYS
        and len(by_day) >= RESIDENT_MIN_DAYS
    )

    if not has_both:
        bucket = UNCLASSIFIED
        only = "arriving" if ev["in_passes"] else "leaving"
        reason.append(f"only ever caught {only} — can't tell its travel pattern apart")
        score = 0.0
    elif resident_by_rt or resident_by_daily:
        bucket = RESIDENT
        if resident_by_rt:
            reason.append(
                f"leaves and returns {net_out_rt} more times than coincidence "
                f"({len(out_first)} round trips vs ~{chance or 0} by chance)"
            )
            score = 0.5 + 0.3 * _clamp(net_out_rt / 8.0) + 0.1 * _clamp(len(by_day) / 30.0)
        else:
            reason.append(
                f"leaves first on {days_out_first} of {len(by_day)} days "
                f"(vs {days_in_first} arrivals-first)"
            )
            score = 0.5 + 0.1 * _clamp(len(by_day) / 30.0)
    elif is_brief:
        bucket = BRIEF
        reason.append(
            f"{len(brief_pairs)}/{len(pairs)} passes were in-and-out "
            f"(~{statistics.median(p.dwell_min for p in brief_pairs):.0f} min)"
        )
        score = 0.45 + 0.3 * _clamp(len(brief_pairs) / 3.0)
    elif staffish and (median_dwell is None or median_dwell > VISITOR_MAX_MIN):
        bucket = YMCA_STAFF
        arrives = ev.get("median_arrival")
        if median_dwell is not None:
            reason.append(f"arrives {arrives} on weekdays, stays ~{median_dwell / 60:.1f} h")
        else:
            reason.append(f"arrives {arrives} on most weekdays (work-rota pattern)")
        score = (
            0.4
            + 0.2 * _clamp(net_pairs / 3.0)
            + 0.2 * _clamp(len(by_day) / 6.0)
            + (0.15 if median_dwell is not None else 0.0)
        )
    else:
        bucket = VISITOR
        if days_out_first > days_in_first:
            reason.append("appears to leave-and-return, but no more than coincidence")
        elif median_dwell is not None:
            reason.append(f"arrives, stays ~{median_dwell:.0f} min, then leaves")
        else:
            reason.append("arrives then leaves (short-stay pattern)")
        score = 0.4 + 0.2 * _clamp(len(by_day) / 20.0) + (0.1 if median_dwell is not None else 0.0)

    score = _clamp(score)
    certainty = "low" if bucket == UNCLASSIFIED else _band(score)
    ev["score"] = round(score, 3)
    label = BUCKET_LABEL[bucket]
    reason_str = f"{'; '.join(reason)} → {label} ({certainty})."
    return VehicleClassification(bucket, certainty, round(score, 3), reason_str, ev)
