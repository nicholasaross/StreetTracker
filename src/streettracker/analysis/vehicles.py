"""Plate-anchored per-vehicle aggregation.

Folds a session's per-track records (``data.json``) and per-track
best ALPR reads (``alpr_by_track.json``) into a per-vehicle view
keyed by plate string. Tracks whose plate read confidence is above
the threshold are grouped under the canonical plate; everything
else is treated as a one-off (anonymous) visit.

Output schema (per vehicle):

::

    {
      "plate": "AB12CDE" | null,            # null = no high-conf read
      "plate_conf": 0.99,
      "n_visits": 2,                        # distinct tracks attributed
      "track_ids": [42, 187],
      "first_seen": "2026-05-26T13:21:42+01:00",
      "last_seen":  "2026-05-26T17:08:15+01:00",
      "gap_minutes_max": 226.5,             # max gap between consecutive visits
      "gap_minutes_min": 226.5,             # min same (= max when n=2)
      "directions": {"right to left": 2, "left to right": 0},
      "colors":     {"white": 2},
      "make": "FORD",                       # from the DVSA MOT label;
      "model": "FOCUS",                     #   null if plate unread,
      "year": 2017,                         #   <3yr car, or dvsa-label
      "make_model_source": "dvsa",          #   not run
      "visits": [
        {
          "track_id": 42,
          "time_start": "...",
          "time_end":   "...",
          "duration_visible": 8.3,
          "direction": "right to left",
          "speed_px_s": 87.2,
          "color": "white",
          "lane": "middle",
          "asset_prefix": "vehicle",
          "n_snaps": 3,
          "best_image": "vehicle_42_main_2.jpg"
        },
        ...
      ]
    }

Single-visit vehicles dominate residential traffic; recurring
vehicles (n_visits >= 2) are the operationally interesting subset
-- commuters, residents leaving and returning, delivery loops.

Run via the CLI::

    streettracker vehicles output/session_20260526_124704
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from streettracker.analysis.dvsa import is_canonical_uk_plate
from streettracker.analysis.parked import (
    ParkedDetection,
    best_unsuppressed_read,
    detect_parked,
    load_alpr_entries,
    merge_episodes,
    normalize_plate,
)

CONF_THRESHOLD = 0.9

# Minimum confidence-weighted vote share for a track's CNN make
# prediction (``<session>_makemodel_by_track.json``) to fill a vehicle's
# make when no DVSA ground truth exists. The rollup's ``conf`` is the
# winning make's share of the track's confidence-weighted votes across
# its snaps; consistent tracks sit near 1.0, mixed/garbage tracks fall
# well below. 0.5 = an absolute majority of the track's evidence.
CNN_MAKE_MIN_CONF = 0.5

# Default Levenshtein similarity (rapidfuzz) threshold for collapsing
# OCR variants of the same physical plate. Only same-length plates
# are eligible.
#
# 85 catches 1-character OCR diffs on 7-char UK plates (ratio 85.7)
# but NOT on 6-char plates (ratio 83.3). Deliberately conservative:
# on this scene, dropping the threshold to 80 over-merges across UK
# regional prefixes (e.g. an LX7751 cluster absorbed 8 distinct
# Newcastle-area plates whose 4th-6th chars happened to be close).
# False merges are operationally worse than missed merges; pass
# ``--fuzzy-ratio 80`` to opt in to the looser threshold for a session
# where the LX/LD/L4 prefix density is known to be low.
FUZZY_RATIO_DEFAULT = 85


@dataclass(slots=True)
class VehicleVisit:
    """A single track attributed to a vehicle."""

    track_id: int
    time_start: str
    time_end: str
    time_start_unix: float
    time_end_unix: float
    duration_visible: float
    direction: str
    speed_px_s: float
    color: str
    lane: str
    asset_prefix: str
    n_snaps: int
    best_image: str | None  # filename of the snap that produced the best ALPR read

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Vehicle:
    """A vehicle keyed by canonical plate string (or ``None`` if no
    high-conf read was achieved for any of its tracks)."""

    plate: str | None
    plate_conf: float | None
    n_visits: int
    track_ids: list[int]
    first_seen: str
    last_seen: str
    gap_minutes_max: float  # max gap between consecutive visits, 0.0 if n_visits==1
    gap_minutes_min: float  # min gap, 0.0 if n_visits==1
    directions: dict[str, int] = field(default_factory=dict)
    colors: dict[str, int] = field(default_factory=dict)
    visits: list[VehicleVisit] = field(default_factory=list)
    # When fuzzy plate clustering merged OCR variants into one
    # vehicle, ``plate_variants`` lists the alternate strings (each
    # with its max OCR confidence across the merged visits).
    # ``plate`` is the highest-conf variant. Empty list when no
    # clustering happened (either disabled or no near-neighbours).
    plate_variants: list[tuple[str, float]] = field(default_factory=list)
    # Make/model/year from the DVSA MOT label harvest
    # (``<session>_dvsa_labels.json``, produced by ``streettracker
    # dvsa-label``). Ground truth for readable, >=3yr-old plates; None
    # when no DVSA label exists (plate unread, <3yr car not yet on the
    # MOT register, or dvsa-label not run). ``make_model_source``
    # records provenance ("dvsa") so a later CNN path can populate the
    # same fields without ambiguity about where a value came from.
    make: str | None = None
    model: str | None = None
    year: int | None = None
    make_model_source: str | None = None
    # Parked-car stints attributed to this plate by the stationary-beacon
    # detector (``analysis.parked``): the car sat in view while its static
    # plate was read across many passing tracks. Those reads are
    # suppressed from ``visits``; the episode records the parked window
    # instead (JSON dicts, see :meth:`ParkedEpisode.to_json_dict`).
    parked_episodes: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["visits"] = [v.to_json_dict() if isinstance(v, VehicleVisit) else v for v in d["visits"]]
        # asdict turns tuples into tuples; JSON encoder accepts those
        # but for explicit shape we normalise to [plate, conf] lists.
        d["plate_variants"] = [[p, c] for p, c in d.get("plate_variants", [])]
        return d


@dataclass(slots=True)
class CrossVehicle:
    """A vehicle pooled across >=2 sessions, keyed by fuzzy-canonical
    plate. ``kind`` classifies repeat behaviour by distinct LOCAL
    calendar date: ``different-day`` (>=2 dates -- a regular, incl. any
    car seen in multiple sessions), ``same-day`` (>=2 visits on one
    date), ``one-off`` (single visit)."""

    plate: str
    make: str | None
    model: str | None
    year: int | None
    make_model_source: str | None
    n_visits: int
    n_dates: int
    dates: list[str]
    sessions: list[str]
    kind: str
    directions: dict[str, int] = field(default_factory=dict)
    first_seen: str = ""
    last_seen: str = ""
    # OCR variants merged in (within- and cross-session); the canonical
    # ``plate`` is the highest-conf string across all sessions.
    plate_variants: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


# Colour vocabulary -> coarse group, shared by the observed-colour votes
# (common.color.vote_color values) and DVSA ``primary_colour`` strings.
# Groups are deliberately broad: white/silver/grey are near-impossible to
# tell apart at this camera's distance, so they corroborate each other.
_COLOUR_GROUPS = {
    "white": "light",
    "silver": "light",
    "grey": "light",
    "gray": "light",
    "beige": "light",
    "cream": "light",
    "black": "black",
    "red": "red",
    "orange": "red",
    "maroon": "red",
    "burgundy": "red",
    "pink": "red",
    "blue": "blue",
    "navy": "blue",
    "turquoise": "blue",
    "green": "green",
    "yellow": "yellow",
    "gold": "yellow",
    "brown": "brown",
    "bronze": "brown",
    "purple": "purple",
    "violet": "purple",
}


def _colour_group(colour: Any) -> str | None:
    return _COLOUR_GROUPS.get(str(colour or "").strip().lower())


def _majority_colour_group(colours: Counter) -> str | None:
    """Coarse group of the most-seen groupable colour, or ``None`` when
    every observation is unknown/ungroupable."""
    known: Counter = Counter()
    for c, n in colours.items():
        g = _colour_group(c)
        if g:
            known[g] += n
    if not known:
        return None
    return known.most_common(1)[0][0]


def make_distinct_vehicle_checker(
    dvsa_rows: Mapping[str, Mapping[str, Any]],
    colours_by_plate: Mapping[str, Counter],
) -> Callable[[str, str], bool]:
    """Build the fuzzy-merge veto: are two spellings provably DIFFERENT
    registered vehicles?

    A 1-char-apart pair is *usually* one physical car plus an OCR
    misread -- but sometimes both plates are real (GU65UGK, a red VW UP,
    was swallowed by GU65UGM, a white VW GOLF; ratio 85.7). Two spellings
    are vetoed from merging only when ALL hold:

    1. both have DVSA rows;
    2. their registered ``primary_colour`` falls in DIFFERENT coarse
       groups (same-colour twins are indistinguishable from a misread,
       so we stay merged -- conservative);
    3. each spelling's sightings have a known majority colour group and
       those groups DIFFER from each other. This is the one-car
       protection: a misread's sightings are of the true car, so both
       spellings observe the same colour and the merge proceeds (the
       LD22BWG/BMG silver hatchback stays one vehicle) -- even when the
       misread string happens to resolve to some other real car on the
       register (LX11PXR does);
    4. at least one side's observed group matches its own register.
       Deliberately not BOTH: cameras drift dark colours (the red UP
       that motivated this reads black-majority in some sessions);
       requiring both sides to match let exactly those sessions
       false-merge two real cars.
    """

    def distinct(a: str, b: str) -> bool:
        ra, rb = dvsa_rows.get(a), dvsa_rows.get(b)
        if not ra or not rb:
            return False
        ga = _colour_group(ra.get("primary_colour"))
        gb = _colour_group(rb.get("primary_colour"))
        if not ga or not gb or ga == gb:
            return False
        oa = _majority_colour_group(colours_by_plate.get(a, Counter()))
        ob = _majority_colour_group(colours_by_plate.get(b, Counter()))
        if not oa or not ob or oa == ob:
            return False
        return oa == ga or ob == gb

    return distinct


def _cluster_plates_by_similarity(
    plates_with_conf: list[tuple[str, float]],
    *,
    ratio: int,
    is_distinct: Callable[[str, str], bool] | None = None,
    support: dict[str, float] | None = None,
) -> dict[str, str]:
    """Cluster plate strings by Levenshtein similarity.

    ``plates_with_conf`` is the list of distinct plate strings with
    their max OCR confidence (across all tracks the plate appears in).
    ``ratio`` is the rapidfuzz similarity floor (0-100); pairs at or
    above this AND of equal length are merged.

    ``is_distinct`` is an optional veto: when it returns ``True`` for a
    (seed, plate) pair, the pair is NOT merged even at/above the ratio
    (see :func:`make_distinct_vehicle_checker` -- two near spellings
    that are provably different registered vehicles).

    ``support`` optionally maps a plate to how much read evidence backs
    it (e.g. total sightings). When given it is the PRIMARY seed order:
    the most-read spelling anchors the cluster, so a one-off high-conf
    misread can't hijack the canonical from a plate read far more often
    (OCR confidence commonly ties at 1.0, and the old alphabetical
    tie-break then picked the wrong spelling -- e.g. a single ``FD51PVX``
    read beating an 84-read ``FD61PVX`` because '5' < '6'). Without
    ``support`` the order is by confidence, as before.

    Returns a map ``{plate -> canonical_plate}``. The canonical for a
    cluster is its seed; ties fall through to confidence then the
    alphabetically-first string for determinism.

    Greedy seed-and-grow: process plates in seed order. For each plate,
    check if it matches any existing cluster's seed at >= ratio (with
    same length); if yes, attach; otherwise start a new cluster.
    """
    if not plates_with_conf or ratio is None:
        return {p: p for p, _c in plates_with_conf}

    # Lazy import so the module loads on Python without rapidfuzz.
    from rapidfuzz import fuzz

    # Seed order: most-supported first (when support given), then highest
    # confidence, then alphabetical for determinism.
    sup = support or {}
    ordered = sorted(plates_with_conf, key=lambda x: (-sup.get(x[0], 0.0), -x[1], x[0]))
    canonical: dict[str, str] = {}  # plate -> canonical
    seeds: list[str] = []  # cluster seeds (canonical plates)

    for plate, _conf in ordered:
        matched_seed: str | None = None
        for seed in seeds:
            if len(seed) != len(plate):
                continue
            if fuzz.ratio(seed, plate) >= ratio:
                if is_distinct is not None and is_distinct(seed, plate):
                    continue
                matched_seed = seed
                break
        if matched_seed is None:
            seeds.append(plate)
            canonical[plate] = plate
        else:
            canonical[plate] = matched_seed

    return canonical


def _load_dvsa_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Return ``{plate -> label-row}`` from a ``_dvsa_labels.json``
    harvest, or ``{}`` if the file is absent / unreadable. Plate keys
    are registration strings (uppercase, no spaces) -- the same
    normalisation applied to OCR reads, so they join directly."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    labels = payload.get("labels")
    return labels if isinstance(labels, dict) else {}


def _attach_dvsa_labels(vehicles: list[Vehicle], labels: dict[str, dict[str, Any]]) -> int:
    """Fill make/model/year on each plated vehicle from the DVSA harvest.

    Tries the canonical plate first, then any fuzzy ``plate_variants``
    (the canonical read may have 404'd while a lower-conf variant
    resolved). Returns the number of vehicles labelled. Empty
    ``labels`` is a no-op -- fields stay ``None``.
    """
    if not labels:
        return 0
    n = 0
    for v in vehicles:
        if v.plate is None:
            continue
        for cand in (v.plate, *(p for p, _c in v.plate_variants)):
            row = labels.get(cand)
            if not row:
                continue
            v.make = row.get("make") or None
            v.model = row.get("model") or None
            v.year = row.get("year")
            v.make_model_source = "dvsa"
            n += 1
            break
    return n


def _load_cnn_by_track(path: Path) -> dict[int, dict[str, Any]]:
    """``{track_id -> rollup row}`` from a ``_makemodel_by_track.json``
    (``streettracker makemodel`` output), or ``{}`` when absent/unreadable."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("tracks", []) if isinstance(payload, dict) else []:
        tid = row.get("track_id")
        if tid is not None:
            out[int(tid)] = row
    return out


def _attach_cnn_makes(
    vehicles: list[Vehicle],
    by_tid: dict[int, dict[str, Any]],
    *,
    min_conf: float = CNN_MAKE_MIN_CONF,
) -> int:
    """Fill ``make`` from the CNN rollup on vehicles without DVSA truth.

    DVSA always wins -- only ``make is None`` vehicles are touched, which
    is exactly the unread-plate majority (plus young plated cars with no
    MOT record). Among a vehicle's tracks, the highest-vote-share row at
    or above ``min_conf`` decides; ``make_model_source`` becomes
    ``"cnn"``. Returns the number of vehicles filled."""
    if not by_tid:
        return 0
    n = 0
    for v in vehicles:
        if v.make is not None:
            continue
        best: dict[str, Any] | None = None
        for tid in v.track_ids:
            row = by_tid.get(tid)
            if not row or not row.get("make"):
                continue
            conf = row.get("conf") or 0.0
            if conf < min_conf:
                continue
            if best is None or conf > (best.get("conf") or 0.0):
                best = row
        if best is not None:
            v.make = best["make"]
            v.model = best.get("model") or None
            v.year = best.get("year")
            v.make_model_source = "cnn"
            n += 1
    return n


def build_vehicles(
    session_dir: Path,
    *,
    conf_threshold: float = CONF_THRESHOLD,
    include_unread: bool = True,
    fuzzy_ratio: int | None = FUZZY_RATIO_DEFAULT,
    canonical_only: bool = True,
    dvsa_labels_path: Path | None = None,
    suppress_parked: bool = True,
    cnn_make: bool = True,
) -> list[Vehicle]:
    """Build per-vehicle aggregations from a closed session's outputs.

    ``conf_threshold`` controls which ALPR reads are treated as
    "anchor" plate identities (default 0.9). Reads below that are
    discarded and the track is treated as unread.

    ``include_unread`` controls whether tracks without an anchor read
    are emitted as plate=None vehicles. Set False to focus on the
    plate-anchored subset only.

    ``fuzzy_ratio`` controls plate clustering. When set (default 85),
    OCR variants of the same physical plate (same length, Levenshtein
    similarity >= ``fuzzy_ratio``) collapse into one Vehicle. The
    highest-conf plate string becomes canonical; lower-conf variants
    are recorded under ``Vehicle.plate_variants``. Set to ``None`` to
    disable clustering (strict-string equality only, legacy
    behaviour).

    ``canonical_only`` (default ``True``) discards anchor reads that
    don't match a UK plate shape (current 'LL00 LLL', 1983-2001
    prefix, 1963-1983 suffix). Those tracks are then bucketed as
    unread (``plate=None``) rather than seeding their own synthetic
    cluster. The 2026-05-29 threshold-curve analysis showed ~55-65 %
    of high-conf 'reads' are OCR garbage that the OCR confidence
    score cannot itself reject; this filter prevents that garbage
    from spawning ghost vehicles or contaminating fuzzy clusters.

    ``dvsa_labels_path`` overrides where the DVSA MOT label harvest is
    read from (default ``<session>_dvsa_labels.json``). When present,
    each plated vehicle's ``make``/``model``/``year`` are filled from
    the plate-keyed labels (``make_model_source="dvsa"``); an absent
    file leaves them ``None``.

    ``suppress_parked`` (default ``True``) runs the stationary-beacon
    detector (:mod:`streettracker.analysis.parked`) over the session's
    per-image ``_alpr.json``. A plate read at a fixed image position
    across many moving tracks is a parked car aliasing onto passing
    traffic; those reads are dropped as identity anchors (each affected
    track falls back to its own best remaining read, or unread) and the
    parked stint is attached to the plate's Vehicle as a
    ``parked_episodes`` entry instead. Sessions without ``_alpr.json``
    are unaffected.

    ``cnn_make`` (default ``True``) fills ``make`` from the session's
    ``_makemodel_by_track.json`` (the ``streettracker makemodel`` CNN
    rollup) on vehicles with no DVSA label -- i.e. the unread-plate
    majority plus young plated cars that 404 on the MOT register. DVSA
    ground truth always wins; CNN fills carry
    ``make_model_source="cnn"``. Absent file -> no-op.

    Cars only -- ``class_name == "car"`` tracks. Person tracks are
    skipped (snaps are anatomically wrong for ALPR).

    Vehicles are returned sorted by ``first_seen`` ascending.
    """
    session = session_dir.name
    data_path = session_dir / f"{session}_data.json"
    alpr_by_track_path = session_dir / f"{session}_alpr_by_track.json"

    data = json.loads(data_path.read_text())
    if alpr_by_track_path.exists():
        alpr_rollup = json.loads(alpr_by_track_path.read_text())
    else:
        alpr_rollup = {"tracks": []}

    # Stationary-beacon detection over the per-image reads. Empty
    # detection (no _alpr.json, or suppression disabled) is a no-op.
    detection = ParkedDetection()
    if suppress_parked:
        entries = load_alpr_entries(session_dir)
        if entries:
            detection = detect_parked(entries, data)

    # tid -> anchor read dict. Use the per-image best as the default
    # anchor (max-conf single read). Substitute the consensus rollup
    # ONLY when (a) it agrees with the best on plate text and (b) its
    # confidence is higher -- this is the "multi-frame agreement
    # boost" case where multiple low-conf reads all support the same
    # answer. On this camera's data the per-image reads frequently
    # capture different parked / passing vehicles, so a disagreeing
    # consensus is unreliable and the best-of-N pick is the right
    # anchor. See ``measure_consensus.py`` for the empirical study.
    # Bespoke pipeline is not the read authority (Step 6 of the ANPR
    # tuning loop).
    best_by_tid: dict[int, dict[str, Any]] = {}
    for t in alpr_rollup.get("tracks", []):
        best = t.get("best_preferred")
        if not best or (best.get("ocr_conf") or 0) < conf_threshold:
            continue
        if canonical_only and not is_canonical_uk_plate(
            (best.get("ocr_text") or "").strip().upper().replace(" ", "")
        ):
            # Non-canonical-shape "read" -- treat the track as unread
            # so it falls into the plate=None bucket and doesn't seed
            # a synthetic-cluster vehicle. Older sessions (pre-PR #37)
            # whose alpr_by_track.json doesn't carry
            # ``canonical_uk_shape`` still work: we re-derive it from
            # ``ocr_text`` here rather than trusting the absent field.
            continue
        tid = int(t["track_id"])
        try:
            best_key = (tid, int(best.get("snap_index")))
        except (TypeError, ValueError):
            best_key = None
        if best_key is not None and best_key in detection.suppressed:
            # The rollup anchor is a parked-car beacon read -- the plate
            # belongs to a stationary car, not this passing track.
            # Re-anchor to the track's own best remaining read (the
            # consensus is not trusted here either: it aggregated the
            # same contaminated reads).
            fallback = best_unsuppressed_read(
                detection.reads_by_track.get(tid, []),
                detection.suppressed,
                conf_threshold=conf_threshold,
                canonical_only=canonical_only,
            )
            if fallback is not None:
                best_by_tid[t["track_id"]] = fallback
            continue
        consensus = t.get("consensus_preferred")
        anchor = best
        if (
            consensus
            and consensus.get("ocr_text") == best.get("ocr_text")
            and (consensus.get("ocr_conf") or 0) > (best.get("ocr_conf") or 0)
        ):
            anchor = dict(consensus)
            anchor.setdefault("image", consensus.get("best_image"))
            anchor.setdefault("snap_index", consensus.get("best_snap_index"))
        best_by_tid[t["track_id"]] = anchor

    # First pass: group by strict-string plate equality. Tracks whose
    # best read is below threshold (or absent) collect under ``None``.
    strict_groups: dict[str | None, list[tuple[dict, dict | None]]] = {}
    for rec in data:
        if rec.get("class_name") != "car":
            continue
        tid = rec.get("track_id")
        best = best_by_tid.get(tid)
        plate = best["ocr_text"] if best else None
        strict_groups.setdefault(plate, []).append((rec, best))

    # Second pass: fuzzy plate clustering on the plated subset.
    # ``canonical`` maps each strict plate to its cluster representative
    # (the highest-conf plate in the cluster). The plate=None bucket
    # is not eligible -- those tracks are anonymous already.
    #
    # Note: an earlier version of this code rejected merges whose
    # tracks overlapped in wall-clock time, on the theory that the
    # same physical car cannot be in frame twice at the same
    # instant. That logic was wrong on this setup: BotSORT routinely
    # ID-switches mid-transit (brief occlusion / detection gap),
    # producing two overlapping tracks for ONE physical vehicle --
    # often with disagreeing direction labels because the new track's
    # motion vector takes a couple of frames to settle. A real
    # example surfaced visually on session_20260526_124704 tracks
    # 1516 (LD22BMG, R→L, "black") and 1517 (LD22BWG, L→R,
    # "unknown") -- both 4K snaps show the same silver hatchback
    # driving R→L, one second apart. Rejecting that merge produced
    # a false negative more often than the temporal-overlap heuristic
    # caught a true positive, so the rejection is gone.
    dvsa_labels_file = (
        dvsa_labels_path
        if dvsa_labels_path is not None
        else session_dir / f"{session}_dvsa_labels.json"
    )
    dvsa_rows = _load_dvsa_labels(dvsa_labels_file)

    if fuzzy_ratio is not None:
        plated_plates: list[tuple[str, float]] = []
        colours_by_plate: dict[str, Counter] = {}
        for plate, items in strict_groups.items():
            if plate is None:
                continue
            max_conf = max((b.get("ocr_conf") or 0.0) for _r, b in items if b)
            plated_plates.append((plate, max_conf))
            colours_by_plate[plate] = Counter((rec.get("color") or "unknown") for rec, _b in items)
        canonical = _cluster_plates_by_similarity(
            plated_plates,
            ratio=fuzzy_ratio,
            is_distinct=make_distinct_vehicle_checker(dvsa_rows, colours_by_plate),
        )
    else:
        canonical = {p: p for p in strict_groups if p is not None}

    # Fold strict groups into clusters (by canonical plate).
    clusters: dict[str | None, list[tuple[dict, dict | None]]] = {}
    cluster_variants: dict[str, list[tuple[str, float]]] = {}
    for plate, items in strict_groups.items():
        if plate is None:
            clusters[None] = items
            continue
        can = canonical.get(plate, plate)
        clusters.setdefault(can, []).extend(items)
        if can != plate:
            # Record this OCR variant under the cluster's canonical.
            variant_conf = max((b.get("ocr_conf") or 0.0) for _r, b in items if b)
            cluster_variants.setdefault(can, []).append((plate, variant_conf))

    out: list[Vehicle] = []
    for plate, items in clusters.items():
        if plate is None and not include_unread:
            continue
        if plate is None:
            for rec, _best in items:
                out.append(_make_single_vehicle(rec, best=None))
            continue
        v = _make_grouped_vehicle(plate, items)
        variants = cluster_variants.get(plate)
        if variants:
            # Stable order: highest-conf variant first, alphabetical
            # tie-break.
            v.plate_variants = sorted(variants, key=lambda x: (-x[1], x[0]))
        out.append(v)

    out.sort(key=lambda v: v.first_seen)

    # DVSA-first make/model: join the plate-keyed MOT label harvest
    # (``streettracker dvsa-label``) onto the plated vehicles. Absent
    # file -> make/model/year stay None.
    _attach_dvsa_labels(out, dvsa_rows)

    if cnn_make:
        _attach_cnn_makes(
            out,
            _load_cnn_by_track(session_dir / f"{session}_makemodel_by_track.json"),
        )

    if detection.episodes:
        _attach_parked_episodes(out, detection, fuzzy_ratio)
    return out


def _attach_parked_episodes(
    vehicles: list[Vehicle],
    detection: ParkedDetection,
    fuzzy_ratio: int | None,
) -> None:
    """Attach merged parked episodes to the Vehicle owning that plate.

    OCR spellings of one parked plate (the beacon is often read under 2+
    variants) are folded with the same fuzzy clustering used for visit
    grouping, seeded by the vehicles' anchor plates so an episode variant
    lands on the surviving canonical spelling. Episodes whose plate has
    no surviving Vehicle (the parked car never produced a genuine moving
    read in this session) are currently dropped from the per-vehicle
    output -- they still reach callers via ``detect_parked`` directly.
    """
    anchor_confs: dict[str, float] = {}
    for v in vehicles:
        if v.plate is None:
            continue
        norm = normalize_plate(v.plate)
        anchor_confs[norm] = max(anchor_confs.get(norm, 0.0), v.plate_conf or 0.0)
    joint = list(anchor_confs.items()) + [
        (ep.plate, 0.0) for ep in detection.episodes if ep.plate not in anchor_confs
    ]
    if fuzzy_ratio is not None:
        canonical = _cluster_plates_by_similarity(joint, ratio=fuzzy_ratio)
    else:
        canonical = {p: p for p, _c in joint}

    by_norm: dict[str, Vehicle] = {}
    for v in vehicles:
        if v.plate is None:
            continue
        by_norm.setdefault(normalize_plate(v.plate), v)
        for pv, _c in v.plate_variants:
            by_norm.setdefault(normalize_plate(pv), v)

    for ep in merge_episodes(detection.episodes, canonical):
        target = by_norm.get(ep.plate)
        if target is not None:
            target.parked_episodes.append(ep.to_json_dict())


def build_cross_session(
    session_dirs: list[Path],
    *,
    conf_threshold: float = CONF_THRESHOLD,
    fuzzy_ratio: int | None = FUZZY_RATIO_DEFAULT,
    canonical_only: bool = True,
    suppress_parked: bool = True,
    cnn_make: bool = True,
) -> list[CrossVehicle]:
    """Pool plated vehicles across sessions, classifying repeats by date.

    Calls :func:`build_vehicles` per session (so the DVSA make/model
    join + within-session fuzzy clustering apply), then runs one more
    fuzzy pass across the per-session canonical plates so a 1-char OCR
    diff doesn't split one physical car across two sessions. Unread
    (plate=None) vehicles are dropped -- cross-session identity needs a
    plate. Returns ``CrossVehicle`` records sorted regulars-first
    (different-day, then same-day, then one-off; ties by visit count).
    """
    plated: list[tuple[str, Vehicle]] = []
    conf_by_plate: dict[str, float] = {}
    for d in session_dirs:
        for v in build_vehicles(
            d,
            conf_threshold=conf_threshold,
            fuzzy_ratio=fuzzy_ratio,
            canonical_only=canonical_only,
            suppress_parked=suppress_parked,
            cnn_make=cnn_make,
        ):
            if v.plate is None:
                continue
            plated.append((d.name, v))
            conf_by_plate[v.plate] = max(conf_by_plate.get(v.plate, 0.0), v.plate_conf or 0.0)

    if fuzzy_ratio is not None:
        # Cross-session veto evidence: DVSA rows pooled across the cohort
        # + each spelling's observed colours pooled across its sessions.
        labels_pool: dict[str, dict[str, Any]] = {}
        for d in session_dirs:
            for plate, row in _load_dvsa_labels(d / f"{d.name}_dvsa_labels.json").items():
                labels_pool.setdefault(plate, row)
        colours_pool: dict[str, Counter] = {}
        for _sname, v in plated:
            if v.plate is not None:
                colours_pool.setdefault(v.plate, Counter()).update(v.colors)
        canonical = _cluster_plates_by_similarity(
            list(conf_by_plate.items()),
            ratio=fuzzy_ratio,
            is_distinct=make_distinct_vehicle_checker(labels_pool, colours_pool),
        )
    else:
        canonical = {p: p for p in conf_by_plate}

    pool: dict[str, dict[str, Any]] = {}
    for sname, v in plated:
        if v.plate is None:  # filtered above; narrows the type for mypy
            continue
        key = canonical.get(v.plate, v.plate)
        rec = pool.setdefault(
            key,
            {
                "make": None,
                "model": None,
                "year": None,
                "make_model_source": None,
                "sessions": set(),
                "variants": set(),
                "visits": [],
            },
        )
        rec["sessions"].add(sname)
        if v.plate != key:
            rec["variants"].add(v.plate)
        for pv, _c in v.plate_variants:
            rec["variants"].add(pv)
        if v.make and rec["make"] is None:
            rec["make"], rec["model"], rec["year"], rec["make_model_source"] = (
                v.make,
                v.model,
                v.year,
                v.make_model_source,
            )
        for vis in v.visits:
            rec["visits"].append((vis.time_start, vis.direction))

    out: list[CrossVehicle] = []
    for plate, rec in pool.items():
        starts = sorted(t for t, _d in rec["visits"] if t)
        dates = sorted({t[:10] for t, _d in rec["visits"] if t})
        n_visits = len(rec["visits"])
        kind = "different-day" if len(dates) >= 2 else "same-day" if n_visits >= 2 else "one-off"
        out.append(
            CrossVehicle(
                plate=plate,
                make=rec["make"],
                model=rec["model"],
                year=rec["year"],
                make_model_source=rec["make_model_source"],
                n_visits=n_visits,
                n_dates=len(dates),
                dates=dates,
                sessions=sorted(rec["sessions"]),
                kind=kind,
                directions=dict(Counter(d for _t, d in rec["visits"])),
                first_seen=starts[0] if starts else "",
                last_seen=starts[-1] if starts else "",
                plate_variants=sorted(rec["variants"]),
            )
        )
    _kind_rank = {"different-day": 0, "same-day": 1, "one-off": 2}
    out.sort(key=lambda c: (_kind_rank[c.kind], -c.n_visits, c.plate))
    return out


def _make_single_vehicle(rec: dict, best: dict | None) -> Vehicle:
    visit = _make_visit(rec, best)
    plate = best["ocr_text"] if best else None
    plate_conf = best.get("ocr_conf") if best else None
    return Vehicle(
        plate=plate,
        plate_conf=plate_conf,
        n_visits=1,
        track_ids=[rec["track_id"]],
        first_seen=rec["time_start"],
        last_seen=rec["time_end"],
        gap_minutes_max=0.0,
        gap_minutes_min=0.0,
        directions={rec["direction"]: 1},
        colors={rec["color"]: 1},
        visits=[visit],
    )


def _make_grouped_vehicle(plate: str, items: list[tuple[dict, dict | None]]) -> Vehicle:
    # Sort visits chronologically by track start.
    items_sorted = sorted(items, key=lambda p: p[0]["time_start_unix"])
    visits = [_make_visit(rec, best) for rec, best in items_sorted]

    # Plate confidence: max across visits' best reads.
    plate_conf = max((b.get("ocr_conf") or 0.0) for _r, b in items_sorted if b)
    track_ids = [v.track_id for v in visits]
    first_seen = visits[0].time_start
    last_seen = visits[-1].time_end

    # Visit gaps: end of visit i to start of visit i+1 (minutes).
    gaps_min: list[float] = []
    for prev, nxt in zip(visits, visits[1:], strict=False):
        gap_s = nxt.time_start_unix - prev.time_end_unix
        gaps_min.append(gap_s / 60.0)
    gap_max = max(gaps_min) if gaps_min else 0.0
    gap_min = min(gaps_min) if gaps_min else 0.0

    directions = dict(Counter(v.direction for v in visits))
    colors = dict(Counter(v.color for v in visits))

    return Vehicle(
        plate=plate,
        plate_conf=plate_conf,
        n_visits=len(visits),
        track_ids=track_ids,
        first_seen=first_seen,
        last_seen=last_seen,
        gap_minutes_max=gap_max,
        gap_minutes_min=gap_min,
        directions=directions,
        colors=colors,
        visits=visits,
    )


def _make_visit(rec: dict, best: dict | None) -> VehicleVisit:
    return VehicleVisit(
        track_id=rec["track_id"],
        time_start=rec["time_start"],
        time_end=rec["time_end"],
        time_start_unix=rec["time_start_unix"],
        time_end_unix=rec["time_end_unix"],
        duration_visible=rec["duration_visible"],
        direction=rec["direction"],
        speed_px_s=rec["speed_px_s"],
        color=rec["color"],
        lane=rec["lane"],
        asset_prefix=rec.get("asset_prefix", "vehicle"),
        n_snaps=len(rec.get("main_snaps") or []),
        best_image=(best.get("image") if best else None),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="streettracker vehicles",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_dir", type=Path)
    ap.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help=(
            "Minimum OCR confidence to treat a plate read as a "
            "vehicle-identity anchor (default 0.9)."
        ),
    )
    ap.add_argument(
        "--no-unread",
        action="store_true",
        help=(
            "Skip cars whose plate was never read above --conf. "
            "Default emits them as plate=None vehicles."
        ),
    )
    ap.add_argument(
        "--fuzzy-ratio",
        type=int,
        default=FUZZY_RATIO_DEFAULT,
        help=(
            "Levenshtein similarity floor (0-100) for collapsing OCR "
            f"variants of the same physical plate. Default "
            f"{FUZZY_RATIO_DEFAULT}. Only same-length plates are "
            "eligible. Set with --no-fuzzy to disable."
        ),
    )
    ap.add_argument(
        "--no-fuzzy",
        action="store_true",
        help="Disable fuzzy plate clustering (strict-string only).",
    )
    ap.add_argument(
        "--include-non-canonical",
        action="store_true",
        help=(
            "Treat OCR reads that don't match a UK plate shape as valid "
            "anchors anyway. Off by default to keep OCR garbage out of "
            "the per-vehicle output. Mirrors `streettracker dvsa-label`'s "
            "flag of the same name."
        ),
    )
    ap.add_argument(
        "--no-cnn-make",
        action="store_true",
        help=(
            "Don't fill make from the CNN classifier rollup "
            "(_makemodel_by_track.json) on vehicles without a DVSA "
            "label. Default fills them with make_model_source='cnn'; "
            "DVSA ground truth always wins either way."
        ),
    )
    ap.add_argument(
        "--no-parked-suppression",
        action="store_true",
        help=(
            "Disable the stationary-beacon detector. By default a plate "
            "read at a fixed image position across >=4 distinct moving "
            "tracks is treated as a PARKED car aliasing onto passing "
            "traffic: those reads stop counting as visits (affected "
            "tracks re-anchor to their own next-best read) and the "
            "parked stint is reported as a parked_episodes entry on the "
            "plate's vehicle instead."
        ),
    )
    ap.add_argument(
        "--across",
        type=Path,
        nargs="+",
        default=None,
        metavar="SESSION_DIR",
        help=(
            "Cross-session mode: pool the primary session_dir with these "
            "additional session dir(s) into a repeat-vehicle summary "
            "(regulars seen on >=2 calendar dates). Reuses the fuzzy "
            "clustering so OCR variants merge across sessions. Writes "
            "cross_session_repeats.json (see --out) and prints the summary."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path for the --across JSON (default: "
            "<session_dir parent>/cross_session_repeats.json)."
        ),
    )
    return ap


def _run_cross(args: argparse.Namespace) -> int:
    dirs = [args.session_dir, *args.across]
    for d in dirs:
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 2
    cross = build_cross_session(
        dirs,
        conf_threshold=args.conf,
        fuzzy_ratio=None if args.no_fuzzy else args.fuzzy_ratio,
        canonical_only=not args.include_non_canonical,
        suppress_parked=not args.no_parked_suppression,
    )
    out_path = args.out or (args.session_dir.parent / "cross_session_repeats.json")
    out_path.write_text(json.dumps([c.to_json_dict() for c in cross], indent=2))
    _print_cross_summary(cross, dirs, out_path)
    return 0


def _print_cross_summary(cross: list[CrossVehicle], dirs: list[Path], out_path: Path) -> None:
    diff = [c for c in cross if c.kind == "different-day"]
    same = [c for c in cross if c.kind == "same-day"]
    both = [c for c in cross if len(c.sessions) >= 2]
    labelled = sum(1 for c in cross if c.make)

    def _mm(c: CrossVehicle) -> str:
        if not c.make:
            return ""
        yr = f" {c.year}" if c.year else ""
        return f"{c.make} {(c.model or '')[:18]}{yr}"

    print(f"[vehicles --across] wrote {out_path}  ({len(cross)} distinct vehicles)")
    print(f"  sessions:               {', '.join(d.name for d in dirs)}")
    print(f"  one-off:                {sum(1 for c in cross if c.kind == 'one-off')}")
    print(f"  same-day repeats:       {len(same)}")
    print(f"  DIFFERENT-day repeats:  {len(diff)}  (seen on >=2 dates)")
    print(f"  seen in >=2 sessions:   {len(both)}")
    print(f"  with DVSA make/model:   {labelled}/{len(cross)}")
    if diff:
        print("\n  DIFFERENT-DAY repeats (regulars):")
        print(f"    {'plate':<9} {'vis':>3} {'days':>4} {'sess':>4}  make/model")
        for c in diff:
            var = f"  ~{','.join(c.plate_variants)}" if c.plate_variants else ""
            print(
                f"    {c.plate:<9} {c.n_visits:>3} {c.n_dates:>4} "
                f"{len(c.sessions):>4}  {_mm(c)}{var}"
            )
    if same:
        print(f"\n  SAME-DAY repeats (first 20 of {len(same)}):")
        print(f"    {'plate':<9} {'vis':>3}  make/model")
        for c in same[:20]:
            print(f"    {c.plate:<9} {c.n_visits:>3}  {_mm(c)}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.across:
        return _run_cross(args)
    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2

    vehicles = build_vehicles(
        session_dir,
        conf_threshold=args.conf,
        include_unread=not args.no_unread,
        fuzzy_ratio=None if args.no_fuzzy else args.fuzzy_ratio,
        canonical_only=not args.include_non_canonical,
        suppress_parked=not args.no_parked_suppression,
        cnn_make=not args.no_cnn_make,
    )

    session = session_dir.name
    out_path = session_dir / f"{session}_vehicles.json"
    out_path.write_text(json.dumps([v.to_json_dict() for v in vehicles], indent=2))
    print(f"[vehicles] wrote {out_path}  ({len(vehicles)} vehicles)")

    # Headline numbers.
    plated = [v for v in vehicles if v.plate is not None]
    recurring = [v for v in plated if v.n_visits >= 2]
    print(f"  plated:    {len(plated)}")
    print(f"  recurring: {len(recurring)}  (>=2 visits in session)")
    dvsa_labelled = [v for v in vehicles if v.make_model_source == "dvsa"]
    if dvsa_labelled:
        print(f"  make/model: {len(dvsa_labelled)}  (DVSA-labelled)")
    cnn_filled = [v for v in vehicles if v.make_model_source == "cnn"]
    if cnn_filled:
        print(f"  cnn-make:   {len(cnn_filled)}  (classifier fill, no DVSA truth)")
    merged = [v for v in plated if v.plate_variants]
    if merged:
        print(f"  fuzzy-merged: {len(merged)}  (OCR variants collapsed)")
    episodes = [(v.plate, ep) for v in plated for ep in v.parked_episodes]
    if episodes:
        print(f"  parked episodes: {len(episodes)}  (beacon reads excluded from visits)")
        for plate, ep in episodes[:5]:
            print(
                f"    {plate}  {ep['first_seen'][:16]} -> {ep['last_seen'][11:16]}"
                f"  ~{ep['duration_minutes']:.0f} min  "
                f"read {ep['n_reads']}x across {ep['n_tracks']} passing tracks"
            )
    if recurring:
        print("  top recurring:")
        for v in sorted(recurring, key=lambda x: -x.n_visits)[:5]:
            extras = ""
            if v.plate_variants:
                variant_strs = ", ".join(f"{p}@{c:.2f}" for p, c in v.plate_variants)
                extras = f"  variants=[{variant_strs}]"
            print(
                f"    {v.plate}  n_visits={v.n_visits}  "
                f"gap_min_max={v.gap_minutes_max:.1f}min  "
                f"directions={dict(v.directions)}{extras}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
