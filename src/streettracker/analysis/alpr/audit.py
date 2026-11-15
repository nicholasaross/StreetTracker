"""The plate-read audit set (review E1.2): which reads a person checks.

Every plate-read rate quoted before this set existed counts a car as read
when a snap produced a UK-shaped string. Nothing checked that the characters
were right, or that the plate was on the tracked car and not on an oncoming or
parked one (findings R1-R3). The audit set is a sample of reads a person
verifies against the picture, drawn so that the answers scale back up to
every read: each item belongs to a stratum whose size in the population is
recorded, and :mod:`streettracker.analysis.alpr.audit_score` weights by it.

Four samples, each over car tracks:

``best``
    A track's best read (``best_preferred`` in the rollup) when it is
    UK-shaped: the read every consumer uses. Stratified by direction, by
    light and by how the read was reached (see :func:`crop_path`), with the
    reads that fail the plate gate kept apart from those that pass.
``colour_rejected`` / ``static_rejected``
    A UK-shaped read that would have been the track's best but was set aside
    by the plate-colour check or the static-plate filter. These calibrate the
    two filters: a rejected plate that is on the tracked car is a lost read.
``dvsa``
    One read per DVSA-labelled car (a distinct plate), stratified by how many
    tracks anywhere read that plate. The labeller also says whether the
    register's make, colour and body type fit the car in the picture.

This module reads session files and draws the sample. It opens no image
beyond one header per session (for the snap size); rendering the pictures
and serving the labelling page is ``.claude/audit_e12.py``.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from streettracker.analysis.alpr.base import OCR_CONF_METHOD, normalize_label_text
from streettracker.analysis.alpr.fullframe import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_PAD_PX,
    rank_vehicle_candidates,
)
from streettracker.analysis.alpr.gate import PlateGate
from streettracker.analysis.snap_assets import (
    hint_window,
    image_size,
    load_bbox_index,
    load_done_bbox_index,
    scale_bbox,
)
from streettracker.common.fileio import read_json
from streettracker.common.session_files import SessionFiles

Box = tuple[float, float, float, float]
IntBox = tuple[int, int, int, int]
Candidate = dict[str, Any]

DIRECTIONS = ("right to left", "left to right")
LIGHT_BANDS = ("day", "dusk", "dark")
# How a best read was reached, as the strata see it. ``rank0`` is split by
# the plate gate because it holds nearly every read: without the split the
# sample would have too few gate-failing reads to calibrate the gate on.
READ_GROUPS = ("rank0_pass", "rank0_fail", "rank1", "fallback")
SUPPORT_BANDS = ("1", "2-4", "5+")
SAMPLES = ("best", "colour_rejected", "static_rejected", "dvsa")

# The sun's elevation, in degrees, that separates the light bands. Below
# +6 degrees the light falls fast and the camera lengthens its exposure
# (the blur that limits low-light reads); below -6 degrees (the end of civil
# twilight) the street lights are all there is. "dusk" covers dawn too.
DAY_MIN_ELEVATION = 6.0
DARK_MAX_ELEVATION = -6.0

# Fewest best reads drawn for a direction in a light band (all, if it holds fewer).
DOMAIN_FLOOR = 10

# What the labeller can say about the boxed plate.
ATTRIBUTIONS: dict[str, str] = {
    "tracked": "On the tracked car",
    "other_moving": "On another moving car",
    "parked": "On a parked or stopped car",
    "not_plate": "Not a number plate",
    "unsure": "Can't tell which car is tracked",
}
DVSA_FIELDS = ("make", "colour", "body")
DVSA_ANSWERS = ("yes", "no", "unsure")
RECONCILED = ("kept", "ocr_right")
_MAX_NOTE_CHARS = 500


# ----------------------------------------------------------------------
# Light


def solar_elevation_deg(unix: float, latitude: float, longitude: float) -> float:
    """The sun's elevation above the horizon, in degrees, at a UTC time.

    The Astronomical Almanac's low-precision formulas, good to a small
    fraction of a degree this century. No refraction, which only matters
    within a degree of the horizon and not for banding light.
    """
    days = unix / 86400.0 - 10957.5  # since 2000-01-01 12:00 UTC
    mean_longitude = math.radians((280.460 + 0.9856474 * days) % 360.0)
    mean_anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    ecliptic_longitude = (
        mean_longitude
        + math.radians(1.915) * math.sin(mean_anomaly)
        + math.radians(0.020) * math.sin(2.0 * mean_anomaly)
    )
    obliquity = math.radians(23.439 - 0.0000004 * days)
    declination = math.asin(math.sin(obliquity) * math.sin(ecliptic_longitude))
    right_ascension = math.atan2(
        math.cos(obliquity) * math.sin(ecliptic_longitude), math.cos(ecliptic_longitude)
    )
    sidereal_hours = (18.697374558 + 24.06570982441908 * days) % 24.0
    hour_angle = math.radians(sidereal_hours * 15.0 + longitude) - right_ascension
    lat = math.radians(latitude)
    sin_elevation = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(
        declination
    ) * math.cos(hour_angle)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elevation))))


def light_band(elevation_deg: float) -> str:
    """``"day"``, ``"dusk"`` (low sun, morning or evening) or ``"dark"``."""
    if elevation_deg >= DAY_MIN_ELEVATION:
        return "day"
    if elevation_deg >= DARK_MAX_ELEVATION:
        return "dusk"
    return "dark"


# ----------------------------------------------------------------------
# How a read was reached


def _centre_inside(
    plate_bbox: Sequence[float], box: Sequence[float], pad: float = DEFAULT_PAD_PX
) -> bool:
    cx = (plate_bbox[0] + plate_bbox[2]) / 2.0
    cy = (plate_bbox[1] + plate_bbox[3]) / 2.0
    return box[0] - pad <= cx <= box[2] + pad and box[1] - pad <= cy <= box[3] + pad


def crop_path(
    plate_bbox: Sequence[float],
    vehicle_boxes: Sequence[Sequence[float]],
    size: tuple[int, int],
    *,
    hint: IntBox | None,
    road_polygon_frac: list[tuple[float, float]] | None,
    bottom_retry: bool = False,
) -> tuple[str, Box | None]:
    """Which of ``alpr-run``'s crops a plate detection came from, and the
    vehicle box it sits on.

    ``alpr-run`` does not record this, so it is worked out again from the
    snap's cached vehicle boxes with the rule ``TrajectoryCropDetector``
    uses: ``"rank0"`` is the on-road vehicle nearest the hint, ``"rank1"``
    the second nearest (tried too, and it wins when its plate detection is
    the more confident), ``"hint_crop"`` the fallback to the stale hint
    window, ``"bottom_retry"`` the tall-vehicle retry (recorded on the read)
    and ``"unmatched"`` anything else, including the whole-image fallback.

    The answer is close, not exact: the cached boxes include motorcycles and
    were detected without the ghost mask, where ``alpr-run`` had neither.
    """
    boxes: list[Box] = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in vehicle_boxes]
    if bottom_retry:
        on = [b for b in boxes if _centre_inside(plate_bbox, b)]
        return "bottom_retry", min(on, key=_area) if on else None
    width, height = size
    ranked = rank_vehicle_candidates(
        boxes, width, height, bbox_hint=hint, road_polygon_frac=road_polygon_frac
    )[:DEFAULT_MAX_CANDIDATES]
    for rank, box in enumerate(ranked):
        if _centre_inside(plate_bbox, box):
            return f"rank{rank}", box
    if hint is not None and _centre_inside(plate_bbox, hint):
        return "hint_crop", None
    return "unmatched", None


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def read_group(path: str, gate_pass: bool) -> str:
    """The :data:`READ_GROUPS` stratum of a best read."""
    if path == "rank0":
        return "rank0_pass" if gate_pass else "rank0_fail"
    return "rank1" if path == "rank1" else "fallback"


def support_band(n_tracks: int) -> str:
    """The :data:`SUPPORT_BANDS` stratum of a plate read by ``n_tracks``
    tracks anywhere."""
    if n_tracks <= 1:
        return "1"
    return "2-4" if n_tracks <= 4 else "5+"


# ----------------------------------------------------------------------
# One session's candidates


@dataclass(slots=True)
class SessionPool:
    """Every read of one session that could be sampled, by sample, and the
    car counts the rates are quoted against (keyed ``"direction|light"``)."""

    session: str
    best: list[Candidate] = field(default_factory=list)
    colour_rejected: list[Candidate] = field(default_factory=list)
    static_rejected: list[Candidate] = field(default_factory=list)
    dvsa: list[Candidate] = field(default_factory=list)
    cars: Counter[str] = field(default_factory=Counter)
    cars_with_snaps: Counter[str] = field(default_factory=Counter)


def skip_reason(session_dir: Path) -> str | None:
    """Why a session can't be sampled, or ``None`` if it can.

    The audit measures the pipeline as it runs now, so a session must have
    been through it: fullframe crops, the min-character confidence and the
    plate-colour check. It also needs the vehicle-box cache, which any of
    the classifier commands leaves behind.
    """
    files = SessionFiles(session_dir)
    for path in (files.data, files.alpr, files.alpr_by_track, files.vehicle_boxes):
        if not path.is_file():
            return f"no {path.name}"
    stamps = read_json(files.static_plates, {})
    if not isinstance(stamps, dict) or stamps.get("crop_mode") != "fullframe":
        return "plates not read with fullframe crops (re-run enrich)"
    if stamps.get("ocr_conf") != OCR_CONF_METHOD:
        return "confidence not re-scored (alpr-rescore)"
    if "plate_colour" not in stamps:
        return "no plate-colour check (alpr-colour)"
    return None


def _car_tracks(
    records: Iterable[Any], latitude: float, longitude: float
) -> dict[int, dict[str, Any]]:
    """Car tracks that count toward a rate, with their direction and light."""
    cars: dict[int, dict[str, Any]] = {}
    for r in records:
        if not isinstance(r, dict) or r.get("class_name") != "car" or r.get("class_suspect"):
            continue
        direction, tid, started = r.get("direction"), r.get("track_id"), r.get("time_start_unix")
        if direction not in DIRECTIONS or tid is None or started is None:
            continue
        elevation = solar_elevation_deg(float(started), latitude, longitude)
        light = light_band(elevation)
        cars[int(tid)] = {
            "direction": direction,
            "light": light,
            "domain": f"{direction}|{light}",
            "sun_elevation": round(elevation, 1),
            "time_start": r.get("time_start"),
            "has_snaps": bool(r.get("main_snaps")),
        }
    return cars


@dataclass(frozen=True, slots=True)
class _Scene:
    """What it takes to place a read in its snap: the recorded track boxes,
    the cached vehicle boxes and the snap size, for one session."""

    size: tuple[int, int]
    sub_size: tuple[int, int] | None
    fire_index: dict[tuple[int, int], IntBox]
    done_index: dict[tuple[int, int], IntBox]
    vehicle_boxes: Mapping[str, Sequence[Sequence[float]]]
    road_polygon_frac: list[tuple[float, float]] | None

    def _scaled(self, box: Sequence[float] | None) -> IntBox | None:
        if box is None or self.sub_size is None:
            return None
        return scale_bbox((box[0], box[1], box[2], box[3]), self.sub_size, self.size)

    def place(self, read: Mapping[str, Any]) -> dict[str, Any]:
        """The read's crop path and the boxes the labelling page draws."""
        key = (int(read["track_id"]), int(read["snap_index"]))
        hint = self._scaled(hint_window(*key, self.fire_index, done_index=self.done_index))
        path, vehicle_box = crop_path(
            read["det_bbox"],
            self.vehicle_boxes.get(str(read["image"]), []),
            self.size,
            hint=hint,
            road_polygon_frac=self.road_polygon_frac,
            bottom_retry=bool(read.get("bottom_retry")),
        )
        return {
            "path": path,
            "vehicle_box": [round(v) for v in vehicle_box] if vehicle_box else None,
            "hint": list(hint) if hint else None,
            "fire_box": _as_list(self._scaled(self.fire_index.get(key))),
            "done_box": _as_list(self._scaled(self.done_index.get(key))),
            "image_size": list(self.size),
        }


def _as_list(box: IntBox | None) -> list[int] | None:
    return list(box) if box is not None else None


def _snap_size(session_dir: Path, images: Iterable[str], tries: int = 25) -> tuple[int, int] | None:
    """The session's 4K snap size, from the first of ``images`` that opens.
    Every snap of a session has the one main-stream resolution."""
    for n, name in enumerate(images):
        if n >= tries:
            break
        size = image_size(session_dir / name)
        if size is not None:
            return size
    return None


def _plate(text: object) -> str:
    return str(text or "").replace(" ", "").upper()


@dataclass(frozen=True, slots=True)
class _TrackContext:
    """What every candidate of one track shares."""

    session: str
    car: Mapping[str, Any]
    scene: _Scene
    gate: PlateGate
    support: Mapping[str, int]
    track_best: Mapping[str, Any] | None


def _candidate(
    sample: str, read: Mapping[str, Any], n_agree: int, context: _TrackContext
) -> Candidate:
    """One read as an audit item: what the pipeline said about it, where it
    sits in the snap, and the stratum it is drawn from."""
    car, track_best = context.car, context.track_best
    plate = _plate(read.get("ocr_text"))
    conf = read.get("ocr_conf")
    support = context.support.get(plate, 0)
    gate_pass = context.gate.passes(conf, n_agree=n_agree, support=support)
    placed = context.scene.place(read)
    if sample == "best":
        stratum = f"best|{car['domain']}|{read_group(placed['path'], gate_pass)}"
    else:
        stratum = f"{sample}|{car['direction']}"
    return {
        "key": f"{context.session}/{read['image']}",
        "sample": sample,
        "stratum": stratum,
        "session": context.session,
        "image": read["image"],
        "track_id": int(read["track_id"]),
        "snap_index": int(read["snap_index"]),
        "direction": car["direction"],
        "light": car["light"],
        "sun_elevation": car["sun_elevation"],
        "time_start": car["time_start"],
        "ocr_text": plate,
        "ocr_conf": conf,
        "det_conf": read.get("det_conf"),
        "det_bbox": [int(v) for v in read["det_bbox"]],
        "plate_colour": read.get("plate_colour"),
        "colour_suspect": bool(read.get("colour_suspect")),
        "static_suspect": bool(read.get("static_suspect")),
        "n_agree": n_agree,
        "support": support,
        "gate_pass": gate_pass,
        # The track's best read when the set was drawn: what a later
        # pipeline is compared with.
        "track_best": (
            {
                "ocr_text": _plate(track_best.get("ocr_text")),
                "ocr_conf": track_best.get("ocr_conf"),
                "canonical_uk_shape": bool(track_best.get("canonical_uk_shape")),
            }
            if track_best
            else None
        ),
        **placed,
    }


def _usable_reads(alpr: Iterable[Any], cars: Mapping[int, Any]) -> dict[int, list[dict[str, Any]]]:
    """Each car track's UK-shaped preferred-pipeline reads that have a plate
    box, suspects included."""
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in alpr:
        if not isinstance(e, dict) or e.get("pipeline") not in (None, "preferred"):
            continue
        tid = e.get("track_id")
        if tid is None or int(tid) not in cars:
            continue
        if e.get("ocr_text") and e.get("canonical_uk_shape") and e.get("det_bbox"):
            by_track[int(tid)].append(e)
    return by_track


def _rejected_read(
    reads: Sequence[Mapping[str, Any]], best: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """The suspect read that would have been the track's best, if any: the
    most confident flagged read, when it beats the best the rollup kept."""
    flagged = [r for r in reads if r.get("static_suspect") or r.get("colour_suspect")]
    if not flagged:
        return None
    top = max(flagged, key=lambda r: float(r.get("ocr_conf") or 0.0))
    kept = float((best or {}).get("ocr_conf") or 0.0) if best else -1.0
    return top if float(top.get("ocr_conf") or 0.0) > kept else None


def _dvsa_details(plate: str, row: Mapping[str, Any], n_tracks: int) -> dict[str, Any]:
    from streettracker.analysis.makemodel.bodytype import body_type_for
    from streettracker.analysis.makemodel.colour import colour_class_for

    return {
        "plate": plate,
        "make": row.get("make"),
        "model": row.get("model"),
        "year": row.get("year"),
        "primary_colour": row.get("primary_colour"),
        "colour_class": colour_class_for(row.get("primary_colour")),
        "body_type": body_type_for(row.get("make"), row.get("model")),
        "support": n_tracks,
        "support_band": support_band(n_tracks),
    }


def _dvsa_candidates(
    labels: object, best_by_track: Mapping[int, Candidate], support: Mapping[str, int]
) -> list[Candidate]:
    """One candidate per labelled track whose best read is the labelled
    plate. ``plate`` on each says which car it stands for."""
    out: list[Candidate] = []
    rows = labels.get("labels") if isinstance(labels, dict) else None
    for plate, row in (rows or {}).items():
        if not isinstance(row, dict) or not row.get("make"):
            continue
        plate = _plate(plate)
        for tid in row.get("track_ids") or []:
            base = best_by_track.get(int(tid))
            if base is None or base["ocr_text"] != plate or not base["gate_pass"]:
                continue
            details = _dvsa_details(plate, row, support.get(plate, 0))
            out.append(
                {
                    **base,
                    "sample": "dvsa",
                    "stratum": f"dvsa|{details['support_band']}",
                    "dvsa": details,
                }
            )
    return out


def session_pool(
    session_dir: Path,
    *,
    gate: PlateGate,
    support: Mapping[str, int],
    road_polygon_frac: list[tuple[float, float]] | None,
    latitude: float,
    longitude: float,
) -> SessionPool | None:
    """Every candidate of one session, or ``None`` when none of its snaps
    can be opened (the 4K snaps are gone). Call :func:`skip_reason` first."""
    files = SessionFiles(session_dir)
    session = files.label
    cars = _car_tracks(read_json(files.data, []), latitude, longitude)
    pool = SessionPool(session)
    for car in cars.values():
        pool.cars[car["domain"]] += 1
        if car["has_snaps"]:
            pool.cars_with_snaps[car["domain"]] += 1

    reads = _usable_reads(read_json(files.alpr, []), cars)
    size = _snap_size(session_dir, (r["image"] for rs in reads.values() for r in rs))
    if size is None:
        return None if reads else pool
    fire_index, sub_size = load_bbox_index(session_dir)
    cache = read_json(files.vehicle_boxes, {})
    scene = _Scene(
        size=size,
        sub_size=sub_size,
        fire_index=fire_index,
        done_index=load_done_bbox_index(session_dir),
        vehicle_boxes=cache.get("boxes", {}) if isinstance(cache, dict) else {},
        road_polygon_frac=road_polygon_frac,
    )

    kept = kept_best_reads(read_json(files.alpr_by_track, {}))
    best_by_track: dict[int, Candidate] = {}
    for tid, track_reads in reads.items():
        best = kept.get(tid)
        context = _TrackContext(session, cars[tid], scene, gate, support, best)
        by_image = {r["image"]: r for r in track_reads}
        if best and best.get("canonical_uk_shape") and best.get("image") in by_image:
            best_by_track[tid] = _candidate(
                "best", by_image[best["image"]], int(best.get("n_agree") or 0), context
            )
            pool.best.append(best_by_track[tid])
        rejected = _rejected_read(track_reads, best)
        if rejected is not None:
            sample = "static_rejected" if rejected.get("static_suspect") else "colour_rejected"
            getattr(pool, sample).append(
                _candidate(sample, rejected, _n_agree(rejected, track_reads), context)
            )

    pool.dvsa = _dvsa_candidates(read_json(files.dvsa_labels, {}), best_by_track, support)
    return pool


def kept_best_reads(rollup: object) -> dict[int, dict[str, Any]]:
    """``track_id -> best_preferred`` from a ``_alpr_by_track.json``."""
    tracks = rollup.get("tracks", []) if isinstance(rollup, dict) else []
    return {
        int(t["track_id"]): t["best_preferred"]
        for t in tracks
        if isinstance(t, dict) and t.get("track_id") is not None and t.get("best_preferred")
    }


def _n_agree(read: Mapping[str, Any], track_reads: Sequence[Mapping[str, Any]]) -> int:
    """Other snaps of the track that read the same string, as the rollup
    counts it for the read it keeps."""
    return sum(
        1
        for r in track_reads
        if r.get("ocr_text") == read.get("ocr_text")
        and r.get("snap_index") != read.get("snap_index")
    )


# ----------------------------------------------------------------------
# Drawing the sample


def allocate(
    sizes: Mapping[str, int], n: int, *, power: float = 0.5, floor: int = 3
) -> dict[str, int]:
    """Share ``n`` items among strata of the given sizes.

    Every non-empty stratum first gets ``floor`` items (or all it holds, or
    an equal share of ``n`` when ``n`` is too small for the floors). The rest
    go one at a time to the stratum with the highest ``size ** power`` per
    item already held (Sainte-Lague), and never beyond what a stratum
    holds. ``power`` 1 is proportional, 0 equal; 0.5 lies between, so small
    strata get enough items to say something and large ones still get most.
    """
    live = {k: v for k, v in sizes.items() if v > 0}
    out = dict.fromkeys(sizes, 0)
    if not live or n <= 0:
        return out
    start = min(floor, n // len(live))
    for k, size in live.items():
        out[k] = min(size, start)
    weight = {k: float(v) ** power for k, v in live.items()}
    left = n - sum(out.values())
    while left > 0:
        open_strata = [k for k in sorted(live) if out[k] < live[k]]
        if not open_strata:
            break
        winner = max(open_strata, key=lambda k: weight[k] / (2 * out[k] + 1))
        out[winner] += 1
        left -= 1
    return out


def allocate_by_domain(
    sizes: Mapping[str, int],
    n: int,
    domain_of: Callable[[str], str],
    *,
    power: float = 0.5,
    floor: int = 3,
    domain_floor: int = DOMAIN_FLOOR,
) -> dict[str, int]:
    """Share ``n`` among domains, then within each domain among its strata,
    both by :func:`allocate`. Each domain is a group of strata a rate is
    quoted for (a direction in a light band).

    Domains get square-root shares with a floor of ``domain_floor``, not
    equal shares: low sun holds about 0.5 % of all best reads (2026-11-15,
    60 sessions), and equal shares spent a third of the sample on it. The
    floor still gives every domain enough items to bound its precision."""
    domains: dict[str, dict[str, int]] = defaultdict(dict)
    for stratum, size in sizes.items():
        domains[domain_of(stratum)][stratum] = size
    shares = allocate(
        {d: sum(s.values()) for d, s in domains.items()}, n, power=power, floor=domain_floor
    )
    out: dict[str, int] = {}
    for domain, strata in domains.items():
        out.update(allocate(strata, shares[domain], power=power, floor=floor))
    return out


def draw(
    candidates: Iterable[Candidate],
    allocation: Mapping[str, int],
    rng: random.Random,
    *,
    usable: Callable[[Candidate], bool] = lambda _c: True,
) -> list[Candidate]:
    """Take each stratum's allocation at random from its candidates,
    passing over any that ``usable`` refuses (a snap that is no longer on
    disk). The same ``rng`` state and candidates give the same draw."""
    by_stratum: dict[str, list[Candidate]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: str(c["key"])):
        by_stratum[c["stratum"]].append(c)
    taken: list[Candidate] = []
    for stratum in sorted(by_stratum):
        pool = by_stratum[stratum]
        rng.shuffle(pool)
        want = allocation.get(stratum, 0)
        got = 0
        for c in pool:
            if got >= want:
                break
            if usable(c):
                taken.append(c)
                got += 1
    return taken


@dataclass(frozen=True, slots=True)
class SampleSizes:
    """How many items each sample holds. The defaults are the plan's: about
    300 reads and 100 DVSA-labelled cars, some three hours of labelling."""

    best: int = 220
    colour_rejected: int = 50
    static_rejected: int = 30
    dvsa: int = 100


@dataclass(frozen=True, slots=True)
class AuditDraw:
    """The drawn set: the items in the order they are to be labelled, each
    stratum's population and sample size, and the per-domain car counts."""

    items: list[Candidate]
    strata: dict[str, dict[str, int]]
    domains: dict[str, dict[str, int]]


def _best_domain(stratum: str) -> str:
    _sample, direction, light, _group = stratum.split("|")
    return f"{direction}|{light}"


def _draw_dvsa(
    candidates: Sequence[Candidate],
    n: int,
    rng: random.Random,
    usable: Callable[[Candidate], bool],
    taken_keys: set[str],
) -> tuple[list[Candidate], dict[str, int]]:
    """One read per sampled car. The population is distinct plates, so the
    plates are drawn and then one of the plate's tracks, skipping a read
    another sample already holds."""
    by_plate: dict[str, list[Candidate]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: str(c["key"])):
        by_plate[c["dvsa"]["plate"]].append(c)
    plates_by_stratum: dict[str, list[str]] = defaultdict(list)
    for plate in sorted(by_plate):
        plates_by_stratum[by_plate[plate][0]["stratum"]].append(plate)
    sizes = {s: len(p) for s, p in plates_by_stratum.items()}
    allocation = allocate(sizes, n, power=0.0, floor=0)
    taken: list[Candidate] = []
    for stratum in sorted(plates_by_stratum):
        plates = plates_by_stratum[stratum]
        rng.shuffle(plates)
        got = 0
        for plate in plates:
            if got >= allocation[stratum]:
                break
            options = by_plate[plate]
            rng.shuffle(options)
            pick = next((c for c in options if c["key"] not in taken_keys and usable(c)), None)
            if pick is not None:
                taken.append(pick)
                got += 1
    return taken, sizes


def draw_audit_set(
    pools: Sequence[SessionPool],
    sizes: SampleSizes,
    rng: random.Random,
    *,
    usable: Callable[[Candidate], bool] = lambda _c: True,
) -> AuditDraw:
    """Draw the four samples from every session's candidates and shuffle
    them into one labelling order, so that whatever part of the set has been
    labelled is a random part of every stratum."""
    items: list[Candidate] = []
    population: dict[str, int] = {}

    best = [c for p in pools for c in p.best]
    best_sizes = Counter(c["stratum"] for c in best)
    population.update(best_sizes)
    items += draw(
        best, allocate_by_domain(best_sizes, sizes.best, _best_domain), rng, usable=usable
    )

    for sample, n in (
        ("colour_rejected", sizes.colour_rejected),
        ("static_rejected", sizes.static_rejected),
    ):
        candidates = [c for p in pools for c in getattr(p, sample)]
        sample_sizes = Counter(c["stratum"] for c in candidates)
        population.update(sample_sizes)
        items += draw(candidates, allocate(sample_sizes, n, power=0.0, floor=0), rng, usable=usable)

    dvsa, dvsa_sizes = _draw_dvsa(
        [c for p in pools for c in p.dvsa], sizes.dvsa, rng, usable, {c["key"] for c in items}
    )
    population.update(dvsa_sizes)
    items += dvsa

    rng.shuffle(items)
    sampled = Counter(c["stratum"] for c in items)
    strata = {
        s: {"population": population[s], "sampled": sampled.get(s, 0)} for s in sorted(population)
    }
    return AuditDraw(items, strata, _domain_counts(pools, best))


def _domain_counts(
    pools: Sequence[SessionPool], best: Sequence[Candidate]
) -> dict[str, dict[str, int]]:
    """Per ``"direction|light"``: car tracks, those with a 4K snap, those
    with a UK-shaped best read and those whose best read passes the gate."""
    domains: dict[str, dict[str, int]] = {}
    for direction in DIRECTIONS:
        for light in LIGHT_BANDS:
            key = f"{direction}|{light}"
            mine = [c for c in best if c["direction"] == direction and c["light"] == light]
            domains[key] = {
                "cars": sum(p.cars[key] for p in pools),
                "cars_with_snaps": sum(p.cars_with_snaps[key] for p in pools),
                "best": len(mine),
                "gated": sum(1 for c in mine if c["gate_pass"]),
            }
    return domains


# ----------------------------------------------------------------------
# Labels


def _clean_text(value: object, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text or null")
    text = normalize_label_text(value)
    if not text:
        return None
    if not 2 <= len(text) <= 8:
        raise ValueError(f"{what} must be 2 to 8 characters, got {len(text)}")
    return text


def clean_label(payload: Mapping[str, Any], *, dvsa: bool) -> dict[str, Any]:
    """Check one item's answers as the labelling page sends them, and return
    them in the form they are stored. Raises ``ValueError`` on anything else.

    ``text`` is the plate as the labeller reads it (``.`` for a character
    they can't make out, ``None`` when none of it is legible);
    ``text_first`` is what they typed before seeing the pipeline's read and
    ``reconciled`` how a disagreement with it was settled.
    """
    attribution = payload.get("attribution")
    if attribution not in ATTRIBUTIONS:
        raise ValueError(f"attribution must be one of {list(ATTRIBUTIONS)}")
    is_plate = attribution != "not_plate"
    reconciled = payload.get("reconciled")
    if reconciled is not None and reconciled not in RECONCILED:
        raise ValueError(f"reconciled must be null or one of {list(RECONCILED)}")
    note = payload.get("note") or ""
    if not isinstance(note, str):
        raise ValueError("note must be text")
    out: dict[str, Any] = {
        "attribution": attribution,
        "text": _clean_text(payload.get("text"), "text") if is_plate else None,
        "text_first": _clean_text(payload.get("text_first"), "text_first") if is_plate else None,
        "reconciled": reconciled if is_plate else None,
        "note": note.strip()[:_MAX_NOTE_CHARS],
    }
    if dvsa:
        answers = payload.get("dvsa")
        if not isinstance(answers, dict):
            raise ValueError("a DVSA item needs its dvsa answers")
        cleaned = {f: answers.get(f) for f in DVSA_FIELDS}
        bad = [f for f, a in cleaned.items() if a not in DVSA_ANSWERS]
        if bad:
            raise ValueError(f"dvsa answers for {bad} must be one of {list(DVSA_ANSWERS)}")
        out["dvsa"] = cleaned
    return out
