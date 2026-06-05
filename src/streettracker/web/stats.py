"""Traffic summary statistics for the showcase site.

Computes volume + temporal patterns over **all vehicle tracks** (every
``class_name == "car"`` record), across every session that has a ``data.json``
-- a wider net than the gallery, which only shows the plated subset. (Pruned
JSON-only sessions still carry ``data.json``, so the stats reach further back
than the images do.)

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
# A track needs this many detections to be eligible for the "fastest" board --
# guards against 2-3 frame BotSORT ID-switch glitches that produce a huge
# net_disp / duration spike.
_FASTEST_MIN_DETECTIONS = 6
_L2R = "left to right"
_R2L = "right to left"


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
        speed={"hist": [], "avg_l2r": 0, "avg_r2l": 0, "unit": unit, "fastest": []},
        makes=[],
        colours=[],
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
    makes_by_plate: dict[str, str] = {}
    fastest_raw: list[tuple[float, str, dict[str, Any]]] = []
    total = 0
    dates: set[str] = set()

    for d in sessions:
        name = d.name
        try:
            data = json.loads((d / f"{name}_data.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = []
        for r in data:
            if r.get("class_name") != "car":
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

            total += 1
            dates.add(date)
            daily[date]["total"] += 1
            dow[_DOW[wd]]["total"] += 1
            heatmap[wd][dt.hour] += 1
            hour_totals[dt.hour] += 1
            colours[r.get("color") or "unknown"] += 1
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
        "avg_l2r": _disp(sum(speeds_l2r) / len(speeds_l2r), factor) if speeds_l2r else 0,
        "avg_r2l": _disp(sum(speeds_r2l) / len(speeds_r2l), factor) if speeds_r2l else 0,
        "unit": unit,
        "fastest": _build_fastest(output_root, fastest_raw, factor, unit),
    }

    makes = [[m, n] for m, n in Counter(makes_by_plate.values()).most_common(N_TOP_MAKES)]
    colours_out = [[c, n] for c, n in colours.most_common(N_TOP_COLOURS)]

    return Stats(
        overall=overall,
        daily=daily_list,
        dow=dow,
        by_day_15min=dict(by_day_15),
        heatmap=heatmap,
        speed=speed,
        makes=makes,
        colours=colours_out,
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
