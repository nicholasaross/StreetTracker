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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONF_THRESHOLD = 0.9


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
    gap_minutes_max: float        # max gap between consecutive visits, 0.0 if n_visits==1
    gap_minutes_min: float        # min gap, 0.0 if n_visits==1
    directions: dict[str, int] = field(default_factory=dict)
    colors: dict[str, int] = field(default_factory=dict)
    visits: list[VehicleVisit] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["visits"] = [v.to_json_dict() if isinstance(v, VehicleVisit) else v
                       for v in d["visits"]]
        return d


def build_vehicles(
    session_dir: Path,
    *,
    conf_threshold: float = CONF_THRESHOLD,
    include_unread: bool = True,
) -> list[Vehicle]:
    """Build per-vehicle aggregations from a closed session's outputs.

    ``conf_threshold`` controls which ALPR reads are treated as
    "anchor" plate identities (default 0.9). Reads below that are
    discarded and the track is treated as unread.

    ``include_unread`` controls whether tracks without an anchor read
    are emitted as plate=None vehicles. Set False to focus on the
    plate-anchored subset only.

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

    # Group track records by canonical plate. Tracks whose best read
    # is below threshold (or absent) collect under ``None``.
    tracks_by_plate: dict[str | None, list[tuple[dict, dict | None]]] = {}
    for rec in data:
        if rec.get("class_name") != "car":
            continue
        tid = rec.get("track_id")
        best = best_by_tid.get(tid)
        plate = best["ocr_text"] if best else None
        tracks_by_plate.setdefault(plate, []).append((rec, best))

    out: list[Vehicle] = []
    for plate, items in tracks_by_plate.items():
        if plate is None and not include_unread:
            continue
        if plate is None:
            # Anonymous tracks emit one Vehicle per track (no
            # cross-track grouping is possible without a key).
            for rec, _best in items:
                out.append(_make_single_vehicle(rec, best=None))
            continue
        # Grouped: one Vehicle per distinct plate, with all matching
        # tracks as visits.
        out.append(_make_grouped_vehicle(plate, items))

    out.sort(key=lambda v: v.first_seen)
    return out


def _make_single_vehicle(rec: dict, best: dict | None) -> Vehicle:
    visit = _make_visit(rec, best)
    plate = best["ocr_text"] if best else None
    plate_conf = (best.get("ocr_conf") if best else None)
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


def _make_grouped_vehicle(
    plate: str, items: list[tuple[dict, dict | None]]
) -> Vehicle:
    # Sort visits chronologically by track start.
    items_sorted = sorted(items, key=lambda p: p[0]["time_start_unix"])
    visits = [_make_visit(rec, best) for rec, best in items_sorted]

    # Plate confidence: max across visits' best reads.
    plate_conf = max(
        (b.get("ocr_conf") or 0.0) for _r, b in items_sorted if b
    )
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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2

    vehicles = build_vehicles(
        session_dir,
        conf_threshold=args.conf,
        include_unread=not args.no_unread,
    )

    session = session_dir.name
    out_path = session_dir / f"{session}_vehicles.json"
    out_path.write_text(
        json.dumps([v.to_json_dict() for v in vehicles], indent=2)
    )
    print(f"[vehicles] wrote {out_path}  ({len(vehicles)} vehicles)")

    # Headline numbers.
    plated = [v for v in vehicles if v.plate is not None]
    recurring = [v for v in plated if v.n_visits >= 2]
    print(f"  plated:    {len(plated)}")
    print(f"  recurring: {len(recurring)}  (>=2 visits in session)")
    if recurring:
        print(f"  top recurring:")
        for v in sorted(recurring, key=lambda x: -x.n_visits)[:5]:
            print(
                f"    {v.plate}  n_visits={v.n_visits}  "
                f"gap_min_max={v.gap_minutes_max:.1f}min  "
                f"directions={dict(v.directions)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
