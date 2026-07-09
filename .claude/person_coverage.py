"""Person capture-coverage soak measurement.

The snap gate (RoadGate polygon + pipeline bands) was designed and tuned
entirely for cars. Person tracks flow through the same class-agnostic
planner, so this script answers, before any person-side capture tuning:

  1. What fraction of person tracks get >=1 usable 4K main snap,
     vs the car coverage on the same session?
  2. WHERE do person snaps land on the road axis (t_norm, done-bbox
     preferred) -- i.e. are pavement walkers only captured when they
     drift into car geometry?
  3. What do person speed (m/s via the showcase calibration) and dwell
     distributions look like -- empirical basis for the jogger/walker
     threshold in `streettracker people`.

    uv run python .claude/person_coverage.py output/session_20260619_111111 ...

Pass any number of session dirs; per-session tables plus a pooled
summary. Sessions without completion-time bboxes fall back to fire-time
bboxes for the landing histogram (flagged in the output).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TP = json.loads(Path(".claude/triggers_proposal.json").read_text())
AX, AY = TP["main_axis_xy"]
CXF, CYF = TP["centroid_frac"]
SW, SH = TP["source_size"]
TMIN, TMAX = TP["t_min"], TP["t_max"]
CX, CY = CXF * SW, CYF * SH

# Single global speed calibration (ignores perspective) -- same maths as
# web/stats.py: m_per_px = road_length_m / DEFAULT_ROAD_AXIS_PX.
M_PER_PX = 33.71 / 801.0

T_BINS = [(-0.25, 0.0), (0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
          (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.85), (0.85, 1.25)]
SPEED_BINS_MS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5),
                 (2.5, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 99.0)]


def t_norm(bbox_substream: list[int]) -> float:
    x1, y1, x2, y2 = bbox_substream
    fx = ((x1 + x2) / 2) / 896.0
    fy = ((y1 + y2) / 2) / 512.0
    return (((fx * SW - CX) * AX + (fy * SH - CY) * AY) - TMIN) / (TMAX - TMIN)


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%  ({n:>4}/{d:<5})" if d else "   n/a"


def hist(rows: list[float], bins: list[tuple[float, float]], unit: str) -> None:
    if not rows:
        print(f"  (no {unit} data)")
        return
    for lo, hi in bins:
        b = [v for v in rows if lo <= v < hi]
        if not b:
            continue
        share = 100 * len(b) / len(rows)
        print(f"  {f'{lo:.2f}-{hi:.2f}':>12}  {len(b):>5}  {share:>5.1f}%  {'#' * int(share / 2)}")


def q(rows: list[float], f: float) -> float:
    s = sorted(rows)
    return s[min(len(s) - 1, int(f * len(s)))] if s else 0.0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    pooled = {
        "persons": 0, "persons_snapped": 0, "person_snaps": 0,
        "cars": 0, "cars_snapped": 0,
        "p_l2r": 0, "p_r2l": 0, "p_nodir": 0,
        "land_done": [], "land_fire": [],
        "speeds_ms": [], "dwell": [],
    }

    for arg in argv:
        sd = Path(arg)
        session = sd.name
        data = json.loads((sd / f"{session}_data.json").read_text(encoding="utf-8"))

        persons = [r for r in data if r.get("class_name") == "person"]
        cars = [r for r in data if r.get("class_name") == "car"]
        p_snapped = [r for r in persons if r.get("main_snaps")]
        c_snapped = [r for r in cars if r.get("main_snaps")]
        n_p_snaps = sum(len(r.get("main_snaps") or []) for r in persons)

        land_done: list[float] = []
        land_fire: list[float] = []
        for r in persons:
            sn = r.get("main_snaps") or []
            done = r.get("main_snap_bboxes_done") or []
            fire = r.get("main_snap_bboxes") or []
            for i in range(len(sn)):
                if i < len(done) and done[i]:
                    land_done.append(t_norm(done[i]))
                elif i < len(fire) and fire[i]:
                    land_fire.append(t_norm(fire[i]))

        speeds = [float(r.get("speed_px_s") or 0.0) * M_PER_PX
                  for r in persons if (r.get("num_detections") or 0) >= 6]
        dwell = [float(r.get("duration_visible") or 0.0)
                 for r in persons if (r.get("duration_visible") or 0.0) > 0]
        d_l2r = sum(1 for r in persons if r.get("direction") == "left to right")
        d_r2l = sum(1 for r in persons if r.get("direction") == "right to left")

        print(f"=== {session} ===")
        print(f"  person tracks : {len(persons):>5}   (L->R {d_l2r}, R->L {d_r2l}, "
              f"other {len(persons) - d_l2r - d_r2l})")
        print(f"  >=1 4K snap   : {pct(len(p_snapped), len(persons))}   "
              f"[cars: {pct(len(c_snapped), len(cars))}]")
        print(f"  person snaps  : {n_p_snaps:>5}   "
              f"({n_p_snaps / len(p_snapped):.1f}/snapped person)" if p_snapped
              else f"  person snaps  : {n_p_snaps:>5}")
        cov = len(land_done) + len(land_fire)
        if cov:
            print(f"  landing bbox  : {len(land_done)} done-bbox, "
                  f"{len(land_fire)} fire-bbox-only of {n_p_snaps} snaps")
        print()

        pooled["persons"] += len(persons)
        pooled["persons_snapped"] += len(p_snapped)
        pooled["person_snaps"] += n_p_snaps
        pooled["cars"] += len(cars)
        pooled["cars_snapped"] += len(c_snapped)
        pooled["p_l2r"] += d_l2r
        pooled["p_r2l"] += d_r2l
        pooled["p_nodir"] += len(persons) - d_l2r - d_r2l
        pooled["land_done"].extend(land_done)
        pooled["land_fire"].extend(land_fire)
        pooled["speeds_ms"].extend(speeds)
        pooled["dwell"].extend(dwell)

    print("=" * 60)
    print(f"POOLED over {len(argv)} session(s)\n")
    print(f"  person tracks : {pooled['persons']:>6}   "
          f"(L->R {pooled['p_l2r']}, R->L {pooled['p_r2l']}, other {pooled['p_nodir']})")
    print(f"  person >=1 4K : {pct(pooled['persons_snapped'], pooled['persons'])}")
    print(f"  car    >=1 4K : {pct(pooled['cars_snapped'], pooled['cars'])}   <- reference")
    if pooled["persons_snapped"]:
        print(f"  snaps/snapped : {pooled['person_snaps'] / pooled['persons_snapped']:.1f}")

    print("\nPERSON SNAP LANDING position (t_norm; 1.0 = near camera)")
    if pooled["land_done"]:
        print(f"  completion-time bboxes (n={len(pooled['land_done'])}):")
        hist(pooled["land_done"], T_BINS, "t_norm")
    if pooled["land_fire"]:
        print(f"  fire-time-only bboxes (n={len(pooled['land_fire'])}):")
        hist(pooled["land_fire"], T_BINS, "t_norm")
    if not pooled["land_done"] and not pooled["land_fire"]:
        print("  (no bbox data)")

    sp = pooled["speeds_ms"]
    print(f"\nPERSON SPEED, m/s calibrated (m_per_px={M_PER_PX:.4f}, "
          f"num_detections>=6, n={len(sp)})")
    print(f"  p50={q(sp, .5):.2f}  p75={q(sp, .75):.2f}  p90={q(sp, .9):.2f}  "
          f"p97={q(sp, .97):.2f} m/s")
    hist(sp, SPEED_BINS_MS, "speed")
    print("  (walking ~1.4 m/s, jogging ~2.5-3.5 m/s; single global calibration,")
    print("   ignores perspective -- use for threshold shape, not absolutes)")

    dw = pooled["dwell"]
    print(f"\nDWELL seconds (n={len(dw)}): p50={q(dw, .5):.1f}  p90={q(dw, .9):.1f}  "
          f"max={max(dw) if dw else 0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
