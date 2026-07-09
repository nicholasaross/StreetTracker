"""Per-direction read success vs snap road-position (t_norm).

Tests the operator's 2026-05-31 observation: R->L (front-plate,
approaching) capture moved from "too early" (old band [0.10,0.45]) to
"too late" (new band [0.25,0.60]) -- the optimum is in the middle.

Joins each individual snap's road-position (t_norm from main_snap_bboxes,
1.0 = near camera) to its read outcome (canonical / garbage), split by
direction. Bins t_norm and reports canonical-read rate per bin per
direction -- the curve whose peak tells us where each direction reads
best, hence where to place the band / triggers.

Run from repo root (after alpr-run):
    uv run python .claude/analyze_band_position.py <session_dir>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from streettracker.analysis.dvsa import is_canonical_uk_plate

TP = json.loads(Path(".claude/triggers_proposal.json").read_text())
AX, AY = TP["main_axis_xy"]
CXF, CYF = TP["centroid_frac"]
SW, SH = TP["source_size"]
TMIN, TMAX = TP["t_min"], TP["t_max"]
CX, CY = CXF * SW, CYF * SH


def t_norm(bbox_substream: list[int]) -> float:
    x1, y1, x2, y2 = bbox_substream
    fx = ((x1 + x2) / 2) / 896.0
    fy = ((y1 + y2) / 2) / 512.0
    return (((fx * SW - CX) * AX + (fy * SH - CY) * AY) - TMIN) / (TMAX - TMIN)


def _collect(session_dir: Path) -> list[tuple[float, bool, str]]:
    session = session_dir.name
    data = json.loads((session_dir / f"{session}_data.json").read_text())
    alpr = json.loads((session_dir / f"{session}_alpr.json").read_text())
    outcome: dict[tuple[int, int], bool] = {}
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        txt = (r.get("ocr_text") or "").strip().upper().replace(" ", "")
        conf = r.get("ocr_conf") or 0.0
        outcome[(r["track_id"], r["snap_index"])] = bool(
            txt and conf >= 0.9 and is_canonical_uk_plate(txt)
        )
    snaps: list[tuple[float, bool, str]] = []
    for rec in data:
        if rec.get("class_name") != "car":
            continue
        d = rec.get("direction")
        sn = rec.get("main_snaps") or []
        bx = rec.get("main_snap_bboxes") or []
        for i, s in enumerate(sn):
            if i < len(bx) and bx[i]:
                canon = outcome.get((rec["track_id"], s))
                if canon is not None:
                    snaps.append((t_norm(bx[i]), canon, d))
    return snaps


def main(argv: list[str]) -> int:
    snaps: list[tuple[float, bool, str]] = []
    for arg in argv:
        snaps += _collect(Path(arg))

    print("=== Per-direction read success vs road position (POOLED) ===")
    print(f"Sessions: {', '.join(Path(a).name for a in argv)}")
    print(f"(t_norm 1.0 = near camera; {len(snaps)} snaps with position+outcome)\n")

    bins = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
            (0.50, 0.60), (0.60, 0.70)]
    for d in ("right to left", "left to right"):
        ds = [s for s in snaps if s[2] == d]
        if not ds:
            continue
        tag = "R->L front-plate" if d == "right to left" else "L->R rear-plate"
        print(f"{d}  ({tag}, n={len(ds)} snaps)")
        print(f"  {'t_norm bin':>12}  {'n':>4}  {'canonical%':>10}  bar")
        for lo, hi in bins:
            b = [s for s in ds if lo <= s[0] < hi]
            if not b:
                continue
            rate = 100 * sum(1 for s in b if s[1]) / len(b)
            bar = "#" * int(rate / 3)
            print(f"  {f'{lo:.2f}-{hi:.2f}':>12}  {len(b):>4}  {rate:>9.0f}%  {bar}")
        print()

    print("Peak bin per direction = optimal snap position for that direction.")
    print("Current band: t_usable_frac=[0.25,0.60].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
