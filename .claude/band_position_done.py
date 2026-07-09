"""Read-rate vs LANDING position (completion-time bbox), per direction.

analyze_band_position.py bins by FIRE-time bbox. This bins by the
completion-time bbox (main_snap_bboxes_done) = where the car actually
was when the 4K image was captured. Answers: does R->L read poorly
because it LANDS far (narrow forward band starves near landings), or
because it reads poorly even where it lands (geometry)?

    uv run python .claude/band_position_done.py output/session_20260613_090331
"""

from __future__ import annotations

import json
import sys
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


def main(argv: list[str]) -> int:
    sd = Path(argv[0])
    session = sd.name
    data = json.loads((sd / f"{session}_data.json").read_text())
    alpr = json.loads((sd / f"{session}_alpr.json").read_text())
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
        bx = rec.get("main_snap_bboxes_done") or []
        for i, s in enumerate(sn):
            if i < len(bx) and bx[i]:
                canon = outcome.get((rec["track_id"], s))
                if canon is not None:
                    snaps.append((t_norm(bx[i]), canon, d))

    print(f"=== Read vs LANDING position (done-bbox), {session} ===")
    print(f"(t_norm 1.0 = near camera; {len(snaps)} snaps)\n")
    bins = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
            (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.85)]
    for d in ("right to left", "left to right"):
        ds = [s for s in snaps if s[2] == d]
        if not ds:
            continue
        tag = "R->L front" if d == "right to left" else "L->R rear"
        print(f"{d}  ({tag}, n={len(ds)})")
        print(f"  {'land t_norm':>12}  {'n':>4}  {'canon%':>7}  bar")
        for lo, hi in bins:
            b = [s for s in ds if lo <= s[0] < hi]
            if not b:
                continue
            rate = 100 * sum(1 for s in b if s[1]) / len(b)
            print(f"  {f'{lo:.2f}-{hi:.2f}':>12}  {len(b):>4}  {rate:>6.0f}%  {'#' * int(rate / 3)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
