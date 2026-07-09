"""Anti-Smearing exposure assessment (2026-06-01).

Single-variable test of the Reolink shutter/exposure change
(Auto, shutter max 125  ->  Anti-Smearing, shutter max 32) applied
2026-05-31. The validation soak session_20260531_182353 ran the wide
band [0.10,0.60] so it samples the full position range, incl. the
0.50-0.60 near-zone that COLLAPSED under Auto (band-2 falsification).

Reuses the t_norm + canonical-read methodology verbatim from
.claude/analyze_band_position.py so numbers are comparable to prior
steps (canonical = preferred pipeline, conf>=0.9, is_canonical_uk_plate).

Two comparisons (each holds position constant -> isolates exposure):

  STEP 2  EXPOSURE A/B   : Anti-Smear vs Auto-baseline in the OVERLAP
                           window t_norm [0.10,0.45] (the proven band).
                           A lift here = faster shutter helped reads.
  STEP 3  NEAR-ZONE      : Anti-Smear vs Auto-band2 in t_norm [0.50,0.60]
                           (the motion-blur collapse zone). A lift here
                           revives the bigger-plate / near-band idea.

Run from repo root (after alpr-run on the Anti-Smear session):
    uv run python .claude/analyze_antismear.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from streettracker.analysis.dvsa import is_canonical_uk_plate

# --- sessions (override on argv: antismear auto_baseline auto_band2) ---
ANTISMEAR = "session_20260531_182353"      # Anti-Smearing, band [0.10,0.60]
AUTO_BASELINE = "session_20260529_164155"  # Auto, band [0.10,0.45]
AUTO_BAND2 = "session_20260530_165958"     # Auto, band [0.25,0.60]

CONF = 0.9

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


def collect(session_dir: Path) -> list[tuple[float, bool, str]]:
    """[(t_norm, canonical, direction), ...] over per-snap preferred reads of cars."""
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
            txt and conf >= CONF and is_canonical_uk_plate(txt)
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


def per_car_canonical(session_dir: Path) -> tuple[int, int]:
    """(cars with >=1 canonical preferred read, total cars) -- headline metric."""
    session = session_dir.name
    data = json.loads((session_dir / f"{session}_data.json").read_text())
    alpr = json.loads((session_dir / f"{session}_alpr.json").read_text())
    cars = {r["track_id"] for r in data if r.get("class_name") == "car"}
    canon_tids: set[int] = set()
    for r in alpr:
        if r.get("pipeline") != "preferred" or r["track_id"] not in cars:
            continue
        txt = (r.get("ocr_text") or "").strip().upper().replace(" ", "")
        conf = r.get("ocr_conf") or 0.0
        if txt and conf >= CONF and is_canonical_uk_plate(txt):
            canon_tids.add(r["track_id"])
    return len(canon_tids), len(cars)


def _rate(snaps: list[tuple[float, bool, str]], lo: float, hi: float, d: str | None) -> tuple[int, int]:
    sel = [s for s in snaps if lo <= s[0] < hi and (d is None or s[2] == d)]
    return sum(1 for s in sel if s[1]), len(sel)


def _fmt(n: int, dn: int) -> str:
    return f"{n:>4}/{dn:<4} {100 * n / dn:>5.1f}%" if dn else f"{'--':>4}/{'--':<4}  {'n/a':>5}"


def _curve(label: str, snaps: list[tuple[float, bool, str]]) -> None:
    bins = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
            (0.40, 0.50), (0.50, 0.60), (0.60, 0.70)]
    print(f"--- {label}  ({len(snaps)} snaps with position+outcome) ---")
    print(f"  {'t_norm':>11}  {'R->L (front)':>16}  {'L->R (rear)':>16}")
    for lo, hi in bins:
        rl_n, rl_d = _rate(snaps, lo, hi, "right to left")
        lr_n, lr_d = _rate(snaps, lo, hi, "left to right")
        if rl_d == 0 and lr_d == 0:
            continue
        print(f"  {f'{lo:.2f}-{hi:.2f}':>11}  {_fmt(rl_n, rl_d):>16}  {_fmt(lr_n, lr_d):>16}")
    print()


def _ab(title: str, win: tuple[float, float], auto_label: str,
        auto: list[tuple[float, bool, str]], asm: list[tuple[float, bool, str]]) -> None:
    lo, hi = win
    print(f"=== {title}  (t_norm window {lo:.2f}-{hi:.2f}) ===")
    print(f"  {'direction':>14}  {auto_label:>17}  {'Anti-Smearing':>17}  {'delta':>8}")
    for d, tag in (("right to left", "R->L (front)"),
                   ("left to right", "L->R (rear)"),
                   (None, "OVERALL")):
        an, ad = _rate(auto, lo, hi, d)
        sn, sd = _rate(asm, lo, hi, d)
        ar = 100 * an / ad if ad else None
        sr = 100 * sn / sd if sd else None
        delta = f"{sr - ar:+5.1f}pp" if (ar is not None and sr is not None) else "   n/a"
        print(f"  {tag:>14}  {_fmt(an, ad):>17}  {_fmt(sn, sd):>17}  {delta:>8}")
    print()


def main(argv: list[str]) -> int:
    asm_s, auto_b, auto_2 = (argv + [ANTISMEAR, AUTO_BASELINE, AUTO_BAND2][len(argv):])[:3]
    out = Path("output")
    asm = collect(out / asm_s)
    auto = collect(out / auto_b)
    band2 = collect(out / auto_2)

    print("=" * 68)
    print("ANTI-SMEARING EXPOSURE ASSESSMENT")
    print(f"  Anti-Smear : {asm_s}  (shutter max 32, band [0.10,0.60])")
    print(f"  Auto base  : {auto_b}  (shutter max 125, band [0.10,0.45])")
    print(f"  Auto band-2: {auto_2}  (shutter max 125, band [0.25,0.60])")
    print(f"  canonical = preferred pipeline, conf>={CONF}, is_canonical_uk_plate")
    print(f"  t_norm 1.0 = near camera")
    print("=" * 68 + "\n")

    _curve(f"Anti-Smear  {asm_s}", asm)
    _curve(f"Auto base   {auto_b}", auto)
    _curve(f"Auto band-2 {auto_2}", band2)

    _ab("STEP 2  EXPOSURE A/B  (Anti-Smear vs Auto-baseline)",
        (0.10, 0.45), "Auto base", auto, asm)
    _ab("STEP 3  NEAR-ZONE RECOVERY  (Anti-Smear vs Auto-band2)",
        (0.50, 0.60), "Auto band-2", band2, asm)

    print("=== Per-car canonical (headline metric, whole session) ===")
    for label, sdir in (("Anti-Smear", asm_s), ("Auto base", auto_b), ("Auto band-2", auto_2)):
        n, d = per_car_canonical(out / sdir)
        print(f"  {label:>12}: {_fmt(n, d)} of cars read a canonical UK plate")
    print()
    print("Interpretation:")
    print("  STEP 2 +pp  => faster shutter lifts reads at proven positions.")
    print("  STEP 3 +pp  => near-zone motion blur was the collapse cause;")
    print("                 bigger-plate/near-band idea REVIVED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
