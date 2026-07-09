"""Plate-crop padding experiment: pad_frac 0.10 vs 0.30.

The 2026-05-30 garbage analysis found 74 % of high-conf misreads were
exactly 6 chars -- one short of a 7-char UK plate -- implicating a
clipped edge character in the OCR crop. This compares the existing
0.10 run against a 0.30 re-run on the SAME 4K snaps (same session,
same detector, same OCR) to see whether widening the crop recovers
those reads.

Run from repo root after the 0.30 alpr-run completes:
    uv run python .claude/compare_pad.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from streettracker.analysis.dvsa import is_canonical_uk_plate

SESSION = "session_20260529_164155"
BASE = Path("output") / SESSION


def _load(suffix: str) -> list[dict]:
    return json.loads((BASE / f"{SESSION}_alpr{suffix}.json").read_text())


def _per_car(recs: list[dict], car_tids: set[int]) -> dict:
    """Per-car best canonical read + length histogram of garbage."""
    by_track: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for r in recs:
        if r.get("pipeline") != "preferred":
            continue
        t = (r.get("ocr_text") or "").strip().upper().replace(" ", "")
        c = r.get("ocr_conf") or 0.0
        if t and c >= 0.9:
            by_track[r["track_id"]].append((t, c))

    canon_cars: set[int] = set()
    garbage_lens: Counter = Counter()
    n_imgs = 0
    n_high = 0
    n_high_canon = 0
    for r in recs:
        if r.get("pipeline") != "preferred":
            continue
        t = (r.get("ocr_text") or "").strip().upper().replace(" ", "")
        c = r.get("ocr_conf") or 0.0
        n_imgs += 1
        if t and c >= 0.9:
            n_high += 1
            if is_canonical_uk_plate(t):
                n_high_canon += 1

    for tid in car_tids:
        hc = by_track.get(tid, [])
        if not hc:
            continue
        if any(is_canonical_uk_plate(t) for t, _ in hc):
            canon_cars.add(tid)
        else:
            best = max(hc, key=lambda x: x[1])[0]
            garbage_lens[len(best)] += 1

    return {
        "canon_cars": canon_cars,
        "garbage_lens": garbage_lens,
        "n_imgs": n_imgs,
        "n_high": n_high,
        "n_high_canon": n_high_canon,
    }


def main() -> int:
    data = json.loads((BASE / f"{SESSION}_data.json").read_text())
    car_tids = {r["track_id"] for r in data if r.get("class_name") == "car"}
    n = len(car_tids)

    a = _per_car(_load(".pad010"), car_tids)
    b = _per_car(_load(""), car_tids)  # current alpr.json == the 0.30 re-run

    print(f"=== Plate-crop padding: 0.10 vs 0.30  ({SESSION}, {n} car-tracks) ===\n")

    def pct(num: int, den: int) -> str:
        return f"{num}/{den} ({100 * num / den:.1f}%)" if den else "0/0"

    ca, cb = len(a["canon_cars"]), len(b["canon_cars"])
    rows = [
        ("Per-car canonical readable", pct(ca, n), pct(cb, n)),
        ("Per-image high-conf (raw)",
         pct(a["n_high"], a["n_imgs"]), pct(b["n_high"], b["n_imgs"])),
        ("Per-image high-conf (canonical)",
         pct(a["n_high_canon"], a["n_imgs"]), pct(b["n_high_canon"], b["n_imgs"])),
    ]
    print(f"{'metric':<34}  {'pad 0.10':<18}  {'pad 0.30':<18}")
    print(f"{'-' * 34}  {'-' * 18}  {'-' * 18}")
    for label, av, bv in rows:
        print(f"{label:<34}  {av:<18}  {bv:<18}")
    print()

    print("Garbage best-read length histogram (cars with high-conf but no canonical):")
    print(f"  {'len':>4}  {'pad 0.10':>9}  {'pad 0.30':>9}")
    for L in sorted(set(a["garbage_lens"]) | set(b["garbage_lens"])):
        print(f"  {L:>4}  {a['garbage_lens'].get(L, 0):>9}  {b['garbage_lens'].get(L, 0):>9}")
    print()

    # The money number: cars that were garbage at 0.10 and became
    # canonical at 0.30.
    recovered = b["canon_cars"] - a["canon_cars"]
    lost = a["canon_cars"] - b["canon_cars"]
    print(f"Cars RECOVERED (garbage@0.10 -> canonical@0.30): {len(recovered)}")
    print(f"Cars LOST      (canonical@0.10 -> garbage@0.30):  {len(lost)}")
    print(f"Net per-car canonical change: {cb - ca:+d}  ({100*(cb-ca)/n:+.1f} pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
