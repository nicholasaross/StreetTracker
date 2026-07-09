"""Baseline-restore ALPR analysis (Saturday soak).

session_20260529_164155 ran ~22.7h on the REVERTED config (mc=2,
uniform 400ms) -- the regression-passes check after rolling back the
Step 13a / mc=3 experiments. Snap-budget already confirmed it
reproduces Step 11 (see analyze_baseline.py); this script checks the
ALPR read rates land in the same place.

Two caveats baked into the comparison:

1. SATURDAY traffic. Step 11 / 13a / re-soak were weekday. Saturday
   has a later, lower, more-directional flow (Sat 08:00 was 46 R->L /
   7 L->R). Per-direction read rates are therefore NOT cleanly
   comparable to the weekday baselines -- the directional mix differs.
   The hour-matched cut (analyze_baseline_hourbucket.py) is the honest
   comparison for read quality.

2. CANONICAL FILTER now available. alpr-run writes
   ``canonical_uk_shape`` per record (PR #37). The 2026-05-29
   threshold-curve work showed ~55-65 % of high-conf "reads" are OCR
   garbage. This script reports both raw and canonical-filtered
   per-car numbers so the "real readable plates" rate is visible.

Run from repo root:
    uv run python .claude/aggregate_baseline.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from streettracker.analysis.dvsa import is_canonical_uk_plate

CONF_THRESHOLD = 0.9
GHOST_TRACK_MIN = 5


def _is_canon(r: dict) -> bool:
    """Canonical-UK-shape flag for an ALPR record. Prefers the
    persisted ``canonical_uk_shape`` annotation (PR #37+); falls back
    to re-deriving from ocr_text for older sessions whose alpr.json
    predates the annotation."""
    flag = r.get("canonical_uk_shape")
    if flag is not None:
        return bool(flag)
    txt = (r.get("ocr_text") or "").strip().upper().replace(" ", "")
    return is_canonical_uk_plate(txt)

STEP11 = "session_20260526_124704"        # weekday, mc=2 uniform
RESOAK = "session_20260528_103902"        # weekday, mc=3 dir-aware
BASELINE = "session_20260529_164155"      # Sat, mc=2 uniform (revert)


def _compute(session: str) -> dict:
    base = Path("output") / session
    alpr = json.loads((base / f"{session}_alpr.json").read_text())
    data = json.loads((base / f"{session}_data.json").read_text())

    cars = [r for r in data if r.get("class_name") == "car"]
    car_tids = {r["track_id"] for r in cars}

    # Per-track best read (any shape) + best canonical-shape read.
    best_conf: dict[int, float] = {}
    best_text: dict[int, str] = {}
    best_canon_conf: dict[int, float] = {}
    best_canon_text: dict[int, str] = {}
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt:
            continue
        tid = r["track_id"]
        if conf > best_conf.get(tid, -1.0):
            best_conf[tid] = conf
            best_text[tid] = txt
        # canonical_uk_shape annotation may be absent on pre-PR-#37
        # sessions; fall back to None -> treated as non-canonical.
        if _is_canon(r) and conf > best_canon_conf.get(tid, -1.0):
            best_canon_conf[tid] = conf
            best_canon_text[tid] = txt

    tracks_high = sum(1 for t in car_tids if best_conf.get(t, 0.0) >= CONF_THRESHOLD)

    # Ghost detection on canonical high-conf reads.
    plate_tracks: dict[str, set[int]] = defaultdict(set)
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt or conf < CONF_THRESHOLD:
            continue
        plate_tracks[txt].add(r["track_id"])
    plate_counts = Counter({p: len(s) for p, s in plate_tracks.items()})
    ghosts = {p for p, n in plate_counts.items() if n >= GHOST_TRACK_MIN}

    real_top = sum(
        1 for t in car_tids
        if best_text.get(t) and best_conf.get(t, 0) >= CONF_THRESHOLD
        and best_text[t] not in ghosts
    )
    # Canonical-shape aliasing-free: top read clears conf, is UK-shaped,
    # and isn't a ghost.
    canon_top = sum(
        1 for t in car_tids
        if best_canon_text.get(t) and best_canon_conf.get(t, 0) >= CONF_THRESHOLD
        and best_canon_text[t] not in ghosts
    )

    n_imgs = len({(r["track_id"], r["snap_index"]) for r in alpr
                  if r.get("pipeline") == "preferred"})
    n_high = sum(1 for r in alpr if r.get("pipeline") == "preferred"
                 and r.get("ocr_text") and (r.get("ocr_conf") or 0) >= CONF_THRESHOLD)
    n_high_canon = sum(
        1 for r in alpr if r.get("pipeline") == "preferred"
        and r.get("ocr_text") and (r.get("ocr_conf") or 0) >= CONF_THRESHOLD
        and _is_canon(r)
    )

    dir_by_tid = {r["track_id"]: r.get("direction") for r in data}
    d_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        d = dir_by_tid.get(r["track_id"], "?")
        d_counts[d][0] += 1
        if r.get("ocr_text") and (r.get("ocr_conf") or 0) >= CONF_THRESHOLD:
            d_counts[d][1] += 1
            if _is_canon(r):
                d_counts[d][2] += 1

    return {
        "n_cars": len(cars),
        "n_imgs": n_imgs,
        "n_high": n_high,
        "n_high_canon": n_high_canon,
        "tracks_high": tracks_high,
        "real_top": real_top,
        "canon_top": canon_top,
        "ghost_count": len(ghosts),
        "lr": d_counts["left to right"],
        "rl": d_counts["right to left"],
    }


def _f(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (n/a)"


def main() -> int:
    s11 = _compute(STEP11)
    rs = _compute(RESOAK)
    bl = _compute(BASELINE)

    print("=== Baseline-restore (Sat, mc=2 uniform) vs Step 11 (weekday, mc=2 uniform) vs Re-soak (weekday, mc=3 dir-aware) ===")
    print(f"Step 11 : {s11['n_cars']} cars, 5.5h weekday")
    print(f"Re-soak : {rs['n_cars']} cars, 25.8h weekday")
    print(f"Baseline: {bl['n_cars']} cars, 22.6h SATURDAY\n")

    rows = [
        ("Per-image high-conf (raw)", "n_high", "n_imgs"),
        ("Per-image high-conf (canonical)", "n_high_canon", "n_imgs"),
        ("Per-car high-conf (raw, aliased)", "tracks_high", "n_cars"),
        ("Per-car aliasing-free (raw top)", "real_top", "n_cars"),
        ("Per-car aliasing-free (CANONICAL top)", "canon_top", "n_cars"),
    ]
    w = max(len(r[0]) for r in rows)
    cw = 20
    print(f"{'metric':<{w}}  {'Step 11':<{cw}}  {'Re-soak':<{cw}}  {'Baseline(Sat)':<{cw}}")
    print(f"{'-' * w}  {'-' * cw}  {'-' * cw}  {'-' * cw}")
    for label, num, den in rows:
        print(f"{label:<{w}}  {_f(s11[num], s11[den]):<{cw}}  {_f(rs[num], rs[den]):<{cw}}  {_f(bl[num], bl[den]):<{cw}}")
    print()

    print("Per-direction per-image high-conf (raw / canonical):")
    for lab, key in [("L->R (rear)", "lr"), ("R->L (front)", "rl")]:
        print(f"  {lab}")
        print(f"    Step 11 : raw {_f(s11[key][1], s11[key][0])}  canon {_f(s11[key][2], s11[key][0])}")
        print(f"    Re-soak : raw {_f(rs[key][1], rs[key][0])}  canon {_f(rs[key][2], rs[key][0])}")
        print(f"    Baseline: raw {_f(bl[key][1], bl[key][0])}  canon {_f(bl[key][2], bl[key][0])}  <- SATURDAY")
    print()
    print("Note: Baseline is SATURDAY traffic (later/lower/more-directional flow).")
    print("Per-direction rates not cleanly comparable to weekday; see")
    print("analyze_baseline.py hourly cut for the directional-mix difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
