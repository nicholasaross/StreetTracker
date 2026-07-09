"""Compare Step 10 (ghost-mask + padding cap) results against Step 9.

Reads both ``<session>_alpr.json`` (the latest, Step 10) and
``<session>_alpr.step9.json`` (the saved Step 9 baseline) and prints
the diff side-by-side: per-image, per-direction, per-car raw,
per-car aliasing-free.

Run from repo root:
    uv run python .claude/aggregate_step10.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

SESSION = "session_20260525_200916"
SESSION_DIR = Path("output") / SESSION
CONF_THRESHOLD = 0.9
GHOST_TRACK_MIN = 5


def _compute(alpr: list[dict], data: list[dict]) -> dict:
    cars = [r for r in data if r.get("class_name") == "car"]
    n_cars = len(cars)
    car_tids = {r["track_id"] for r in cars}
    dir_by_tid = {r["track_id"]: r.get("direction") for r in data}

    by_track_best_conf: dict[int, float] = {}
    by_track_best_text: dict[int, str] = {}
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt:
            continue
        tid = r["track_id"]
        if conf > by_track_best_conf.get(tid, -1.0):
            by_track_best_conf[tid] = conf
            by_track_best_text[tid] = txt

    tracks_high = sum(
        1 for tid in car_tids
        if by_track_best_conf.get(tid, 0.0) >= CONF_THRESHOLD
    )

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

    car_nonghost: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt or conf < CONF_THRESHOLD or txt in ghosts:
            continue
        car_nonghost[r["track_id"]].append((txt, conf))
    real_any = sum(1 for tid in car_tids if car_nonghost.get(tid))
    real_top = sum(
        1 for tid in car_tids
        if (by_track_best_text.get(tid) and
            by_track_best_conf.get(tid, 0.0) >= CONF_THRESHOLD and
            by_track_best_text[tid] not in ghosts)
    )

    n_imgs = len({(r["track_id"], r["snap_index"])
                  for r in alpr if r.get("pipeline") == "preferred"})
    n_high = sum(
        1 for r in alpr
        if r.get("pipeline") == "preferred"
        and r.get("ocr_text")
        and (r.get("ocr_conf") or 0) >= CONF_THRESHOLD
    )

    d_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        d = dir_by_tid.get(r["track_id"], "?")
        d_counts[d][0] += 1
        if r.get("ocr_text") and (r.get("ocr_conf") or 0) >= CONF_THRESHOLD:
            d_counts[d][1] += 1

    return {
        "n_cars": n_cars,
        "n_imgs": n_imgs,
        "n_high": n_high,
        "tracks_high": tracks_high,
        "real_any": real_any,
        "real_top": real_top,
        "ghost_count": len(ghosts),
        "ghosts": sorted(ghosts),
        "lr": d_counts["left to right"],
        "rl": d_counts["right to left"],
        "top10": plate_counts.most_common(10),
    }


def _fmt(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (n/a)"


def main() -> int:
    data = json.loads((SESSION_DIR / f"{SESSION}_data.json").read_text())
    step9 = _compute(
        json.loads((SESSION_DIR / f"{SESSION}_alpr.step9.json").read_text()),
        data,
    )
    step10 = _compute(
        json.loads((SESSION_DIR / f"{SESSION}_alpr.json").read_text()),
        data,
    )

    print(f"=== Step 10 (mask + padding cap) vs Step 9 (bbox-pipe only) ===")
    print(f"Session: {SESSION}  ({step9['n_cars']} car-class tracks)\n")

    rows = [
        ("Per-image preferred high-conf",
         _fmt(step9["n_high"], step9["n_imgs"]),
         _fmt(step10["n_high"], step10["n_imgs"])),
        ("  L->R per-image",
         _fmt(step9["lr"][1], step9["lr"][0]),
         _fmt(step10["lr"][1], step10["lr"][0])),
        ("  R->L per-image",
         _fmt(step9["rl"][1], step9["rl"][0]),
         _fmt(step10["rl"][1], step10["rl"][0])),
        ("Per-car high-conf (aliased)",
         _fmt(step9["tracks_high"], step9["n_cars"]),
         _fmt(step10["tracks_high"], step10["n_cars"])),
        ("Per-car aliasing-free (any non-ghost)",
         _fmt(step9["real_any"], step9["n_cars"]),
         _fmt(step10["real_any"], step10["n_cars"])),
        ("Per-car aliasing-free (top non-ghost)",
         _fmt(step9["real_top"], step9["n_cars"]),
         _fmt(step10["real_top"], step10["n_cars"])),
        ("Ghost plates (>=5 tracks)",
         str(step9["ghost_count"]),
         str(step10["ghost_count"])),
    ]
    label_w = max(len(r[0]) for r in rows)
    col_w = 22
    print(f"{'metric':<{label_w}}  {'Step 9':<{col_w}}  {'Step 10':<{col_w}}")
    print(f"{'-' * label_w}  {'-' * col_w}  {'-' * col_w}")
    for label, a, b in rows:
        print(f"{label:<{label_w}}  {a:<{col_w}}  {b:<{col_w}}")
    print()
    print(f"Step 9 ghosts:  {step9['ghosts']}")
    print(f"Step 10 ghosts: {step10['ghosts']}")
    print()
    print("Top 10 plate strings by distinct-track count (Step 10):")
    for plate, n in step10["top10"]:
        flag = "  <-- GHOST" if n >= GHOST_TRACK_MIN else ""
        print(f"  {plate:<10}  {n} tracks{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
