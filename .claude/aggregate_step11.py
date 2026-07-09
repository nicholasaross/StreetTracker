"""Compare Step 11 (post-trim) vs Step 10 (mask + bbox-pipe, pre-trim).

Cross-session comparison (different cars / traffic), so the most
informative metrics are per-car / per-image RATES, not absolute
counts. Two sessions:

- Step 10 baseline: ``session_20260525_200916``, captured 2026-05-25
  with t_usable_frac=[0.10, 0.67], analyzed with mask + pre-crop.
- Step 11 (post-trim): ``session_20260526_124704``, captured 2026-05-26
  with t_usable_frac=[0.10, 0.45] (deployed in commit 6a35c36),
  analyzed with the same mask + pre-crop.

Run from repo root:
    uv run python .claude/aggregate_step11.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

STEP10 = "session_20260525_200916"
STEP11 = "session_20260526_124704"
CONF_THRESHOLD = 0.9
GHOST_TRACK_MIN = 5


def _compute(session: str) -> dict:
    base = Path("output") / session
    alpr = json.loads((base / f"{session}_alpr.json").read_text())
    data = json.loads((base / f"{session}_data.json").read_text())
    meta = json.loads((base / f"{session}_meta.json").read_text())

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

    snap_stats = meta.get("snap_stats", {})

    return {
        "session": session,
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
        "fires_per_track": snap_stats.get("fires_per_track", {}),
        "pipeline_fires": snap_stats.get("pipeline_fires"),
        "pipeline_throttled": snap_stats.get("pipeline_throttled"),
        "pipeline_budget_exhausted": snap_stats.get("pipeline_budget_exhausted"),
        "latency_p50": snap_stats.get("latency", {}).get("p50_ms"),
    }


def _fmt(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (n/a)"


def main() -> int:
    s10 = _compute(STEP10)
    s11 = _compute(STEP11)

    print(f"=== Step 11 (post-trim) vs Step 10 ===")
    print(f"Step 10: {STEP10}  ({s10['n_cars']} cars, t_usable=[0.10, 0.67])")
    print(f"Step 11: {STEP11}  ({s11['n_cars']} cars, t_usable=[0.10, 0.45])\n")

    rows = [
        ("Per-image preferred high-conf",
         _fmt(s10["n_high"], s10["n_imgs"]),
         _fmt(s11["n_high"], s11["n_imgs"])),
        ("  L->R per-image",
         _fmt(s10["lr"][1], s10["lr"][0]),
         _fmt(s11["lr"][1], s11["lr"][0])),
        ("  R->L per-image",
         _fmt(s10["rl"][1], s10["rl"][0]),
         _fmt(s11["rl"][1], s11["rl"][0])),
        ("Per-car high-conf (aliased)",
         _fmt(s10["tracks_high"], s10["n_cars"]),
         _fmt(s11["tracks_high"], s11["n_cars"])),
        ("Per-car aliasing-free (any non-ghost)",
         _fmt(s10["real_any"], s10["n_cars"]),
         _fmt(s11["real_any"], s11["n_cars"])),
        ("Per-car aliasing-free (top non-ghost)",
         _fmt(s10["real_top"], s10["n_cars"]),
         _fmt(s11["real_top"], s11["n_cars"])),
        ("Ghost plates (>=5 tracks)",
         str(s10["ghost_count"]),
         str(s11["ghost_count"])),
    ]
    label_w = max(len(r[0]) for r in rows)
    col_w = 22
    print(f"{'metric':<{label_w}}  {'Step 10':<{col_w}}  {'Step 11':<{col_w}}")
    print(f"{'-' * label_w}  {'-' * col_w}  {'-' * col_w}")
    for label, a, b in rows:
        print(f"{label:<{label_w}}  {a:<{col_w}}  {b:<{col_w}}")
    print()

    # Budget redistribution evidence (mean fires per track).
    s10_f = s10["fires_per_track"] or {}
    s11_f = s11["fires_per_track"] or {}
    print("Budget redistribution (from snap_stats):")
    print(f"  mean fires/track:           "
          f"Step 10 = {s10_f.get('mean')}    Step 11 = {s11_f.get('mean')}")
    print(f"  pipeline_fires:             "
          f"Step 10 = {s10['pipeline_fires']}     Step 11 = {s11['pipeline_fires']}")
    print(f"  pipeline_budget_exhausted:  "
          f"Step 10 = {s10['pipeline_budget_exhausted']}      Step 11 = {s11['pipeline_budget_exhausted']}")
    print(f"  latency p50 ms:             "
          f"Step 10 = {s10['latency_p50']}      Step 11 = {s11['latency_p50']}")
    print()

    print(f"Step 10 ghosts:  {s10['ghosts']}")
    print(f"Step 11 ghosts:  {s11['ghosts']}")
    print()
    print("Top 10 plate strings by distinct-track count (Step 11):")
    for plate, n in s11["top10"]:
        flag = "  <-- GHOST" if n >= GHOST_TRACK_MIN else ""
        print(f"  {plate:<10}  {n} tracks{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
