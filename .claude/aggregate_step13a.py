"""Compare Step 13a (direction-aware pipeline interval) vs Step 11.

Step 11 baseline: ``session_20260526_124704`` (2026-05-26, t_usable=[0.10, 0.45],
uniform pipeline_interval_ms=400).
Step 13a soak:   ``session_20260527_160139`` (2026-05-27 -> 28, same t_usable,
pipeline_interval_ms_by_direction={forward:300, reverse:400}, max_concurrent=2).

Both analyses use the SAME methodology: --pre-crop + --ghost-mask, preferred
pipeline only. Sessions are different traffic, so per-image / per-car RATES
are the informative metric (counts differ because Step 13a soak ran 18.4h
vs Step 11's 5.5h).

Run from repo root:
    uv run python .claude/aggregate_step13a.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

STEP11 = "session_20260526_124704"
STEP13A = "session_20260527_160139"
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
        "latency_p99": snap_stats.get("latency", {}).get("p99_ms"),
        "attempts": snap_stats.get("attempts"),
        "successes": snap_stats.get("successes"),
    }


def _fmt(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (n/a)"


def main() -> int:
    s11 = _compute(STEP11)
    s13 = _compute(STEP13A)

    print("=== Step 13a (direction-aware pipeline) vs Step 11 (uniform 400ms) ===")
    print(f"Step 11:  {STEP11}   ({s11['n_cars']} cars, 5.5h, uniform 400ms)")
    print(f"Step 13a: {STEP13A}  ({s13['n_cars']} cars, 18.4h, fwd=300/rev=400ms)\n")

    rows = [
        ("Per-image preferred high-conf",
         _fmt(s11["n_high"], s11["n_imgs"]),
         _fmt(s13["n_high"], s13["n_imgs"])),
        ("  L->R per-image (rear plate)",
         _fmt(s11["lr"][1], s11["lr"][0]),
         _fmt(s13["lr"][1], s13["lr"][0])),
        ("  R->L per-image (front plate)",
         _fmt(s11["rl"][1], s11["rl"][0]),
         _fmt(s13["rl"][1], s13["rl"][0])),
        ("Per-car high-conf (aliased)",
         _fmt(s11["tracks_high"], s11["n_cars"]),
         _fmt(s13["tracks_high"], s13["n_cars"])),
        ("Per-car aliasing-free (any non-ghost)",
         _fmt(s11["real_any"], s11["n_cars"]),
         _fmt(s13["real_any"], s13["n_cars"])),
        ("Per-car aliasing-free (top non-ghost)",
         _fmt(s11["real_top"], s11["n_cars"]),
         _fmt(s13["real_top"], s13["n_cars"])),
        ("Ghost plates (>=5 tracks)",
         str(s11["ghost_count"]),
         str(s13["ghost_count"])),
    ]
    label_w = max(len(r[0]) for r in rows)
    col_w = 22
    print(f"{'metric':<{label_w}}  {'Step 11':<{col_w}}  {'Step 13a':<{col_w}}")
    print(f"{'-' * label_w}  {'-' * col_w}  {'-' * col_w}")
    for label, a, b in rows:
        print(f"{label:<{label_w}}  {a:<{col_w}}  {b:<{col_w}}")
    print()

    s11_f = s11["fires_per_track"] or {}
    s13_f = s13["fires_per_track"] or {}
    print("Snap-stats (from meta.json):")
    print(f"  HTTP attempts/successes:      Step 11 = {s11['attempts']}/{s11['successes']}     Step 13a = {s13['attempts']}/{s13['successes']}")
    print(f"  mean fires/track (all tracks):  Step 11 = {s11_f.get('mean')}    Step 13a = {s13_f.get('mean')}")
    print(f"  pipeline_fires:                 Step 11 = {s11['pipeline_fires']}     Step 13a = {s13['pipeline_fires']}")
    print(f"  pipeline_throttled:             Step 11 = {s11['pipeline_throttled']}     Step 13a = {s13['pipeline_throttled']}")
    print(f"  pipeline_budget_exhausted:      Step 11 = {s11['pipeline_budget_exhausted']}      Step 13a = {s13['pipeline_budget_exhausted']}")
    print(f"  latency p50 / p99 (ms):         Step 11 = {s11['latency_p50']}/{s11['latency_p99']}     Step 13a = {s13['latency_p50']}/{s13['latency_p99']}")
    print()

    print(f"Step 11 ghosts:  {s11['ghosts']}")
    print(f"Step 13a ghosts: {s13['ghosts']}")
    print()
    print("Top 10 plate strings by distinct-track count (Step 13a):")
    for plate, n in s13["top10"]:
        flag = "  <-- GHOST" if n >= GHOST_TRACK_MIN else ""
        print(f"  {plate:<10}  {n} tracks{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
