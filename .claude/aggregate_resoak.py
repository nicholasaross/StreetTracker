"""Re-soak ALPR analysis: 3-way vs Step 11 vs Step 13a.

The validation soak for max_concurrent=3 + dir-aware cadence (300/400ms).
Step 13a's apparent regression turned out to be hour-of-day confound;
the matched-hour cut (.claude/analyze_step13a_hourbucket.py) showed
Step 13a as decisively better in hours 17, 18. Re-soak unblocks
max_concurrent and validates whether the dir-aware cadence finally
binds the way it was supposed to.

Run from repo root:
    uv run python .claude/aggregate_resoak.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

STEP11 = "session_20260526_124704"
STEP13A = "session_20260527_160139"
RESOAK = "session_20260528_103902"
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
        1 for tid in car_tids if by_track_best_conf.get(tid, 0.0) >= CONF_THRESHOLD
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
        1
        for tid in car_tids
        if (
            by_track_best_text.get(tid)
            and by_track_best_conf.get(tid, 0.0) >= CONF_THRESHOLD
            and by_track_best_text[tid] not in ghosts
        )
    )

    n_imgs = len(
        {(r["track_id"], r["snap_index"]) for r in alpr if r.get("pipeline") == "preferred"}
    )
    n_high = sum(
        1
        for r in alpr
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
        "dropped": snap_stats.get("dropped"),
    }


def _fmt(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (n/a)"


def main() -> int:
    s11 = _compute(STEP11)
    s13 = _compute(STEP13A)
    rs = _compute(RESOAK)

    print(
        "=== Re-soak (max_concurrent=3 + dir-aware) vs Step 13a (mc=2 + dir-aware) vs Step 11 (mc=2 + uniform) ==="
    )
    print(f"Step 11:  {STEP11}   ({s11['n_cars']} cars, 5.5h, mc=2 + uniform 400ms)")
    print(f"Step 13a: {STEP13A}   ({s13['n_cars']} cars, 18.4h, mc=2 + 300/400ms)")
    print(f"Re-soak:  {RESOAK}   ({rs['n_cars']} cars, 25.8h, mc=3 + 300/400ms)\n")

    rows = [
        (
            "Per-image preferred high-conf",
            _fmt(s11["n_high"], s11["n_imgs"]),
            _fmt(s13["n_high"], s13["n_imgs"]),
            _fmt(rs["n_high"], rs["n_imgs"]),
        ),
        (
            "  L->R per-image (rear)",
            _fmt(s11["lr"][1], s11["lr"][0]),
            _fmt(s13["lr"][1], s13["lr"][0]),
            _fmt(rs["lr"][1], rs["lr"][0]),
        ),
        (
            "  R->L per-image (front)",
            _fmt(s11["rl"][1], s11["rl"][0]),
            _fmt(s13["rl"][1], s13["rl"][0]),
            _fmt(rs["rl"][1], rs["rl"][0]),
        ),
        (
            "Per-car high-conf (aliased)",
            _fmt(s11["tracks_high"], s11["n_cars"]),
            _fmt(s13["tracks_high"], s13["n_cars"]),
            _fmt(rs["tracks_high"], rs["n_cars"]),
        ),
        (
            "Per-car aliasing-free (any)",
            _fmt(s11["real_any"], s11["n_cars"]),
            _fmt(s13["real_any"], s13["n_cars"]),
            _fmt(rs["real_any"], rs["n_cars"]),
        ),
        (
            "Per-car aliasing-free (top)",
            _fmt(s11["real_top"], s11["n_cars"]),
            _fmt(s13["real_top"], s13["n_cars"]),
            _fmt(rs["real_top"], rs["n_cars"]),
        ),
        (
            "Ghost plates (>=5 tracks)",
            str(s11["ghost_count"]),
            str(s13["ghost_count"]),
            str(rs["ghost_count"]),
        ),
    ]
    label_w = max(len(r[0]) for r in rows)
    col_w = 22
    print(
        f"{'metric':<{label_w}}  {'Step 11':<{col_w}}  {'Step 13a':<{col_w}}  {'Re-soak':<{col_w}}"
    )
    print(f"{'-' * label_w}  {'-' * col_w}  {'-' * col_w}  {'-' * col_w}")
    for row in rows:
        label, a, b, c = row
        print(f"{label:<{label_w}}  {a:<{col_w}}  {b:<{col_w}}  {c:<{col_w}}")
    print()

    print("Snap-stats (from meta.json):")
    print(
        f"  HTTP attempts/successes:  s11={s11['attempts']}/{s11['successes']}  "
        f"s13={s13['attempts']}/{s13['successes']}  rs={rs['attempts']}/{rs['successes']}"
    )
    print(
        f"  dropped (mc cap):         s11={s11['dropped']}  s13={s13['dropped']}  rs={rs['dropped']}"
    )
    print(
        f"  mean fires/track:         s11={s11['fires_per_track'].get('mean')}  "
        f"s13={s13['fires_per_track'].get('mean')}  rs={rs['fires_per_track'].get('mean')}"
    )
    print(
        f"  pipeline_fires:           s11={s11['pipeline_fires']}  "
        f"s13={s13['pipeline_fires']}  rs={rs['pipeline_fires']}"
    )
    print(
        f"  pipeline_throttled:       s11={s11['pipeline_throttled']}  "
        f"s13={s13['pipeline_throttled']}  rs={rs['pipeline_throttled']}"
    )
    print(
        f"  pipeline_budget_exhaust:  s11={s11['pipeline_budget_exhausted']}  "
        f"s13={s13['pipeline_budget_exhausted']}  rs={rs['pipeline_budget_exhausted']}"
    )
    print(
        f"  latency p50 / p99 (ms):   s11={s11['latency_p50']}/{s11['latency_p99']}  "
        f"s13={s13['latency_p50']}/{s13['latency_p99']}  "
        f"rs={rs['latency_p50']}/{rs['latency_p99']}"
    )
    print()

    print(f"Re-soak ghosts: {rs['ghosts']}")
    print()
    print("Top 10 plate strings by distinct-track count (Re-soak):")
    for plate, n in rs["top10"]:
        flag = "  <-- GHOST" if n >= GHOST_TRACK_MIN else ""
        print(f"  {plate:<10}  {n} tracks{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
