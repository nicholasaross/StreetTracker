"""Hour-bucketed Step 13a vs Step 11.

Question: was the apparent regression (per-car aliasing-free 78.3% ->
73.2%, R->L per-image 28.7% -> 19.6%) driven by traffic mix (Step 13a
spans 18h incl. overnight + dawn) or by the cadence change itself?

Method: bucket both sessions by local hour-of-day. For each hour where
BOTH sessions have >= 10 cars, compare per-image rates and per-car
aliasing-free rates side by side. If the regression disappears when
matched on hour-of-day, traffic mix did it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

STEP11 = "session_20260526_124704"
STEP13A = "session_20260527_160139"
CONF_THRESHOLD = 0.9
LOCAL_TZ_OFFSET = 1  # BST


def _load_session(session: str) -> dict:
    base = Path("output") / session
    data = json.loads((base / f"{session}_data.json").read_text())
    alpr = json.loads((base / f"{session}_alpr.json").read_text())

    # Per-car high-conf and per-car best read.
    by_tid_dir: dict[int, str] = {}
    by_tid_hour: dict[int, int] = {}
    car_tids: set[int] = set()
    for r in data:
        if r.get("class_name") != "car":
            continue
        tid = r["track_id"]
        car_tids.add(tid)
        by_tid_dir[tid] = r.get("direction", "?")
        dt = datetime.fromtimestamp(r["time_start_unix"], tz=timezone.utc)
        by_tid_hour[tid] = (dt.hour + LOCAL_TZ_OFFSET) % 24

    by_tid_best: dict[int, float] = {}
    img_rows: list[dict] = []
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        tid = r["track_id"]
        conf = r.get("ocr_conf") or 0.0
        txt = r.get("ocr_text")
        if txt and conf > by_tid_best.get(tid, -1.0):
            by_tid_best[tid] = conf
        img_rows.append(
            {
                "tid": tid,
                "high": bool(txt and conf >= CONF_THRESHOLD),
                "hour": by_tid_hour.get(tid),
                "dir": by_tid_dir.get(tid),
            }
        )

    return {
        "car_tids": car_tids,
        "tid_to_hour": by_tid_hour,
        "tid_to_dir": by_tid_dir,
        "tid_to_best": by_tid_best,
        "img_rows": img_rows,
    }


def _bucket(stats: dict) -> dict[int, dict]:
    out: dict[int, dict] = defaultdict(lambda: {
        "cars": 0,
        "cars_high": 0,
        "rl_cars": 0,
        "rl_cars_high": 0,
        "lr_cars": 0,
        "lr_cars_high": 0,
        "imgs": 0,
        "imgs_high": 0,
        "rl_imgs": 0,
        "rl_imgs_high": 0,
        "lr_imgs": 0,
        "lr_imgs_high": 0,
    })

    for tid in stats["car_tids"]:
        h = stats["tid_to_hour"][tid]
        d = stats["tid_to_dir"].get(tid, "?")
        is_high = stats["tid_to_best"].get(tid, 0.0) >= CONF_THRESHOLD
        out[h]["cars"] += 1
        if is_high:
            out[h]["cars_high"] += 1
        if d == "right to left":
            out[h]["rl_cars"] += 1
            if is_high:
                out[h]["rl_cars_high"] += 1
        elif d == "left to right":
            out[h]["lr_cars"] += 1
            if is_high:
                out[h]["lr_cars_high"] += 1

    for row in stats["img_rows"]:
        h = row["hour"]
        if h is None:
            continue
        out[h]["imgs"] += 1
        if row["high"]:
            out[h]["imgs_high"] += 1
        if row["dir"] == "right to left":
            out[h]["rl_imgs"] += 1
            if row["high"]:
                out[h]["rl_imgs_high"] += 1
        elif row["dir"] == "left to right":
            out[h]["lr_imgs"] += 1
            if row["high"]:
                out[h]["lr_imgs_high"] += 1

    return out


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "  n/a "


def main() -> int:
    s11 = _load_session(STEP11)
    s13 = _load_session(STEP13A)
    b11 = _bucket(s11)
    b13 = _bucket(s13)

    print("=== Hour-bucketed Step 11 vs Step 13a (BST hour-of-day) ===")
    print(
        "       per-image high-conf      |    per-car (aliasing-free)     |    R->L per-image  "
    )
    print(
        "hour | s11           s13       | s11           s13            | s11        s13       "
    )
    print(
        "-----+-------------------------+--------------------------------+--------------------"
    )
    hours = sorted(set(b11.keys()) | set(b13.keys()))
    for h in hours:
        a = b11[h]
        b = b13[h]
        if a["cars"] < 10 and b["cars"] < 10:
            continue
        cell_a_img = f"{a['imgs_high']:4d}/{a['imgs']:4d} {_pct(a['imgs_high'], a['imgs'])}"
        cell_b_img = f"{b['imgs_high']:4d}/{b['imgs']:4d} {_pct(b['imgs_high'], b['imgs'])}"
        cell_a_car = f"{a['cars_high']:3d}/{a['cars']:3d} {_pct(a['cars_high'], a['cars'])}"
        cell_b_car = f"{b['cars_high']:3d}/{b['cars']:3d} {_pct(b['cars_high'], b['cars'])}"
        cell_a_rl = f"{a['rl_imgs_high']:3d}/{a['rl_imgs']:3d} {_pct(a['rl_imgs_high'], a['rl_imgs'])}"
        cell_b_rl = f"{b['rl_imgs_high']:3d}/{b['rl_imgs']:3d} {_pct(b['rl_imgs_high'], b['rl_imgs'])}"
        marker = "  <-- s11 hours" if a["cars"] >= 10 else ""
        print(
            f"  {h:>2d} | {cell_a_img}  {cell_b_img}  | {cell_a_car}  {cell_b_car} | {cell_a_rl}  {cell_b_rl}{marker}"
        )

    # Overlapping daytime (12-18 BST = ~Step 11's natural window).
    overlap_hours = [h for h in hours if 12 <= h <= 17]
    if overlap_hours:
        def _sum(buckets, key):
            return sum(buckets[h][key] for h in overlap_hours)

        print()
        print("=== Daytime-only roll-up (BST 12-17, matched on s11's window) ===")
        rows = [
            ("Per-image high-conf",
             _sum(b11, "imgs_high"), _sum(b11, "imgs"),
             _sum(b13, "imgs_high"), _sum(b13, "imgs")),
            ("Per-car aliasing-free",
             _sum(b11, "cars_high"), _sum(b11, "cars"),
             _sum(b13, "cars_high"), _sum(b13, "cars")),
            ("  R->L per-image",
             _sum(b11, "rl_imgs_high"), _sum(b11, "rl_imgs"),
             _sum(b13, "rl_imgs_high"), _sum(b13, "rl_imgs")),
            ("  R->L per-car",
             _sum(b11, "rl_cars_high"), _sum(b11, "rl_cars"),
             _sum(b13, "rl_cars_high"), _sum(b13, "rl_cars")),
            ("  L->R per-image",
             _sum(b11, "lr_imgs_high"), _sum(b11, "lr_imgs"),
             _sum(b13, "lr_imgs_high"), _sum(b13, "lr_imgs")),
        ]
        label_w = max(len(r[0]) for r in rows)
        for label, na, da, nb, db in rows:
            print(
                f"  {label:<{label_w}}  Step 11: {na:4d}/{da:4d} ({100*na/da if da else 0:5.1f}%)   "
                f"Step 13a: {nb:4d}/{db:4d} ({100*nb/db if db else 0:5.1f}%)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
