"""Hour-bucketed Re-soak vs Step 13a vs Step 11.

The headline aggregate shows Re-soak per-car aliasing-free at 61.0% vs
Step 11's 78.3% -- a 17pp regression. But Step 13a's apparent regression
(73.2%) turned out to be hour-of-day confound; matched-hour cuts showed
Step 13a was decisively better than Step 11 in hours 17 and 18.

Question: in matched hours, does the Re-soak beat Step 13a (and thus
both beat Step 11)? If yes, the 17pp aggregate hit is overnight/dawn
sampling, not a real regression. If no, max_concurrent=3 is harmful.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

STEP11 = "session_20260526_124704"
STEP13A = "session_20260527_160139"
RESOAK = "session_20260528_103902"
CONF_THRESHOLD = 0.9
LOCAL_TZ_OFFSET = 1  # BST


def _load_session(session: str) -> dict:
    base = Path("output") / session
    data = json.loads((base / f"{session}_data.json").read_text())
    alpr = json.loads((base / f"{session}_alpr.json").read_text())

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
    out: dict[int, dict] = defaultdict(
        lambda: {
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
        }
    )

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
    rs = _load_session(RESOAK)
    b11 = _bucket(s11)
    b13 = _bucket(s13)
    bRS = _bucket(rs)

    print("=== Hour-bucketed Re-soak vs Step 13a vs Step 11 (BST hour-of-day) ===")
    print("Per-car aliasing-free rate by hour, only hours where all three have >= 30 cars OR Step 11 has data\n")

    hours = sorted(set(b11.keys()) | set(b13.keys()) | set(bRS.keys()))
    print(
        f"{'hr':>3} | {'s11 cars':>10}  {'s13a cars':>11}  {'resoak cars':>13} || "
        f"{'s11':>8} {'s13a':>8} {'resoak':>8}"
    )
    print("-" * 90)
    for h in hours:
        a = b11[h]
        b = b13[h]
        c = bRS[h]
        if a["cars"] < 10 and b["cars"] < 10 and c["cars"] < 10:
            continue
        cell_a_car = f"{a['cars_high']:3d}/{a['cars']:3d}" if a["cars"] else "  -/- "
        cell_b_car = f"{b['cars_high']:3d}/{b['cars']:3d}" if b["cars"] else "  -/- "
        cell_c_car = f"{c['cars_high']:3d}/{c['cars']:3d}" if c["cars"] else "  -/- "
        cell_a_pct = _pct(a["cars_high"], a["cars"])
        cell_b_pct = _pct(b["cars_high"], b["cars"])
        cell_c_pct = _pct(c["cars_high"], c["cars"])
        s11_mark = " <-s11" if a["cars"] >= 10 else "      "
        print(
            f" {h:>2} | {cell_a_car:>10}  {cell_b_car:>11}  {cell_c_car:>13} || "
            f"{cell_a_pct} {cell_b_pct} {cell_c_pct}{s11_mark}"
        )

    print("\nR->L per-image rate by hour:\n")
    print(f"{'hr':>3} | {'s11':>16}  {'s13a':>16}  {'resoak':>16}")
    print("-" * 80)
    for h in hours:
        a = b11[h]
        b = b13[h]
        c = bRS[h]
        if a["rl_imgs"] < 10 and b["rl_imgs"] < 10 and c["rl_imgs"] < 10:
            continue
        cell_a = f"{a['rl_imgs_high']:3d}/{a['rl_imgs']:3d} {_pct(a['rl_imgs_high'], a['rl_imgs'])}"
        cell_b = f"{b['rl_imgs_high']:3d}/{b['rl_imgs']:3d} {_pct(b['rl_imgs_high'], b['rl_imgs'])}"
        cell_c = f"{c['rl_imgs_high']:3d}/{c['rl_imgs']:3d} {_pct(c['rl_imgs_high'], c['rl_imgs'])}"
        print(f" {h:>2} | {cell_a:>16}  {cell_b:>16}  {cell_c:>16}")

    # Roll-up across hours where Step 11 has data (its window: 12-18 BST)
    overlap = [h for h in hours if 12 <= h <= 17]

    def _sum(buckets, key):
        return sum(buckets[h][key] for h in overlap)

    print("\n=== Daytime-only roll-up (BST 12-17, matched on s11's window) ===")
    rows = [
        (
            "Per-image high-conf",
            _sum(b11, "imgs_high"),
            _sum(b11, "imgs"),
            _sum(b13, "imgs_high"),
            _sum(b13, "imgs"),
            _sum(bRS, "imgs_high"),
            _sum(bRS, "imgs"),
        ),
        (
            "Per-car aliasing-free",
            _sum(b11, "cars_high"),
            _sum(b11, "cars"),
            _sum(b13, "cars_high"),
            _sum(b13, "cars"),
            _sum(bRS, "cars_high"),
            _sum(bRS, "cars"),
        ),
        (
            "  R->L per-image",
            _sum(b11, "rl_imgs_high"),
            _sum(b11, "rl_imgs"),
            _sum(b13, "rl_imgs_high"),
            _sum(b13, "rl_imgs"),
            _sum(bRS, "rl_imgs_high"),
            _sum(bRS, "rl_imgs"),
        ),
        (
            "  R->L per-car",
            _sum(b11, "rl_cars_high"),
            _sum(b11, "rl_cars"),
            _sum(b13, "rl_cars_high"),
            _sum(b13, "rl_cars"),
            _sum(bRS, "rl_cars_high"),
            _sum(bRS, "rl_cars"),
        ),
        (
            "  L->R per-image",
            _sum(b11, "lr_imgs_high"),
            _sum(b11, "lr_imgs"),
            _sum(b13, "lr_imgs_high"),
            _sum(b13, "lr_imgs"),
            _sum(bRS, "lr_imgs_high"),
            _sum(bRS, "lr_imgs"),
        ),
    ]
    label_w = max(len(r[0]) for r in rows)
    for label, na, da, nb, db, nc, dc in rows:
        pa = 100 * na / da if da else 0
        pb = 100 * nb / db if db else 0
        pc = 100 * nc / dc if dc else 0
        print(
            f"  {label:<{label_w}}  s11: {na:4d}/{da:4d} ({pa:5.1f}%)   "
            f"s13a: {nb:4d}/{db:4d} ({pb:5.1f}%)   "
            f"resoak: {nc:4d}/{dc:4d} ({pc:5.1f}%)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
