"""Are cars missed after dark? Weeknight car counts by camera mode (review R16).

Two tables from every session under ``output/``:

1. Car tracks per fully covered hour, Mon-Thu, in three periods: June-July
   (light until ~21:30), the IR weeks (18 Aug-27 Sep, when the camera's IR
   mode made the runtime skip inference after dark) and forced colour
   (28 Sep-1 Oct, dark from ~19:30). If forced colour matches summer
   daylight hour for hour, dark-hour cars aren't being missed.
2. Track quality, Mon-Thu, 12-16 h vs 20-23 h in summer and forced colour:
   mean detection confidence, visible seconds, detections, net displacement
   and the R->L share. Missed detections offset by split tracks would show as
   shorter tracks with less displacement.

An hour counts as covered when sessions span >= 90 % of it (session start from
``_meta.json`` to the last track end). Car tracks exclude ``class_suspect``.

    uv run python .claude/lowlight_counts.py [--output-root output]

Read-only. Result (2026-10-05): forced-colour dark hours match summer
daylight (20 h 43.5 vs 49.6, 22 h 1.8 vs 1.5); dark tracks have lower
confidence (0.77 vs 0.84) and are visible ~2 s less.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

PERIODS = [
    ("Jun-Jul (light)", date(2026, 6, 1), date(2026, 7, 31)),
    ("IR weeks", date(2026, 8, 18), date(2026, 9, 27)),
    ("forced colour", date(2026, 9, 28), date(2026, 10, 4)),
]
HOURS = [5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2]


def period(dd: date) -> str | None:
    for name, a, b in PERIODS:
        if a <= dd <= b:
            return name
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    args = ap.parse_args()

    cars: Counter = Counter()
    cov: dict = defaultdict(float)
    quality: dict = defaultdict(list)
    for meta_path in sorted(args.output_root.glob("session_*/session_*_meta.json")):
        try:
            data = json.loads(
                meta_path.with_name(meta_path.name.replace("_meta", "_data")).read_text()
            )
        except (OSError, ValueError):
            continue
        if not data:
            continue
        t0 = json.loads(meta_path.read_text()).get("session_start_unix")
        t1 = max(r.get("time_end_unix") or 0 for r in data)
        if not t0 or not t1:
            continue
        t = t0
        while t < t1:
            dt = datetime.fromtimestamp(t)
            nxt = min(t1, t + (3600 - dt.minute * 60 - dt.second - dt.microsecond / 1e6))
            cov[(dt.date(), dt.hour)] += (nxt - t) / 3600
            t = nxt + 1e-3
        for r in data:
            if r.get("class_name") != "car" or r.get("class_suspect") or not r.get("time_start"):
                continue
            dt = datetime.fromisoformat(r["time_start"])
            cars[(dt.date(), dt.hour)] += 1
            p = period(dt.date())
            if p in ("Jun-Jul (light)", "forced colour") and dt.weekday() <= 3:
                if 12 <= dt.hour < 16:
                    quality[(p, "12-16")].append(r)
                elif 20 <= dt.hour < 23:
                    quality[(p, "20-23")].append(r)

    agg: dict = defaultdict(lambda: [0, 0])
    for (dd, h), c in cov.items():
        p = period(dd)
        if not p or dd.weekday() > 3 or c < 0.9:
            continue
        agg[(p, h)][0] += cars[(dd, h)]
        agg[(p, h)][1] += 1
    names = [p[0] for p in PERIODS]
    print("car tracks per hour, Mon-Thu, fully covered hours (n hours)")
    print("hour " + "".join(f"{p:>22}" for p in names))
    for h in HOURS:
        cells = []
        for p in names:
            n, k = agg[(p, h)]
            cells.append(f"{(n / k if k else float('nan')):>17.1f} ({k:>2})")
        print(f"{h:>4} " + "".join(cells))

    print("\ncar track quality, Mon-Thu (medians)")
    print(f"{'set':<24}{'n':>6}{'conf':>7}{'vis_s':>7}{'dets':>6}{'net_px':>8}{'R->L%':>7}")
    for key in sorted(quality):
        rs = quality[key]

        def med(field: str, rs: list = rs) -> float:
            return float(np.median([r[field] for r in rs]))

        rl = 100 * np.mean([r["direction"] == "right to left" for r in rs])
        print(
            f"{key[0] + ' ' + key[1]:<24}{len(rs):>6}{med('avg_confidence'):>7.3f}"
            f"{med('duration_visible'):>7.1f}{med('num_detections'):>6.0f}"
            f"{med('net_displacement_px'):>8.0f}{rl:>7.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
