"""Night-exposure lever verdict: does the day->night ANPR drop come from
UNDEREXPOSURE (a fixable capture lever) or something else?

Decomposes the fullframe ALPR funnel by camera-local hour across the
night-spanning soaks:

    snap  ->  plate DETECTED (det_bbox present)  ->  canonical read
              (conf>=0.9 + canonical UK shape)

Two failure modes separate cleanly (when a plate is detected OCR always
yields text, so the only stages are detection and the canonical/OCR gate):

* detection fails   = plate not found in the crop (too dark? glare? gone)
* non-canonical OCR = plate found but unreadable (noise / blur / low-res)

If night loss is DETECTION-dominated + snaps are dark -> underexposure,
exposure/IR-illumination is the lever. If it's OCR-dominated on well-exposed
crops -> motion blur / sensor noise, a different fix.

    uv run python .claude/night_exposure_verdict.py [session ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from streettracker.analysis.dvsa import is_canonical_uk_plate

DEFAULT_SESSIONS = [
    "output/session_20260608_125639",
    "output/session_20260721_115918",
    "output/session_20260709_103920",
]

# Coarse lighting buckets for an August UK scene (sunrise ~05:45, sunset
# ~20:15, full dark ~21:15). Fine enough to show the dusk gradient.
BUCKETS = [
    ("00-05 dark", range(0, 5)),
    ("05-07 dawn", range(5, 7)),
    ("07-10 day", range(7, 10)),
    ("10-16 day", range(10, 16)),
    ("16-19 day", range(16, 19)),
    ("19-21 dusk", range(19, 21)),
    ("21-24 dark", range(21, 24)),
]


def bucket_for(hour: int) -> str:
    for name, hrs in BUCKETS:
        if hour in hrs:
            return name
    return "?"


def has_det(row: dict) -> bool:
    b = row.get("det_bbox")
    return bool(b) and isinstance(b, (list, tuple)) and len(b) >= 4


def main(argv: list[str]) -> int:
    sessions = argv or DEFAULT_SESSIONS

    # Per-bucket snap funnel: [n_snaps, n_detected, n_canonical]
    snap = defaultdict(lambda: [0, 0, 0])
    # Per-bucket per-car canonical, split by direction: dir -> [n_read, n_cars]
    car = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    is_night = {"day": [0, 0, 0], "night": [0, 0, 0]}  # snap funnel day/night

    for sess in sessions:
        sd = Path(sess)
        name = sd.name
        data = json.loads((sd / f"{name}_data.json").read_text())
        alpr = json.loads((sd / f"{name}_alpr.json").read_text())

        # track_id -> (bucket, direction, is_night_flag)
        tinfo: dict[int, tuple[str, str, bool]] = {}
        for r in data:
            if r.get("class_name") != "car" or not r.get("main_snaps"):
                continue
            ts = r.get("time_start")
            if not ts:
                continue
            try:
                h = datetime.fromisoformat(ts).hour
            except ValueError:
                continue
            night = h >= 19 or h < 7
            tinfo[r["track_id"]] = (bucket_for(h), r.get("direction") or "?", night)

        # snap-level funnel
        outcome: dict[tuple[int, int], bool] = {}
        for row in alpr:
            if row.get("pipeline") != "preferred":
                continue
            tid = row.get("track_id")
            ti = tinfo.get(tid)
            if ti is None:
                continue
            b, _d, night = ti
            det = has_det(row)
            txt = (row.get("ocr_text") or "").strip().upper().replace(" ", "")
            canon = bool(txt and (row.get("ocr_conf") or 0.0) >= 0.9 and is_canonical_uk_plate(txt))
            snap[b][0] += 1
            snap[b][1] += det
            snap[b][2] += canon
            key = "night" if night else "day"
            is_night[key][0] += 1
            is_night[key][1] += det
            is_night[key][2] += canon
            outcome[(tid, row["snap_index"])] = canon

        # per-car canonical
        for r in data:
            if r.get("class_name") != "car" or not r.get("main_snaps"):
                continue
            ti = tinfo.get(r["track_id"])
            if ti is None:
                continue
            b, d, _night = ti
            read = any(outcome.get((r["track_id"], s), False) for s in r["main_snaps"])
            for grp in (d, "all"):
                car[b][grp][1] += 1
                car[b][grp][0] += read

    def pct(a: int, n: int) -> str:
        return f"{100 * a / n:5.1f}% ({a}/{n})" if n else "   n/a"

    order = [name for name, _ in BUCKETS]
    print(f"=== Night-exposure verdict: {len(sessions)} session(s) ===\n")
    print("SNAP FUNNEL by lighting bucket (det% = plate found; canon% = readable):")
    print(f"  {'bucket':<11} {'snaps':>6}  {'det%':>16}  {'canon/snap':>18}  {'canon|det':>16}")
    for b in order:
        n, det, canon = snap[b]
        cd = f"{100 * canon / det:.1f}%" if det else "n/a"
        print(f"  {b:<11} {n:>6}  {pct(det, n):>16}  {pct(canon, n):>18}  {cd:>16}")

    print("\nPER-CAR canonical (>=1 readable read) by lighting bucket:")
    print(f"  {'bucket':<11} {'all':>16}  {'L->R (rear)':>18}  {'R->L (front)':>18}")
    for b in order:
        c = car[b]
        print(
            f"  {b:<11} {pct(*c['all'][::-1][::-1]):>16}  "
            f"{pct(c['left to right'][0], c['left to right'][1]):>18}  "
            f"{pct(c['right to left'][0], c['right to left'][1]):>18}"
        )

    print("\nDAY (07-19) vs NIGHT (19-07) snap funnel:")
    for k in ("day", "night"):
        n, det, canon = is_night[k]
        cd = f"{100 * canon / det:.1f}%" if det else "n/a"
        print(f"  {k:<6} snaps={n:<6} det={pct(det, n)}  canon/snap={pct(canon, n)}  canon|det={cd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
