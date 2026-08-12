"""Characterize the DETECTED plate pixels day vs night, to tell WHICH
capture problem kills night reads: underexposure, retroreflective/IR
blow-out, or motion blur.

The funnel (night_exposure_verdict.py) showed night plates are still
DETECTED (det% 81-92%) but the glyphs don't resolve. This crops each
detected plate's det_bbox out of the 4K snap and measures:

* luma           mean grayscale 0-255  (low => underexposed / dark)
* over%          fraction of px > 240   (high => blown out / glare / IR bloom)
* under%         fraction of px < 40    (high => crushed shadows)
* sharp          variance of Laplacian  (low => motion/defocus blur)
* chroma         mean |max-min| across BGR per px (low => IR monochrome)

Groups: day/night x canonical(read)/failed(detected-but-unread). The
contrast between night-failed and day-canonical names the lever.

    uv run python .claude/night_exposure_pixels.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from streettracker.analysis.dvsa import is_canonical_uk_plate

SESSIONS = [
    "output/session_20260608_125639",
    "output/session_20260721_115918",
    "output/session_20260709_103920",
]
PER_GROUP = 400  # sampled detected plates per group (decode-bounded)


def plate_stats(img: np.ndarray, bbox: list[int]) -> dict | None:
    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 6:
        return None
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    b, g, r = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    chroma = float(np.mean(np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)))
    return {
        "luma": float(gray.mean()),
        "over": float((gray > 240).mean()),
        "under": float((gray < 40).mean()),
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "chroma": chroma,
        "width": x2 - x1,
    }


def main(argv: list[str]) -> int:
    random.seed(0)
    # Collect candidate rows per group first (no decode), then sample + decode.
    cand: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for sess in argv or SESSIONS:
        sd = Path(sess)
        name = sd.name
        data = json.loads((sd / f"{name}_data.json").read_text())
        alpr = json.loads((sd / f"{name}_alpr.json").read_text())
        night_tid = {}
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
            night_tid[r["track_id"]] = h >= 19 or h < 7
        for row in alpr:
            if row.get("pipeline") != "preferred":
                continue
            b = row.get("det_bbox")
            if not (b and isinstance(b, (list, tuple)) and len(b) >= 4):
                continue  # detection failed -> no plate pixels to characterize
            tid = row.get("track_id")
            if tid not in night_tid:
                continue
            night = night_tid[tid]
            txt = (row.get("ocr_text") or "").strip().upper().replace(" ", "")
            canon = bool(txt and (row.get("ocr_conf") or 0.0) >= 0.9 and is_canonical_uk_plate(txt))
            grp = f"{'night' if night else 'day'}-{'canon' if canon else 'fail'}"
            ip = row.get("image_path")
            if ip:
                cand[grp].append((ip, list(b[:4])))

    groups = ["day-canon", "day-fail", "night-canon", "night-fail"]
    print(f"candidate detected plates: {{ {', '.join(f'{g}:{len(cand[g])}' for g in groups)} }}\n")

    agg: dict[str, list[dict]] = defaultdict(list)
    for grp in groups:
        rows = cand[grp]
        random.shuffle(rows)
        for ip, bbox in rows[:PER_GROUP]:
            img = cv2.imread(ip)
            if img is None:
                continue
            st = plate_stats(img, bbox)
            if st:
                agg[grp].append(st)

    def med(vals: list[float]) -> float:
        return float(np.median(vals)) if vals else float("nan")

    print(f"{'group':<12} {'n':>4}  {'luma':>6} {'over%':>6} {'under%':>7} {'sharp':>7} {'chroma':>7} {'width':>6}")
    for grp in groups:
        s = agg[grp]
        if not s:
            print(f"{grp:<12} {0:>4}   (no samples)")
            continue
        print(
            f"{grp:<12} {len(s):>4}  "
            f"{med([x['luma'] for x in s]):>6.1f} "
            f"{100 * med([x['over'] for x in s]):>5.1f}% "
            f"{100 * med([x['under'] for x in s]):>6.1f}% "
            f"{med([x['sharp'] for x in s]):>7.0f} "
            f"{med([x['chroma'] for x in s]):>7.1f} "
            f"{med([x['width'] for x in s]):>6.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
