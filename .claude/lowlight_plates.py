"""Low-light plate loss by hour and direction, from production sessions (review R16).

For snapped car tracks (class car, has 4K snaps), bucketed by local start hour:

* per hour: % with a plate detected on any snap, % with a read;
* per hour group x direction: the same two rates;
* per hour group: detected plates' UK-shaped and read shares, plus plate-pixel
  stats on a sample (luma, raw and width-normalised Laplacian variance, width).

A read is a non-suspect row (no ``static_suspect`` / ``colour_suspect``) with a
plate box, ``ocr_conf`` (min-character confidence) >= 0.90 and a UK shape.
Width-normalised sharpness resizes each plate crop to 128 px wide first, so
near and far plates compare. ``plate_pixels`` is shared with
``.claude/ir_capture_plates.py``.

    uv run python .claude/lowlight_plates.py                       # the 30 Sep-4 Oct week
    uv run python .claude/lowlight_plates.py output/session_A ...

Read-only. Result (2026-10-05): L->R plates are still found at 07-08 / 18-19 h
(81-100 %) but none read; R->L plate detection drops to 17-41 % (day 91 %);
low-light plates are brighter than day (luma 128-171 vs 105) and 2-4x less
sharp (width-normalised 484-954 vs 1,982).
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

SESSIONS = ["output/session_20260930_211033", "output/session_20261001_190826"]
GATE = 0.90
NORM_W = 128
PX_SAMPLE = 250


def plate_pixels(img: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    """Pixel stats of an unpadded plate box in a 4K snap."""
    x1, y1, x2, y2 = bbox
    crop = img[max(0, y1) : y2, max(0, x1) : x2]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    w = g.shape[1]
    interp = cv2.INTER_AREA if w > NORM_W else cv2.INTER_CUBIC
    gn = cv2.resize(g, (NORM_W, max(8, round(g.shape[0] * NORM_W / w))), interpolation=interp)
    return {
        "luma": round(float(g.mean()), 1),
        "over": round(float((g > 240).mean()), 3),
        "sharp": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
        "sharp_n": round(float(cv2.Laplacian(gn, cv2.CV_64F).var()), 1),
        "pw": int(x2 - x1),
        "ph": int(y2 - y1),
    }


def group(h: int) -> str:
    if 10 <= h < 16:
        return "day 10-16"
    if h >= 21 or h < 6:
        return "night 21-06"
    if 19 <= h < 21:
        return "19-21"
    return f"{h:02d}-{h + 1:02d}"


def is_read(a: dict) -> bool:
    return bool(
        a.get("det_bbox") and a.get("canonical_uk_shape") and (a.get("ocr_conf") or 0) >= GATE
    )


def main(argv: list[str]) -> int:
    random.seed(0)
    per_hour: dict = defaultdict(lambda: [0, 0, 0])
    per_dir: dict = defaultdict(lambda: [0, 0, 0])
    per_grp: dict = defaultdict(lambda: [0, 0, 0])
    plates: dict = defaultdict(list)
    for sd in map(Path, argv or SESSIONS):
        data = json.loads((sd / f"{sd.name}_data.json").read_text())
        alpr = json.loads((sd / f"{sd.name}_alpr.json").read_text())
        tracks = {
            r["track_id"]: r for r in data if r.get("class_name") == "car" and r.get("main_snaps")
        }
        rows = defaultdict(list)
        for a in alpr:
            if a.get("pipeline") != "preferred" or a["track_id"] not in tracks:
                continue
            if a.get("static_suspect") or a.get("colour_suspect"):
                continue
            rows[a["track_id"]].append(a)
        for tid, r in tracks.items():
            h = datetime.fromisoformat(r["time_start"]).hour
            g = group(h)
            rs = rows.get(tid, [])
            det = any(a.get("det_bbox") for a in rs)
            read = any(is_read(a) for a in rs)
            for acc in (per_hour[h], per_dir[(g, r["direction"])], per_grp[g]):
                acc[0] += 1
                acc[1] += det
                acc[2] += read
            for a in rs:
                if a.get("det_bbox"):
                    plates[g].append(
                        (
                            a["image_path"],
                            a["det_bbox"],
                            bool(a.get("canonical_uk_shape")),
                            is_read(a),
                        )
                    )

    def pct(a: int, b: int) -> float:
        return 100 * a / b if b else float("nan")

    print("snapped car tracks by start hour")
    print(f"{'hour':>5} {'tracks':>7} {'plate found':>11} {'read':>7}")
    for h in sorted(per_hour):
        n, d, rd = per_hour[h]
        print(f"{h:>5} {n:>7} {pct(d, n):>10.1f}% {pct(rd, n):>6.1f}%")

    print("\nby hour group and direction")
    for (g, direction), (n, d, rd) in sorted(per_dir.items()):
        print(
            f"{g:<12} {direction:<14} n={n:>4}  "
            f"plate found {pct(d, n):5.1f}%  read {pct(rd, n):5.1f}%"
        )

    print(
        f"\n{'group':<12} {'tracks':>6} {'found%':>7} {'read%':>6} | {'plates':>6} {'UK%':>6} "
        f"{'read%':>6} {'n_px':>5} {'luma':>6} {'sharp':>7} {'sharp_n':>8} {'pw':>5}"
    )
    for g in sorted(per_grp):
        n, d, rd = per_grp[g]
        pl = plates[g]
        px = []
        for ip, bb, _uk, _rd in random.sample(pl, min(PX_SAMPLE, len(pl))):
            img = cv2.imread(ip)
            if img is not None:
                px.append(plate_pixels(img, tuple(int(v) for v in bb[:4])))

        def med(k: str, px: list = px) -> float:
            return float(np.median([x[k] for x in px])) if px else float("nan")

        print(
            f"{g:<12} {n:>6} {pct(d, n):>6.1f}% {pct(rd, n):>5.1f}% | {len(pl):>6} "
            f"{pct(sum(p[2] for p in pl), len(pl)):>5.1f}% "
            f"{pct(sum(p[3] for p in pl), len(pl)):>5.1f}% {len(px):>5} {med('luma'):>6.1f} "
            f"{med('sharp'):>7.0f} {med('sharp_n'):>8.0f} {med('pw'):>5.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
