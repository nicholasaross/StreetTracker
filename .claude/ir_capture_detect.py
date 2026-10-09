"""Offline detection + motion pass over one ``ir_capture.py`` window's sub.mp4.

Step 1 of the capture analysis (review R16 "Results", 2026-10-05). One decode,
three detector views per frame, with the live runtime's classes and image size:

  A: YOLOv8m + BotSORT at the live conf (0.30)
  B: YOLOv8m + BotSORT at conf 0.10
  C: YOLOv8m raw predict at conf 0.05 (no tracker)

plus a detector-free MOG2 foreground count inside the operator-traced road
polygon (``.claude/road_polygon_user.json``), on a half-size grey frame. The
point is to find cars the live tracker never saw: a moving car shows up in the
motion count whatever the detector makes of it.

    uv run python .claude/ir_capture_detect.py <capture>/1004_2130 [--max-frames N]

Writes ``<window>/detect.json`` (next to the capture, never into the repo).
~25 min per 20-min window on the dev-box 3080; the BotSORT motion compensation
on the CPU is the bottleneck. Then run ``.claude/ir_capture_report.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
CLASSES = [0, 1, 2, 3, 5, 7, 16]  # the live vehicle_classes


def road_mask(w: int, h: int) -> np.ndarray:
    poly = json.loads((REPO / ".claude/road_polygon_user.json").read_text())["vertices_frac"]
    pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m


def boxes(r, with_id: bool) -> list:
    """``[tid?, cls, conf, x1, y1, x2, y2]`` rows; tid -1 for untracked boxes."""
    b = r.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy().round(1).tolist()
    conf = b.conf.cpu().numpy().round(3).tolist()
    cls = b.cls.cpu().numpy().astype(int).tolist()
    ids = b.id.cpu().numpy().astype(int).tolist() if (with_id and b.id is not None) else None
    out = []
    for i in range(len(xyxy)):
        row = [cls[i], conf[i], *xyxy[i]]
        if with_id:
            row.insert(0, ids[i] if ids is not None else -1)
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("window", type=Path)
    ap.add_argument("--max-frames", type=int, default=0, help="stop early (smoke test)")
    args = ap.parse_args()
    wdir: Path = args.window

    cap = cv2.VideoCapture(str(wdir / "sub.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    weights = str(REPO / "yolov8m.pt")
    mA, mB, mC = (YOLO(weights) for _ in range(3))
    sw, sh = W // 2, H // 2
    rmask = cv2.resize(road_mask(W, H), (sw, sh), interpolation=cv2.INTER_NEAREST)
    road_px = int((rmask > 0).sum())
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    common = {"iou": 0.45, "imgsz": 640, "classes": CLASSES, "verbose": False}

    frames = []
    fi = 0
    t0 = time.time()
    while not args.max_frames or fi < args.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        rA = mA.track(f, persist=True, tracker="botsort.yaml", conf=0.30, **common)[0]
        rB = mB.track(f, persist=True, tracker="botsort.yaml", conf=0.10, **common)[0]
        rC = mC.predict(f, conf=0.05, **common)[0]
        g = cv2.cvtColor(cv2.resize(f, (sw, sh)), cv2.COLOR_BGR2GRAY)
        fg = cv2.morphologyEx(mog.apply(g), cv2.MORPH_OPEN, k)
        fg_road = cv2.bitwise_and(fg, rmask)
        n_fg = int((fg_road > 0).sum())
        blob = None
        if n_fg > 0:
            n, _lab, st, _ = cv2.connectedComponentsWithStats(fg_road, connectivity=8)
            if n > 1:
                j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                x, y, w, h, a = st[j]
                blob = [int(x) * 2, int(y) * 2, int(x + w) * 2, int(y + h) * 2, int(a) * 4]
        frames.append(
            {
                "i": fi,
                "ms": round(pts_ms, 1),
                "luma": round(float(g.mean()), 1),
                "A": boxes(rA, True),
                "B": boxes(rB, True),
                "C": boxes(rC, False),
                "fg": n_fg * 4,
                "blob": blob,
            }
        )
        fi += 1
        if fi % 1000 == 0:
            print(f"{wdir.name} {fi} frames {fi / (time.time() - t0):.1f} fps", flush=True)
    out = wdir / "detect.json"
    out.write_text(
        json.dumps(
            {
                "window": wdir.name,
                "fps": fps,
                "w": W,
                "h": H,
                "road_px": road_px * 4,
                "frames": frames,
            }
        )
    )
    print(f"done {wdir.name}: {fi} frames in {time.time() - t0:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
