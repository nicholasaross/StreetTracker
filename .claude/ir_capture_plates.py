"""Full-frame plate pass over one ``ir_capture.py`` window's periodic 4K snaps.

Step 2 of the capture analysis (review R16 "Results", 2026-10-05). The snaps
fire every 2 s whatever the detector does, so every car that passes gets a
couple of views. Per snap, as ``alpr-run --crop-mode fullframe`` would:

1. zero the parked-car ghost-mask rect (``.claude/ghost_mask.json``);
2. YOLOv8m @1920 vehicle boxes (car/motorcycle/bus/truck, conf 0.2);
3. for EVERY on-road box >= 45 px tall (not just the top two): plate detection
   (yolo-v9-t-640, the alpr-run default) on the box padded 30 px, OCR
   (fast-plate-ocr, min-character confidence) on the plate padded 10 %, plate
   colour, and plate-pixel stats (``lowlight_plates.plate_pixels``).

    uv run python .claude/ir_capture_plates.py <capture>/1005_0700

Writes ``<window>/plates.json`` (next to the capture: it holds plate text, so
never into the repo). ~2-3 min per 600 snaps on the dev-box 3080.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from streettracker.analysis.alpr.base import crop_with_padding
from streettracker.analysis.alpr.plate_colour import classify_plate_colour
from streettracker.analysis.alpr.preferred import FastPlateOcrRecognizer, OpenImageModelsDetector
from streettracker.analysis.dvsa import is_canonical_uk_plate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowlight_plates import plate_pixels  # noqa: E402

VEHICLES = [2, 3, 5, 7]
MIN_VEHICLE_H = 45
VEHICLE_PAD = 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("window", type=Path)
    args = ap.parse_args()
    wdir: Path = args.window

    meta = json.loads((wdir / "capture.json").read_text())
    poly = json.loads((REPO / ".claude/road_polygon_user.json").read_text())["vertices_frac"]
    ghost = json.loads((REPO / ".claude/ghost_mask.json").read_text())
    yolo = YOLO(str(REPO / "yolov8m.pt"))
    det = OpenImageModelsDetector("yolo-v9-t-640-license-plate-end2end")
    ocr = FastPlateOcrRecognizer()
    rows = []
    t0 = time.time()
    for k, s in enumerate(meta["snaps"]):
        img = cv2.imread(str(wdir / "snaps" / s["file"]))
        if img is None:
            rows.append({**s, "error": "imread"})
            continue
        H, W = img.shape[:2]
        sx, sy = W / ghost["source_size"][0], H / ghost["source_size"][1]
        for x1, y1, x2, y2 in ghost["rects_4k"]:
            img[int(y1 * sy) : int(y2 * sy), int(x1 * sx) : int(x2 * sx)] = 0
        pts = np.array([[x * W, y * H] for x, y in poly], np.float32)
        r = yolo.predict(img, classes=VEHICLES, conf=0.2, imgsz=1920, verbose=False)[0]
        vehicles = []
        for (bx1, by1, bx2, by2), c, cl in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            strict=True,
        ):
            centre = (float((bx1 + bx2) / 2), float((by1 + by2) / 2))
            on_road = cv2.pointPolygonTest(pts, centre, False) >= 0
            v = {
                "box": [int(bx1), int(by1), int(bx2), int(by2)],
                "conf": round(float(c), 3),
                "cls": int(cl),
                "on_road": bool(on_road),
            }
            if on_road and by2 - by1 >= MIN_VEHICLE_H:
                px1, py1 = max(0, int(bx1 - VEHICLE_PAD)), max(0, int(by1 - VEHICLE_PAD))
                px2, py2 = min(W, int(bx2 + VEHICLE_PAD)), min(H, int(by2 + VEHICLE_PAD))
                d = det.detect(img[py1:py2, px1:px2])
                if d is not None:
                    pb = (
                        int(d.bbox[0] + px1),
                        int(d.bbox[1] + py1),
                        int(d.bbox[2] + px1),
                        int(d.bbox[3] + py1),
                    )
                    v["plate"] = {"bbox": list(pb), "det_conf": round(float(d.det_confidence), 3)}
                    if pb[2] - pb[0] >= 8 and pb[3] - pb[1] >= 6:
                        crop = crop_with_padding(img, pb, pad_frac=0.10)
                        rd = ocr.recognize(crop)
                        if rd is not None:
                            v["plate"].update(
                                {
                                    "text": rd.text,
                                    "conf": round(float(rd.ocr_confidence), 4),
                                    "canon": bool(is_canonical_uk_plate(rd.text)),
                                }
                            )
                        v["plate"]["colour"] = classify_plate_colour(crop).label
                        v["plate"].update(plate_pixels(img, pb))
            vehicles.append(v)
        g = cv2.cvtColor(cv2.resize(img, (W // 8, H // 8)), cv2.COLOR_BGR2GRAY)
        rows.append({**s, "luma": round(float(g.mean()), 1), "vehicles": vehicles})
        if (k + 1) % 100 == 0:
            print(
                f"{wdir.name} {k + 1}/{len(meta['snaps'])} {(k + 1) / (time.time() - t0):.1f}/s",
                flush=True,
            )

    on = [(s, v) for s in rows for v in s.get("vehicles", []) if v["on_road"]]
    on = [(s, v) for s, v in on if v["box"][3] - v["box"][1] >= MIN_VEHICLE_H]
    found = [v for _s, v in on if "plate" in v]
    read = [v for v in found if v["plate"].get("canon") and v["plate"].get("conf", 0) >= 0.90]
    out = wdir / "plates.json"
    out.write_text(json.dumps({"window": wdir.name, "snaps": rows}))
    print(
        f"done {wdir.name} in {time.time() - t0:.0f} s: {len(rows)} snaps, "
        f"{len({s['file'] for s, _v in on})} with an on-road vehicle, {len(on)} vehicle views, "
        f"{len(found)} plates found, {len(read)} read (>= 0.90, UK-shaped) -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
