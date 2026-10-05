"""Pixelate number plates inside photos on the arch_capture.py screenshots, then export them.

For every <img> rect saved by arch_capture.py, crop it from the screenshot, run the project's
plate detector (open-image-models YOLOv9-t-640) at 1-4x scale, plus tiled passes over the page at
three tile sizes, and pixelate the union of hits. Hits are kept only if plate-shaped, inside a
photo, and not already under a yellow text bar. Manual boxes come last:
.claude/arch_work/manual_boxes.json, {page: [[x1, y1, x2, y2], ...]} in CSS px.

The detector misses motion-blurred and oblique front plates. After every capture, LOOK at each
redacted image at full resolution (crop it into ~1400 px tiles) and add manual boxes for misses
before publishing. The boxes are specific to one capture.

    uv run --no-sync python .claude/arch_redact.py [page ...]   # redact + export to docs/architecture/assets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from open_image_models import LicensePlateDetector
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / ".claude" / "arch_work"
SHOTS = WORK / "shots"
OUT = WORK / "shots_red"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = REPO / "docs" / "architecture" / "assets"
EXPORT = {  # capture name -> published asset name
    "showcase_gallery": "site_gallery",
    "showcase_car": "site_car",
    "showcase_stats": "site_stats",
    "showcase_people": "site_people",
    "showcase_schedule": "site_schedule",
    "control_dashboard": "site_control",
    "control_training": "site_training",
    "orin_dashboard": "site_orin",
}

DET = LicensePlateDetector(
    detection_model="yolo-v9-t-640-license-plate-end2end",
    conf_thresh=0.12,
    providers=["CPUExecutionProvider"],
)


def detect(img: Image.Image) -> list[tuple[float, float, float, float, float]]:
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()  # BGR like cv2
    out = []
    for d in DET.predict(arr) or []:
        bb = d.bounding_box
        out.append((bb.x1, bb.y1, bb.x2, bb.y2, float(d.confidence)))
    return out


def pixelate(im: Image.Image, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(im.width, x2), min(im.height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    region = im.crop((x1, y1, x2, y2))
    small = region.resize((max(1, (x2 - x1) // 10), max(1, (y2 - y1) // 10)), Image.BILINEAR)
    im.paste(small.resize(region.size, Image.NEAREST), (x1, y1))


def pad_box(x1, y1, x2, y2, frac=0.35, minpad=6):
    w, h = x2 - x1, y2 - y1
    px, py = max(minpad, w * frac), max(minpad, h * frac * 1.6)
    return int(x1 - px), int(y1 - py), int(x2 + px), int(y2 + py)


def run(name: str, manual: dict) -> None:
    meta = json.loads((SHOTS / f"{name}.json").read_text())
    dsf = meta["dsf"]
    im = Image.open(SHOTS / f"{name}.png").convert("RGB")
    hits = []
    for r in meta["imgs"]:
        x1, y1 = int(r["x"] * dsf), int(r["y"] * dsf)
        x2, y2 = int((r["x"] + r["w"]) * dsf), int((r["y"] + r["h"]) * dsf)
        x2, y2 = min(x2, im.width), min(y2, im.height)
        if x2 - x1 < 40 or y2 - y1 < 30:
            continue
        crop = im.crop((x1, y1, x2, y2))
        for scale in (1.0, 2.0, 3.0, 4.0):
            if crop.width * scale > 2600:
                continue
            up = crop if scale == 1.0 else crop.resize(
                (int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
            for bx1, by1, bx2, by2, c in detect(up):
                hits.append((x1 + bx1 / scale, y1 + by1 / scale, x1 + bx2 / scale,
                             y1 + by2 / scale, c, "img"))
    # Tiled whole-page second net.
    for T, O in ((640, 160), (900, 250), (1100, 300)):
        for ty in range(0, im.height, T - O):
            for tx in range(0, im.width, T - O):
                tile = im.crop((tx, ty, min(im.width, tx + T), min(im.height, ty + T)))
                if tile.width < 200 or tile.height < 200:
                    continue
                for bx1, by1, bx2, by2, c in detect(tile):
                    hits.append((tx + bx1, ty + by1, tx + bx2, ty + by2, c, "tile"))
    red = [[v * dsf for v in r] for r in meta.get("red_rects", [])]

    def covered(h):
        x1, y1, x2, y2 = h[:4]
        a = max(1.0, (x2 - x1) * (y2 - y1))
        for rx1, ry1, rx2, ry2 in red:
            ix = max(0.0, min(x2, rx2 + 8) - max(x1, rx1 - 8))
            iy = max(0.0, min(y2, ry2 + 8) - max(y1, ry1 - 8))
            if ix * iy / a > 0.4:
                return True
        return False

    def plate_shaped(h):
        w, hh = h[2] - h[0], h[3] - h[1]
        return 6 <= w <= 230 and 3 <= hh <= 110 and 1.2 <= w / max(1.0, hh) <= 8.0

    dropped = [h for h in hits if covered(h) or not plate_shaped(h)]
    rects = [(r["x"] * dsf, r["y"] * dsf, (r["x"] + r["w"]) * dsf, (r["y"] + r["h"]) * dsf) for r in meta["imgs"]]

    def in_photo(h):
        cx, cy = (h[0] + h[2]) / 2, (h[1] + h[3]) / 2
        return any(a <= cx <= c and b <= cy <= d for a, b, c, d in rects)

    hits = [h for h in hits if not covered(h) and plate_shaped(h) and in_photo(h)]
    print('   dropped', [(int(h[2]-h[0]), int(h[3]-h[1]), round(h[4], 2), h[5]) for h in dropped])
    for x1, y1, x2, y2, c, src in hits:
        pixelate(im, pad_box(x1, y1, x2, y2))
    for b in manual.get(name, []):  # CSS px [x1, y1, x2, y2]
        pixelate(im, tuple(int(v * dsf) for v in b))
    im.save(OUT / f"{name}.png")
    (OUT / f"{name}.hits.json").write_text(json.dumps(hits))
    print(name, "plate hits:", len(hits), "img:", sum(1 for h in hits if h[5] == "img"),
          "manual:", len(manual.get(name, [])), flush=True)


def export(name: str) -> None:
    """1600 px WebP for the pop-up, plus a 640 px top crop for the card thumbnail."""
    im = Image.open(OUT / f"{name}.png").convert("RGB")
    w, h = im.size
    im.resize((1600, int(h * 1600 / w)), Image.LANCZOS).save(
        ASSETS / f"{EXPORT[name]}.webp", quality=80, method=6
    )
    im.crop((0, 0, w, int(w * 0.62))).resize((640, 397), Image.LANCZOS).save(
        ASSETS / f"{EXPORT[name]}_th.webp", quality=78, method=6
    )


if __name__ == "__main__":
    mf = WORK / "manual_boxes.json"
    manual = json.loads(mf.read_text()) if mf.exists() else {}
    names = sys.argv[1:] or sorted(p.stem for p in SHOTS.glob("*.png"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    for n in names:
        run(n, manual)
        if n in EXPORT:
            export(n)
