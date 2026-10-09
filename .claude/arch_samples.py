"""Build the redacted pipeline sample images for docs/architecture/ from one track.

Track 2676 of session_20261001_190826 (white Ford, L->R, 8 snaps, no pedestrians in view):
scene_overlay.jpg (road polygon, bands, triggers, ghost mask, fire-time vs landing box),
track_snaps.jpg, sample_hq.jpg and plate_masked.png. Plates are pixelated from the track's own
ALPR boxes plus detector hits, the ghost-mask region is pixelated in every frame, and the HQ crop
gets hand-placed boxes (the detector misses plates at that size). Look at every output before use.

    uv run --no-sync python .claude/arch_samples.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arch_redact import detect, pad_box, pixelate  # noqa: E402

REPO = HERE.parent
WORK = REPO / ".claude" / "arch_work"
SESS = REPO / "output" / "session_20261001_190826"
TID = 2676
OUT = REPO / "docs" / "architecture" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
SUB_W, SUB_H = 896, 512
W4, H4 = 4512, 2512
SX, SY = W4 / SUB_W, H4 / SUB_H

rec = next(r for r in json.load(open(SESS / "session_20261001_190826_data.json", encoding="utf-8"))
           if r["track_id"] == TID)
reads = {r["snap_index"]: r for r in json.load(open(SESS / "session_20261001_190826_alpr.json", encoding="utf-8"))
         if r["track_id"] == TID}
gate = json.load(open(REPO / ".claude" / "snap_gate.json"))
ghost = json.load(open(REPO / ".claude" / "ghost_mask.json"))
# Hand-placed boxes in the 3x-upscaled HQ crop: the car's rear plate and a parked car's plate.
HQ_BOXES = [(165, 440, 255, 505), (425, 95, 515, 145)]
manual: dict = {}

try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 64)
    FONT_S = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 52)
except OSError:
    FONT = FONT_S = ImageFont.load_default()


def redact_4k(im: Image.Image, snap: int) -> list:
    """Pixelate the tracked car's plate (from its own read) plus any detector hit."""
    boxes = []
    r = reads.get(snap)
    if r and r.get("det_bbox"):
        boxes.append(tuple(r["det_bbox"]))
    T, O = 640, 160
    for ty in range(0, im.height, T - O):
        for tx in range(0, im.width, T - O):
            tile = im.crop((tx, ty, min(im.width, tx + T), min(im.height, ty + T)))
            if tile.width < 200 or tile.height < 200:
                continue
            for x1, y1, x2, y2, c in detect(tile):
                w, h = x2 - x1, y2 - y1
                if 8 <= w <= 400 and 1.2 <= w / max(1, h) <= 8 and ty + y1 > 110:
                    boxes.append((tx + x1, ty + y1, tx + x2, ty + y2))
    for b in manual.get(str(snap), []):
        boxes.append(tuple(b))
    for g in ghost["rects_4k"]:
        pixelate(im, tuple(g))
    for b in boxes:
        pixelate(im, pad_box(*b, frac=0.25, minpad=10))
    return boxes


# ---- road geometry, computed exactly as RoadGate.from_config does, in sub-stream px ----
poly = [(fx * SUB_W, fy * SUB_H) for fx, fy in gate["polygon_frac"]]
n = len(poly)
cx = sum(p[0] for p in poly) / n
cy = sum(p[1] for p in poly) / n
sxx = sum((x - cx) ** 2 for x, _ in poly) / n
syy = sum((y - cy) ** 2 for _, y in poly) / n
sxy = sum((x - cx) * (y - cy) for x, y in poly) / n
tr, det = sxx + syy, sxx * syy - sxy * sxy
lam = tr / 2 + math.sqrt(max(0.0, tr * tr / 4 - det))
ex, ey = lam - syy, sxy
nrm = math.hypot(ex, ey)
ex, ey = ex / nrm, ey / nrm
if ey < 0:
    ex, ey = -ex, -ey
ts = [(x - cx) * ex + (y - cy) * ey for x, y in poly]
tmin, tmax = min(ts), max(ts)


def t_norm(px, py):
    return ((px - cx) * ex + (py - cy) * ey - tmin) / (tmax - tmin)


def line_at(tn):
    """Two far-apart points on the perpendicular line at absolute t_norm, in 4K px."""
    t = tmin + tn * (tmax - tmin)
    bx, by = cx + ex * t, cy + ey * t
    px, py = -ey, ex
    a = (bx - px * 2000, by - py * 2000)
    b = (bx + px * 2000, by + py * 2000)
    return [(a[0] * SX, a[1] * SY), (b[0] * SX, b[1] * SY)]


def band_poly(lo, hi):
    a1, a2 = line_at(lo)
    b1, b2 = line_at(hi)
    return [a1, a2, b2, b1]


def scene_overlay(snap: int) -> dict:
    im = Image.open(SESS / f"vehicle_{TID}_main_{snap}.jpg").convert("RGB")
    redact_4k(im, snap)
    poly4 = [(x * SX, y * SY) for x, y in poly]
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).polygon(poly4, fill=255)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # Base usable band, then per-direction pipeline bands.
    d.polygon(band_poly(0.30, 0.60), fill=(91, 157, 255, 80))   # reverse (L->R) pipeline band
    d.polygon(band_poly(0.10, 0.20), fill=(31, 182, 168, 80))   # forward (R->L) pipeline band
    lo, hi = gate["t_usable_frac"]
    for tp, dirn in zip(gate["trigger_t_prime"], gate["trigger_directions"]):
        tn = lo + tp * (hi - lo)
        col = (31, 182, 168, 255) if dirn == "forward" else (91, 157, 255, 255)
        d.line(line_at(tn), fill=col, width=7)
    for tn in (lo, hi):
        d.line(line_at(tn), fill=(245, 197, 24, 230), width=5)
    clipped = Image.new("RGBA", im.size, (0, 0, 0, 0))
    clipped.paste(layer, (0, 0), mask)
    base = im.convert("RGBA")
    base.alpha_composite(clipped)
    d = ImageDraw.Draw(base)
    d.line(poly4 + [poly4[0]], fill=(245, 197, 24, 255), width=8)
    gx1, gy1, gx2, gy2 = ghost["rects_4k"][0]
    d.rectangle((gx1, gy1, gx2, gy2), fill=(20, 20, 24, 255), outline=(255, 107, 107, 255), width=8)
    for k in range(gx1 - (gy2 - gy1), gx2, 28):
        d.line([(max(gx1, k), gy1 + max(0, gx1 - k)), (min(gx2, k + (gy2 - gy1)), min(gy2, gy1 + (gx2 - k)))],
               fill=(255, 107, 107, 150), width=3)
    d.text((gx1 - 60, gy2 + 14), "ghost mask", fill=(255, 140, 140, 255), font=FONT_S)
    i = snap_idx(snap)
    fb = [v * s for v, s in zip(rec["main_snap_bboxes"][i], (SX, SY, SX, SY))]
    db = [v * s for v, s in zip(rec["main_snap_bboxes_done"][i], (SX, SY, SX, SY))]
    dashed_rect(d, fb, (245, 158, 11, 255))
    d.rectangle(db, outline=(236, 240, 246, 255), width=8)
    d.text((fb[0] + 8, fb[3] + 8), "fire-time box", fill=(245, 170, 40, 255), font=FONT)
    d.text((db[0] + 8, db[1] - 76), "landing box", fill=(236, 240, 246, 255), font=FONT)
    pb = reads[snap]["det_bbox"]
    d.rectangle(pad_box(*pb, frac=0.25, minpad=10), outline=(245, 197, 24, 255), width=6)
    out = base.convert("RGB").resize((1800, int(1800 * H4 / W4)), Image.LANCZOS)
    out.save(OUT / "scene_overlay.jpg", quality=86)
    c = ((db[0] + db[2]) / 2 / SX, (db[1] + db[3]) / 2 / SY)
    return {"snap": snap, "t_norm_landing": round(t_norm(*c), 3)}


def dashed_rect(d, b, col, dash=34, gap=22, w=8):
    x1, y1, x2, y2 = b
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        L = math.hypot(bx - ax, by - ay)
        k = 0.0
        while k < L:
            e = min(L, k + dash)
            d.line([(ax + (bx - ax) * k / L, ay + (by - ay) * k / L),
                    (ax + (bx - ax) * e / L, ay + (by - ay) * e / L)], fill=col, width=w)
            k += dash + gap


SNAPS = sorted(int(p.stem.rsplit("_", 1)[1]) for p in SESS.glob(f"vehicle_{TID}_main_*.jpg"))


def snap_idx(snap: int) -> int:
    return SNAPS.index(snap)


def strip(snaps):
    tw = 900
    th = int(tw * H4 / W4)
    sheet = Image.new("RGB", (2 * tw + 12, 2 * th + 12), (15, 17, 21))
    for k, s in enumerate(snaps):
        im = Image.open(SESS / f"vehicle_{TID}_main_{s}.jpg").convert("RGB")
        redact_4k(im, s)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), ((k % 2) * (tw + 12), (k // 2) * (th + 12)))
    sheet.save(OUT / "track_snaps.jpg", quality=84)


def hq():
    im = Image.open(SESS / f"vehicle_{TID}_hq.jpg").convert("RGB")
    up = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
    hits = detect(up)
    for x1, y1, x2, y2, c in hits:
        pixelate(up, pad_box(x1, y1, x2, y2, frac=0.3, minpad=8))
    for b in HQ_BOXES:  # coarse 4x3 blocks: finer pixelation still hinted at characters
        r = up.crop(b)
        up.paste(r.resize((4, 3), Image.BILINEAR).resize(r.size, Image.NEAREST), b[:2])
    up.save(OUT / "sample_hq.jpg", quality=88)
    return {"hq_size": im.size, "hq_hits": len(hits)}


def plate_crop(snap: int):
    p = SESS / "alpr_crops" / "preferred" / f"vehicle_{TID}_main_{snap}.jpg"
    im = Image.open(p).convert("RGB")
    big = im.resize((im.width * 4, im.height * 4), Image.NEAREST)
    small = big.resize((max(1, big.width // 18), max(1, big.height // 6)), Image.BILINEAR)
    small.resize(big.size, Image.NEAREST).save(OUT / "plate_masked.png")
    return {"crop_size": im.size}


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    info = {"snaps": SNAPS}
    info["overlay"] = scene_overlay(5)
    strip([SNAPS[0], SNAPS[2], SNAPS[4], SNAPS[-2]])
    info["hq"] = hq()
    info["plate"] = plate_crop(8)
    info["read"] = {k: {kk: reads[k].get(kk) for kk in ("det_conf", "ocr_conf", "ocr_char_probs", "plate_colour")}
                    for k in reads}
    info["t_norm_done"] = [round(t_norm((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), 3) for b in rec["main_snap_bboxes_done"]]
    info["t_norm_fire"] = [round(t_norm((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), 3) for b in rec["main_snap_bboxes"]]
    (WORK / "sample_track.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "read"}, indent=1))
