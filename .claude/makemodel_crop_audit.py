"""Audit the make/colour/body-type training crops: is the labelled car in them?

`makemodel-build-uk` crops each 4K snap around the FIRE-TIME sub-stream
bbox (`snap_assets.resolve_bbox_hint`). The snap lands ~0.7 s after the
fire decision, so the car may have left that box -- the same stale-bbox
failure that capped R->L ANPR until the 2026-07-28 fullframe crop path.

Ground truth (plate-anchored): a corpus crop's source snap is
"verifiable" when that snap's own `_alpr.json` record read the car's
labelled plate (fuzzy >= 85, not a static/parked suspect). The plate bbox
then pins which physical vehicle is the labelled car: the full-frame YOLO
vehicle box containing the plate centre.

Per verifiable snap we measure:
  * coverage  = |true car  crop| / |true car|  for the builder's crop box
                (pad_frac 0.1 = the build CLI default, and 0.25);
  * fullframe = does the candidate fix (nearest on-road vehicle to the
                stale hint, TrajectoryCropDetector's rank-0 rule) pick
                the true car (IoU >= 0.5)?

Also counts, over the WHOLE corpus (no image IO), how many crops/cars
are plate-anchored -- the size of a "certain-identity" training set.

Usage:
    uv run python .claude/makemodel_crop_audit.py [--corpus runs/uk_crops_0730_576] [--n 800]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from streettracker.analysis.alpr.fullframe import (
    DEFAULT_MIN_VEHICLE_H_PX,
    DEFAULT_VEHICLE_IMGSZ,
    _point_in_polygon,
)
from streettracker.analysis.snap_assets import load_bbox_index, resolve_bbox_hint

_NAME_RE = re.compile(
    r"^(?P<car>.+)_(?P<session>session_\d{8}_\d{6})_(?P<tid>\d+)_(?P<n>\d+)\.jpg$"
)
_VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck (fullframe.py's set)


def _crop_box(hint, pad_frac, w, h):
    x1, y1, x2, y2 = hint
    pad = int(round(pad_frac * max(x2 - x1, y2 - y1)))
    return (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _iou(a, b):
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def _plate_matches(read: str | None, plate: str) -> bool:
    if not read:
        return False
    read = read.replace(" ", "").upper()
    return read == plate or (len(read) == len(plate) and fuzz.ratio(read, plate) >= 85)


def _load_alpr(session_dir: Path) -> dict[tuple[int, int], list[dict]]:
    p = session_dir / f"{session_dir.name}_alpr.json"
    out: dict[tuple[int, int], list[dict]] = defaultdict(list)
    if not p.exists():
        return out
    for r in json.loads(p.read_text()):
        if r.get("track_id") is None or r.get("snap_index") is None:
            continue
        out[(int(r["track_id"]), int(r["snap_index"]))].append(r)
    return out


def _anchor(recs: list[dict], plate: str) -> list[float] | None:
    """Plate bbox of the best non-static read of ``plate`` on this snap."""
    best = None
    for r in recs:
        if r.get("static_suspect") or not r.get("det_bbox"):
            continue
        if _plate_matches(r.get("ocr_text"), plate) and (
            best is None or (r.get("ocr_conf") or 0) > (best.get("ocr_conf") or 0)
        ):
            best = r
    return best["det_bbox"] if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("runs/uk_crops_0730_576"))
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--road-polygon", type=Path, default=Path(".claude/triggers_proposal.json"))
    ap.add_argument("--n", type=int, default=800, help="verifiable snaps to image-audit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="per-snap results JSON")
    args = ap.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    samples = manifest["samples"]
    poly_frac = json.loads(args.road_polygon.read_text()).get("vertices_frac")

    # ---- pass 1: whole-corpus plate-anchor coverage (JSON only) ----
    alpr_cache: dict[str, dict] = {}
    bbox_cache: dict[str, tuple] = {}
    parsed = []
    for s in samples:
        m = _NAME_RE.match(Path(s["path"]).name)
        if not m:
            continue
        sess = m["session"]
        if sess not in alpr_cache:
            sd = args.output_root / sess
            alpr_cache[sess] = _load_alpr(sd)
            bbox_cache[sess] = load_bbox_index(sd)
        tid, n = int(m["tid"]), int(m["n"])
        anchor = _anchor(alpr_cache[sess].get((tid, n), []), m["car"])
        parsed.append((s, m["car"], sess, tid, n, anchor))

    n_all = len(parsed)
    anchored = [p for p in parsed if p[5] is not None]
    cars_all = {p[1] for p in parsed}
    cars_anch = {p[1] for p in anchored}
    makes_anch = Counter()
    car_make = {p[0]["car"]: p[0]["make"] for p in parsed}
    for car in cars_anch:
        makes_anch[car_make[car]] += 1
    print(f"corpus {args.corpus.name}: {n_all} crops / {len(cars_all)} cars")
    print(
        f"  plate-anchored (snap read its own plate): {len(anchored)} crops "
        f"({len(anchored) / n_all:.1%}) / {len(cars_anch)} cars "
        f"({len(cars_anch) / len(cars_all):.1%}); "
        f"makes with >=5 anchored cars: {sum(1 for v in makes_anch.values() if v >= 5)}"
    )

    # ---- pass 2: image audit on a random verifiable sample ----
    import cv2
    from ultralytics import YOLO

    yolo = YOLO("yolov8m.pt")
    rng = random.Random(args.seed)
    pick = rng.sample(anchored, min(args.n, len(anchored)))

    rows = []
    t0 = time.time()
    for i, (s, car, sess, tid, n, plate_bb) in enumerate(pick, 1):
        img_path = None
        for prefix in ("vehicle", "person"):
            cand = args.output_root / sess / f"{prefix}_{tid}_main_{n}.jpg"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            rows.append({"path": s["path"], "status": "missing_image"})
            continue
        bbox_index, sub_size = bbox_cache[sess]
        hint = resolve_bbox_hint(img_path, tid, n, bbox_index, sub_size)
        img = cv2.imread(str(img_path))
        if img is None or hint is None:
            rows.append({"path": s["path"], "status": "no_image_or_hint"})
            continue
        h, w = img.shape[:2]
        res = yolo.predict(
            img, classes=_VEHICLE_CLASSES, conf=0.2, imgsz=DEFAULT_VEHICLE_IMGSZ, verbose=False
        )[0]
        boxes = [tuple(float(v) for v in b[:4]) for b in res.boxes.xyxy.cpu().numpy()]

        pcx, pcy = (plate_bb[0] + plate_bb[2]) / 2, (plate_bb[1] + plate_bb[3]) / 2
        containing = [b for b in boxes if b[0] <= pcx <= b[2] and b[1] <= pcy <= b[3]]
        if not containing:
            rows.append({"path": s["path"], "status": "no_vehicle_at_plate"})
            continue
        true_car = min(containing, key=_area)  # tightest box around the plate

        row = {"path": s["path"], "session": sess, "status": "ok", "car": car, "make": s["make"]}
        for pf in (0.1, 0.25):
            crop = _crop_box(hint, pf, w, h)
            row[f"cov_{pf}"] = round(_inter(true_car, crop) / _area(true_car), 3)
            row[f"plate_in_{pf}"] = crop[0] <= pcx <= crop[2] and crop[1] <= pcy <= crop[3]

        # Candidate fix: TrajectoryCropDetector's rank-0 rule.
        hx, hy = (hint[0] + hint[2]) / 2, (hint[1] + hint[3]) / 2
        poly = [(px * w, py * h) for px, py in poly_frac] if poly_frac else None
        cands = []
        for b in boxes:
            if b[3] - b[1] < DEFAULT_MIN_VEHICLE_H_PX:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if poly is not None and not _point_in_polygon(cx, cy, poly):
                continue
            cands.append((((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5, b))
        cands.sort(key=lambda c: c[0])
        row["ff_found"] = bool(cands)
        row["ff_correct"] = bool(cands) and _iou(cands[0][1], true_car) >= 0.5
        row["ff_top2"] = any(_iou(b, true_car) >= 0.5 for _, b in cands[:2])
        row["hint_offset_px"] = round(
            (
                (hx - (true_car[0] + true_car[2]) / 2) ** 2
                + (hy - (true_car[1] + true_car[3]) / 2) ** 2
            )
            ** 0.5
        )
        row["car_w_px"] = round(true_car[2] - true_car[0])
        rows.append(row)
        if i % 100 == 0:
            print(f"  {i}/{len(pick)}  ({time.time() - t0:.0f}s)", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    print(
        f"\nimage audit: {len(pick)} sampled, {len(ok)} measurable; "
        f"skipped {Counter(r['status'] for r in rows if r['status'] != 'ok')}"
    )
    if not ok:
        return 1

    def bucket(c):
        return (
            "good (>=80% of car)"
            if c >= 0.8
            else ("partial (30-80%)" if c >= 0.3 else "miss (<30%)")
        )

    for pf in (0.1, 0.25):
        b = Counter(bucket(r[f"cov_{pf}"]) for r in ok)
        plate_in = sum(r[f"plate_in_{pf}"] for r in ok) / len(ok)
        print(f"\ncurrent builder crop, pad_frac={pf}:")
        for k in ("good (>=80% of car)", "partial (30-80%)", "miss (<30%)"):
            print(f"  {k:22s} {b[k] / len(ok):6.1%}  (n={b[k]})")
        print(f"  labelled plate inside crop: {plate_in:.1%}")

    offs = sorted(r["hint_offset_px"] for r in ok)
    ratio = sorted(r["hint_offset_px"] / max(1, r["car_w_px"]) for r in ok)
    print(
        f"\nstale-hint centre offset from true car: median {offs[len(offs) // 2]} px "
        f"(p25 {offs[len(offs) // 4]}, p75 {offs[3 * len(offs) // 4]}); "
        f"median {ratio[len(ratio) // 2]:.2f} car-widths"
    )

    print("\ncandidate fix (fullframe nearest on-road vehicle to hint):")
    print(f"  rank-0 = true car: {sum(r['ff_correct'] for r in ok) / len(ok):.1%}")
    print(f"  true car in top-2: {sum(r['ff_top2'] for r in ok) / len(ok):.1%}")
    print(f"  no on-road candidate: {sum(not r['ff_found'] for r in ok) / len(ok):.1%}")

    # By session era (pre/post 2026-06-13 completion-bbox deploy is irrelevant to
    # the builder -- it only uses fire bboxes -- but month shows drift).
    by_month = defaultdict(list)
    for r in ok:
        by_month[r["session"][8:14]].append(r)
    print("\nby month (good @0.1 / fullframe rank-0 correct):")
    for mth in sorted(by_month):
        rs = by_month[mth]
        print(
            f"  {mth}: n={len(rs):4d}  good {sum(r['cov_0.1'] >= 0.8 for r in rs) / len(rs):6.1%}  "
            f"ff {sum(r['ff_correct'] for r in rs) / len(rs):6.1%}"
        )

    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"\nper-snap rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
