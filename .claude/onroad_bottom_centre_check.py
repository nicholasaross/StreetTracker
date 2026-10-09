"""On-road test for the fullframe crop path: box centre vs bottom-centre (2026-10-09).

``rank_vehicle_candidates`` keeps a vehicle box only when its CENTRE is
inside the road polygon. A tall van in the far zone has its centre over the
houses behind the road, so it was dropped: a van arriving with its front
plate in plain view (track 80526, session_20261004_095219) got no plate
detection at all.

Testing the bottom-centre instead was tried first and REJECTED: on
session_20261004_095219 it changed the candidate list on 3,077 / 14,986
snaps, and 693 of its 1,209 new or changed reads sat at a static
(parked-plate) spot or were read on >= 3 other tracks -- cars parked at the
kerb have their wheels on the road edge, so they took candidate slots from
the tracked car (one track's 0.96 read became another car's plate).

What shipped instead is a RETRY: keep the centre rule, and only when neither
its candidates nor the hint crop hold a plate, try vehicles whose
bottom-centre is on the road AND that overlap the hint (IoU >= 0.1). That
only touches snaps that read nothing today. Its reads carry
``bottom_retry: true``, and the static filter checks them but doesn't learn
spots from them. Each guard was added after a measured failure on
session_20261004_095219:

* retry with no overlap check: +14 / -9 tracks passing the plate gate. It
  read cars parked mid-street, and the static filter's new spots from those
  reads flagged moving cars' plates (one was read at two positions);
* + IoU >= 0.1 (every correct new read had IoU 0.13-0.35): +13 / -4;
* + retry reads not seeding spots: +13 / -0. 12 of the 13 are the tracked
  vehicle's plate, mostly vans and SUVs near the camera, whose box centre
  sits over the pavement. The wrong one is a bicycle track that read a
  parked Royal Mail van. session_20260930_211033: +0 / -0, because the
  static and colour filters caught all 34 retry reads.

The van itself (80526) now gets a 6-character read at 0.84 (one character
dropped), which isn't UK-shaped, so it still doesn't pass the gate.

This script measures it on one enriched session:

1. from the cached ``_vehicle_boxes.json`` (the same YOLOv8m@1920 conf 0.2
   the crop path runs, plus motorcycles), pick the snaps that had no plate
   detection, have a hint, and have a bottom-only candidate to retry;
2. re-run the real preferred pipeline (``alpr-run`` defaults: fullframe
   crop, yolo-v9-t-640 plates, ghost mask, motion-window hint) on them with
   the current code;
3. put the baseline (production ``_alpr.json``) and the variant (those snaps
   replaced) through ``alpr-run``'s own post-processing -- static-plate
   filter, plate-colour check, by-track rollup -- and compare the tracks
   whose best read passes the live plate gate (``configs/alpr.json``, with
   cross-session support from every rollup under the output root).

Usage:
    uv run python .claude/onroad_bottom_centre_check.py output/<session>
        [--montage out.jpg] [--json out.json]
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import streettracker.cli.alpr_run as ar
from streettracker.analysis.alpr.fullframe import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RETRY_MIN_HINT_IOU,
    TrajectoryCropDetector,
    _iou,
    load_road_polygon,
    rank_vehicle_candidates,
)
from streettracker.analysis.alpr.gate import load_plate_support, plate_gate, read_passes
from streettracker.analysis.alpr.plate_colour import mark_colour_suspects
from streettracker.analysis.alpr.preferred import FastPlateOcrRecognizer, OpenImageModelsDetector
from streettracker.analysis.alpr.runner import PipelineRunner
from streettracker.analysis.alpr.staticfilter import find_static_spots, mark_static_suspects
from streettracker.analysis.dvsa import is_canonical_uk_plate

SNAP_W, SNAP_H = 4512, 2512
DERIVED = ("static_suspect", "colour_suspect", "plate_colour")


def post_process(records, session_dir, bbox_index, done_index, sub_size, directions):
    """alpr-run's post-loop on a copy: static filter, colour check, rollup."""
    recs = copy.deepcopy(records)
    for r in recs:
        for k in DERIVED:
            r.pop(k, None)
    spots, consistent = find_static_spots(
        recs, {**bbox_index, **done_index}, sub_size, (SNAP_W, SNAP_H)
    )
    mark_static_suspects(recs, spots, consistent)
    # A session dir with no alpr_crops/ so every read uses its recorded
    # crop_path (new reads' crops live outside the session).
    mark_colour_suspects(recs, session_dir / "_no_such_dir", directions)
    rollup = ar._rollup_by_track(recs)
    best = {
        t["track_id"]: t["best_preferred"] for t in rollup["tracks"] if "best_preferred" in t
    }
    return recs, best


def support_for(base_support, old_best, new_best):
    """Cross-session plate support with this session's rollup swapped."""
    s = Counter(base_support)
    for best, sign in ((old_best, -1), (new_best, 1)):
        for b in best.values():
            p = str(b.get("ocr_text") or "").replace(" ", "").upper()
            if p and is_canonical_uk_plate(p):
                s[p] += sign
    return s


def passing(best, gate, support):
    return {
        tid: b["ocr_text"]
        for tid, b in best.items()
        if b.get("ocr_text")
        and is_canonical_uk_plate(str(b["ocr_text"]))
        and read_passes(gate, b, support)
    }


def montage(path, gained, var_best, var_recs, sd):
    tiles = []
    for t, p in sorted(gained.items()):
        b = var_best[t]
        rec = next(x for x in var_recs if x["image"] == b["image"])
        im = cv2.imread(rec["image_path"].replace("\\", "/"))
        x1, y1, x2, y2 = rec["det_bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ctx = im[max(0, cy - 260) : cy + 140, max(0, cx - 350) : cx + 350]
        ctx = cv2.resize(ctx, (420, int(ctx.shape[0] * 420 / ctx.shape[1])))
        hq = cv2.imread(str(sd / f"vehicle_{t}_hq.jpg"))
        if hq is None:
            hq = np.full((ctx.shape[0], 200, 3), 200, np.uint8)
        hq = cv2.resize(hq, (int(hq.shape[1] * ctx.shape[0] / hq.shape[0]), ctx.shape[0]))
        tile = np.hstack([ctx, hq])
        tile = cv2.copyMakeBorder(tile, 26, 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(
            tile, f"{t} {p} {b['ocr_conf']:.2f}", (4, 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 160), 2,
        )
        tiles.append(cv2.resize(tile, (640, int(tile.shape[0] * 640 / tile.shape[1]))))
    rows = []
    for k in range(0, len(tiles), 2):
        chunk = tiles[k : k + 2]
        hh = max(x.shape[0] for x in chunk)
        chunk = [
            cv2.copyMakeBorder(x, 0, hh - x.shape[0], 0, 4, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
            for x in chunk
        ]
        while len(chunk) < 2:
            chunk.append(np.full((hh, 644, 3), 255, np.uint8))
        rows.append(np.hstack(chunk))
    cv2.imwrite(str(path), np.vstack(rows))
    print(f"[check] montage (4K context | tracker HQ crop) -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--road-polygon", type=Path, default=Path(".claude/triggers_proposal.json"))
    ap.add_argument("--ghost-mask", type=Path, default=Path(".claude/ghost_mask.json"))
    ap.add_argument("--montage", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    sd = args.session_dir
    label = sd.name
    prod = [
        r
        for r in json.loads((sd / f"{label}_alpr.json").read_text())
        if r.get("pipeline") == "preferred"
    ]
    boxes_by_img = json.loads((sd / f"{label}_vehicle_boxes.json").read_text())["boxes"]
    data = {r["track_id"]: r for r in json.loads((sd / f"{label}_data.json").read_text())}
    directions = {tid: str(r.get("direction") or "") for tid, r in data.items()}
    poly = load_road_polygon(args.road_polygon)
    assert poly, "road polygon needed"
    bbox_index, sub_size = ar._load_bbox_index(sd)
    done_index = ar._load_done_bbox_index(sd)
    ghost_rects, ghost_src = ar._load_ghost_mask(args.ghost_mask)

    # 1. snaps the retry can reach.
    todo = []
    for i, r in enumerate(prod):
        if r.get("det_bbox") is not None or r.get("error"):
            continue
        cached = boxes_by_img.get(r["image"])
        if not cached:
            continue
        hint = ar._resolve_bbox_hint_window(
            Path(r["image_path"]), r["track_id"], r["snap_index"], bbox_index, sub_size,
            lookahead=3, done_index=done_index,
        )
        if hint is None:
            continue
        b4 = [tuple(b[:4]) for b in cached]
        kw = {"bbox_hint": hint, "road_polygon_frac": poly}
        tried = rank_vehicle_candidates(b4, SNAP_W, SNAP_H, **kw)[:DEFAULT_MAX_CANDIDATES]
        retry = [
            b
            for b in rank_vehicle_candidates(b4, SNAP_W, SNAP_H, on_road_point="bottom", **kw)
            if b not in tried and _iou(b, hint) >= DEFAULT_RETRY_MIN_HINT_IOU
        ]
        if retry:
            todo.append((i, hint))
    n_nodet = sum(1 for r in prod if r.get("det_bbox") is None and not r.get("error"))
    print(
        f"[check] {label}: {len(prod)} snaps, {n_nodet} with no plate detection, "
        f"{len(todo)} reachable by the bottom-centre retry"
    )

    # 2. re-run those snaps with the current code.
    runner = PipelineRunner(
        name="preferred",
        detector=TrajectoryCropDetector(
            plate_detector=OpenImageModelsDetector("yolo-v9-t-640-license-plate-end2end"),
            road_polygon_frac=poly,
        ),
        recognizer=FastPlateOcrRecognizer(),
    )
    crop_dir = Path(".claude/onroad_check_crops") / label
    variant = list(prod)
    n_det = n_read = 0
    for k, (i, hint) in enumerate(todo, 1):
        r = prod[i]
        res = runner.run(
            Path(r["image_path"]), r["track_id"], r["snap_index"], r["class_name"], crop_dir,
            bbox_hint=hint, ghost_rects=ghost_rects, ghost_source_size=ghost_src,
        ).to_json()
        n_det += res.get("det_bbox") is not None
        n_read += bool(res.get("ocr_text"))
        variant[i] = res
        if k % 100 == 0:
            print(f"  {k}/{len(todo)}")
    print(f"[check] retry: {n_det} plate detections, {n_read} reads on {len(todo)} snaps")

    # 3. alpr-run post-processing on both, then the plate gate.
    base_recs, base_best = post_process(prod, sd, bbox_index, done_index, sub_size, directions)
    var_recs, var_best = post_process(variant, sd, bbox_index, done_index, sub_size, directions)
    retry_reads = [var_recs[i] for i, _h in todo if var_recs[i].get("ocr_text")]
    print(
        f"[check] of the retry's {len(retry_reads)} reads: "
        f"{sum(bool(r.get('static_suspect')) for r in retry_reads)} static_suspect "
        f"(parked spot), {sum(bool(r.get('colour_suspect')) for r in retry_reads)} "
        f"colour_suspect"
    )
    print(
        f"[check] static_suspect reads overall: "
        f"{sum(bool(r.get('static_suspect')) for r in base_recs)} -> "
        f"{sum(bool(r.get('static_suspect')) for r in var_recs)}"
    )

    gate = plate_gate()
    support0 = load_plate_support(sd.parent)
    base_pass = passing(base_best, gate, support0)
    var_pass = passing(var_best, gate, support_for(support0, base_best, var_best))
    gained = {t: p for t, p in var_pass.items() if t not in base_pass}
    lost = {t: p for t, p in base_pass.items() if t not in var_pass}
    changed = {
        t: (base_pass[t], p) for t, p in var_pass.items() if t in base_pass and base_pass[t] != p
    }
    best_changed = sum(
        1
        for t in set(base_best) | set(var_best)
        if (base_best.get(t) or {}).get("ocr_text") != (var_best.get(t) or {}).get("ocr_text")
    )
    print(f"[check] gate: {gate.describe()}")
    print(f"[check] tracks whose best read changed: {best_changed}")
    print(
        f"[check] tracks passing the gate: {len(base_pass)} -> {len(var_pass)} "
        f"(+{len(gained)} gained, -{len(lost)} lost, {len(changed)} changed plate)"
    )
    print(
        "[check]   gained by class: "
        f"{dict(Counter(data.get(t, {}).get('class_name', '?') for t in gained))}"
    )
    known = set()
    for f in sd.parent.glob("session_*/session_*_dvsa_labels.json"):
        try:
            labels = json.loads(f.read_text()).get("labels", {})
        except (OSError, json.JSONDecodeError):
            continue
        known |= {p for p, v in labels.items() if isinstance(v, dict) and v.get("make")}
    here = Counter(b["ocr_text"] for b in var_best.values() if b.get("ocr_text"))
    for t, p in sorted(gained.items()):
        b = var_best[t]
        rec = data.get(t, {})
        print(
            f"  +{t} {rec.get('class_name', '?'):5s} {(rec.get('direction') or '?')[:5]:5s} "
            f"{p} conf={b['ocr_conf']} n_agree={b.get('n_agree')} on_register={p in known} "
            f"tracks_here={here.get(p, 0)} snap={b.get('image')}"
        )
    for t, p in lost.items():
        print(f"  -{t} {p}")
    for t, (a, b) in changed.items():
        print(f"  ~{t} {a} -> {b}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "gained": gained,
                    "lost": lost,
                    "changed": changed,
                    "retry_snaps": [prod[i]["image"] for i, _h in todo],
                },
                indent=1,
            )
        )
    if args.montage and gained:
        montage(args.montage, gained, var_best, var_recs, sd)


if __name__ == "__main__":
    main()
