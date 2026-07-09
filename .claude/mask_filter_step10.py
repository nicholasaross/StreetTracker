"""Step 10 (option A only): post-hoc mask of the parked-car ghost
region and re-aggregate per-car / aliasing-free read rates.

The parked FD61PVX car's plate detections clustered at
(853, 780) 4K-coords with width 87 / height 50 across 165 ghost
detections (computed 2026-05-26). The mask used here is a 200x140
px rect centred on that point -- generous enough to catch
OCR-variant ghosts (FD51PVX, FD61PWX, etc.) which are the same
physical plate misread differently.

Run from repo root:
    uv run python .claude/mask_filter_step10.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

SESSION = "session_20260525_200916"
SESSION_DIR = Path("output") / SESSION
CONF_THRESHOLD = 0.9
GHOST_TRACK_MIN = 5

# Parked-car mask in 4K snap coords. Centred on FD61PVX plate
# cluster centroid (853, 780). The rect is generous (200x140) to
# catch OCR-variant misreads of the same physical plate.
MASK_X1, MASK_Y1, MASK_X2, MASK_Y2 = 750, 700, 960, 860


def in_mask(bbox: list[int]) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return MASK_X1 <= cx <= MASK_X2 and MASK_Y1 <= cy <= MASK_Y2


def main() -> int:
    alpr_path = SESSION_DIR / f"{SESSION}_alpr.json"
    data_path = SESSION_DIR / f"{SESSION}_data.json"
    alpr = json.loads(alpr_path.read_text())
    data = json.loads(data_path.read_text())

    cars = [r for r in data if r.get("class_name") == "car"]
    car_tids = {r["track_id"] for r in cars}
    n_cars = len(cars)

    # Apply mask filter: any preferred-pipeline detection whose plate
    # bbox centre lies in the mask is wiped (ocr cleared). Counts how
    # many detections fall inside the mask, and what fraction of those
    # were ghost OCRs vs (rare) misclassified real plates.
    n_masked = 0
    n_masked_was_ghost = 0
    GHOSTS_PRE = {"FD61PVX", "FD61PWX", "FD51PVX", "FD61PYX", "FD61PXX"}
    filtered: list[dict] = []
    for r in alpr:
        r2 = dict(r)
        if (
            r.get("pipeline") == "preferred"
            and r.get("det_bbox")
            and in_mask(r["det_bbox"])
        ):
            n_masked += 1
            if r.get("ocr_text") in GHOSTS_PRE:
                n_masked_was_ghost += 1
            r2["det_bbox"] = None
            r2["det_conf"] = None
            r2["ocr_text"] = None
            r2["ocr_raw"] = None
            r2["ocr_conf"] = None
        filtered.append(r2)
    print(
        f"mask: {n_masked} detections dropped "
        f"({n_masked_was_ghost} were known ghost OCRs)"
    )

    # Per-track best preferred-pipe read after masking.
    by_track_best_conf: dict[int, float] = {}
    by_track_best_text: dict[int, str] = {}
    for r in filtered:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt:
            continue
        tid = r["track_id"]
        if conf > by_track_best_conf.get(tid, -1.0):
            by_track_best_conf[tid] = conf
            by_track_best_text[tid] = txt

    tracks_with_high_read = sum(
        1 for tid in car_tids
        if by_track_best_conf.get(tid, 0.0) >= CONF_THRESHOLD
    )

    # Recompute ghost distribution on the filtered set.
    plate_tracks: dict[str, set[int]] = defaultdict(set)
    for r in filtered:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt or conf < CONF_THRESHOLD:
            continue
        plate_tracks[txt].add(r["track_id"])
    plate_counts = Counter({p: len(tids) for p, tids in plate_tracks.items()})
    ghosts_post = {p for p, n in plate_counts.items() if n >= GHOST_TRACK_MIN}

    # Aliasing-free: any non-ghost high-conf read per car.
    car_nonghost: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for r in filtered:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt or conf < CONF_THRESHOLD or txt in ghosts_post:
            continue
        car_nonghost[r["track_id"]].append((txt, conf))
    tracks_real_any = sum(
        1 for tid in car_tids if car_nonghost.get(tid)
    )
    tracks_real_top = 0
    for tid in car_tids:
        txt = by_track_best_text.get(tid)
        conf = by_track_best_conf.get(tid, 0.0)
        if txt and conf >= CONF_THRESHOLD and txt not in ghosts_post:
            tracks_real_top += 1

    # Per-image rates after masking.
    n_images = len({(r["track_id"], r["snap_index"])
                    for r in filtered if r.get("pipeline") == "preferred"})
    n_image_high = sum(
        1 for r in filtered
        if r.get("pipeline") == "preferred"
        and r.get("ocr_text")
        and (r.get("ocr_conf") or 0) >= CONF_THRESHOLD
    )

    # Per-direction breakdown.
    dir_by_tid = {r["track_id"]: r.get("direction") for r in data}
    dcounts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in filtered:
        if r.get("pipeline") != "preferred":
            continue
        d = dir_by_tid.get(r["track_id"], "?")
        dcounts[d][0] += 1
        if r.get("ocr_text") and (r.get("ocr_conf") or 0) >= CONF_THRESHOLD:
            dcounts[d][1] += 1

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (n/a)"

    print()
    print("=== Step 10 (option A): parked-car mask -- post-hoc ===")
    print(f"Mask: 4K rect x=[{MASK_X1},{MASK_X2}]  y=[{MASK_Y1},{MASK_Y2}]")
    print()
    print(f"--- Per-image preferred-pipeline ---")
    print(f"  any high-conf read:             {pct(n_image_high, n_images)}")
    print(f"  L->R:                           {pct(dcounts['left to right'][1], dcounts['left to right'][0])}")
    print(f"  R->L:                           {pct(dcounts['right to left'][1], dcounts['right to left'][0])}")
    print()
    print(f"--- Per-car rates ---")
    print(f"  high-conf any read (with aliasing) : {pct(tracks_with_high_read, n_cars)}  (Step 9: 90.4%)")
    print(f"  high-conf, ghost-filtered (any):     {pct(tracks_real_any, n_cars)}  (Step 9: 78.5%)")
    print(f"  high-conf, ghost-filtered (top):     {pct(tracks_real_top, n_cars)}  (Step 9: 55.4%)")
    print()
    print(f"--- Ghost-plate distribution (top 10 by distinct-track count) ---")
    for plate, n in plate_counts.most_common(10):
        flag = "  <-- GHOST" if n >= GHOST_TRACK_MIN else ""
        print(f"  {plate:<10}  {n} tracks{flag}")
    print()
    print(f"Step 9 ghosts (>=5 tracks):  5 -- FD61PVX(106), FD61PWX(12), FD51PVX(10), FD61PYX(6), FD61PXX(5)")
    if ghosts_post:
        print(f"Step 10 ghosts (>=5 tracks): {len(ghosts_post)} -- {sorted(ghosts_post)}")
    else:
        print(f"Step 10 ghosts (>=5 tracks): 0  (clean)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
