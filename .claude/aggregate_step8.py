"""Aggregate Step 8 (bbox-pipe) ALPR results vs Step 7 baseline.

Compares per-car high-confidence read rate and ghost-plate
distribution for ``session_20260525_200916`` (bbox-pipe live) against
the Step 7 numbers recorded in CLAUDE.md (91.5% with aliasing,
``FD61PVX`` in 363/410 tracks).

Run from repo root:
    uv run python .claude/aggregate_step8.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

SESSION = "session_20260525_200916"
SESSION_DIR = Path("output") / SESSION
CONF_THRESHOLD = 0.9
GHOST_TRACK_MIN = 5  # plate seen in this many distinct tracks => ghost


def main() -> int:
    alpr_path = SESSION_DIR / f"{SESSION}_alpr.json"
    data_path = SESSION_DIR / f"{SESSION}_data.json"
    meta_path = SESSION_DIR / f"{SESSION}_meta.json"

    alpr = json.loads(alpr_path.read_text())
    data = json.loads(data_path.read_text())
    meta = json.loads(meta_path.read_text())

    # 1. Population: count cars + cars with at least one vehicle-prefix
    #    main snap on disk (so ALPR could have run on them).
    cars = [r for r in data if r.get("class_name") == "car"]
    n_cars = len(cars)
    cars_with_snaps = sum(
        1 for r in cars
        if (r.get("main_snaps") or []) and r.get("asset_prefix") == "vehicle"
    )
    cars_with_bboxes = sum(
        1 for r in cars
        if r.get("main_snap_bboxes")
        and any(b is not None for b in (r.get("main_snap_bboxes") or []))
    )

    # 2. Per-track best read on the preferred pipeline.
    by_track_best_conf: dict[int, float] = {}
    by_track_best_text: dict[int, str] = {}
    for r in alpr:
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

    car_tids = {r["track_id"] for r in cars}
    tracks_with_any_read = sum(
        1 for tid in car_tids if tid in by_track_best_text
    )
    tracks_with_high_read = sum(
        1 for tid in car_tids
        if by_track_best_conf.get(tid, 0.0) >= CONF_THRESHOLD
    )

    # 3. Ghost detection: count distinct tracks per plate string at
    #    high confidence on the preferred pipeline.
    plate_tracks: dict[str, set[int]] = defaultdict(set)
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt or conf < CONF_THRESHOLD:
            continue
        plate_tracks[txt].add(r["track_id"])

    plate_track_counts = Counter({p: len(tids) for p, tids in plate_tracks.items()})
    ghosts = {p for p, n in plate_track_counts.items() if n >= GHOST_TRACK_MIN}

    # 4. Aliasing-free read rate: for each car, count it as "read" if
    #    it has ANY non-ghost high-conf read across its snaps. Matches
    #    Step 7's reported approach ("ghost-filtered").
    car_nonghost_reads: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        txt = r.get("ocr_text")
        conf = r.get("ocr_conf") or 0.0
        if not txt or conf < CONF_THRESHOLD:
            continue
        if txt in ghosts:
            continue
        car_nonghost_reads[r["track_id"]].append((txt, conf))
    tracks_with_real_read = sum(
        1 for tid in car_tids if car_nonghost_reads.get(tid)
    )

    # 4b. Strict variant: best-read-per-car, then check non-ghost.
    tracks_with_real_top_read = 0
    for tid in car_tids:
        txt = by_track_best_text.get(tid)
        conf = by_track_best_conf.get(tid, 0.0)
        if txt and conf >= CONF_THRESHOLD and txt not in ghosts:
            tracks_with_real_top_read += 1

    # 5. Per-image preferred-pipeline detection / OCR rates (for the
    #    Step 7 comparison table).
    n_images = len({(r["track_id"], r["snap_index"])
                    for r in alpr if r.get("pipeline") == "preferred"})
    n_image_dets = sum(
        1 for r in alpr
        if r.get("pipeline") == "preferred" and r.get("det_bbox")
    )
    n_image_high_ocr = sum(
        1 for r in alpr
        if r.get("pipeline") == "preferred"
        and r.get("ocr_text")
        and (r.get("ocr_conf") or 0) >= CONF_THRESHOLD
    )

    snap_stats = meta.get("snap_stats", {})

    print(f"=== Step 8 (bbox-pipe) results vs Step 7 baseline ===")
    print(f"Session:         {SESSION}")
    print(f"frame_size:      {meta.get('frame_size')}")
    print(f"  uptime hours:  {snap_stats.get('latency', {}).get('count')} HTTP successes")
    print()
    print(f"--- Population ---")
    print(f"  car-class tracks:                {n_cars}")
    print(f"  cars with vehicle_*_main snaps:  {cars_with_snaps}")
    print(f"  cars with main_snap_bboxes set:  {cars_with_bboxes}")
    print()
    print(f"--- Per-image rates (preferred pipeline) ---")
    if n_images:
        pct_det = 100 * n_image_dets / n_images
        pct_ocr = 100 * n_image_high_ocr / n_images
    else:
        pct_det = pct_ocr = 0.0
    print(f"  images processed:                {n_images}")
    print(f"  with det_bbox:                   {n_image_dets} ({pct_det:.1f}%)")
    print(f"  with OCR conf >= {CONF_THRESHOLD}:           {n_image_high_ocr} ({pct_ocr:.1f}%)")
    print(f"  (Step 7 was 99% / 99% on 2391 vehicle snaps)")
    print()
    print(f"--- Per-car read rates ---")
    if n_cars:
        any_pct = 100 * tracks_with_any_read / n_cars
        high_pct = 100 * tracks_with_high_read / n_cars
        real_pct = 100 * tracks_with_real_read / n_cars
    else:
        any_pct = high_pct = real_pct = 0.0
    print(f"  any OCR text:                    {tracks_with_any_read}/{n_cars} ({any_pct:.1f}%)")
    print(f"  conf >= {CONF_THRESHOLD} (with aliasing):     {tracks_with_high_read}/{n_cars} ({high_pct:.1f}%)")
    real_top_pct = 100 * tracks_with_real_top_read / n_cars if n_cars else 0.0
    print(f"  conf >= {CONF_THRESHOLD} aliasing-free (any):  {tracks_with_real_read}/{n_cars} ({real_pct:.1f}%)  <-- Step 7 method")
    print(f"  conf >= {CONF_THRESHOLD} aliasing-free (top):  {tracks_with_real_top_read}/{n_cars} ({real_top_pct:.1f}%)  <-- strict: best read must be non-ghost")
    print()
    print(f"  Step 7 baseline (aliased):       91.5% (366/400)")
    print(f"  Step 8 predicted (aliasing-free):>=95%")
    print()
    print(f"--- Ghost-plate distribution (top 10 by distinct-track count) ---")
    for plate, n in plate_track_counts.most_common(10):
        flag = "  <-- GHOST" if n >= GHOST_TRACK_MIN else ""
        print(f"  {plate:<10}  {n} tracks{flag}")
    print()
    print(f"  Step 7 worst:    FD61PVX in 363/410 tracks (massive aliasing)")
    n_ghosts = len(ghosts)
    if ghosts:
        print(f"  Step 8 ghosts (>= {GHOST_TRACK_MIN} tracks):     {n_ghosts} plates -- {sorted(ghosts)[:10]}")
    else:
        print(f"  Step 8 ghosts (>= {GHOST_TRACK_MIN} tracks):     0  (clean)")
    print()
    print(f"--- snap_stats (from _meta.json) ---")
    lat = snap_stats.get("latency", {})
    print(f"  HTTP attempts/successes/failures: "
          f"{snap_stats.get('attempts')}/{snap_stats.get('successes')}/"
          f"{snap_stats.get('failures')}")
    print(f"  latency p50/p90/p99/max ms:       "
          f"{lat.get('p50_ms')} / {lat.get('p90_ms')} / "
          f"{lat.get('p99_ms')} / {lat.get('max_ms')}")
    print(f"  pipeline_fires / throttled / exhausted: "
          f"{snap_stats.get('pipeline_fires')} / "
          f"{snap_stats.get('pipeline_throttled')} / "
          f"{snap_stats.get('pipeline_budget_exhausted')}")
    print(f"  blur_skipped_frames:              {snap_stats.get('blur_skipped_frames')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
