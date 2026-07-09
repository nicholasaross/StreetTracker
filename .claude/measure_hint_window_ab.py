"""A/B measurement for the motion-window pre-crop hint (Step 16).

Compares per-image and per-car read metrics, split by direction, between
a baseline ``_alpr.json`` (single-bbox hints) and the current one
(window hints), for one session. Also runs the parked-beacon detector
on both to confirm the wider windows didn't inflate aliasing.

Usage (from repo root):
    uv run python .claude/measure_hint_window_ab.py output/session_20260530_165958
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from streettracker.analysis.dvsa import is_canonical_uk_plate
from streettracker.analysis.parked import detect_parked, normalize_plate


def _metrics(alpr_path: Path, recs: dict[int, dict]) -> dict:
    entries = json.loads(alpr_path.read_text())
    out: dict[str, dict] = {}
    per_track_best: dict[tuple[str, int], float] = {}
    per_track_canon: dict[tuple[str, int], bool] = {}
    for e in entries:
        if e.get("pipeline") not in (None, "preferred"):
            continue
        tid = e.get("track_id")
        rec = recs.get(int(tid)) if tid is not None else None
        if rec is None or rec.get("class_name") != "car":
            continue
        d = "R->L" if rec.get("direction") == "right to left" else "L->R"
        m = out.setdefault(d, {"images": 0, "det": 0, "read90": 0, "read95": 0, "canon90": 0})
        m["images"] += 1
        if e.get("det_bbox"):
            m["det"] += 1
        conf = e.get("ocr_conf") or 0.0
        text = normalize_plate(e.get("ocr_text"))
        if conf >= 0.9:
            m["read90"] += 1
            if is_canonical_uk_plate(text):
                m["canon90"] += 1
        if conf >= 0.95:
            m["read95"] += 1
        key = (d, int(tid))
        if conf > per_track_best.get(key, 0.0):
            per_track_best[key] = conf
            per_track_canon[key] = is_canonical_uk_plate(text)
    for d in out:
        tids = [k for k in per_track_best if k[0] == d]
        out[d]["cars"] = len({t for _d, t in tids})
        out[d]["cars_read90"] = sum(1 for k in tids if per_track_best[k] >= 0.9)
        out[d]["cars_canon90"] = sum(
            1 for k in tids if per_track_best[k] >= 0.9 and per_track_canon.get(k)
        )
    det = detect_parked(entries, list(recs.values()))
    out["_beacons"] = {
        "episodes": len(det.episodes),
        "suppressed_reads": len(det.suppressed),
        "suppressed_tracks": len({k[0] for k in det.suppressed}),
    }
    return out


def main() -> int:
    sd = Path(sys.argv[1])
    label = sd.name
    recs = {
        int(r["track_id"]): r
        for r in json.loads((sd / f"{label}_data.json").read_text())
        if r.get("track_id") is not None
    }
    base = _metrics(sd / f"{label}_alpr.baseline_prewindow.json", recs)
    new = _metrics(sd / f"{label}_alpr.json", recs)

    n_cars = {d: sum(1 for r in recs.values() if r.get("class_name") == "car" and (
        ("right to left" if d == "R->L" else "left to right") == r.get("direction"))) for d in ("R->L", "L->R")}

    print(f"{label}: baseline (single-bbox hint) vs window hint")
    for d in ("R->L", "L->R"):
        b, n = base.get(d, {}), new.get(d, {})
        if not b or not n:
            continue
        print(f"\n  {d}  ({b['images']} images, {n_cars[d]} car tracks in session)")
        for k, desc in (
            ("det", "plate detected"),
            ("read90", "read conf>=0.90"),
            ("read95", "read conf>=0.95"),
            ("canon90", "canonical read >=0.90"),
        ):
            pb, pn = 100 * b[k] / b["images"], 100 * n[k] / n["images"]
            print(f"    {desc:24} {pb:5.1f}%  ->  {pn:5.1f}%   ({pn - pb:+.1f} pp)")
        print(f"    {'snapped cars w/ read>=0.9':24} {b['cars_read90']:4d}/{b['cars']}  ->  "
              f"{n['cars_read90']:4d}/{n['cars']}")
        print(f"    {'  of which canonical':24} {b['cars_canon90']:4d}      ->  {n['cars_canon90']:4d}")
    bb, nb = base["_beacons"], new["_beacons"]
    print(f"\n  parked beacons: episodes {bb['episodes']} -> {nb['episodes']}, "
          f"suppressed reads {bb['suppressed_reads']} -> {nb['suppressed_reads']} "
          f"(aliasing pressure check; suppression layer absorbs these)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
