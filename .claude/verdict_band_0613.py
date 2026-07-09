"""Band-deploy verdict on the post-deploy soak (session_20260613_090331).

Per-image + per-snapped-car canonical-read rates, split by direction, plus
completion-time-bbox coverage. Compares against the pre-deploy baseline and
the checkpoint's expected numbers.

    uv run python .claude/verdict_band_0613.py output/session_20260613_090331
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from streettracker.analysis.dvsa import is_canonical_uk_plate


def main(argv: list[str]) -> int:
    sd = Path(argv[0])
    session = sd.name
    data = json.loads((sd / f"{session}_data.json").read_text())
    alpr = json.loads((sd / f"{session}_alpr.json").read_text())

    outcome: dict[tuple[int, int], bool] = {}
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        txt = (r.get("ocr_text") or "").strip().upper().replace(" ", "")
        conf = r.get("ocr_conf") or 0.0
        outcome[(r["track_id"], r["snap_index"])] = bool(
            txt and conf >= 0.9 and is_canonical_uk_plate(txt)
        )

    # per-image + per-car, split by direction
    img = {"right to left": [0, 0], "left to right": [0, 0], "all": [0, 0]}
    car = {"right to left": [0, 0], "left to right": [0, 0], "all": [0, 0]}
    done_cov = [0, 0]  # (snaps with done-bbox, total snaps)
    snapped_cars = 0

    for rec in data:
        if rec.get("class_name") != "car":
            continue
        sn = rec.get("main_snaps") or []
        if not sn:
            continue
        snapped_cars += 1
        d = rec.get("direction")
        done = rec.get("main_snap_bboxes_done") or []
        car_canon = False
        for i, s in enumerate(sn):
            done_cov[1] += 1
            if i < len(done) and done[i]:
                done_cov[0] += 1
            o = outcome.get((rec["track_id"], s))
            if o is None:
                continue
            img["all"][1] += 1
            if d in img:
                img[d][1] += 1
            if o:
                img["all"][0] += 1
                car_canon = True
                if d in img:
                    img[d][0] += 1
        car["all"][1] += 1
        if d in car:
            car[d][1] += 1
        if car_canon:
            car["all"][0] += 1
            if d in car:
                car[d][0] += 1

    def pct(n: list[int]) -> str:
        return f"{100 * n[0] / n[1]:5.1f}%  ({n[0]:>4}/{n[1]:<5})" if n[1] else "   n/a"

    print(f"=== Band-deploy verdict: {session} ===\n")
    print(f"snapped car tracks: {snapped_cars}")
    print(f"completion-time bbox coverage: {pct(done_cov)} of car-snaps\n")
    print("PER-IMAGE canonical (conf>=0.9, canonical UK plate):")
    print(f"  R->L (front): {pct(img['right to left'])}   [expected ~37 -> ~56%+]")
    print(f"  L->R (rear) : {pct(img['left to right'])}   [expected ~52 -> 75-85%]")
    print(f"  all         : {pct(img['all'])}\n")
    print("PER-SNAPPED-CAR canonical (>=1 canonical read):")
    print(f"  R->L (front): {pct(car['right to left'])}")
    print(f"  L->R (rear) : {pct(car['left to right'])}")
    print(f"  all         : {pct(car['all'])}   [baseline 63.9%, expected mid-70s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
