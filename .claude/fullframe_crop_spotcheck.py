"""Independent check of the fullframe TRAJECTORY rule on snaps with no plate read.

The make/colour/body-type inference path (`vehicle_locator`, mode
"fullframe") is plate-anchored where the snap carries a plate read, else
picks the on-road vehicle nearest the stale fire-time hint. The crop audit
(`.claude/makemodel_crop_audit.py`) measured that rule at 90.7 % on
plate-anchored snaps, but that population is circular: the fullframe
ALPR produced those reads from its own top-2 nearest-to-hint candidates.

This script checks the rule where it actually matters -- snaps WITHOUT a
plate read -- with two plate-free consistency proxies:

  * parked pick: the rule picks (IoU >= 0.8) the same box on another snap
    of the same track >= 3 snap indices (~1.2 s) away. A moving car can't
    hold still that long, so the pick is a stationary (parked) vehicle.
  * behind: the pick's centre lies BEHIND the stale hint along the
    track's travel direction by > 0.25 hint-widths. The real car can only
    have moved forward during the ~0.7 s snap latency.

Both proxies are calibrated on plate-anchored snaps (truth known), giving
their flag rates on correct vs wrong picks, then applied to the
unanchored population. Also writes a visual sheet of unanchored picks
(red = stale hint, green = pick).

Usage:
    uv run python .claude/fullframe_crop_spotcheck.py [--sessions output/session_A ...] [--n 500]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from streettracker.analysis.alpr.fullframe import load_road_polygon, rank_vehicle_candidates
from streettracker.analysis.snap_assets import (
    discover_vehicle_snaps,
    load_bbox_index,
    resolve_bbox_hint,
)
from streettracker.analysis.vehicle_locator import (
    VehicleBoxCache,
    anchor_plate_bbox,
    load_alpr_reads,
    plate_anchored_box,
)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class _Session:
    def __init__(self, sd: Path, poly) -> None:
        self.sd = sd
        self.bbox_index, self.sub_size = load_bbox_index(sd)
        self.reads = load_alpr_reads(sd)
        self.cache = VehicleBoxCache(sd)
        self.poly = poly
        self.direction: dict[int, str] = {}
        data = sd / f"{sd.name}_data.json"
        if data.exists():
            for r in json.loads(data.read_text()):
                if r.get("class_name") == "car":
                    self.direction[int(r["track_id"])] = r.get("direction") or ""
        self.snaps: dict[int, dict[int, Path]] = defaultdict(dict)
        for path, tid, n, _cls in discover_vehicle_snaps(sd):
            if tid in self.direction:
                self.snaps[tid][n] = path

    def analyse(self, tid: int, n: int) -> dict | None:
        path = self.snaps[tid][n]
        hint = resolve_bbox_hint(path, tid, n, self.bbox_index, self.sub_size)
        if hint is None:
            return None
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        boxes = self.cache.boxes(path.name, img)
        ranked = rank_vehicle_candidates(boxes, w, h, bbox_hint=hint, road_polygon_frac=self.poly)
        anchor = anchor_plate_bbox(self.reads.get((tid, n), []), None)
        truth = plate_anchored_box(boxes, anchor) if anchor is not None else None
        return {
            "path": path,
            "img": img,
            "hint": hint,
            "pick": ranked[0] if ranked else None,
            "truth": truth,
            "anchored": anchor is not None,
        }


def _proxies(s: _Session, tid: int, n: int, a: dict) -> dict:
    """Parked / behind flags for pick ``a`` of snap ``n`` on track ``tid``."""
    pick, hint = a["pick"], a["hint"]
    out = {"parked": None, "behind": None}
    # behind: along travel direction, relative to the stale hint.
    d = s.direction.get(tid, "")
    hcx = (hint[0] + hint[2]) / 2
    pcx = (pick[0] + pick[2]) / 2
    tol = 0.25 * (hint[2] - hint[0])
    if d == "left to right":
        out["behind"] = pcx < hcx - tol
    elif d == "right to left":
        out["behind"] = pcx > hcx + tol
    # parked: same pick on a snap >= 3 indices away.
    others = [m for m in s.snaps[tid] if abs(m - n) >= 3]
    if others:
        b = s.analyse(tid, min(others, key=lambda m: abs(m - n)))
        if b is not None and b["pick"] is not None:
            out["parked"] = _iou(pick, b["pick"]) >= 0.8
    return out


def _rate(xs: list[bool | None]) -> str:
    v = [x for x in xs if x is not None]
    return f"{sum(v) / len(v):6.1%} (n={len(v)})" if v else "   n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=Path, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=500, help="unanchored snaps to check")
    ap.add_argument("--n-calib", type=int, default=400, help="anchored snaps for calibration")
    ap.add_argument("--road-polygon", type=Path, default=Path(".claude/triggers_proposal.json"))
    ap.add_argument("--sheet", type=Path, default=None, help="visual sheet JPEG path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    poly = load_road_polygon(args.road_polygon, log_prefix="[spotcheck]")
    sessions = (
        args.sessions
        or sorted(Path(p).parent for p in Path("output").glob("session_*/session_*_alpr.json"))[-4:]
    )
    print("sessions:", ", ".join(s.name for s in sessions))
    sess = [_Session(sd, poly) for sd in sessions]

    # Snap population split by whether the snap carries a canonical read.
    anchored, unanchored = [], []
    for s in sess:
        for tid, by_n in s.snaps.items():
            for n in by_n:
                key = (s, tid, n)
                has = anchor_plate_bbox(s.reads.get((tid, n), []), None) is not None
                (anchored if has else unanchored).append(key)
    print(f"car snaps: {len(anchored)} with a plate read, {len(unanchored)} without")
    rng = random.Random(args.seed)

    # ---- calibration on anchored snaps (truth = plate-anchored box) ----
    calib = {"correct": defaultdict(list), "wrong": defaultdict(list)}
    for s, tid, n in rng.sample(anchored, min(args.n_calib, len(anchored))):
        a = s.analyse(tid, n)
        if a is None or a["pick"] is None or a["truth"] is None:
            continue
        ok = _iou(a["pick"], a["truth"]) >= 0.5
        for k, v in _proxies(s, tid, n, a).items():
            calib["correct" if ok else "wrong"][k].append(v)
    n_ok = len(calib["correct"]["behind"])
    n_bad = len(calib["wrong"]["behind"])
    print(
        f"\ncalibration (anchored): rule correct {n_ok}, wrong {n_bad} "
        f"({n_ok / max(1, n_ok + n_bad):.1%} correct)"
    )
    for k in ("parked", "behind"):
        print(
            f"  {k:6s} flag rate | correct picks {_rate(calib['correct'][k])} "
            f"| wrong picks {_rate(calib['wrong'][k])}"
        )

    # ---- the population that matters: no plate read on the snap ----
    rows, sheet = [], []
    no_cand = 0
    for s, tid, n in rng.sample(unanchored, min(args.n, len(unanchored))):
        a = s.analyse(tid, n)
        if a is None:
            continue
        if a["pick"] is None:
            no_cand += 1
            continue
        p = _proxies(s, tid, n, a)
        rows.append(p)
        if len(sheet) < 24:
            img = a["img"].copy()
            hx1, hy1, hx2, hy2 = a["hint"]
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 0, 255), 12)
            px1, py1, px2, py2 = (int(v) for v in a["pick"])
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 12)
            t = cv2.resize(img, (564, 314))
            label = f"{s.direction.get(tid, '?')[:2].upper()} park={p['parked']} bh={p['behind']}"
            cv2.putText(t, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            sheet.append(t)
    for c in (s.cache for s in sess):
        c.save()

    print(
        f"\nunanchored snaps: {len(rows)} picked, {no_cand} with no on-road candidate "
        f"({no_cand / max(1, len(rows) + no_cand):.1%})"
    )
    any_flag = [
        (r["parked"] is True) or (r["behind"] is True)
        for r in rows
        if r["parked"] is not None or r["behind"] is not None
    ]
    for k in ("parked", "behind"):
        print(f"  {k:6s} flag rate {_rate([r[k] for r in rows])}")
    print(f"  either flag      {_rate(any_flag)}")

    # Correct the raw flag rate with the calibration (Rogan-Gladen style):
    # err ~= (flag - fa) / (sens - fa), per proxy, clipped to [0, 1].
    for k in ("parked", "behind"):
        fa = [x for x in calib["correct"][k] if x is not None]
        se = [x for x in calib["wrong"][k] if x is not None]
        fl = [r[k] for r in rows if r[k] is not None]
        if fa and se and fl:
            fa_r, se_r, fl_r = sum(fa) / len(fa), sum(se) / len(se), sum(fl) / len(fl)
            if se_r > fa_r:
                est = min(1.0, max(0.0, (fl_r - fa_r) / (se_r - fa_r)))
                print(f"  est. wrong-pick rate via {k}: {est:.1%} (sens {se_r:.0%}, fa {fa_r:.0%})")

    if args.sheet and sheet:
        while len(sheet) % 3:
            sheet.append(np.zeros_like(sheet[0]))
        grid = np.vstack([np.hstack(sheet[i : i + 3]) for i in range(0, len(sheet), 3)])
        cv2.imwrite(str(args.sheet), grid)
        print(f"\nsheet -> {args.sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
