"""E1: offline re-crop & re-score of R->L snaps with corrected crop targeting.

For every R->L car snap in the given sessions:
  1. full-frame vehicle detection (ultralytics, GPU) on the 4K image;
  2. candidate = detected vehicles inside the road polygon, ranked by
     along-road distance to the track's completion-time bbox position;
  3. production plate detector + OCR on the candidate crop(s);
  4. plate detections on the session's static-plate map (E6) rejected.

Outputs one JSONL row per snap (resumable) plus a per-session summary:
honest per-car canonical rate with correct crops, landing curve keyed
by the DETECTED car position, and the detected-vs-recorded bbox offset
distribution (E3-lite: how stale are the done-bboxes really?).

Run detached (WMI) -- takes GPU-hours:
  .venv/Scripts/python.exe .claude/rl_rescore_e1.py <session> [<session> ...]
"""

from __future__ import annotations

import ctypes
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from streettracker.analysis.alpr.preferred import (  # noqa: E402
    FastPlateOcrRecognizer,
    OpenImageModelsDetector,
)
from streettracker.analysis.alpr.staticfilter import (  # noqa: E402
    find_static_spots,
)
from streettracker.analysis.dvsa import is_canonical_uk_plate  # noqa: E402
from streettracker.analysis.snap_assets import (  # noqa: E402
    load_bbox_index,
    load_done_bbox_index,
)

OUT_DIR = REPO / ".claude" / "e1_rescore"
K4 = (4512, 2512)
PAD = 30
MIN_CAR_H = 45
STATIC_REJECT_PX = 40.0

TP = json.loads((REPO / ".claude" / "triggers_proposal.json").read_text())
AX, AY = TP["main_axis_xy"]
SW, SH = TP["source_size"]
CX, CY = TP["centroid_frac"][0] * SW, TP["centroid_frac"][1] * SH
TMIN, TMAX = TP["t_min"], TP["t_max"]
POLY = [(x * K4[0], y * K4[1]) for x, y in TP["vertices_frac"]]


def t_norm_4k(x: float, y: float) -> float:
    fx, fy = x / K4[0], y / K4[1]
    return (((fx * SW - CX) * AX + (fy * SH - CY) * AY) - TMIN) / (TMAX - TMIN)


def in_poly(x: float, y: float) -> bool:
    inside = False
    n = len(POLY)
    for i in range(n):
        x1, y1 = POLY[i]
        x2, y2 = POLY[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def main() -> None:
    # keep the box awake for the duration (dev box idle-sleeps)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    OUT_DIR.mkdir(exist_ok=True)

    from ultralytics import YOLO

    try:
        veh = YOLO("yolov8m.pt")
    except Exception:  # noqa: BLE001 - any load/download failure -> fallback
        veh = YOLO("yolov8n.pt")
    plate_det = OpenImageModelsDetector("yolo-v9-t-640-license-plate-end2end")
    ocr = FastPlateOcrRecognizer("global-plates-mobile-vit-v2-model")

    for session in sys.argv[1:]:
        sd = REPO / "output" / session
        data = json.loads((sd / f"{session}_data.json").read_text())
        alpr = [
            r
            for r in json.loads((sd / f"{session}_alpr.json").read_text())
            if r.get("pipeline") == "preferred"
        ]
        bbox_index, sub_size = load_bbox_index(sd)
        done_index = load_done_bbox_index(sd)
        car_index = {**bbox_index, **done_index}
        spots, _cons = find_static_spots(alpr, car_index, sub_size, K4)
        spot_cs = [(s["cx"], s["cy"]) for s in spots]
        print(f"[{session}] static spots: {len(spots)}", flush=True)

        sx = K4[0] / (sub_size[0] if sub_size else 896)
        sy = K4[1] / (sub_size[1] if sub_size else 512)

        out_path = OUT_DIR / f"{session}_e1.jsonl"
        done: set[tuple[int, int]] = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    done.add((row["tid"], row["snap"]))
                except (json.JSONDecodeError, KeyError):
                    pass

        jobs = []
        for rec in data:
            if rec.get("class_name") != "car" or rec.get("direction") != "right to left":
                continue
            for i, s in enumerate(rec.get("main_snaps") or []):
                if (rec["track_id"], s) in done:
                    continue
                p = sd / f"vehicle_{rec['track_id']}_main_{s}.jpg"
                if p.exists():
                    jobs.append((rec["track_id"], s, p))
        import os

        limit = int(os.environ.get("E1_LIMIT", "0"))
        if limit:
            jobs = jobs[:limit]
        print(f"[{session}] {len(jobs)} R->L snaps to process ({len(done)} already done)", flush=True)

        t0 = time.time()
        with out_path.open("a", encoding="utf-8") as fh:
            for k, (tid, snap, p) in enumerate(jobs, 1):
                try:
                    img = np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1]  # BGR
                except OSError:
                    continue
                res = veh.predict(img, classes=[2, 5, 7], conf=0.2, imgsz=1920, verbose=False)[0]
                cands = []
                ref = car_index.get((tid, snap))
                ref_t = None
                if ref is not None:
                    rcx, rcy = (ref[0] + ref[2]) / 2 * sx, (ref[1] + ref[3]) / 2 * sy
                    ref_t = t_norm_4k(rcx, rcy)
                for bb in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = (float(v) for v in bb)
                    ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
                    if y2 - y1 < MIN_CAR_H or not in_poly(ccx, ccy):
                        continue
                    ct = t_norm_4k(ccx, ccy)
                    rank = abs(ct - ref_t) if ref_t is not None else -(x2 - x1) * (y2 - y1)
                    cands.append((rank, (x1, y1, x2, y2), ct, (ccx, ccy)))
                cands.sort(key=lambda c: c[0])

                row: dict = {
                    "tid": tid,
                    "snap": snap,
                    "n_cand": len(cands),
                    "ref_t": round(ref_t, 4) if ref_t is not None else None,
                }
                best = None
                for _rank, (x1, y1, x2, y2), ct, (ccx, ccy) in cands[:2]:
                    cx1, cy1 = max(0, int(x1 - PAD)), max(0, int(y1 - PAD))
                    cx2, cy2 = min(K4[0], int(x2 + PAD)), min(K4[1], int(y2 + PAD))
                    crop = img[cy1:cy2, cx1:cx2]
                    det = plate_det.detect(crop)
                    if det is None:
                        continue
                    px1, py1, px2, py2 = det.bbox
                    pcx, pcy = (px1 + px2) / 2 + cx1, (py1 + py2) / 2 + cy1
                    if any(
                        abs(pcx - scx) <= STATIC_REJECT_PX and abs(pcy - scy) <= STATIC_REJECT_PX
                        for scx, scy in spot_cs
                    ):
                        continue
                    pad = max(2, int((px2 - px1) * 0.10))
                    pcrop = crop[
                        max(0, py1 - pad) : py2 + pad, max(0, px1 - pad) : px2 + pad
                    ]
                    read = ocr.recognize(pcrop)
                    if read is None:
                        continue
                    txt = (read.text or "").strip().upper().replace(" ", "")
                    conf = read.ocr_confidence or 0.0
                    canon = bool(txt and conf >= 0.9 and is_canonical_uk_plate(txt))
                    cand_row = {
                        "car_t": round(ct, 4),
                        "car_h": round(y2 - y1, 1),
                        "off_px": round(
                            ((ccx - rcx) ** 2 + (ccy - rcy) ** 2) ** 0.5, 1
                        )
                        if ref is not None
                        else None,
                        "plate_w": int(px2 - px1),
                        "text": txt,
                        "conf": round(conf, 3),
                        "canon": canon,
                    }
                    if best is None or (cand_row["canon"], conf) > (best["canon"], best["conf"]):
                        best = cand_row
                if best:
                    row.update(best)
                elif cands:
                    row["car_t"] = round(cands[0][2], 4)
                    row["car_h"] = round(cands[0][1][3] - cands[0][1][1], 1)
                fh.write(json.dumps(row) + "\n")
                if k % 100 == 0:
                    rate = k / (time.time() - t0)
                    print(
                        f"[{session}] {k}/{len(jobs)} ({rate:.1f}/s, eta {(len(jobs) - k) / rate / 60:.0f} min)",
                        flush=True,
                    )
                    fh.flush()

        summarize(session, out_path, data)

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print("ALL DONE", flush=True)


def summarize(session: str, out_path: Path, data: list[dict]) -> None:
    rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    tracks_all = {
        r["track_id"]
        for r in data
        if r.get("class_name") == "car"
        and r.get("direction") == "right to left"
        and r.get("main_snaps")
    }
    canon_tracks = {r["tid"] for r in rows if r.get("canon")}
    off = [r["off_px"] for r in rows if r.get("off_px") is not None]
    print(f"\n=== E1 {session} ===", flush=True)
    print(
        f"snaps rescored: {len(rows)}; R->L snapped cars: {len(tracks_all)}; "
        f"cars w/ canonical read (corrected crops, static-rejected): "
        f"{len(canon_tracks)} ({100 * len(canon_tracks) / max(1, len(tracks_all)):.1f}%)",
        flush=True,
    )
    if off:
        q = statistics.quantiles(off, n=4)
        print(
            f"detected-car vs done-bbox offset (4K px): p25/p50/p75 = "
            f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}  (n={len(off)})",
            flush=True,
        )
    bins: dict[int, list] = defaultdict(list)
    for r in rows:
        t = r.get("car_t")
        if t is not None:
            bins[min(int(max(t, 0) * 10), 9)].append(r)
    print("landing (TRUE car position) curve:", flush=True)
    for b in sorted(bins):
        rs = bins[b]
        n_canon = sum(1 for r in rs if r.get("canon"))
        n_read = sum(1 for r in rs if r.get("text"))
        pw = [r["plate_w"] for r in rs if r.get("plate_w")]
        print(
            f"  t {b / 10:.1f}-{b / 10 + 0.1:.1f}: n={len(rs):>5} read={n_read:>4} "
            f"canon={n_canon:>4} ({100 * n_canon / len(rs):.1f}%)"
            + (f" plate_w p50={statistics.median(pw):.0f}px" if pw else ""),
            flush=True,
        )
    summary = {
        "session": session,
        "snaps": len(rows),
        "cars": len(tracks_all),
        "cars_canonical": len(canon_tracks),
    }
    (OUT_DIR / f"{session}_e1_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
