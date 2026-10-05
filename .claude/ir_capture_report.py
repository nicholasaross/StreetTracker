"""Turn ``ir_capture_detect.py`` output into tracks, motion events and a contact sheet.

Step 3 of the capture analysis (review R16 "Results", 2026-10-05).

* Tracks: for views A (conf 0.30) and B (conf 0.10), group boxes by BotSORT id,
  vote the class by summed confidence, and apply the runtime's finalize
  filters (``compute_attributes``: >= 1 s and >= 50 px net displacement; person
  0.5 s / 30 px, dog 0.5 s / 20 px, as the live ``tracking.by_class``).
* Raw episodes: view C's vehicle boxes (conf >= 0.05) with their centre on the
  road, minus static boxes (present in >= 20 % of sampled frames: parked
  cars), grouped in time; "covered" when a kept track overlaps it.
* Motion events: road-polygon MOG2 foreground above max(1500 px, 3x the 90th
  percentile), >= 4 frames, after a 300-frame warm-up; each is checked against
  the kept tracks and the raw episodes. An event with neither is a candidate
  missed car: look at it on the contact sheet.
* With ``--live-events``, the live session's records in the window (by
  ``capture.json``'s ffmpeg start; the video runs ~4 s behind wall-clock time
  there) are listed beside the offline tracks.

    uv run python .claude/ir_capture_report.py <capture>/1005_0700 \
        --live-events <session>_events.jsonl

Writes ``<window>/report.json`` and ``<window>/sheet.jpg`` (frames of every
motion event, every uncovered raw episode and every kept B track) next to the
capture, never into the repo.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 16: "dog"}
MINIMA = {0: (0.5, 30), 16: (0.5, 20)}  # class -> (min_duration_s, parked_px); default (1.0, 50)
VEHICLES = {2, 3, 5, 7}
WARMUP = 300


def build_tracks(frames: list, key: str) -> list[dict]:
    pts = defaultdict(list)
    for f in frames:
        for tid, cls, conf, x1, y1, x2, y2 in f[key]:
            if tid >= 0:
                pts[tid].append((f["ms"] / 1000.0, f["i"], cls, conf, x1, y1, x2, y2))
    out = []
    for tid, p in pts.items():
        votes: Counter = Counter()
        for row in p:
            votes[row[2]] += row[3]
        cls = votes.most_common(1)[0][0]
        t0, tN = p[0][0], p[-1][0]
        c0 = ((p[0][4] + p[0][6]) / 2, (p[0][5] + p[0][7]) / 2)
        cN = ((p[-1][4] + p[-1][6]) / 2, (p[-1][5] + p[-1][7]) / 2)
        net = ((cN[0] - c0[0]) ** 2 + (cN[1] - c0[1]) ** 2) ** 0.5
        min_dur, parked = MINIMA.get(cls, (1.0, 50))
        biggest = max(p, key=lambda r: (r[6] - r[4]) * (r[7] - r[5]))
        out.append(
            {
                "tid": tid,
                "name": NAMES.get(cls, str(cls)),
                "t0": t0,
                "t1": tN,
                "n": len(p),
                "net": round(net, 1),
                "kept": len(p) >= 2 and (tN - t0) >= min_dur and net >= parked,
                "dir": "L->R" if cN[0] > c0[0] else "R->L",
                "max_conf": round(max(r[3] for r in p), 3),
                "mean_conf": round(sum(r[3] for r in p) / len(p), 3),
                "peak_i": biggest[1],
                "peak_box": [round(v) for v in biggest[4:8]],
            }
        )
    return sorted(out, key=lambda r: r["t0"])


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def runs(flags: list[bool], gap: int, min_len: int) -> list[tuple[int, int]]:
    """Index spans where ``flags`` is set, bridging gaps of <= ``gap`` frames."""
    out, start, last = [], None, None
    for i, f in enumerate(flags):
        if not f:
            continue
        if start is None:
            start = i
        elif i - last > gap:
            if last - start + 1 >= min_len:
                out.append((start, last))
            start = i
        last = i
    if start is not None and last - start + 1 >= min_len:
        out.append((start, last))
    return out


def contact_sheet(wdir: Path, items: list, W: int, H: int) -> int:
    cap = cv2.VideoCapture(str(wdir / "sub.mp4"))
    tiles = []
    for kind, e in items[:60]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["peak_i"])
        ok, img = cap.read()
        if not ok:
            continue
        box = e.get("blob") or e.get("box") or e.get("peak_box")
        if box:
            cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:4])), (0, 0, 255), 2)
        if kind == "M":
            label = (
                f"M {e['t0']:.0f}s fg{e['peak_fg']} "
                f"trk{int(e['covered_track'])} raw{int(e['any_raw_vehicle'])}"
            )
        elif kind == "C":
            label = f"C {e['t0']:.0f}s {e['cls']} {e['max_conf']:.2f} n{e['n_dets']}"
        else:
            label = f"B {e['t0']:.0f}s {e['name']} {e['dir']} c{e['max_conf']:.2f}"
        cv2.rectangle(img, (0, H - 34), (W, H), (0, 0, 0), -1)
        cv2.putText(img, label, (6, H - 10), 0, 0.7, (0, 255, 255), 2)
        tiles.append(cv2.resize(img, (W // 2, H // 2)))
    if not tiles:
        return 0
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(wdir / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return len(items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("window", type=Path)
    ap.add_argument("--live-events", type=Path, help="the live session's _events.jsonl")
    args = ap.parse_args()
    wdir: Path = args.window

    d = json.loads((wdir / "detect.json").read_text())
    frames, W, H = d["frames"], d["w"], d["h"]
    poly = json.loads((REPO / ".claude/road_polygon_user.json").read_text())["vertices_frac"]
    road = np.array([[x * W, y * H] for x, y in poly], np.float32)

    def on_road(b) -> bool:
        return cv2.pointPolygonTest(road, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), False) >= 0

    luma = np.array([f["luma"] for f in frames])
    rep: dict = {
        "window": wdir.name,
        "frames": len(frames),
        "minutes": round(frames[-1]["ms"] / 60000, 1),
        "luma_min_median_max": [float(luma.min()), float(np.median(luma)), float(luma.max())],
    }
    tracks = {k: build_tracks(frames, k) for k in ("A", "B")}
    for k, tr in tracks.items():
        kept = [t for t in tr if t["kept"]]
        rep[f"{k}_raw_tracks"] = len(tr)
        rep[f"{k}_kept"] = dict(Counter(f"{t['name']} {t['dir']}" for t in kept))
        rep[f"{k}_kept_list"] = kept
    kept_spans = [(t["t0"] - 1, t["t1"] + 1) for k in ("A", "B") for t in tracks[k] if t["kept"]]

    def covered(a: float, b: float) -> bool:
        return any(a <= y and b >= x for x, y in kept_spans)

    # Raw low-confidence vehicle boxes, minus static ones (parked cars).
    raw = [[r for r in f["C"] if r[0] in VEHICLES and on_road(r[2:6])] for f in frames]
    every = [tuple(r[2:6]) for rs in raw for r in rs]
    sampled = raw[::20]
    static: list[tuple] = []
    for b in every[:: max(1, len(every) // 3000)]:
        if any(iou(b, s) > 0.6 for s in static):
            continue
        if sum(any(iou(b, r[2:6]) > 0.6 for r in rs) for rs in sampled) >= 0.2 * len(sampled):
            static.append(b)
    rep["static_boxes"] = [list(map(round, s)) for s in static]
    moving = [[r for r in rs if not any(iou(r[2:6], s) > 0.5 for s in static)] for rs in raw]
    episodes = []
    for s, e in runs([bool(r) for r in moving], gap=10, min_len=1):
        rows = [r for i in range(s, e + 1) for r in moving[i]]
        best = max(rows, key=lambda r: r[1])
        peak = next(i for i in range(s, e + 1) if best in moving[i])
        t0, t1 = frames[s]["ms"] / 1000, frames[e]["ms"] / 1000
        episodes.append(
            {
                "t0": t0,
                "t1": t1,
                "frames": e - s + 1,
                "n_dets": len(rows),
                "max_conf": best[1],
                "cls": NAMES[best[0]],
                "peak_i": peak,
                "box": [round(v) for v in best[2:6]],
                "covered": covered(t0, t1),
            }
        )
    rep["raw_episodes"] = episodes

    fg = np.array([f["fg"] for f in frames], float)
    noise = float(np.percentile(fg[WARMUP:], 90)) if len(fg) > WARMUP + 100 else 0.0
    thr = max(1500.0, 3 * noise)
    rep["motion_thr_px"] = thr
    rep["motion_fg_max_px"] = int(fg[WARMUP:].max()) if len(fg) > WARMUP else 0
    events = []
    for s, e in runs(list(fg > thr), gap=10, min_len=4):
        if s < WARMUP:
            continue
        peak = s + int(np.argmax(fg[s : e + 1]))
        t0, t1 = frames[s]["ms"] / 1000, frames[e]["ms"] / 1000
        events.append(
            {
                "t0": t0,
                "t1": t1,
                "frames": e - s + 1,
                "peak_fg": int(fg[peak]),
                "peak_i": peak,
                "blob": frames[peak]["blob"],
                "covered_track": covered(t0, t1),
                "any_raw_vehicle": any(moving[i] for i in range(s, e + 1)),
            }
        )
    rep["motion_events"] = events

    print(
        f"== {wdir.name}: {rep['frames']} frames, {rep['minutes']} min, "
        f"luma {rep['luma_min_median_max']}"
    )
    for k in ("A", "B"):
        print(f"  {k}: raw tracks {rep[f'{k}_raw_tracks']}, kept {rep[f'{k}_kept']}")
        for t in rep[f"{k}_kept_list"]:
            print(
                f"     {t['t0']:7.0f}-{t['t1']:5.0f}s {t['name']:<10} {t['dir']} "
                f"max conf {t['max_conf']}"
            )
    print(f"  static boxes {rep['static_boxes']}")
    print(
        f"  raw vehicle episodes (conf >= 0.05, moving, on road): {len(episodes)}, "
        f"not covered by a kept track: {sum(not e['covered'] for e in episodes)}"
    )
    unexplained = [e for e in events if not e["covered_track"] and not e["any_raw_vehicle"]]
    print(
        f"  motion events (thr {thr:.0f} px, largest blob {rep['motion_fg_max_px']} px): "
        f"{len(events)}; "
        f"with a kept track {sum(e['covered_track'] for e in events)}, raw detection only "
        f"{sum((not e['covered_track']) and e['any_raw_vehicle'] for e in events)}, "
        f"unexplained {len(unexplained)}"
    )

    if args.live_events:
        cap_meta = json.loads((wdir / "capture.json").read_text())
        w0 = cap_meta["ffmpeg_start_unix"]
        w1 = cap_meta.get("end_unix", w0 + frames[-1]["ms"] / 1000)
        live = []
        for line in args.live_events.read_text().splitlines():
            e = json.loads(line)
            if e["time_start_unix"] < w1 and e["time_end_unix"] > w0:
                live.append(e)
        rep["live"] = [
            {
                "t0": round(e["time_start_unix"] - w0, 1),
                "t1": round(e["time_end_unix"] - w0, 1),
                "class": e["class_name"],
                "direction": e["direction"],
                "avg_confidence": e["avg_confidence"],
                "n_main_snaps": len(e.get("main_snaps") or []),
            }
            for e in sorted(live, key=lambda e: e["time_start_unix"])
        ]
        print(
            f"  live tracker: {len(live)} records, {dict(Counter(e['class_name'] for e in live))}"
        )
        for e in rep["live"]:
            print(
                f"     {e['t0']:7.0f}-{e['t1']:5.0f}s {e['class']:<10} {e['direction']} "
                f"({e['n_main_snaps']} snaps)"
            )

    (wdir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    items = [("M", e) for e in events] + [("C", e) for e in episodes if not e["covered"]]
    items += [("B", t) for t in tracks["B"] if t["kept"]]
    n = contact_sheet(wdir, items, W, H)
    if n:
        print(f"  sheet: {n} items -> {wdir / 'sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
