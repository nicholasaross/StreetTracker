"""Door-origin calibration: plot the entry/exit point cloud.

The door zone has to contain the first/last detection **centroid**, not
the door's pixels -- and for someone clipped by the frame edge that
centroid sits an unknown distance from the door itself. This renders
where person tracks actually enter and leave the view, with the
configured zone overlaid, so the polygon can be tuned against reality
instead of a hand-trace.

Needs sessions recorded by the post-2026-07-19 runtime (the one that
captures ``entry_point_frac`` / ``exit_point_frac``); earlier sessions
report as unusable rather than silently plotting nothing.

    uv run python .claude/door_cloud.py output/session_XXXX [more...]
    uv run python .claude/door_cloud.py output/session_XXXX --track 24647

``--track`` highlights specific track ids (the operator-labelled ground
truth) in magenta with their id, so you can check that a known door trip
actually lands inside the zone.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from streettracker.analysis.walks import (
    ORIGINATED,
    PASSING,
    RETURNED,
    ROUND_TRIP,
    UNKNOWN,
    DoorZone,
    classify_walk_origin,
)

REFERENCE_FRAME = Path(".claude/live_frame.jpg")

# BGR, and deliberately distinct at a glance.
COLOURS = {
    ORIGINATED: (0, 220, 255),  # amber -- left the door
    RETURNED: (0, 200, 0),  # green  -- arrived at the door
    ROUND_TRIP: (255, 200, 0),  # cyan   -- both, within one track
    PASSING: (110, 110, 110),  # grey   -- through-traffic
}


def _records(session_dir: Path) -> list[dict[str, Any]]:
    """Person records from data.json, falling back to events.jsonl for a
    session that is still open (the live one has no data.json yet)."""
    name = session_dir.name
    data = session_dir / f"{name}_data.json"
    if data.exists():
        recs = json.loads(data.read_text(encoding="utf-8"))
    else:
        events = session_dir / f"{name}_events.jsonl"
        if not events.exists():
            return []
        lines = events.read_text(encoding="utf-8").splitlines()
        recs = [json.loads(line) for line in lines if line]
    return [r for r in recs if r.get("class_name") == "person" and not r.get("class_suspect")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--zone", type=Path, default=Path("configs/door_zone.json"))
    ap.add_argument("--frame", type=Path, default=REFERENCE_FRAME)
    ap.add_argument("--out", type=Path, default=Path(".claude/door_cloud.jpg"))
    ap.add_argument("--track", type=int, nargs="*", default=[], help="track ids to highlight")
    args = ap.parse_args()

    zone = DoorZone.load(args.zone)
    if zone is None:
        print(f"no door zone at {args.zone} -- nothing to calibrate against")
        return 1

    rows: list[tuple[int, str, list[float] | None, list[float] | None, str]] = []
    n_persons = 0
    for session_dir in args.sessions:
        for r in _records(session_dir):
            n_persons += 1
            entry, exit_ = r.get("entry_point_frac"), r.get("exit_point_frac")
            kind = classify_walk_origin(entry, exit_, zone)
            rows.append((int(r.get("track_id", -1)), session_dir.name, entry, exit_, kind))

    counts = Counter(k for *_, k in rows)
    usable = [row for row in rows if row[4] != UNKNOWN]
    print(f"person tracks: {n_persons}   with entry/exit points: {len(usable)}")
    if not usable:
        print(
            "\nNo track carries entry/exit points. These sessions predate the\n"
            "capture change -- deploy the post-2026-07-19 runtime and re-check."
        )
        return 1
    for kind in (ORIGINATED, RETURNED, ROUND_TRIP, PASSING):
        n = counts.get(kind, 0)
        print(f"  {kind:<11} {n:6d}  ({100 * n / len(usable):5.1f} % of usable)")
    own = sum(counts.get(k, 0) for k in (ORIGINATED, RETURNED, ROUND_TRIP))
    print(f"  {'OWN TRIPS':<11} {own:6d}  ({100 * own / len(usable):5.1f} %)")

    # x-histogram of entry points: where the zone's left edge should sit.
    print("\nentry-point x distribution (all person tracks, per 0.05):")
    buckets = [0] * 20
    for _, _, entry, _, _ in usable:
        if entry:
            buckets[min(19, int(float(entry[0]) * 20))] += 1
    peak = max(buckets) or 1
    zx0 = min(p[0] for p in zone.polygon_frac)
    for i, c in enumerate(buckets):
        lo = i / 20
        mark = " <-- zone" if lo >= zx0 else ""
        print(f"  {lo:.2f}-{lo + 0.05:.2f} {c:6d} {'#' * int(50 * c / peak)}{mark}")

    for tid, session, entry, exit_, kind in rows:
        if tid in args.track:
            print(f"\nhighlighted track {tid} ({session}): {kind}")
            print(f"  entry={entry}  exit={exit_}")

    if not args.frame.exists():
        print(f"\n(no reference frame at {args.frame}; skipping the render)")
        return 0

    frame = cv2.imread(str(args.frame))
    h, w = frame.shape[:2]
    img = frame.copy()
    pts = np.array(
        [[int(x * (w - 1)), int(y * (h - 1))] for x, y in zone.polygon_frac], dtype=np.int32
    )
    shade = img.copy()
    cv2.fillPoly(shade, [pts], (255, 255, 255))
    img = cv2.addWeighted(shade, 0.18, img, 0.82, 0)
    cv2.polylines(img, [pts], True, (255, 255, 255), 5)

    for tid, _, entry, exit_, kind in rows:
        if kind == UNKNOWN:
            continue
        colour = COLOURS[kind]
        hot = tid in args.track
        for point, filled in ((entry, -1), (exit_, 3)):
            if not point:
                continue
            c = (int(float(point[0]) * (w - 1)), int(float(point[1]) * (h - 1)))
            cv2.circle(img, c, 16 if hot else 7, (255, 0, 255) if hot else colour, filled)
        if hot and entry:
            cv2.putText(
                img,
                str(tid),
                (int(float(entry[0]) * (w - 1)) + 20, int(float(entry[1]) * (h - 1))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (255, 0, 255),
                4,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    print(f"\nwrote {args.out}  (filled = entry, ring = exit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
