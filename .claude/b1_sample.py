"""B1 diagnostic sample: pick a representative subset of no-read cars
from session_20260524_075630 for manual failure-mode classification."""
from __future__ import annotations

import json
import random
from pathlib import Path

session = Path("output/session_20260524_075630")
data = json.load(open(session / "session_20260524_075630_data.json"))
by_track = json.load(open(session / "session_20260524_075630_alpr_by_track.json"))
read_tids = {t["track_id"] for t in by_track["tracks"]}

cars = [r for r in data if r["class_name"] == "car" and r.get("main_snaps")]

no_read = []
for r in cars:
    tid = r["track_id"]
    if tid in read_tids:
        continue
    veh_files = [n for n in r["main_snaps"] if (session / f"vehicle_{tid}_main_{n}.jpg").exists()]
    if veh_files:
        no_read.append({
            "track_id": tid,
            "direction": r["direction"],
            "speed_px_s": r["speed_px_s"],
            "duration_visible": r["duration_visible"],
            "num_detections": r["num_detections"],
            "avg_confidence": r["avg_confidence"],
            "main_snaps": veh_files,
        })

print(f"cars (with any main_snaps): {len(cars)}")
print(f"cars with at least one vehicle-prefix file on disk and read in alpr_by_track: {sum(1 for r in cars if r['track_id'] in read_tids)}")
print(f"cars with vehicle-prefix files but no OCR read: {len(no_read)}")

# Reproducible sample.
random.seed(42)
sample_size = min(15, len(no_read))
sample = random.sample(no_read, sample_size)
sample.sort(key=lambda r: r["track_id"])

print()
print(f"sampled {len(sample)} tracks for visual inspection:")
for r in sample:
    pick = r["main_snaps"][len(r["main_snaps"]) // 2]
    print(
        f"  track={r['track_id']:>4} "
        f"dir={r['direction']:<14} "
        f"speed={r['speed_px_s']:>5.1f} px/s "
        f"dur={r['duration_visible']:>5.2f}s "
        f"dets={r['num_detections']:>3} "
        f"conf={r['avg_confidence']:.2f} "
        f"pick=vehicle_{r['track_id']}_main_{pick}.jpg"
    )

with_pick = []
for r in sample:
    pick = r["main_snaps"][len(r["main_snaps"]) // 2]
    with_pick.append({**r, "pick_snap": pick, "pick_path": f"vehicle_{r['track_id']}_main_{pick}.jpg"})

out = Path(".claude/b1_sample.json")
out.parent.mkdir(exist_ok=True)
with out.open("w") as f:
    json.dump(with_pick, f, indent=2)
print()
print(f"sample list written to {out}")
