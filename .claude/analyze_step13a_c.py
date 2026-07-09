"""Step 13a follow-up #2: traffic mix + temporal load by hour-of-day.

Step 13a soak spans 18h (overnight + morning); Step 11 was 5.5h daytime.
If overnight has near-zero traffic AND morning rush is much denser, snap
concurrency throttling could be far worse during rush than during the
Step 11 window, masking any per-track-interval improvement.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOAK = Path(".claude/soak_events.jsonl")
LOCAL_TZ_OFFSET = 1  # BST


def main() -> int:
    recs = [json.loads(l) for l in SOAK.read_text().splitlines() if l.strip()]
    vehicles = [r for r in recs if r.get("asset_prefix") == "vehicle"]

    # group by local hour-of-day
    by_hour: dict[int, list[dict]] = defaultdict(list)
    for r in vehicles:
        dt = datetime.fromtimestamp(r["time_start_unix"], tz=timezone.utc)
        local_hr = (dt.hour + LOCAL_TZ_OFFSET) % 24
        by_hour[local_hr].append(r)

    print("Hourly vehicle volume + direction split (local BST):")
    print(f"  {'hour':>4}  {'n':>4}  {'R->L':>5}  {'L->R':>5}  {'snaps/veh':>10}  {'R->L sps':>10}  {'L->R sps':>10}")
    for h in sorted(by_hour):
        bucket = by_hour[h]
        rl = [r for r in bucket if r["direction"] == "right to left"]
        lr = [r for r in bucket if r["direction"] == "left to right"]
        avg_snaps = statistics.mean(len(r.get("main_snaps") or []) for r in bucket)

        def _sps(xs: list[dict]) -> float:
            ok = [r for r in xs if r["duration_visible"] > 0]
            if not ok:
                return 0.0
            return statistics.mean(
                len(r.get("main_snaps") or []) / r["duration_visible"] for r in ok
            )

        print(
            f"  {h:>4d}  {len(bucket):>4d}  {len(rl):>5d}  {len(lr):>5d}  "
            f"{avg_snaps:>10.2f}  {_sps(rl):>10.3f}  {_sps(lr):>10.3f}"
        )

    # also: at-or-near-cap (>=8) -- proxy for "could have used more snaps"
    near_cap_rl = [r for r in vehicles if r["direction"] == "right to left" and len(r.get("main_snaps") or []) >= 8]
    near_cap_lr = [r for r in vehicles if r["direction"] == "left to right" and len(r.get("main_snaps") or []) >= 8]
    print()
    print(
        f"Vehicles with >=8 snaps:  R->L {len(near_cap_rl)}/278 ({100*len(near_cap_rl)/278:.1f}%)   "
        f"L->R {len(near_cap_lr)}/309 ({100*len(near_cap_lr)/309:.1f}%)"
    )

    # zero-snap cars: are they short-dwell? probably skipped by t_usable.
    z_rl = [r for r in vehicles if r["direction"] == "right to left" and not r.get("main_snaps")]
    z_lr = [r for r in vehicles if r["direction"] == "left to right" and not r.get("main_snaps")]
    if z_rl:
        d = [r["duration_visible"] for r in z_rl]
        print(f"Zero-snap R->L (n={len(z_rl)}): duration mean={statistics.mean(d):.2f}s max={max(d):.2f}s")
    if z_lr:
        d = [r["duration_visible"] for r in z_lr]
        print(f"Zero-snap L->R (n={len(z_lr)}): duration mean={statistics.mean(d):.2f}s max={max(d):.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
