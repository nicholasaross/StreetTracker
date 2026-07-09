"""Baseline-restore soak: snap-budget + hourly volume.

session_20260529_164155 ran ~22.7h on the REVERTED config:
  max_concurrent = 2, pipeline_interval_ms = 400 (uniform),
  pipeline_interval_ms_by_direction = {} (disabled)

This is the regression-passes check after rolling back the Step 13a /
mc=3 experiment. It should reproduce the Step 11 snap-budget profile.

Caveat: this soak spans Fri 16:41 -> Sat 15:22. Saturday daytime
traffic differs from weekday (later start, lower commuter peak), so
the right comparison is hour-matched daytime, not the raw aggregate.

Run from repo root:
    uv run python .claude/analyze_baseline.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASELINE = ".claude/baseline_events.jsonl"
STEP11_DATA = "output/session_20260526_124704/session_20260526_124704_data.json"
LOCAL_TZ_OFFSET = 1  # BST


def _load(p: str) -> list[dict]:
    path = Path(p)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def _summary(label: str, records: list[dict]) -> None:
    vehicles = [r for r in records if r.get("asset_prefix") == "vehicle"]
    t0 = min(r["time_start_unix"] for r in records)
    t1 = max(r["time_end_unix"] for r in records)
    print(f"=== {label} ===")
    print(
        f"  Window: {datetime.fromtimestamp(t0, tz=timezone.utc).isoformat()} -> "
        f"{datetime.fromtimestamp(t1, tz=timezone.utc).isoformat()} "
        f"({(t1 - t0) / 3600:.1f}h, {len(vehicles)} vehicles)"
    )
    snaps = [len(r.get("main_snaps") or []) for r in vehicles]
    print(
        f"  snaps/vehicle: mean={statistics.mean(snaps):.2f} "
        f"median={statistics.median(snaps)} max={max(snaps)} "
        f">=1={sum(1 for n in snaps if n >= 1)}/{len(snaps)} "
        f"({100 * sum(1 for n in snaps if n >= 1) / len(snaps):.1f}%)"
    )
    by_dir: dict[str, list[int]] = defaultdict(list)
    for r in vehicles:
        by_dir[r["direction"]].append(len(r.get("main_snaps") or []))
    for d in sorted(by_dir):
        ns = by_dir[d]
        rates = [
            len(r.get("main_snaps") or []) / r["duration_visible"]
            for r in vehicles
            if r["direction"] == d and r["duration_visible"] > 0
        ]
        print(
            f"    {d:18s}: n={len(ns):4d} snaps mean={statistics.mean(ns):.2f} "
            f"snaps/sec mean={statistics.mean(rates):.3f}"
        )
    print()


def _hourly(label: str, records: list[dict]) -> None:
    vehicles = [r for r in records if r.get("asset_prefix") == "vehicle"]
    by_hour: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in vehicles:
        dt = datetime.fromtimestamp(r["time_start_unix"], tz=timezone.utc)
        local = dt.hour + LOCAL_TZ_OFFSET
        # day-of-week + local hour
        dow = dt.strftime("%a")
        hr = local % 24
        by_hour[(dow, hr)].append(r)
    print(f"=== {label} hourly volume (local BST) ===")
    print(f"  {'day':>4} {'hr':>3} {'n':>4} {'R->L':>5} {'L->R':>5} {'snaps/veh':>10} {'R->L sps':>9}")
    for (dow, hr) in sorted(by_hour, key=lambda k: (k[0] != "Fri", k[1])):
        bucket = by_hour[(dow, hr)]
        rl = [r for r in bucket if r["direction"] == "right to left"]
        lr = [r for r in bucket if r["direction"] == "left to right"]
        avg = statistics.mean(len(r.get("main_snaps") or []) for r in bucket)

        def _sps(xs: list[dict]) -> float:
            ok = [r for r in xs if r["duration_visible"] > 0]
            return (
                statistics.mean(len(r.get("main_snaps") or []) / r["duration_visible"] for r in ok)
                if ok else 0.0
            )

        print(
            f"  {dow:>4} {hr:>3} {len(bucket):>4} {len(rl):>5} {len(lr):>5} "
            f"{avg:>10.2f} {_sps(rl):>9.3f}"
        )
    print()


def main() -> int:
    base = _load(BASELINE)
    _summary("Baseline-restore soak (mc=2, uniform 400ms)", base)
    if Path(STEP11_DATA).exists():
        _summary("Step 11 anchor (mc=2, uniform 400ms, weekday)", _load(STEP11_DATA))
    _hourly("Baseline-restore soak", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
