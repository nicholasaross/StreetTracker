"""Re-soak final snap-budget analysis (no ALPR yet).

Three-way comparison from events.jsonl alone:
- Step 11 anchor (max_concurrent=2, uniform 400ms, 5.5h)
- Step 13a soak (max_concurrent=2, dir-aware 300/400ms, 18.4h)
- Re-soak full (max_concurrent=3, dir-aware 300/400ms, 25.8h)

Runs before alpr-run to confirm the snap-budget story matches the
mid-stream signal (R->L snaps/sec 0.788, mean snaps/vehicle 5.33).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _load(p: Path) -> list[dict]:
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return json.loads(p.read_text())


def _summarize(label: str, records: list[dict]) -> None:
    vehicles = [r for r in records if r.get("asset_prefix") == "vehicle"]
    if not vehicles:
        print(f"{label}: no vehicles")
        return
    t0 = min(r["time_start_unix"] for r in records)
    t1 = max(r["time_end_unix"] for r in records)
    print(f"=== {label} ===")
    print(
        f"  Window: {datetime.fromtimestamp(t0, tz=timezone.utc).isoformat()} -> "
        f"{datetime.fromtimestamp(t1, tz=timezone.utc).isoformat()} "
        f"({(t1 - t0) / 3600:.2f}h, {len(vehicles)} vehicles)"
    )
    snaps = [len(r.get("main_snaps") or []) for r in vehicles]
    cap_hits = sum(1 for n in snaps if n >= 15)
    print(
        f"  All vehicles: mean={statistics.mean(snaps):.2f}  "
        f"median={statistics.median(snaps)}  max={max(snaps)}  "
        f">=1={sum(1 for n in snaps if n >= 1)}/{len(snaps)} "
        f"({100 * sum(1 for n in snaps if n >= 1) / len(snaps):.1f}%)  "
        f"at_cap_15={cap_hits}"
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
        cov = sum(1 for n in ns if n >= 1)
        print(
            f"    {d:18s}: n={len(ns):4d}  snaps mean={statistics.mean(ns):.2f}  "
            f"median={statistics.median(ns)}  snaps/sec mean={statistics.mean(rates):.3f}  "
            f">=1={cov}/{len(ns)} ({100 * cov / len(ns):.1f}%)"
        )
    print()


def main() -> int:
    _summarize(
        "Step 11 anchor (max_concurrent=2, uniform 400ms)",
        _load(Path("output/session_20260526_124704/session_20260526_124704_data.json")),
    )
    _summarize(
        "Step 13a soak (max_concurrent=2, dir-aware 300/400ms)",
        _load(Path("output/session_20260527_160139/session_20260527_160139_data.json")),
    )
    _summarize(
        "Re-soak final (max_concurrent=3, dir-aware 300/400ms)",
        _load(Path(".claude/resoak_events_final.jsonl")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
