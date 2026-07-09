"""Step 13a follow-up: why is per-vehicle snap count flat vs Step 11?

Hypothesis to test: pipeline_max_per_track=15 is non-binding, so the
1.33x cadence cannot show up as more snaps -- t_usable dwell time is the
binding constraint. If true, snaps/sec-of-track-life should rise ~1.33x
for R->L while count is flat, and the cap (=15) should rarely hit.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

SOAK = Path(".claude/soak_events.jsonl")
STEP11 = Path("output/session_20260526_124704/session_20260526_124704_data.json")


def _load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())


def _stats(label: str, records: list[dict]) -> None:
    vehicles = [r for r in records if r.get("asset_prefix") == "vehicle"]
    print(f"=== {label} ===")
    for direction in ("right to left", "left to right"):
        ds = [r for r in vehicles if r["direction"] == direction]
        if not ds:
            continue
        durs = [r["duration_visible"] for r in ds]
        snaps = [len(r.get("main_snaps") or []) for r in ds]
        rates = [
            (len(r.get("main_snaps") or []) / r["duration_visible"])
            if r["duration_visible"] > 0
            else 0.0
            for r in ds
        ]
        at_cap = sum(1 for n in snaps if n >= 15)
        zero = sum(1 for n in snaps if n == 0)
        print(f"  {direction}:")
        print(
            f"    n={len(ds):4d}  "
            f"duration_visible  mean={statistics.mean(durs):.2f}s "
            f"median={statistics.median(durs):.2f}s "
            f"p90={statistics.quantiles(durs, n=10)[-1]:.2f}s"
        )
        print(
            f"    snaps         mean={statistics.mean(snaps):.2f}  "
            f"median={statistics.median(snaps)}  max={max(snaps)}  "
            f"at_cap_15={at_cap}  zero_snap={zero}"
        )
        print(
            f"    snaps/sec     mean={statistics.mean(rates):.3f}  "
            f"median={statistics.median(rates):.3f}"
        )
    print()


def main() -> int:
    _stats("Step 13a soak (live, 18h)", _load(SOAK))
    if STEP11.exists():
        _stats("Step 11 anchor (session_20260526_124704, 5.5h)", _load(STEP11))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
