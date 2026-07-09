"""Step 13a (direction-aware pipeline throttling) — soak-test analysis.

Step 13a deployed 2026-05-27 with::

    pipeline_interval_ms_by_direction = {"forward": 300, "reverse": 400}

forward = R->L approach (front-plate visible).
reverse = L->R depart  (rear-plate visible).

Hypothesis: R->L cars get ~1.33x snap rate; L->R unchanged. R->L coverage
should be the layer that moves, since it has been the bottlenecked
direction (28.7% per-image high-conf vs L->R 88.6% at Step 11).

Source: session_20260527_160139, soak still active. Uses events.jsonl
(crash-safe, written per-finalized-track) since data.json / meta.json
only land at session end. Step 11 anchor: session_20260526_124704
(5.5h, 1.45 mean fires/track post-trim).

Run from repo root:
    uv run python .claude/analyze_step13a.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOAK = Path(".claude/soak_events.jsonl")
STEP11_DATA = Path("output/session_20260526_124704/session_20260526_124704_data.json")
STEP11_META = Path("output/session_20260526_124704/session_20260526_124704_meta.json")


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _summarize(label: str, records: list[dict]) -> dict:
    vehicles = [r for r in records if r.get("asset_prefix") == "vehicle"]
    t0 = min(r["time_start_unix"] for r in records)
    t1 = max(r["time_end_unix"] for r in records)
    by_dir: dict[str, list[int]] = defaultdict(list)
    for r in vehicles:
        by_dir[r["direction"]].append(len(r.get("main_snaps") or []))
    return {
        "label": label,
        "t_start": datetime.fromtimestamp(t0, tz=timezone.utc),
        "t_end": datetime.fromtimestamp(t1, tz=timezone.utc),
        "hours": (t1 - t0) / 3600.0,
        "n_records": len(records),
        "n_vehicles": len(vehicles),
        "class_counts": Counter(r["class_name"] for r in records),
        "snaps_per": [len(r.get("main_snaps") or []) for r in vehicles],
        "by_dir": dict(by_dir),
    }


def _fmt_dir(ns: list[int]) -> str:
    if not ns:
        return "n=0 (no data)"
    cov = sum(1 for n in ns if n >= 1)
    return (
        f"n={len(ns):4d}  mean={statistics.mean(ns):.2f}  "
        f"median={statistics.median(ns)}  >=1={cov}/{len(ns)} "
        f"({100 * cov / len(ns):.1f}%)  total_snaps={sum(ns)}"
    )


def main() -> int:
    soak = _summarize("Step 13a soak (live)", _load_jsonl(SOAK))

    print(f"=== {soak['label']} ===")
    print(f"  Window: {soak['t_start'].isoformat()} -> {soak['t_end'].isoformat()}")
    print(f"  Span:   {soak['hours']:.2f}h  ({soak['n_records']} finalized records)")
    print(f"  Classes: {dict(soak['class_counts'])}")
    print()

    print(f"All vehicles (n={soak['n_vehicles']})")
    print(f"  {_fmt_dir(soak['snaps_per'])}")
    print()

    print("By direction:")
    for d in sorted(soak["by_dir"]):
        print(f"  {d:18s}: {_fmt_dir(soak['by_dir'][d])}")
    print()

    # Step 11 anchor (post-trim, pre-Step 13a; same t_usable_frac).
    if STEP11_DATA.exists():
        s11 = json.loads(STEP11_DATA.read_text())
        s11_vehicles = [r for r in s11 if r.get("asset_prefix") == "vehicle"]
        s11_by_dir: dict[str, list[int]] = defaultdict(list)
        for r in s11_vehicles:
            s11_by_dir[r["direction"]].append(len(r.get("main_snaps") or []))

        meta = (
            json.loads(STEP11_META.read_text()) if STEP11_META.exists() else {}
        )
        s11_stats = meta.get("snap_stats", {})

        print("=== Anchor: Step 11 (session_20260526_124704, 5.5h, pre-13a) ===")
        print(f"  All vehicles (n={len(s11_vehicles)}):")
        print(
            f"  {_fmt_dir([len(r.get('main_snaps') or []) for r in s11_vehicles])}"
        )
        print("  By direction:")
        for d in sorted(s11_by_dir):
            print(f"  {d:18s}: {_fmt_dir(s11_by_dir[d])}")
        print()
        if s11_stats:
            print("  Step 11 snap_stats (from meta.json):")
            fpt = s11_stats.get("fires_per_track", {})
            print(
                f"    mean fires/track={fpt.get('mean')} "
                f"  pipeline_fires={s11_stats.get('pipeline_fires')} "
                f"  throttled={s11_stats.get('pipeline_throttled')} "
                f"  exhausted={s11_stats.get('pipeline_budget_exhausted')}"
            )
        print()

        # Direction-ratio comparison (the Step 13a primary signal).
        def _mean(xs: list[int]) -> float:
            return statistics.mean(xs) if xs else 0.0

        s11_fwd = _mean(s11_by_dir.get("right to left", []))
        s11_rev = _mean(s11_by_dir.get("left to right", []))
        s13_fwd = _mean(soak["by_dir"].get("right to left", []))
        s13_rev = _mean(soak["by_dir"].get("left to right", []))

        print("=== Direction-aware delta (Step 13a hypothesis) ===")
        print(
            "                      Step 11 (uniform 400ms)   "
            "Step 13a (fwd=300 / rev=400)"
        )
        print(
            f"  R->L mean snaps:    {s11_fwd:6.2f}                   "
            f"{s13_fwd:6.2f}    "
            f"({(s13_fwd / s11_fwd - 1) * 100:+.1f}%)"
            if s11_fwd
            else f"  R->L: insufficient Step 11 data"
        )
        print(
            f"  L->R mean snaps:    {s11_rev:6.2f}                   "
            f"{s13_rev:6.2f}    "
            f"({(s13_rev / s11_rev - 1) * 100:+.1f}%)"
            if s11_rev
            else f"  L->R: insufficient Step 11 data"
        )
        print()
        print(
            "  Expected from config: R->L +33%, L->R ~0%. "
            "Note: bounded by pipeline_max_per_track=15 and by t_usable dwell time."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
