"""ANPR confidence-threshold quality curve.

Reads ``<session>_alpr_by_track.json`` from a pulled session, sweeps
the OCR confidence threshold from 0.50 to 0.99, and reports at each
threshold:

- ``n_total``      -- distinct plate strings (post-grouping)
- ``n_canonical``  -- those matching one of the UK plate schemes
- ``garbage_rate`` -- ``1 - n_canonical / n_total``

A sharp drop in garbage_rate at threshold X tells us: tighten the
high-conf bar to X and the per-image / per-car ALPR aggregates stop
counting OCR noise as 'reads'. Cheap diagnostic -- no DVSA API calls,
just regex matching on the existing alpr_by_track.json.

Run from repo root::

    uv run python .claude/threshold_curve.py output/session_20260528_103902
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from streettracker.analysis.dvsa import is_canonical_uk_plate


def _collect_at(by_track: dict, threshold: float) -> tuple[int, int]:
    seen: set[str] = set()
    for track in by_track.get("tracks", []):
        best = track.get("best_preferred")
        if not best:
            continue
        plate = (best.get("ocr_text") or "").strip().upper().replace(" ", "")
        conf = float(best.get("ocr_conf") or 0.0)
        if not plate or conf < threshold:
            continue
        seen.add(plate)
    n_total = len(seen)
    n_canonical = sum(1 for p in seen if is_canonical_uk_plate(p))
    return n_total, n_canonical


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: threshold_curve.py <session_dir>", file=sys.stderr)
        return 2
    session = Path(argv[0]).resolve()
    by_track_path = session / f"{session.name}_alpr_by_track.json"
    if not by_track_path.exists():
        print(f"missing {by_track_path}", file=sys.stderr)
        return 2

    by_track = json.loads(by_track_path.read_text(encoding="utf-8"))

    print(f"=== ANPR threshold curve for {session.name} ===")
    print()
    print(
        f"{'thresh':>7} | {'n_total':>8} {'n_canon':>8} {'garbage':>8}  yield_keep%"
    )
    print("-" * 60)
    baseline_total = None
    for tenths in range(50, 100, 5):
        threshold = tenths / 100.0
        n_total, n_canon = _collect_at(by_track, threshold)
        garbage_pct = (1 - n_canon / n_total) * 100 if n_total else 0.0
        if baseline_total is None:
            baseline_total = max(n_canon, 1)
        yield_pct = 100 * n_canon / baseline_total
        print(
            f"  {threshold:>5.2f} | {n_total:>8} {n_canon:>8} {garbage_pct:>7.1f}% "
            f"  {yield_pct:>7.1f}%"
        )

    # Finer grid in the typical high-conf region.
    print()
    print("Finer grid near the operationally interesting region:")
    print(
        f"{'thresh':>7} | {'n_total':>8} {'n_canon':>8} {'garbage':>8}"
    )
    print("-" * 50)
    for tenths in range(80, 100, 1):
        threshold = tenths / 100.0
        n_total, n_canon = _collect_at(by_track, threshold)
        garbage_pct = (1 - n_canon / n_total) * 100 if n_total else 0.0
        print(
            f"  {threshold:>5.2f} | {n_total:>8} {n_canon:>8} {garbage_pct:>7.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
