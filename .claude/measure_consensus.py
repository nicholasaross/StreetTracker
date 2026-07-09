"""Measure the lift from multi-frame plate consensus on existing
sessions, with no re-run of the ALPR pipelines needed.

Reads each session's per-image ``alpr.json``, applies
``consensus_plate`` per track, and prints the per-car aliasing-free
read rate using:
  (a) per-image best-of-N at conf >= 0.9 (current production rollup)
  (b) consensus at conf >= 0.9 (new rollup)
  (c) consensus at conf >= 0.7 (looser threshold -- consensus is
      already more reliable than any individual read at the same
      threshold; useful for the R→L direction)

Run from repo root:
    uv run python .claude/measure_consensus.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from streettracker.analysis.alpr.consensus import consensus_by_track

SESSIONS = [
    "session_20260525_200916",  # Step 10 baseline
    "session_20260526_124704",  # Step 11 post-trim
]
GHOST_TRACK_MIN = 5


def _measure(session: str) -> dict:
    base = Path("output") / session
    alpr = json.loads((base / f"{session}_alpr.json").read_text())
    data = json.loads((base / f"{session}_data.json").read_text())
    cars = [r for r in data if r.get("class_name") == "car"]
    car_tids = {r["track_id"] for r in cars}
    dir_by_tid = {r["track_id"]: r.get("direction") for r in data}

    # --- (a) per-image best-of-N at conf >= 0.9 (production rollup)
    best_by_tid: dict[int, tuple[str, float]] = {}
    for r in alpr:
        if r.get("pipeline") != "preferred":
            continue
        if not r.get("ocr_text"):
            continue
        c = r.get("ocr_conf") or 0.0
        if c > best_by_tid.get(r["track_id"], ("", -1.0))[1]:
            best_by_tid[r["track_id"]] = (r["ocr_text"], c)

    # --- (b)/(c) consensus
    consensus = consensus_by_track(alpr, pipeline="preferred")

    def ghost_set(threshold: float, source: dict) -> set[str]:
        """Re-compute ghosts at the given threshold for either rollup."""
        plate_tracks: dict[str, set[int]] = defaultdict(set)
        for tid, info in source.items():
            text, conf = (info if isinstance(info, tuple)
                          else (info["ocr_text"], info["ocr_conf"]))
            if conf >= threshold and tid in car_tids:
                plate_tracks[text].add(tid)
            elif isinstance(info, dict) and info.get("ocr_conf", 0) >= threshold:
                plate_tracks[info["ocr_text"]].add(tid)
        return {p for p, s in plate_tracks.items() if len(s) >= GHOST_TRACK_MIN}

    def rate_best(threshold: float) -> tuple[int, int, int, int]:
        """Return (any-non-ghost, top-non-ghost, n_with_high_conf, n_cars)."""
        ghosts = {
            p for p, n in Counter(
                {p: sum(1 for tid, (txt, c) in best_by_tid.items()
                        if txt == p and c >= threshold and tid in car_tids)
                 for p in {txt for txt, _ in best_by_tid.values()}}
            ).items() if n >= GHOST_TRACK_MIN
        }
        top_clean = 0; with_hc = 0
        for tid in car_tids:
            entry = best_by_tid.get(tid)
            if entry is None:
                continue
            txt, c = entry
            if c >= threshold:
                with_hc += 1
                if txt not in ghosts:
                    top_clean += 1
        return top_clean, top_clean, with_hc, len(car_tids)

    def rate_consensus(threshold: float) -> tuple[int, int, int, int]:
        ghosts = {
            p for p, n in Counter(
                {p: sum(1 for tid, info in consensus.items()
                        if info["ocr_text"] == p
                        and info["ocr_conf"] >= threshold
                        and tid in car_tids)
                 for p in {info["ocr_text"] for info in consensus.values()}}
            ).items() if n >= GHOST_TRACK_MIN
        }
        top_clean = 0; with_hc = 0
        for tid in car_tids:
            info = consensus.get(tid)
            if info is None:
                continue
            if info["ocr_conf"] >= threshold:
                with_hc += 1
                if info["ocr_text"] not in ghosts:
                    top_clean += 1
        return top_clean, top_clean, with_hc, len(car_tids)

    # Per-direction breakdown.
    dir_split = {"left to right": {"with_best": 0, "with_consensus": 0,
                                   "total": 0},
                 "right to left": {"with_best": 0, "with_consensus": 0,
                                   "total": 0}}
    for tid in car_tids:
        d = dir_by_tid.get(tid)
        if d not in dir_split:
            continue
        dir_split[d]["total"] += 1
        b = best_by_tid.get(tid)
        if b and b[1] >= 0.9:
            dir_split[d]["with_best"] += 1
        c = consensus.get(tid)
        if c and c["ocr_conf"] >= 0.9:
            dir_split[d]["with_consensus"] += 1

    return {
        "session": session,
        "n_cars": len(car_tids),
        "best_09": rate_best(0.9),
        "consensus_09": rate_consensus(0.9),
        "consensus_07": rate_consensus(0.7),
        "dir_split": dir_split,
    }


def main() -> int:
    for s in SESSIONS:
        r = _measure(s)
        n = r["n_cars"]
        b = r["best_09"]; c9 = r["consensus_09"]; c7 = r["consensus_07"]
        print(f"=== {r['session']} ({n} cars) ===")
        print(f"  per-car high-conf (best-of-N >= 0.9):     {b[0]}/{n} = {100*b[0]/n:.1f}%")
        print(f"  per-car high-conf (consensus  >= 0.9):    {c9[0]}/{n} = {100*c9[0]/n:.1f}%")
        print(f"  per-car high-conf (consensus  >= 0.7):    {c7[0]}/{n} = {100*c7[0]/n:.1f}%")
        print(f"  per-direction (best vs consensus @ 0.9):")
        for d, s_ in r["dir_split"].items():
            t = s_["total"]; b1 = s_["with_best"]; c1 = s_["with_consensus"]
            if t == 0: continue
            print(f"    {d}: best {b1}/{t}={100*b1/t:.1f}%  consensus {c1}/{t}={100*c1/t:.1f}%  delta={100*(c1-b1)/t:+.1f} pp")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
