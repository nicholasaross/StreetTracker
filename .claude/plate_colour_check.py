"""E1.3: plate colour vs track direction over every read (review R2).

A right-to-left car shows its white front plate, a left-to-right car its
yellow rear plate (``analysis.alpr.plate_colour``). A read whose plate colour
contradicts its track's direction belongs to another car, or the track's
direction is wrong. Read-only over ``output/``; writes only ``--json``.

    uv run python .claude/plate_colour_check.py
    uv run python .claude/plate_colour_check.py output/session_A --json .claude/e13.json

Reports:
- per UK-shaped read: colour labels and the inconsistent share, by direction
  and light (day 10-15 h, shoulder 07-09 / 16-18 h, dark 19-06 h);
- per track's best read (what dvsa-label uses; parked beacons left out):
  inconsistent share by confidence, snap agreement and cross-track support;
- E0.5's concurrent opposite-direction collisions (two tracks live together
  with one plate): whether the plate's colour picks exactly one owner;
- for inconsistent best reads, whether another snap of the track holds a
  different UK-shaped read whose colour fits (a fallback exists).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from streettracker.analysis.alpr.base import plate_conf_threshold  # noqa: E402
from streettracker.analysis.alpr.plate_colour import (  # noqa: E402
    classify_plate_colour,
    colour_consistent,
)
from streettracker.analysis.dvsa import is_canonical_uk_plate  # noqa: E402
from streettracker.analysis.parked import detect_parked  # noqa: E402

_FUZZY = 85
_WINDOW_S = 10.0
_CONCURRENT_S = 1.0


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _plate(t: Any) -> str:
    return str(t or "").replace(" ", "").upper()


def _light(hour: int) -> str:
    if 10 <= hour < 16:
        return "day 10-15"
    if 7 <= hour < 10 or 16 <= hour < 19:
        return "shoulder 07-09/16-18"
    return "dark 19-06"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = _wilson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def _fmt(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "n=0"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:5.1f}% ({r['k']}/{r['n']}, CI {100 * lo:.1f}-{100 * hi:.1f})"


def main() -> int:
    from rapidfuzz import fuzz

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path)
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    sessions = args.sessions or sorted(
        d for d in args.output_root.glob("session_*") if (d / f"{d.name}_alpr.json").exists()
    )
    gate = plate_conf_threshold()

    # Pass 1: cross-track support (tracks anywhere whose UK-shaped best read is the plate).
    support: Counter[str] = Counter()
    for d in sessions:
        for t in (_load(d / f"{d.name}_alpr_by_track.json") or {}).get("tracks", []):
            p = _plate((t.get("best_preferred") or {}).get("ocr_text"))
            if p and is_canonical_uk_plate(p):
                support[p] += 1

    per_read: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    best_rows: list[dict[str, Any]] = []
    pair_outcomes: Counter[str] = Counter()
    fallback: Counter[str] = Counter()
    for d in sessions:
        n = d.name
        entries = _load(d / f"{n}_alpr.json")
        data = _load(d / f"{n}_data.json")
        by_track = _load(d / f"{n}_alpr_by_track.json")
        if (
            not isinstance(entries, list)
            or not isinstance(data, list)
            or not isinstance(by_track, dict)
        ):
            continue
        rec = {int(r["track_id"]): r for r in data if r.get("track_id") is not None}
        suppressed = detect_parked(entries, data).suppressed
        colour_of: dict[tuple[int, int], str] = {}
        reads_by_track: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for e in entries:
            if e.get("pipeline") != "preferred" or e.get("static_suspect"):
                continue
            p = _plate(e.get("ocr_text"))
            if not p or not is_canonical_uk_plate(p) or not e.get("crop_path"):
                continue
            r = rec.get(int(e["track_id"]))
            if not r:
                continue
            img = cv2.imread(str(e["crop_path"]).replace("\\", "/"))
            if img is None:
                continue
            label = classify_plate_colour(img).label
            key = (int(e["track_id"]), int(e["snap_index"]))
            colour_of[key] = label
            reads_by_track[key[0]].append((key[1], p))
            light = _light(datetime.fromtimestamp(float(r["time_start_unix"])).hour)
            c = per_read[(str(r.get("direction")), light)]
            c[label] += 1
            ok = colour_consistent(label, r.get("direction"))
            if ok is not None:
                c["decided"] += 1
                c["inconsistent"] += not ok

        bests = []
        for t in by_track.get("tracks", []):
            b = t.get("best_preferred")
            if not isinstance(b, dict):
                continue
            tid, snap = int(t["track_id"]), int(b.get("snap_index", -1))
            p = _plate(b.get("ocr_text"))
            r = rec.get(tid)
            if not p or not is_canonical_uk_plate(p) or not r or (tid, snap) in suppressed:
                continue
            label = colour_of.get((tid, snap))
            if label is None:
                continue
            others = [x for s, x in reads_by_track.get(tid, []) if s != snap]
            row = {
                "session": n,
                "track_id": tid,
                "plate": p,
                "conf": float(b.get("ocr_conf") or 0.0),
                "direction": r.get("direction"),
                "start": float(r["time_start_unix"]),
                "end": float(r["time_end_unix"]),
                "label": label,
                "consistent": colour_consistent(label, r.get("direction")),
                "agree": (p in others) if others else None,
                "support": support.get(p, 0),
            }
            bests.append(row)
            if row["consistent"] is False:
                alt = [
                    (s, x)
                    for s, x in reads_by_track.get(tid, [])
                    if x != p and colour_consistent(colour_of.get((tid, s), ""), r.get("direction"))
                ]
                fallback["has a consistent other read" if alt else "no consistent other read"] += 1
        best_rows.extend(bests)

        gated = sorted((b for b in bests if b["conf"] >= gate), key=lambda b: b["start"])
        for i, a in enumerate(gated):
            for j in range(i + 1, len(gated)):
                b = gated[j]
                if b["start"] > a["end"] + _WINDOW_S:
                    break
                if a["direction"] == b["direction"] or len(a["plate"]) != len(b["plate"]):
                    continue
                if fuzz.ratio(a["plate"], b["plate"]) < _FUZZY:
                    continue
                if min(a["end"], b["end"]) - max(a["start"], b["start"]) <= _CONCURRENT_S:
                    continue
                ca, cb = a["consistent"], b["consistent"]
                if ca is None or cb is None:
                    pair_outcomes["a crop undecided (mono/unsure)"] += 1
                elif ca != cb:
                    pair_outcomes["colour picks one owner"] += 1
                elif ca and cb:
                    pair_outcomes["both fit (front + rear of one plate string)"] += 1
                else:
                    pair_outcomes["neither fits"] += 1

    print(f"== E1.3 plate colour vs direction: {len(sessions)} sessions ==")
    print("per UK-shaped read (labels; inconsistent share of decided reads):")
    out: dict[str, Any] = {"sessions": len(sessions), "gate": gate, "per_read": {}}
    for (direction, light), c in sorted(per_read.items()):
        r = _rate(c["inconsistent"], c["decided"])
        out["per_read"][f"{direction} | {light}"] = {**dict(c), "inconsistent_rate": r}
        labels = ", ".join(f"{k} {c[k]}" for k in ("yellow", "white", "unsure", "mono"))
        print(f"  {direction:14s} {light:21s} {labels:44s} inconsistent {_fmt(r)}")

    def table(title: str, key: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[key(row)].append(row)
        res = {}
        print(f"  by {title}:")
        for g in sorted(groups):
            dec = [x for x in groups[g] if x["consistent"] is not None]
            res[g] = _rate(sum(1 for x in dec if not x["consistent"]), len(dec))
            print(f"    {g:18s} {_fmt(res[g])}")
        return res

    print("per track's best read (inconsistent share of decided):")
    dec = [b for b in best_rows if b["consistent"] is not None]
    gated_rows = [b for b in best_rows if b["conf"] >= gate]
    dec_g = [b for b in gated_rows if b["consistent"] is not None]
    out["best_all"] = _rate(sum(1 for b in dec if not b["consistent"]), len(dec))
    out["best_gated"] = _rate(sum(1 for b in dec_g if not b["consistent"]), len(dec_g))
    print(f"  all UK-shaped: {_fmt(out['best_all'])}; gated (>= {gate}): {_fmt(out['best_gated'])}")
    out["by_direction"] = table("direction (gated)", lambda b: str(b["direction"]), gated_rows)
    out["by_conf"] = table(
        "min-char confidence",
        lambda b: ">=0.95" if b["conf"] >= 0.95 else "0.90-0.95" if b["conf"] >= 0.9 else "<0.90",
        best_rows,
    )
    out["by_agreement"] = table(
        "snap agreement (gated)",
        lambda b: {None: "single snap", True: "a snap agrees", False: "no snap agrees"}[b["agree"]],
        gated_rows,
    )
    out["by_support"] = table(
        "cross-track support (gated)",
        lambda b: (
            "1 track" if b["support"] <= 1 else "2-4 tracks" if b["support"] <= 4 else ">=5 tracks"
        ),
        gated_rows,
    )
    out["concurrent_opposite_pairs"] = dict(pair_outcomes)
    print(f"concurrent opposite-direction collisions (gated): {dict(pair_outcomes)}")
    out["inconsistent_best_fallback"] = dict(fallback)
    print(f"inconsistent best reads (all): {dict(fallback)}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
