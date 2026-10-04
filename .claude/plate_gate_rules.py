"""E1.4 first cut: compare candidate plate gates on label-quality signals.

For every track's UK-shaped best read (parked beacons left out, as
dvsa-label does) it gathers four signals and scores candidate gates on how
many tracks they keep and how dirty the kept labels look:

- ``conf``: min-character OCR confidence (today's gate: >= 0.90);
- ``agree``: another snap of the same track read the same string;
- ``support``: tracks anywhere whose best read is this exact plate;
- ``colour``: E1.3, the plate's colour fits the track's direction
  (``analysis.alpr.plate_colour``; ``None`` when the crop is mono/unsure).

Quality measures (lower is better; none is ground truth, so read them
together):

- not on register: old current-format plates DVSA has no MOT record for
  (``.claude/ocr_conf_calibration.py``), over every session;
- colour mismatch: DVSA colour group vs the colour CNN, only on sessions
  recorded after the production corpus (no head trained on those images);
- plate colour inconsistent: E1.3 (misattribution, which the register can't
  see).

    uv run python .claude/plate_gate_rules.py --json .claude/gate_rules.json

Read-only on session data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import cv2

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))
import ocr_conf_calibration as occ  # noqa: E402
import phase0_checks as p0  # noqa: E402

from streettracker.analysis.alpr.plate_colour import (  # noqa: E402
    classify_plate_colour,
    colour_consistent,
)
from streettracker.analysis.dvsa import is_canonical_uk_plate  # noqa: E402

Rule = Callable[[dict[str, Any]], bool]

RULES: dict[str, Rule] = {
    "A  conf>=0.90 (today)": lambda r: r["conf"] >= 0.90,
    "B  conf>=0.95": lambda r: r["conf"] >= 0.95,
    "C  conf>=0.90 & (agree|sup>=2)": lambda r: (
        r["conf"] >= 0.90 and (r["agree"] or r["support"] >= 2)
    ),
    "D  (agree|sup>=2) & conf>=0.80": lambda r: (
        (r["agree"] or r["support"] >= 2) and r["conf"] >= 0.80
    ),
    "E  agree|sup>=2 (any conf)": lambda r: bool(r["agree"] or r["support"] >= 2),
    "F  sup>=2 & conf>=0.80": lambda r: r["support"] >= 2 and r["conf"] >= 0.80,
    "G  A & colour not wrong": lambda r: r["conf"] >= 0.90 and r["colour"] is not False,
    "H  D & colour not wrong": lambda r: (
        (r["agree"] or r["support"] >= 2) and r["conf"] >= 0.80 and r["colour"] is not False
    ),
    "I  C & colour not wrong": lambda r: (
        r["conf"] >= 0.90 and (r["agree"] or r["support"] >= 2) and r["colour"] is not False
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    ns = argparse.Namespace(corpus=None, runs=args.runs)
    corpus = p0._production_corpus(ns)
    post = {
        d.name
        for d in p0._post_corpus_sessions(
            args.output_root, p0._corpus_sessions(p0._manifest_samples(corpus))
        )
    }
    sessions = p0._sessions(args.output_root)
    dvsa = occ.dvsa_outcomes(sessions)
    support: dict[str, int] = {}
    for d in sessions:
        for t in (p0._load_json(d / f"{d.name}_alpr_by_track.json") or {}).get("tracks", []):
            p = p0._plate((t.get("best_preferred") or {}).get("ocr_text"))
            if p and is_canonical_uk_plate(p):
                support[p] = support.get(p, 0) + 1
    old_before = date.fromordinal(date.today().toordinal() - int(round(4 * 365.25)))

    rows: list[dict[str, Any]] = []
    for d in sessions:
        got = occ.session_reads(d, dvsa, old_before=old_before, exclude_colour=set())
        if got is None:
            continue
        data = {int(r["track_id"]): r for r in p0._load_json(d / f"{d.name}_data.json") or []}
        bt = {
            int(t["track_id"]): t.get("best_preferred") or {}
            for t in (p0._load_json(d / f"{d.name}_alpr_by_track.json") or {}).get("tracks", [])
        }
        for r in got:
            best = bt.get(r.track_id, {})
            crop = best.get("crop_path")
            colour = None
            if not crop:  # by_track rows may lack it; fall back to the per-image entry's path
                crop = str(d / "alpr_crops" / "preferred" / str(best.get("image", "")))
            img = cv2.imread(str(crop).replace("\\", "/"))
            if img is not None:
                colour = colour_consistent(
                    classify_plate_colour(img).label, data.get(r.track_id, {}).get("direction")
                )
            rows.append(
                {
                    "session": r.session,
                    "conf": r.conf["min"],
                    "agree": r.no_agreement is False,
                    "support": support.get(r.plate, 0),
                    "colour": colour,
                    "not_on_register": r.not_on_register,
                    "colour_mismatch": r.colour_mismatch if r.session in post else None,
                }
            )
    n_all = len(rows)
    print(
        f"{n_all} UK-shaped best reads over {len(sessions)} sessions; post-corpus: {sorted(post)}"
    )
    print(
        f"{'rule':34s} {'kept':>14s}  {'not on register':>26s}  {'colour mismatch (unseen)':>26s}  "
        f"{'plate colour wrong':>26s}"
    )
    out: dict[str, Any] = {"n": n_all, "post_corpus": sorted(post), "rules": {}}
    for name, rule in RULES.items():
        kept = [r for r in rows if rule(r)]

        def rate(metric: str, kept: list[dict[str, Any]] = kept) -> dict[str, Any]:
            vals = [r[metric] for r in kept if r[metric] is not None]
            return p0._rate(sum(1 for v in vals if v), len(vals))

        nor = rate("not_on_register")
        cm = rate("colour_mismatch")
        cw = p0._rate(
            sum(1 for r in kept if r["colour"] is False),
            sum(1 for r in kept if r["colour"] is not None),
        )
        out["rules"][name] = {
            "kept": len(kept),
            "not_on_register": nor,
            "colour_mismatch": cm,
            "plate_colour_wrong": cw,
        }
        print(
            f"{name:34s} {len(kept):6d} ({100 * len(kept) / n_all:4.1f} %)  "
            f"{p0._fmt(nor):>26s}  {p0._fmt(cm):>26s}  {p0._fmt(cw):>26s}"
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
