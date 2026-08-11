#!/usr/bin/env python
"""Score StreetTracker's colour labels against DVSA ground truth.

The live colour label (``TrackRecord.color``, from ``common.color.vote_color``)
is a hand-tuned HSV vote on a *low-res sub-stream* crop with an 8-colour
palette. DVSA ``primary_colour`` (in every ``*_dvsa_labels.json``) is
owner-registered ground truth for the readable ~25-30 % of cars. This joins
the two per plated car and reports how good (or rubbish) the detector is --
the baseline any improvement must beat.

Two comparison levels:

* **grouped** -- both sides mapped through ``vehicles._colour_group``
  (white/silver/grey -> "light", red/orange/maroon -> "red", ...). Fair to
  the HSV voter, which cannot tell silver from white at this distance.
* **fine** -- raw lowercased strings. Exposes the palette gap: the voter can
  never emit orange/brown/gold/purple, so every such car is a guaranteed miss.

Usage::

    uv run python .claude/color_accuracy.py                 # all output/session_*
    uv run python .claude/color_accuracy.py output/session_20260709_103920 ...
    uv run python .claude/color_accuracy.py --cnn           # score CNN sidecar
        # (_colour_by_track.json) instead of the HSV `color` field
    uv run python .claude/color_accuracy.py --cnn --val-cars runs/uk_crops_colour
        # restrict to the trained model's held-out val cars (leakage-free
        # head-to-head; needs the corpus manifest that produced the model)

Read-only. Prints a report; writes nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Reuse the ONLY bridge across the detected + DVSA vocabularies. Import the
# grouping directly so there's a single source of truth for the mapping.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from streettracker.analysis.vehicles import _colour_group  # noqa: E402


def _norm(colour: Any) -> str:
    """Lowercase/strip a colour string; '' for None/empty/unknown."""
    s = str(colour or "").strip().lower()
    return "" if s in ("", "unknown", "none", "not known") else s


def _load_data_records(session: Path) -> list[dict[str, Any]]:
    matches = list(session.glob("*_data.json"))
    if not matches:
        return []
    try:
        d = json.loads(matches[0].read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return d if isinstance(d, list) else d.get("records") or d.get("tracks") or []


def _load_dvsa(session: Path) -> dict[str, dict[str, Any]]:
    matches = list(session.glob("*_dvsa_labels.json"))
    if not matches:
        return {}
    try:
        return json.loads(matches[0].read_text()).get("labels", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _load_cnn_by_track(session: Path) -> dict[int, str]:
    """track_id -> CNN colour (from ``*_colour_by_track.json``), or {}."""
    matches = list(session.glob("*_colour_by_track.json"))
    if not matches:
        return {}
    try:
        payload = json.loads(matches[0].read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, str] = {}
    for t in payload.get("tracks", []):
        col = t.get("colour")
        if col:
            out[int(t["track_id"])] = str(col)
    return out


def _val_cars(crops_dir: Path) -> set[str] | None:
    """Reconstruct the deterministic by-car val split for the colour target,
    so a CNN head-to-head scores only cars the model never trained on."""
    try:
        from streettracker.analysis.makemodel.uk_dataset import UKMakeDataset
    except Exception as exc:  # torch/torchvision may be absent
        print(f"[warn] cannot import UKMakeDataset ({exc}); scoring all cars")
        return None
    ds = UKMakeDataset(crops_dir, "val", label_field="colour", seed=0)
    return {s.car for s in ds._samples}  # noqa: SLF001 - intentional split reuse


def _group(colour: str) -> str:
    """Coarse group, or '' when ungroupable/unknown."""
    return _colour_group(colour) or ""


class Tally:
    """Accumulate predicted-vs-truth pairs and emit accuracy + confusion.

    ``mode`` picks how each colour is normalised before comparison:
    ``"group"`` -> coarse ``_colour_group`` (light/black/red/...),
    ``"fine"`` -> raw lowercased string, ``"identity"`` -> already-mapped
    (the caller pre-grouped both sides). A truth that normalises to '' is
    unscoreable and dropped; a pred that does becomes ``"unknown"``.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.pairs: list[tuple[str, str]] = []  # (truth, pred), both mapped

    def _map(self, colour: str) -> str:
        if self.mode == "identity":
            return colour
        if not colour:
            return ""
        return _group(colour) if self.mode == "group" else _norm(colour)

    def add(self, truth: str, pred: str) -> None:
        t = self._map(truth)
        if not t:
            return  # ungroupable / unlabelled truth -> not scoreable
        self.pairs.append((t, self._map(pred) or "unknown"))

    def report(self, title: str) -> None:
        n = len(self.pairs)
        if not n:
            print(f"\n{title}: no scoreable pairs")
            return
        correct = sum(1 for t, p in self.pairs if t == p)
        print(f"\n{title}: {correct}/{n} = {100 * correct / n:.1f}% accuracy")

        truths = sorted({t for t, _ in self.pairs})
        preds = sorted({p for _, p in self.pairs})
        cm: dict[str, Counter] = defaultdict(Counter)
        for t, p in self.pairs:
            cm[t][p] += 1

        print("  per-class recall (truth -> % correct, n):")
        for t in truths:
            tot = sum(cm[t].values())
            hit = cm[t].get(t, 0)
            top = cm[t].most_common(3)
            spread = " ".join(f"{k}:{v}" for k, v in top)
            print(f"    {t:<8} {100 * hit / tot:5.1f}%  n={tot:<5} [{spread}]")

        # Compact confusion matrix (rows=truth, cols=pred).
        col_w = max(6, *(len(p) for p in preds))
        header = "truth\\pred".ljust(10) + "".join(p.rjust(col_w) for p in preds)
        print("  confusion matrix:")
        print("    " + header)
        for t in truths:
            row = t.ljust(10) + "".join(str(cm[t].get(p, 0)).rjust(col_w) for p in preds)
            print("    " + row)


def score_sessions(sessions: list[Path], *, use_cnn: bool, val_cars: set[str] | None) -> None:
    grouped = Tally(mode="group")
    fine = Tally(mode="fine")
    grouped_car = Tally(mode="identity")  # caller pre-groups both sides

    n_cars = n_tracks = n_skipped_val = 0
    src = "CNN (_colour_by_track.json)" if use_cnn else "HSV (data.json `color`)"
    print(f"Scoring source: {src}")
    if val_cars is not None:
        print(f"Restricting to {len(val_cars)} held-out val cars")

    for session in sessions:
        records = _load_data_records(session)
        dvsa = _load_dvsa(session)
        if not records or not dvsa:
            continue
        pred_by_track: dict[int, str]
        if use_cnn:
            pred_by_track = _load_cnn_by_track(session)
        else:
            pred_by_track = {
                int(r["track_id"]): _norm(r.get("color"))
                for r in records
                if r.get("track_id") is not None
            }

        for plate, row in dvsa.items():
            truth = str(row.get("primary_colour") or "")
            if not truth:
                continue
            if val_cars is not None and plate not in val_cars:
                n_skipped_val += 1
                continue
            car_preds: Counter = Counter()
            for tid in row.get("track_ids", []):
                pred = pred_by_track.get(int(tid))
                if not pred:
                    continue
                n_tracks += 1
                grouped.add(truth, pred)
                fine.add(truth, pred)
                g = _group(pred)
                if g:
                    car_preds[g] += 1
            truth_g = _group(truth)
            if car_preds and truth_g:
                n_cars += 1
                # Per-car: majority grouped prediction vs truth group (both
                # already grouped -> identity tally, no double-mapping).
                grouped_car.add(truth_g, car_preds.most_common(1)[0][0])

    print(f"\nJoined {n_cars} cars / {n_tracks} tracks with a prediction + DVSA colour.")
    if n_skipped_val:
        print(f"(skipped {n_skipped_val} non-val plated cars)")
    grouped.report("GROUPED per-track")
    fine.report("FINE per-track (raw strings)")
    grouped_car.report("GROUPED per-car (majority vote)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions", nargs="*", type=Path, help="session dirs (default: output/session_*)"
    )
    parser.add_argument(
        "--cnn", action="store_true", help="score the CNN sidecar instead of HSV `color`"
    )
    parser.add_argument(
        "--val-cars",
        type=Path,
        default=None,
        help="colour crop-corpus dir; restrict to its held-out val cars (leakage-free)",
    )
    args = parser.parse_args(argv)

    sessions = list(args.sessions)
    if not sessions:
        sessions = sorted(
            {Path(g).parent for g in glob.glob("output/session_*/*_dvsa_labels.json")}
        )
    if not sessions:
        print("no DVSA-labelled sessions found")
        return 1

    val_cars = _val_cars(args.val_cars) if args.val_cars else None
    score_sessions(sessions, use_cnn=args.cnn, val_cars=val_cars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
