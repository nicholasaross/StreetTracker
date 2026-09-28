"""Calibrate the plate-read confidence threshold from data already on disk.

Until 2026-09-28 ``ocr_conf`` was the most confident OCR slot (~1.0 for
every read), so ``dvsa-label --conf-threshold 0.9`` let every UK-shaped
read through. It is now the probability of the weakest character
(``alpr-rescore`` fixes existing sessions), and the 0.9 carried over
without anyone choosing it for the new score. This script shows where the
threshold should go, with no hand-labelling.

The old bug helps: because the gate passed everything, almost every
UK-shaped read has already been looked up on DVSA. So for each track's
best read (the read dvsa-label uses), grouped by its new confidence, three
misread signals can be measured. All three are "bad" rates (lower is
better):

* **Not on register.** A current-format plate (``LL00LLL``) whose age
  identifier says it is older than ``--mot-age-years`` (default 4) should
  have an MOT record. "Not found" there almost always means a misread (a
  few are exported, scrapped, MOT-exempt or on cherished plates).
* **Colour mismatch.** The DVSA register colour vs the colour classifier's
  read of the track, in coarse groups (white/silver/grey = light, ...). A
  misread that lands on a real car gets an unrelated colour. The
  classifier is itself imperfect, so the level has a floor; only the change
  across confidence groups carries the signal.
* **No other snap agrees.** Among tracks with >= 2 reads, whether no other
  snap of the same track read exactly the same string. Two independent
  reads agreeing are very likely right. A second car in the frame also
  produces disagreement, so this rate has a floor too.

The threshold belongs where the rates stop improving: the lowest group
whose rates match the most confident groups'. Both scorers are shown,
computed from the stored per-character probabilities: ``min`` (the weakest
character; what ``ocr_conf`` now is) and ``product`` (all characters
jointly). A sweep then shows, for each candidate cut-off, how many tracks
pass and the rates among the tracks kept and dropped. A last table asks
whether agreement between two snaps makes low-confidence reads safe to
keep.

Needs sessions re-scored with ``alpr-rescore`` (reads carry
``ocr_char_probs``); others are skipped and listed. Tracks whose best read
is a parked-car beacon are left out, as dvsa-label leaves them out. Only
UK-shaped best reads count, as only those are looked up.

    uv run python .claude/ocr_conf_calibration.py                  # all output/session_*
    uv run python .claude/ocr_conf_calibration.py output/session_A output/session_B
    uv run python .claude/ocr_conf_calibration.py --exclude-corpus runs/uk_crops_0924_576
        # colour check only on cars the colour model never trained on
    uv run python .claude/ocr_conf_calibration.py --json .claude/ocr_calibration.json

Read-only on session data; writes only the optional --json summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from streettracker.analysis.dvsa import is_canonical_uk_plate  # noqa: E402
from streettracker.analysis.makemodel.colour import colour_class_for  # noqa: E402
from streettracker.analysis.parked import detect_parked  # noqa: E402
from streettracker.analysis.vehicles import _colour_group  # noqa: E402

# Confidence groups (lower edge inclusive). Finer near the top, where the
# decision sits.
BIN_EDGES = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
SWEEP = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
SCORERS = ("min", "product")
METRICS = ("not_on_register", "colour_mismatch", "no_agreement")
METRIC_LABELS = {
    "not_on_register": "not on register",
    "colour_mismatch": "colour mismatch",
    "no_agreement": "no snap agrees",
}
_CURRENT_FORMAT = re.compile(r"^[A-Z]{2}(\d{2})[A-Z]{3}$")


@dataclass(slots=True)
class TrackRead:
    """One track's best read, with the misread signals that apply to it.
    A signal is ``None`` where it can't be measured for this track."""

    session: str
    track_id: int
    plate: str
    conf: dict[str, float]  # scorer -> confidence
    looked_up: bool
    not_on_register: bool | None
    colour_mismatch: bool | None
    no_agreement: bool | None


# ----------------------------------------------------------------------
# Inputs


def registration_date(plate: str) -> date | None:
    """First-registration half-year from a current-format plate's age
    identifier: 02-49 = March of 20NN, 51-99 = September of 20(NN-50)."""
    m = _CURRENT_FORMAT.match(plate)
    if not m:
        return None
    nn = int(m.group(1))
    if 1 <= nn <= 49:
        return date(2000 + nn, 3, 1)
    if 51 <= nn <= 99:
        return date(2000 + nn - 50, 9, 1)
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dvsa_outcomes(session_dirs: list[Path]) -> dict[str, dict[str, Any] | None]:
    """``plate -> DVSA row`` for hits and ``plate -> None`` for not-found,
    pooled over every session (a hit anywhere wins over a not-found)."""
    out: dict[str, dict[str, Any] | None] = {}
    for d in session_dirs:
        payload = _read_json(d / f"{d.name}_dvsa_labels.json")
        if not isinstance(payload, dict):
            continue
        for plate in payload.get("unknown") or []:
            out.setdefault(str(plate), None)
        for plate, row in (payload.get("labels") or {}).items():
            if isinstance(row, dict):
                out[str(plate)] = row
    return out


def corpus_plates(corpus_dirs: list[Path]) -> set[str]:
    plates: set[str] = set()
    for c in corpus_dirs:
        manifest = _read_json(c / "manifest.json")
        if not isinstance(manifest, dict):
            sys.exit(f"--exclude-corpus {c}: no readable manifest.json")
        plates |= {s["car"] for s in manifest.get("samples", []) if s.get("car")}
    return plates


def session_reads(
    d: Path,
    dvsa: dict[str, dict[str, Any] | None],
    *,
    old_before: date,
    exclude_colour: set[str],
) -> list[TrackRead] | None:
    """The session's UK-shaped best reads, or ``None`` when the session
    hasn't been re-scored (no per-character probabilities)."""
    label = d.name
    by_track = _read_json(d / f"{label}_alpr_by_track.json")
    entries = _read_json(d / f"{label}_alpr.json")
    if not isinstance(by_track, dict) or not isinstance(entries, list):
        return None
    bests = [
        (int(t["track_id"]), t["best_preferred"])
        for t in by_track.get("tracks", [])
        if isinstance(t.get("best_preferred"), dict)
    ]
    if not any(b.get("ocr_char_probs") for _tid, b in bests):
        return None

    # Parked-car beacons: excluded exactly as dvsa-label excludes them.
    suppressed: set[tuple[int, int]] = set()
    records = _read_json(d / f"{label}_data.json")
    if isinstance(records, list) and records:
        suppressed = detect_parked(entries, records).suppressed

    # Every other read of each track, for the agreement signal.
    reads_by_track: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for r in entries:
        if r.get("pipeline") != "preferred" or r.get("static_suspect") or not r.get("ocr_text"):
            continue
        reads_by_track[int(r["track_id"])].append((int(r["snap_index"]), str(r["ocr_text"])))

    colour_by_track: dict[int, str] = {}
    colour_doc = _read_json(d / f"{label}_colour_by_track.json")
    if isinstance(colour_doc, dict):
        for t in colour_doc.get("tracks", []):
            if t.get("colour") and t.get("track_id") is not None:
                colour_by_track[int(t["track_id"])] = str(t["colour"])

    out: list[TrackRead] = []
    for tid, best in bests:
        plate = str(best.get("ocr_text") or "").replace(" ", "").upper()
        probs = best.get("ocr_char_probs")
        if not plate or not probs or not is_canonical_uk_plate(plate):
            continue
        if (tid, int(best.get("snap_index", -1))) in suppressed:
            continue

        conf = {"min": min(probs), "product": math.prod(probs)}

        looked_up = plate in dvsa
        row = dvsa.get(plate)
        reg = registration_date(plate)
        not_on_register: bool | None = None
        if looked_up and reg is not None and reg <= old_before:
            not_on_register = row is None

        colour_mismatch: bool | None = None
        if row is not None and plate not in exclude_colour:
            g_dvsa = _colour_group(colour_class_for(row.get("primary_colour")))
            g_cnn = _colour_group(colour_by_track.get(tid))
            if g_dvsa and g_cnn:
                colour_mismatch = g_dvsa != g_cnn

        others = [text for n, text in reads_by_track.get(tid, []) if n != best.get("snap_index")]
        no_agreement = (plate not in others) if others else None

        out.append(
            TrackRead(
                label, tid, plate, conf, looked_up, not_on_register, colour_mismatch, no_agreement
            )
        )
    return out


# ----------------------------------------------------------------------
# Statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(rows: list[TrackRead], metric: str) -> dict[str, Any]:
    vals = [getattr(r, metric) for r in rows if getattr(r, metric) is not None]
    n, k = len(vals), sum(1 for v in vals if v)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def summarise(rows: list[TrackRead]) -> dict[str, Any]:
    return {
        "tracks": len(rows),
        "looked_up": sum(1 for r in rows if r.looked_up),
        **{m: rate(rows, m) for m in METRICS},
    }


def bin_label(i: int) -> str:
    lo = BIN_EDGES[i]
    hi = BIN_EDGES[i + 1] if i + 1 < len(BIN_EDGES) else 1.0
    return f"{lo:.2f}-{hi:.2f}" if i + 1 < len(BIN_EDGES) else f"{lo:.2f}-1.00"


def bin_index(x: float) -> int:
    i = 0
    for j, edge in enumerate(BIN_EDGES):
        if x >= edge:
            i = j
    return i


# ----------------------------------------------------------------------
# Report


def fmt_rate(s: dict[str, Any], min_n: int) -> str:
    if not s["n"]:
        return "-"
    lo, hi = s["ci95"]
    text = f"{100 * s['rate']:5.1f}% ±{50 * (hi - lo):4.1f} ({s['n']})"
    return text if s["n"] >= min_n else text + "*"


def print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(header, widths, strict=True)))
    for r in rows:
        print("  " + "  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)))


def report(reads: list[TrackRead], *, min_n: int, colour_note: str) -> dict[str, Any]:
    result: dict[str, Any] = {"n_tracks": len(reads), "scorers": {}}
    metric_cols = [METRIC_LABELS[m] for m in METRICS]
    for scorer in SCORERS:
        by_bin: dict[int, list[TrackRead]] = defaultdict(list)
        for r in reads:
            by_bin[bin_index(r.conf[scorer])].append(r)
        print(
            f"\n== {scorer}: misread signals by confidence group "
            f"(rate ± half 95% CI (n); * = n < {min_n}) =="
        )
        rows, bins_json = [], []
        for i in range(len(BIN_EDGES)):
            s = summarise(by_bin.get(i, []))
            bins_json.append({"group": bin_label(i), **s})
            looked = f"{100 * s['looked_up'] / s['tracks']:.0f}%" if s["tracks"] else "-"
            rows.append(
                [bin_label(i), str(s["tracks"]), looked] + [fmt_rate(s[m], min_n) for m in METRICS]
            )
        print_table(["confidence", "tracks", "on DVSA", *metric_cols], rows)

        print(f"\n== {scorer}: threshold sweep (kept = best read at or above the cut-off) ==")
        rows, sweep_json = [], []
        for t in SWEEP:
            kept = [r for r in reads if r.conf[scorer] >= t]
            dropped = [r for r in reads if r.conf[scorer] < t]
            sk, sd = summarise(kept), summarise(dropped)
            sweep_json.append({"threshold": t, "kept": sk, "dropped": sd})
            pct = f"{100 * len(kept) / len(reads):.0f}%" if reads else "-"
            rows.append(
                [
                    f"{t:.2f}",
                    f"{len(kept)} ({pct})",
                    fmt_rate(sk["not_on_register"], min_n),
                    fmt_rate(sk["colour_mismatch"], min_n),
                    str(len(dropped)),
                    fmt_rate(sd["not_on_register"], min_n),
                    fmt_rate(sd["colour_mismatch"], min_n),
                ]
            )
        print_table(
            [
                "cut-off",
                "kept",
                "kept: not on reg",
                "kept: colour mm",
                "dropped",
                "dropped: not on reg",
                "dropped: colour mm",
            ],
            rows,
        )
        result["scorers"][scorer] = {"groups": bins_json, "sweep": sweep_json}

    print("\n== min: does another snap agreeing make a low-confidence read safe? ==")
    rows, rescue_json = [], []
    for i in range(len(BIN_EDGES)):
        in_bin = [r for r in reads if bin_index(r.conf["min"]) == i]
        for agreed, name in ((True, "another snap agrees"), (False, "no snap agrees")):
            sub = [r for r in in_bin if r.no_agreement is (not agreed)]
            s = summarise(sub)
            rescue_json.append({"group": bin_label(i), "agreed": agreed, **s})
            rows.append(
                [
                    bin_label(i),
                    name,
                    str(s["tracks"]),
                    fmt_rate(s["not_on_register"], min_n),
                    fmt_rate(s["colour_mismatch"], min_n),
                ]
            )
    print_table(["confidence", "", "tracks", "not on register", "colour mismatch"], rows)
    result["agreement_rescue"] = rescue_json

    print(
        "\nHow to read this: in the first table of each scorer, find the lowest group whose\n"
        "rates match the top groups' (0.95 and up). That group's lower edge is the\n"
        "threshold the data supports. In the sweep, a good cut-off drops tracks whose rates\n"
        "are clearly worse than the kept tracks'; if the dropped tracks look like the kept\n"
        "ones, the cut-off is too strict. The last table shows whether 'another snap agrees'\n"
        "lets a lower bar through safely.\n"
        f"Colour: {colour_note}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "sessions", nargs="*", type=Path, help="session dirs (default: every output/session_*)"
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="where DVSA outcomes are pooled from (default: output)",
    )
    ap.add_argument(
        "--mot-age-years",
        type=float,
        default=4.0,
        help="plates registered at least this long ago should be on the MOT "
        "register (default 4; the first MOT is at 3 years)",
    )
    ap.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="reference date for plate age, YYYY-MM-DD (default today)",
    )
    ap.add_argument(
        "--exclude-corpus",
        type=Path,
        action="append",
        default=[],
        help="corpus dir(s) whose cars the colour model trained on; their tracks "
        "are left out of the colour check (repeatable)",
    )
    ap.add_argument(
        "--min-n", type=int, default=30, help="rates from fewer tracks are marked * (default 30)"
    )
    ap.add_argument("--json", type=Path, help="also write the numbers as JSON")
    args = ap.parse_args(argv)

    all_dirs = sorted(
        Path(p) for p in glob.glob(str(args.output_root / "session_*")) if Path(p).is_dir()
    )
    session_dirs = [Path(s) for s in args.sessions] if args.sessions else all_dirs
    if not session_dirs:
        print(f"no sessions found under {args.output_root}/", file=sys.stderr)
        return 1

    dvsa = dvsa_outcomes(sorted(set(all_dirs) | set(session_dirs)))
    exclude = corpus_plates(args.exclude_corpus)
    days = int(round(args.mot_age_years * 365.25))
    old_before = date.fromordinal(args.as_of.toordinal() - days)

    reads: list[TrackRead] = []
    skipped: list[str] = []
    no_alpr = 0
    for d in session_dirs:
        if not (d / f"{d.name}_alpr.json").is_file():
            no_alpr += 1
            continue
        got = session_reads(d, dvsa, old_before=old_before, exclude_colour=exclude)
        if got is None:
            skipped.append(d.name)
        else:
            reads.extend(got)

    n_used = len(session_dirs) - no_alpr - len(skipped)
    print(f"sessions: {n_used} used, {len(skipped)} not re-scored yet, {no_alpr} without ALPR")
    if skipped:
        print(
            "  not re-scored (run alpr-rescore or the panel's rescore playbook): "
            + ", ".join(skipped)
        )
    if not reads:
        print("no re-scored UK-shaped best reads to calibrate on", file=sys.stderr)
        return 1
    n_old = sum(1 for r in reads if r.not_on_register is not None)
    n_col = sum(1 for r in reads if r.colour_mismatch is not None)
    n_multi = sum(1 for r in reads if r.no_agreement is not None)
    print(
        f"tracks: {len(reads)} UK-shaped best reads; {sum(r.looked_up for r in reads)} looked "
        f"up on DVSA; {n_old} old enough for the register check (registered before "
        f"{old_before.isoformat()}); {n_col} with both colours; {n_multi} with >= 2 reads"
    )
    colour_note = (
        f"{len(exclude)} training-corpus plates left out of the colour check."
        if exclude
        else "the colour model trained on DVSA-labelled cars, some of which may be in this "
        "data; pass --exclude-corpus to check only unseen cars."
    )
    result = report(reads, min_n=args.min_n, colour_note=colour_note)

    if args.json:
        result.update(
            sessions_used=n_used,
            sessions_not_rescored=skipped,
            as_of=args.as_of.isoformat(),
            mot_age_years=args.mot_age_years,
            registered_before=old_before.isoformat(),
            corpus_plates_excluded=len(exclude),
        )
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
