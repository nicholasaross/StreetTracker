"""Re-score a session's plate reads with the corrected OCR confidence.

    streettracker alpr-rescore <session_dir> [--ocr-model NAME]
        [--allow-missing] [--dry-run] [--force]

Until 2026-09-28 ``ocr_conf`` was the probability of the MOST confident
OCR slot (see :func:`streettracker.analysis.alpr.preferred._unpack_ocr_output`),
so it sat near 1.0 for every read, garbage included, and each downstream
``conf >= 0.9`` gate let everything through. It is now the probability of
the read's least certain character.

This command applies the fix to a session that has already been through
``alpr-run``, without redoing plate detection. Each read's plate crop is
saved under ``alpr_crops/<pipeline>/``, so only the OCR runs again (a few
ms per read, vs ~0.1 s of full-frame YOLO per snap). Every fast-plate-ocr
read with a saved crop takes the re-run's text, confidence and
per-character probabilities. The crop is the image the OCR saw first time,
re-encoded as JPEG, so the text occasionally changes; the summary counts
those. Static-plate flags are position-based and stay as they are.

Writes ``_alpr.json`` + ``_alpr_by_track.json`` and stamps
``"ocr_conf": "min_char"`` into ``_static_plates.json``. A session already
carrying that stamp is skipped unless ``--force``. Afterwards re-run
``dvsa-label`` -> ``dvsa-apply`` -> ``vehicles`` so the corrected gates
reach the labels (the control panel's "rescore" playbook does all of it).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from streettracker.analysis.alpr.base import (
    OCR_CONF_METHOD,
    PlateRead,
    atomic_write_text,
    plate_conf_threshold,
    resolve_plate_conf_threshold,
)
from streettracker.cli.alpr_run import _rollup_by_track

if TYPE_CHECKING:
    import numpy as np

# Pipelines whose recognizer is fast-plate-ocr -- the ones whose confidence
# was wrong. The bespoke pipeline's EasyOCR confidence is its own and is
# left alone.
FAST_PLATE_OCR_PIPELINES = frozenset({"preferred", "ablation_bespokedet_fastocr"})
DEFAULT_OCR_MODEL = "global-plates-mobile-vit-v2-model"
_PROGRESS_EVERY = 500


@dataclass(slots=True)
class RescoreStats:
    n_reads: int = 0  # fast-plate-ocr reads with a saved plate crop
    n_rescored: int = 0
    n_missing: int = 0  # crop file absent or unreadable -> confidence cleared
    n_text_changed: int = 0
    n_now_unread: int = 0  # had text before, the re-run read nothing
    old_conf: list[float] = field(default_factory=list)
    new_conf: list[float] = field(default_factory=list)


def is_rescorable(record: dict[str, Any]) -> bool:
    """A fast-plate-ocr record whose plate crop was saved."""
    return record.get("pipeline") in FAST_PLATE_OCR_PIPELINES and bool(record.get("crop_path"))


def crop_path_for(record: dict[str, Any], session_dir: Path) -> Path | None:
    """The saved plate crop: ``alpr_crops/<pipeline>/<image>`` in the session
    (portable if the session moved), else the recorded absolute path."""
    name, pipeline = record.get("image"), record.get("pipeline")
    if name and pipeline:
        p = session_dir / "alpr_crops" / str(pipeline) / str(name)
        if p.is_file():
            return p
    stored = record.get("crop_path")
    if stored and Path(stored).is_file():
        return Path(stored)
    return None


def rescore_records(
    records: list[dict[str, Any]],
    session_dir: Path,
    recognize: Callable[[np.ndarray], PlateRead | None],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> RescoreStats:
    """Re-run the OCR on every rescorable record's crop, updating it in place.

    A record whose crop is missing or unreadable keeps its text but has its
    confidence cleared (``None``, which every gate treats as 0) -- the old
    value is the bug, so it can't be kept.
    """
    import cv2

    from streettracker.analysis.dvsa import is_canonical_uk_plate

    stats = RescoreStats()
    todo = [r for r in records if is_rescorable(r)]
    stats.n_reads = len(todo)
    for i, r in enumerate(todo, 1):
        old_text = r.get("ocr_text") or None
        if r.get("ocr_conf") is not None:
            stats.old_conf.append(float(r["ocr_conf"]))
        path = crop_path_for(r, session_dir)
        image = cv2.imread(str(path)) if path is not None else None
        if image is None:
            stats.n_missing += 1
            r["ocr_conf"] = None
            r["ocr_char_probs"] = None
        else:
            stats.n_rescored += 1
            read = recognize(image)
            if read is None:
                r["ocr_text"] = r["ocr_raw"] = r["ocr_conf"] = r["ocr_char_probs"] = None
                r["canonical_uk_shape"] = None
            else:
                r["ocr_text"] = read.text
                r["ocr_raw"] = read.raw_text
                r["ocr_conf"] = read.ocr_confidence
                r["ocr_char_probs"] = read.char_probs
                r["canonical_uk_shape"] = is_canonical_uk_plate(read.text) if read.text else None
                stats.new_conf.append(read.ocr_confidence)
            new_text = r.get("ocr_text") or None
            if new_text != old_text:
                stats.n_text_changed += 1
                if old_text and not new_text:
                    stats.n_now_unread += 1
        if progress is not None and (i % _PROGRESS_EVERY == 0 or i == len(todo)):
            progress(i, len(todo))
    return stats


def gated_tracks(rollup: dict[str, Any], *, gate: float | None = None) -> int:
    """Tracks whose best preferred read is UK-shaped and clears ``gate``
    (default: the shared plate setting) -- the population ``dvsa-label``
    looks up (before parked-beacon suppression)."""
    gate = plate_conf_threshold(gate)
    n = 0
    for t in rollup.get("tracks", []):
        best = t.get("best_preferred")
        if best and best.get("canonical_uk_shape") and (best.get("ocr_conf") or 0.0) >= gate:
            n += 1
    return n


def _share_at_least(values: list[float], gate: float) -> str:
    if not values:
        return "n/a"
    return f"{100.0 * sum(1 for v in values if v >= gate) / len(values):.1f} %"


def _quantiles(values: list[float]) -> str:
    if not values:
        return "n/a"
    s = sorted(values)

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(len(s) * p))]

    return f"p10 {q(0.1):.2f} / p50 {q(0.5):.2f} / p90 {q(0.9):.2f}"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="streettracker alpr-rescore",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_dir", type=Path)
    ap.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR_MODEL,
        help=f"fast-plate-ocr model -- the one alpr-run used (default {DEFAULT_OCR_MODEL})",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="proceed when some plate crops are missing; those reads have their "
        "confidence cleared (default: refuse, and write nothing)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the before/after summary without writing anything",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-score even if the session is already stamped ocr_conf=min_char",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    session_dir: Path = args.session_dir
    label = session_dir.name
    alpr_path = session_dir / f"{label}_alpr.json"
    by_track_path = session_dir / f"{label}_alpr_by_track.json"
    stamp_path = session_dir / f"{label}_static_plates.json"

    if not alpr_path.is_file():
        print(
            f"[alpr-rescore] missing {alpr_path}; run "
            f"`streettracker alpr-run {session_dir}` first.",
            file=sys.stderr,
        )
        return 2

    stamp: dict[str, Any] | None = None
    if stamp_path.is_file():
        try:
            loaded = json.loads(stamp_path.read_text(encoding="utf-8"))
            stamp = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            stamp = None
    if stamp and stamp.get("ocr_conf") == OCR_CONF_METHOD and not args.force:
        print(
            f"[alpr-rescore] {label}: already re-scored (ocr_conf={OCR_CONF_METHOD}); nothing to do"
        )
        return 0

    records: list[dict[str, Any]] = json.loads(alpr_path.read_text(encoding="utf-8"))
    rescorable = [r for r in records if is_rescorable(r)]
    missing = [r for r in rescorable if crop_path_for(r, session_dir) is None]
    print(f"[alpr-rescore] {label}: {len(rescorable)} fast-plate-ocr reads with saved plate crops")
    if missing and not args.allow_missing:
        print(
            f"[alpr-rescore] {len(missing)} of them have no crop on disk (e.g. "
            f"{missing[0].get('image')}). Nothing written. Re-run `streettracker alpr-run "
            f"{session_dir}` to regenerate the crops, or pass --allow-missing to clear "
            f"those reads' confidence instead.",
            file=sys.stderr,
        )
        return 2

    # The summary reports against the gate dvsa-label will apply next.
    try:
        gate, gate_source = resolve_plate_conf_threshold()
    except ValueError as exc:
        print(f"[alpr-rescore] {exc}", file=sys.stderr)
        return 2

    old_rollup = _rollup_by_track(records)

    from streettracker.analysis.alpr.preferred import FastPlateOcrRecognizer

    recognizer = FastPlateOcrRecognizer(args.ocr_model)
    stats = rescore_records(
        records,
        session_dir,
        recognizer.recognize,
        progress=lambda i, n: print(f"  [batch] {i}/{n} done", flush=True),
    )
    new_rollup = _rollup_by_track(records)

    pct_changed = 100.0 * stats.n_text_changed / stats.n_rescored if stats.n_rescored else 0.0
    print(
        f"[alpr-rescore] re-scored {stats.n_rescored} reads; text changed on "
        f"{stats.n_text_changed} ({pct_changed:.1f} %), {stats.n_now_unread} now unread"
    )
    if stats.n_missing:
        print(f"[alpr-rescore] {stats.n_missing} reads had no usable crop: confidence cleared")
    print(
        f"[alpr-rescore] reads with ocr_conf >= {gate} ({gate_source}): before "
        f"{_share_at_least(stats.old_conf, gate)} -> after "
        f"{_share_at_least(stats.new_conf, gate)}; new ocr_conf {_quantiles(stats.new_conf)}"
    )
    n_old, n_new = gated_tracks(old_rollup, gate=gate), gated_tracks(new_rollup, gate=gate)
    print(
        f"[alpr-rescore] tracks with a UK-shaped best read at ocr_conf >= {gate} "
        f"(what dvsa-label looks up): {n_old} -> {n_new}"
    )

    if args.dry_run:
        print("[alpr-rescore] dry run: nothing written")
        return 0

    atomic_write_text(alpr_path, json.dumps(records, indent=2))
    atomic_write_text(by_track_path, json.dumps(new_rollup, indent=2))
    if stamp is not None:
        stamp["ocr_conf"] = OCR_CONF_METHOD
        stamp["ocr_rescored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_text(stamp_path, json.dumps(stamp, indent=2))
    else:
        print(
            f"[alpr-rescore] note: no provenance stamp ({stamp_path.name}), so this session "
            f"can't be marked as re-scored; a later run will re-score it again"
        )
    print(f"[alpr-rescore] wrote {alpr_path.name} + {by_track_path.name}")
    print(f"[alpr-rescore] next: streettracker dvsa-label {session_dir} -> dvsa-apply -> vehicles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
