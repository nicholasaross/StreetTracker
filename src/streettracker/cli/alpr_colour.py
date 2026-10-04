"""``streettracker alpr-colour <session>`` -- apply the plate-colour check
to a session's existing ALPR output.

``alpr-run`` checks plate colour itself since 2026-10-04 (review E1.3).
For sessions enriched before that, this classifies every saved plate crop
(white front / yellow rear, ``analysis.alpr.plate_colour``), marks reads whose
colour contradicts their track's direction ``colour_suspect``, and rewrites
``<session>_alpr.json`` and ``<session>_alpr_by_track.json`` with the same
rollup ``alpr-run`` uses -- so a track whose best read was another car's plate
falls back to its next read, and every best read gains ``n_agree`` (other snaps
agreeing, for the combined plate gate). No OCR or detection re-runs; only the
saved crops are read. Stamps ``"plate_colour"`` into
``<session>_static_plates.json``. Re-run ``dvsa-label`` -> ``dvsa-apply`` ->
``vehicles`` afterwards (the panel's plate-check playbook does all of it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from streettracker.analysis.alpr.base import atomic_write_text
from streettracker.analysis.alpr.plate_colour import PLATE_COLOUR_METHOD, mark_colour_suspects
from streettracker.cli.alpr_run import _direction_by_track, _rollup_by_track, _stamp


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="streettracker alpr-colour", description=__doc__)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument(
        "--dry-run", action="store_true", help="report what would change; write nothing"
    )
    return ap


def _best_changes(old: dict, new: dict) -> tuple[int, int]:
    """(tracks whose best preferred read changed, tracks that lost it)."""
    o = {t["track_id"]: t.get("best_preferred") for t in old.get("tracks", [])}
    n = {t["track_id"]: t.get("best_preferred") for t in new.get("tracks", [])}
    changed = lost = 0
    for tid, ob in o.items():
        if not ob:
            continue
        nb = n.get(tid)
        if not nb:
            lost += 1
        elif (nb.get("snap_index"), nb.get("ocr_text")) != (
            ob.get("snap_index"),
            ob.get("ocr_text"),
        ):
            changed += 1
    return changed, lost


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    session_dir: Path = args.session_dir
    label = session_dir.name
    alpr_path = session_dir / f"{label}_alpr.json"
    by_track_path = session_dir / f"{label}_alpr_by_track.json"
    if not alpr_path.is_file():
        print(f"[alpr-colour] missing {alpr_path}; run alpr-run first", file=sys.stderr)
        return 2
    records = json.loads(alpr_path.read_text(encoding="utf-8"))
    old_rollup = _rollup_by_track(records)
    stats = mark_colour_suspects(records, session_dir, _direction_by_track(session_dir))
    new_rollup = _rollup_by_track(records)
    changed, lost = _best_changes(old_rollup, new_rollup)
    labels = {k: v for k, v in stats.items() if k not in ("suspect", "no_crop")}
    print(
        f"[alpr-colour] {label}: {sum(labels.values())} reads classified {labels}; "
        f"{stats['no_crop']} without a crop; {stats['suspect']} colour_suspect"
    )
    print(
        f"[alpr-colour] {label}: best read changed on {changed} track(s), "
        f"{lost} track(s) left with no read"
    )
    if args.dry_run:
        print("[alpr-colour] dry run: nothing written")
        return 0
    atomic_write_text(alpr_path, json.dumps(records, indent=2))
    atomic_write_text(by_track_path, json.dumps(new_rollup, indent=2))
    _stamp(session_dir, plate_colour=PLATE_COLOUR_METHOD)
    print(f"[alpr-colour] wrote {alpr_path.name} + {by_track_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
