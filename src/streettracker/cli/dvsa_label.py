"""DVSA MOT label accumulator: per-session plate -> make/model lookups.

    streettracker dvsa-label <session_dir>
        [--config configs/dvsa.json]
        [--conf-threshold 0.9]
        [--limit N]
        [--re-label]

Reads ``<session>_alpr_by_track.json`` (produced by ``streettracker
alpr-run``), picks the tracks whose best preferred-pipeline OCR read
clears ``--conf-threshold``, and calls the DVSA MOT history API for
each distinct plate. Writes the harvest to
``<session>_dvsa_labels.json``.

Re-running is idempotent: by default, plates already present in the
output file are skipped. Pass ``--re-label`` to overwrite all entries
(useful if the DVSA register has updated, e.g. a new MOT lifted a
"not on record" plate into the dataset). Plates that returned 404
are remembered under the ``"unknown"`` list and not retried unless
``--re-label`` is set -- a 3-year-old vehicle does not become MOT-
eligible mid-session.

The accumulator is the ground-truth source for the in-house
make/model classifier's regression test (see the planning doc at
``.claude/makemodel_design.md`` and the approved plan file).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

from streettracker.analysis.dvsa import DvsaClient, DvsaConfig, MotLookupResult

logger = logging.getLogger(__name__)

# Default path matches the camera.json convention.
_DEFAULT_CONFIG_PATH = Path("configs/dvsa.json")
_DEFAULT_CONF_THRESHOLD = 0.9


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="streettracker dvsa-label",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_dir", type=Path)
    ap.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=(
            "Path to the dvsa.json credential file (default: "
            f"{_DEFAULT_CONFIG_PATH}). See configs/dvsa.example.json."
        ),
    )
    ap.add_argument(
        "--conf-threshold",
        type=float,
        default=_DEFAULT_CONF_THRESHOLD,
        help=(
            f"Minimum preferred-pipeline OCR confidence to attempt a lookup "
            f"(default: {_DEFAULT_CONF_THRESHOLD}). Below this we treat the "
            f"plate as unread."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N distinct plates (0 = all). Smoke tests.",
    )
    ap.add_argument(
        "--re-label",
        action="store_true",
        help=(
            "Overwrite all existing labels, including 'unknown' plates. "
            "Default is incremental: only newly-present plates are looked up."
        ),
    )
    return ap


@dataclasses.dataclass(frozen=True, slots=True)
class _PlateRequest:
    """One distinct plate to look up, together with the set of track
    IDs that resolved to it."""

    plate: str
    plate_conf: float
    track_ids: list[int]


def _collect_plate_requests(
    by_track: dict[str, Any], conf_threshold: float
) -> list[_PlateRequest]:
    """Walk ``alpr_by_track.json`` and group its tracks by best-read
    plate. Per-plate aggregation avoids billing 80 lookups for the same
    car that BotSORT split into 80 tracks (the FD61PVX failure mode)."""
    by_plate: dict[str, _PlateRequest] = {}
    for track in by_track.get("tracks", []):
        best = track.get("best_preferred")
        if not best:
            continue
        plate = (best.get("ocr_text") or "").strip().upper().replace(" ", "")
        conf = float(best.get("ocr_conf") or 0.0)
        if not plate or conf < conf_threshold:
            continue
        tid = int(track["track_id"])
        existing = by_plate.get(plate)
        if existing is None:
            by_plate[plate] = _PlateRequest(
                plate=plate,
                plate_conf=conf,
                track_ids=[tid],
            )
        else:
            existing.track_ids.append(tid)
            # Take the highest plate_conf seen across the group; the
            # frozen dataclass means we rebuild rather than mutate.
            if conf > existing.plate_conf:
                by_plate[plate] = dataclasses.replace(existing, plate_conf=conf)
    # Sort for stable output ordering -- humans diff these files.
    return sorted(by_plate.values(), key=lambda r: r.plate)


def _result_to_json(
    res: MotLookupResult, plate_conf: float, track_ids: list[int]
) -> dict[str, Any]:
    return {
        "plate": res.registration,
        "plate_conf": plate_conf,
        "track_ids": sorted(track_ids),
        "make": res.make,
        "model": res.model,
        "year": res.year,
        "primary_colour": res.primary_colour,
        "fuel_type": res.fuel_type,
        "engine_size_cc": res.engine_size_cc,
        "looked_up_at": res.looked_up_at,
    }


def _load_existing(path: Path) -> dict[str, Any]:
    """Return the previous output if present, or an empty skeleton."""
    if not path.exists():
        return {"labels": {}, "unknown": []}
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "[dvsa-label] %s is corrupt (%s) -- starting fresh.", path, exc
        )
        return {"labels": {}, "unknown": []}
    # Be tolerant of older variants that may have omitted "unknown".
    prev.setdefault("labels", {})
    prev.setdefault("unknown", [])
    return prev


def _write_output(
    path: Path,
    *,
    session_label: str,
    labels: dict[str, dict[str, Any]],
    unknown: list[str],
    n_high_conf_plates: int,
) -> None:
    payload: dict[str, Any] = {
        "session": session_label,
        "labelled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "n_high_conf_plates": n_high_conf_plates,
        "n_labelled": len(labels),
        "n_unknown": len(unknown),
        "labels": labels,
        "unknown": sorted(set(unknown)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)

    session_dir: Path = args.session_dir.resolve()
    session_label = session_dir.name
    by_track_path = session_dir / f"{session_label}_alpr_by_track.json"
    out_path = session_dir / f"{session_label}_dvsa_labels.json"

    if not by_track_path.exists():
        print(
            f"[dvsa-label] missing {by_track_path}; "
            f"run `streettracker alpr-run {session_dir}` first.",
            file=sys.stderr,
        )
        return 2

    if not args.config.exists():
        print(
            f"[dvsa-label] missing credentials at {args.config}. "
            f"Copy configs/dvsa.example.json to that path and fill it in.",
            file=sys.stderr,
        )
        return 2

    cfg = DvsaConfig.from_json_file(args.config)
    by_track = json.loads(by_track_path.read_text(encoding="utf-8"))
    requests_ = _collect_plate_requests(by_track, args.conf_threshold)
    print(
        f"[dvsa-label] {len(requests_)} distinct high-conf plates "
        f"(>= {args.conf_threshold}) across {sum(len(r.track_ids) for r in requests_)} tracks"
    )

    existing = _load_existing(out_path)
    existing_labels: dict[str, dict[str, Any]] = dict(existing["labels"])
    existing_unknown: set[str] = set(existing["unknown"])
    if args.re_label:
        # Drop everything; we'll re-query every plate.
        existing_labels.clear()
        existing_unknown.clear()

    client = DvsaClient(cfg)

    processed = 0
    new_labels = 0
    new_unknown = 0
    api_errors = 0
    for req in requests_:
        if args.limit and processed >= args.limit:
            break
        processed += 1
        if req.plate in existing_labels or req.plate in existing_unknown:
            # Already known -- still refresh track_ids in case BotSORT
            # surfaced new tracks pointing at the same plate this run.
            if req.plate in existing_labels:
                row = existing_labels[req.plate]
                row["track_ids"] = sorted(set(row.get("track_ids", [])) | set(req.track_ids))
            continue
        try:
            result = client.lookup_plate(req.plate)
        except Exception as exc:  # pragma: no cover - hard-to-mock network
            logger.error(
                "[dvsa-label] %s lookup failed: %s: %s",
                req.plate, type(exc).__name__, exc,
            )
            api_errors += 1
            continue
        if result is None:
            existing_unknown.add(req.plate)
            new_unknown += 1
        else:
            existing_labels[req.plate] = _result_to_json(
                result, req.plate_conf, req.track_ids
            )
            new_labels += 1
        # Persist after every successful lookup so a SIGKILL doesn't
        # lose the work. Cheap -- the output is < 1 MB even for a
        # multi-day soak.
        _write_output(
            out_path,
            session_label=session_label,
            labels=existing_labels,
            unknown=sorted(existing_unknown),
            n_high_conf_plates=len(requests_),
        )

    print(
        f"[dvsa-label] wrote {out_path}: {len(existing_labels)} labelled, "
        f"{len(existing_unknown)} unknown, {api_errors} api errors "
        f"({new_labels} new labels + {new_unknown} new unknown this run)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
