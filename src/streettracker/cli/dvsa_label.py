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
``docs/makemodel_design.md`` and the approved plan file).
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

from streettracker.analysis.dvsa import (
    DvsaClient,
    DvsaConfig,
    MotLookupResult,
    is_canonical_uk_plate,
)
from streettracker.analysis.parked import (
    ParkedDetection,
    best_unsuppressed_read,
    detect_parked,
    load_alpr_entries,
)

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
    ap.add_argument(
        "--include-non-canonical",
        action="store_true",
        help=(
            "Look up plates whose shape does not match any of the UK car-"
            "plate formats (current 'LL00 LLL', 1983-2001 prefix, "
            "1963-1983 suffix). Off by default because the 2026-05-29 "
            "smoke test found 57 percent of high-conf 'reads' were OCR "
            "misreads (e.g. '123889', '1157WHT') that just burn API budget. "
            "Skipped plates land in the output's 'skipped_non_canonical' "
            "list for inspection."
        ),
    )
    ap.add_argument(
        "--no-parked-suppression",
        action="store_true",
        help=(
            "Disable the stationary-beacon filter. By default, tracks "
            "whose best read came from a PARKED car's static plate "
            "(same plate at a fixed image position across >=4 distinct "
            "moving tracks) are re-anchored to their own next-best read "
            "before plates are grouped -- otherwise dozens of passing "
            "cars inherit the parked car's plate and the make/model "
            "training corpus is mislabelled accordingly."
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
    by_track: dict[str, Any],
    conf_threshold: float,
    *,
    canonical_only: bool = True,
    detection: ParkedDetection | None = None,
) -> tuple[list[_PlateRequest], list[str]]:
    """Walk ``alpr_by_track.json`` and group its tracks by best-read
    plate. Per-plate aggregation avoids billing 80 lookups for the same
    car that BotSORT split into 80 tracks (the FD61PVX failure mode).

    When ``canonical_only`` (the default), plates that fail
    :func:`is_canonical_uk_plate` are returned in the second tuple
    element instead of the request list; the caller surfaces them in
    the output JSON's ``skipped_non_canonical`` field rather than
    billing them to DVSA. Set ``False`` to recover the pre-2026-05-29
    behaviour.
    """
    by_plate: dict[str, _PlateRequest] = {}
    skipped_non_canonical: set[str] = set()
    for track in by_track.get("tracks", []):
        best = track.get("best_preferred")
        if not best:
            continue
        if detection is not None:
            try:
                key = (int(track["track_id"]), int(best.get("snap_index")))
            except (KeyError, TypeError, ValueError):
                key = None
            if key is not None and key in detection.suppressed:
                # Beacon read: the plate belongs to a parked car, not
                # this passing track. Re-anchor to the track's own best
                # remaining read, or drop the track from attribution.
                best = best_unsuppressed_read(
                    detection.reads_by_track.get(key[0], []),
                    detection.suppressed,
                    conf_threshold=conf_threshold,
                    canonical_only=canonical_only,
                )
                if best is None:
                    continue
        plate = (best.get("ocr_text") or "").strip().upper().replace(" ", "")
        conf = float(best.get("ocr_conf") or 0.0)
        if not plate or conf < conf_threshold:
            continue
        if canonical_only and not is_canonical_uk_plate(plate):
            skipped_non_canonical.add(plate)
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
    return sorted(by_plate.values(), key=lambda r: r.plate), sorted(
        skipped_non_canonical
    )


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
        return {"labels": {}, "unknown": [], "skipped_non_canonical": []}
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "[dvsa-label] %s is corrupt (%s) -- starting fresh.", path, exc
        )
        return {"labels": {}, "unknown": [], "skipped_non_canonical": []}
    # Be tolerant of older variants that may have omitted these keys
    # (pre-canonical-filter outputs).
    prev.setdefault("labels", {})
    prev.setdefault("unknown", [])
    prev.setdefault("skipped_non_canonical", [])
    return prev


def _write_output(
    path: Path,
    *,
    session_label: str,
    labels: dict[str, dict[str, Any]],
    unknown: list[str],
    skipped_non_canonical: list[str],
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
        "n_skipped_non_canonical": len(skipped_non_canonical),
        "labels": labels,
        "unknown": sorted(set(unknown)),
        "skipped_non_canonical": sorted(set(skipped_non_canonical)),
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

    # Stationary-beacon suppression: keep parked cars' plates from being
    # attributed (and DVSA-labelled) onto every track that drove past
    # them. Needs the per-image reads + track records; silently skipped
    # when either file is absent.
    detection: ParkedDetection | None = None
    if not args.no_parked_suppression:
        entries = load_alpr_entries(session_dir)
        data_path = session_dir / f"{session_label}_data.json"
        if entries and data_path.exists():
            try:
                records = json.loads(data_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                records = []
            if records:
                detection = detect_parked(entries, records)
                if detection.episodes:
                    n_tracks = len({k[0] for k in detection.suppressed})
                    print(
                        f"[dvsa-label] parked-plate suppression: "
                        f"{len(detection.episodes)} beacon(s); "
                        f"{len(detection.suppressed)} reads across "
                        f"{n_tracks} passing tracks re-anchored or dropped"
                    )

    requests_, skipped_non_canonical = _collect_plate_requests(
        by_track,
        args.conf_threshold,
        canonical_only=not args.include_non_canonical,
        detection=detection,
    )
    n_tracks_billed = sum(len(r.track_ids) for r in requests_)
    print(
        f"[dvsa-label] {len(requests_)} distinct high-conf plates "
        f"(>= {args.conf_threshold}) across {n_tracks_billed} tracks"
        + (
            f"; skipping {len(skipped_non_canonical)} non-canonical "
            f"(use --include-non-canonical to query them)"
            if skipped_non_canonical
            else ""
        )
    )

    existing = _load_existing(out_path)
    existing_labels: dict[str, dict[str, Any]] = dict(existing["labels"])
    existing_unknown: set[str] = set(existing["unknown"])
    # Merge the run's skipped set with any prior skipped record; the
    # output is the cumulative diagnostic, not just this run's.
    existing_skipped: set[str] = set(existing["skipped_non_canonical"]) | set(
        skipped_non_canonical
    )
    if args.re_label:
        # Drop everything; we'll re-query every plate.
        existing_labels.clear()
        existing_unknown.clear()
        existing_skipped = set(skipped_non_canonical)

    # A labelled plate with NO anchored tracks in the current rollup keeps
    # its DVSA data but loses its track attributions. Without this, a
    # parked-beacon plate whose every read was suppressed (no genuine
    # sighting left) would retain its stale host tracks from an older
    # harvest -- and keep feeding mislabelled crops to the training
    # corpus.
    request_plates = {r.plate for r in requests_}
    n_orphaned = 0
    for plate, row in existing_labels.items():
        if plate not in request_plates and row.get("track_ids"):
            row["track_ids"] = []
            n_orphaned += 1
    if n_orphaned:
        print(
            f"[dvsa-label] cleared track_ids on {n_orphaned} labelled "
            f"plate(s) with no remaining anchored tracks"
        )

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
            # Already known -- still refresh track_ids without an API
            # call. REPLACE rather than union: the request set is
            # recomputed in full from the current rollup each run, so a
            # plain re-run both picks up newly-surfaced tracks and scrubs
            # stale attributions (e.g. parked-beacon hosts suppressed by
            # the stationary-plate filter) from an older harvest.
            if req.plate in existing_labels:
                row = existing_labels[req.plate]
                row["track_ids"] = sorted(set(req.track_ids))
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
            skipped_non_canonical=sorted(existing_skipped),
            n_high_conf_plates=len(requests_),
        )

    # Final flush: ensures the output is updated even when the loop
    # ran zero iterations (e.g. every plate was already known but the
    # skipped_non_canonical list grew this run).
    _write_output(
        out_path,
        session_label=session_label,
        labels=existing_labels,
        unknown=sorted(existing_unknown),
        skipped_non_canonical=sorted(existing_skipped),
        n_high_conf_plates=len(requests_),
    )

    print(
        f"[dvsa-label] wrote {out_path}: {len(existing_labels)} labelled, "
        f"{len(existing_unknown)} unknown, {len(existing_skipped)} skipped "
        f"(non-canonical), {api_errors} api errors "
        f"({new_labels} new labels + {new_unknown} new unknown this run)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
