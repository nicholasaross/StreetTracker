"""Apply the DVSA MOT label harvest to a session's per-track records.

``streettracker dvsa-label`` harvests ``<session>_dvsa_labels.json``
(plate -> make/model/year, each label carrying the ``track_ids`` it
resolved to). This pass folds those labels back onto the per-track
records: every car track whose plate was DVSA-labelled gets ``make`` /
``model`` / ``year`` + ``make_model_source="dvsa"`` written into
``<session>_events.jsonl`` and ``<session>_data.json`` (atomic).

The per-vehicle view already joins the harvest at aggregation time
(``streettracker vehicles``); this is the per-track counterpart so
``data.json`` (and any per-track consumer / dashboard) carries
make/model too.

Mirrors ``streettracker recolor``'s rewrite pattern. Local + cheap (no
API calls) and idempotent -- re-run any time the harvest changes. When
the CompCars classifier lands it will write the same fields with
``make_model_source="cnn"``.

    streettracker dvsa-apply <session_dir>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ApplyStats:
    """What an apply pass touched."""

    records_total: int = 0
    cars_total: int = 0
    cars_labelled: int = 0
    track_ids_available: int = 0


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _track_label_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Invert the plate-keyed harvest into ``track_id -> {make,model,year}``
    via each label row's ``track_ids`` list. dvsa-label groups tracks by
    exact plate string, so a track maps to at most one label -- no fuzzy
    ambiguity at the per-track level."""
    out: dict[int, dict[str, Any]] = {}
    for row in (payload.get("labels") or {}).values():
        info = {
            "make": row.get("make") or None,
            "model": row.get("model") or None,
            "year": row.get("year"),
        }
        for tid in row.get("track_ids") or []:
            out[int(tid)] = info
    return out


def apply_dvsa_labels(session_dir: Path) -> ApplyStats:
    """Write make/model/year onto the session's car records from the DVSA
    harvest, rewriting ``*_events.jsonl`` + ``*_data.json`` atomically.

    A missing / empty / unparseable harvest is a no-op (zeroed stats, no
    file touched). Raises ``FileNotFoundError`` only when a harvest IS
    present but the session has no ``*_events.jsonl`` to rewrite.
    """
    labels_paths = sorted(session_dir.glob("*_dvsa_labels.json"))
    tmap: dict[int, dict[str, Any]] = {}
    if labels_paths:
        try:
            tmap = _track_label_map(
                json.loads(labels_paths[0].read_text(encoding="utf-8"))
            )
        except json.JSONDecodeError:
            tmap = {}
    if not tmap:
        return ApplyStats(track_ids_available=0)

    jsonl_paths = sorted(session_dir.glob("*_events.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No *_events.jsonl in {session_dir}")
    jsonl_path = jsonl_paths[0]
    data_paths = sorted(session_dir.glob("*_data.json"))

    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    stats = ApplyStats(records_total=len(records), track_ids_available=len(tmap))
    for r in records:
        if r.get("class_name") != "car":
            continue
        stats.cars_total += 1
        info = tmap.get(int(r["track_id"]))
        if info is None:
            continue
        r["make"] = info["make"]
        r["model"] = info["model"]
        r["year"] = info["year"]
        r["make_model_source"] = "dvsa"
        stats.cars_labelled += 1

    lines = [json.dumps(r, separators=(",", ":")) for r in records]
    _atomic_write(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))
    if data_paths:
        _atomic_write(data_paths[0], json.dumps(records, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="streettracker dvsa-apply",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("session_dir", type=Path, help="closed session directory")
    args = parser.parse_args(argv)

    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"[dvsa-apply] not a directory: {session_dir}")
        return 1
    if not sorted(session_dir.glob("*_dvsa_labels.json")):
        print(
            f"[dvsa-apply] no *_dvsa_labels.json in {session_dir} -- run "
            f"`streettracker dvsa-label {session_dir}` first."
        )
        return 2

    stats = apply_dvsa_labels(session_dir)
    print(
        f"[dvsa-apply] {stats.cars_labelled}/{stats.cars_total} car tracks "
        f"labelled from {stats.track_ids_available} DVSA-mapped track ids "
        f"-> make/model/year written to events.jsonl + data.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
