"""JSON / JSONL output: EventLog (crash-safe append) and save_json helpers.

Ported from NanoTracker's `EventLog` (nano_tracker.py line ~953) and
`save_json` (line ~943). Behaviour is unchanged so existing pulled
sessions remain readable by the new analysers and vice versa.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

from streettracker.common.schema import SessionMeta, TrackRecord

logger = logging.getLogger(__name__)


def format_wall(unix_ts: float) -> str:
    """Local-time ISO-8601 with timezone offset, e.g.
    ``2026-05-17T14:32:05+01:00``. Matches NanoTracker's `format_wall()`.
    """
    return (
        datetime.datetime.fromtimestamp(unix_ts)
        .astimezone()
        .isoformat(timespec="seconds")
    )


class EventLog:
    """Append-on-finalize JSONL event writer.

    Each finalized track is written as a single JSON line and immediately
    flushed + fsynced. A crash loses only tracks still active in memory
    at crash time — never tracks already finalized.

    The file handle is opened in append mode so re-running a session with
    the same output directory cleanly extends the existing log.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        # SIM115: file handle is intentionally long-lived (one open per session).
        self._fh = open(str(path), "a", encoding="utf-8")  # noqa: SIM115
        self.count = 0

    def append(self, record: TrackRecord | dict[str, Any]) -> None:
        d = record.to_json_dict() if isinstance(record, TrackRecord) else record
        self._fh.write(json.dumps(d, separators=(",", ":")) + "\n")
        self._fh.flush()
        # fsync can fail on tmpfs / NFS / etc. — not fatal.
        with contextlib.suppress(OSError):
            os.fsync(self._fh.fileno())
        self.count += 1

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def save_data_json(records: list[TrackRecord | dict[str, Any]], path: Path) -> None:
    """Write the records as a bare top-level array (for jq / pandas / SQL
    ingestion). Pretty-printed; size is small enough (~150 B/row) that
    even a 24h session is a few MB.
    """
    payload = [
        r.to_json_dict() if isinstance(r, TrackRecord) else r
        for r in records
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_meta_json(meta: SessionMeta | dict[str, Any], path: Path) -> None:
    payload = meta.to_json_dict() if isinstance(meta, SessionMeta) else meta
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_json(
    records: list[TrackRecord | dict[str, Any]],
    meta: SessionMeta | dict[str, Any],
    data_path: Path,
    meta_path: Path,
) -> None:
    """Convenience wrapper: write both data and meta JSON in one call.
    Matches NanoTracker's `save_json` signature."""
    save_data_json(records, data_path)
    save_meta_json(meta, meta_path)


def read_data_json(path: Path) -> list[TrackRecord]:
    """Read a `{session}_data.json` produced by `save_data_json`. Tolerates
    extra fields (forward-compatible) and missing optional ones."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TrackRecord.from_json_dict(r) for r in raw]


def read_events_jsonl(path: Path) -> list[TrackRecord]:
    """Read a `{session}_events.jsonl` produced by `EventLog`.

    Tolerates partial-write tails:
    - blank lines are skipped silently
    - a final line that fails JSON parsing is logged and skipped (rather
      than aborting the read). EventLog's per-record flush+fsync narrows
      the corruption window to "last record only" -- everything before
      it is intact and must remain readable so the operator can recover
      crashed-session data.
    """
    out: list[TrackRecord] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[output] skipping torn line %d in %s: %s", n, path, exc
            )
            continue
        out.append(TrackRecord.from_json_dict(payload))
    return out


# ----------------------------------------------------------------------
# IR event log (crash-safe append).
#
# IRPeriod records are accumulated in memory and written to ``meta.json``
# at session end. On graceful SIGTERM the runtime synthesises a closing
# record for an in-progress period via ``_close_open_ir_period``. On a
# SIGKILL (systemd ``TimeoutStopSec`` exceeded mid-drain), nothing
# writes to meta.json and the in-progress IR period is lost entirely --
# the events.jsonl crash-safety guarantee did not extend to IR state.
#
# IREventLog mirrors :class:`EventLog`: one fsynced JSON line per IR
# transition. After a crash the operator can reconstruct the period
# history by reading the sidecar with :func:`read_ir_events_jsonl`
# (interleaved start/end pairs, possibly with a trailing unmatched
# start = the period that was active at crash time).


class IREventLog:
    """Append-on-transition IR event writer.

    Each transition writes one line of
    ``{"event": "start" | "end", "unix": float, "iso": str}`` and is
    flushed + fsynced immediately, matching :class:`EventLog`'s
    crash-safety contract.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        # SIM115: file handle is intentionally long-lived (one open per session).
        self._fh = open(str(path), "a", encoding="utf-8")  # noqa: SIM115
        self.count = 0

    def _append(self, event: str, unix_ts: float) -> None:
        line = json.dumps(
            {"event": event, "unix": unix_ts, "iso": format_wall(unix_ts)},
            separators=(",", ":"),
        )
        self._fh.write(line + "\n")
        self._fh.flush()
        with contextlib.suppress(OSError):
            os.fsync(self._fh.fileno())
        self.count += 1

    def append_start(self, unix_ts: float) -> None:
        """Record a day -> IR transition."""
        self._append("start", unix_ts)

    def append_end(self, unix_ts: float) -> None:
        """Record an IR -> day transition (or a synthesised close at
        graceful shutdown)."""
        self._append("end", unix_ts)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()

    def __enter__(self) -> IREventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_ir_events_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an ``_ir_events.jsonl`` produced by :class:`IREventLog`.

    Same torn-tail tolerance as :func:`read_events_jsonl`. Returns the
    events in file order; an odd-length result (trailing unmatched
    "start") indicates an unfinished IR period at crash time.
    """
    out: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning(
                "[output] skipping torn IR-event line %d in %s: %s", n, path, exc
            )
    return out
