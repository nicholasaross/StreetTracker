"""Rebuild a crashed session's end-of-session files from its crash-safe logs.

The runtime writes ``_data.json`` / ``_meta.json`` / ``_hourly.json`` only on
a clean shutdown (``device.runtime.write_session_outputs``). A power cut,
kernel panic or SIGKILL skips that step and leaves just:

- ``_events.jsonl`` -- one fsynced line per finalised track, the exact dict
  ``_data.json`` would have held (only the tracks still in flight at the
  crash are lost);
- ``_ir_events.jsonl`` -- one fsynced line per IR transition;
- ``_summary.html`` -- rewritten whenever the runtime is idle after a
  finalise, so it embeds the session meta (frame size, snap stats, frame
  count) as of a few seconds before the crash.

Every offline tool keys off ``_data.json``: ``vehicles`` fails without it,
the showcase and ``/stats`` skip the session, and ``alpr-run`` quietly loses
its per-snap bbox hints, track directions and the sub-stream frame size (the
last lives in the meta). So a crashed session is invisible to enrichment
until these files are rebuilt. Surfaced 2026-10-23: the Orin lost power
39.5 h into ``session_20261021_161948``, whose 1,885 finalised tracks then
sat unusable behind a missing ``_data.json``.
"""

from __future__ import annotations

import ast
import datetime
import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from streettracker.common.hourly import build_hourly_rollup
from streettracker.common.output import format_wall

__all__ = [
    "RecoveryResult",
    "ir_periods_from_events",
    "meta_from_summary_html",
    "needs_recovery",
    "recover_session",
]

# generate_html renders the meta as ``k: v · k: v`` inside this div, with
# each value's str() (so dicts/lists are Python reprs) and the whole line
# HTML-escaped.
_META_DIV_RE = re.compile(r'<div class="meta">(.*?)</div>', re.S)
_META_SEP = " · "


@dataclass(slots=True)
class RecoveryResult:
    """What :func:`recover_session` rebuilt, for the caller to report."""

    n_records: int
    meta_source: str  # "summary" (embedded meta) or "label" (minimal fallback)
    frame_size: list[int] | None
    last_track_end: str | None
    written: list[Path] = field(default_factory=list)

    def describe(self) -> str:
        """One-line report for ``recover-session`` and ``pull``."""
        meta = (
            f"meta from the summary (frame_size {self.frame_size})"
            if self.meta_source == "summary"
            else "minimal meta from the label (no frame size, so alpr-run gets no bbox hints)"
        )
        return (
            f"rebuilt {', '.join(p.name for p in self.written)}: {self.n_records} tracks "
            f"(last ended {self.last_track_end}), {meta}"
        )


def needs_recovery(session_dir: Path) -> bool:
    """True when the session has finalised tracks but no ``_data.json``."""
    label = session_dir.name
    return (session_dir / f"{label}_events.jsonl").is_file() and not (
        session_dir / f"{label}_data.json"
    ).exists()


def meta_from_summary_html(text: str) -> dict[str, Any] | None:
    """Parse the session meta that ``generate_html`` embeds in the summary.

    Returns ``None`` when the div is missing or lacks the required keys.
    Values that are not Python literals (the bare session label) are kept
    as strings.
    """
    m = _META_DIV_RE.search(text)
    if not m:
        return None
    meta: dict[str, Any] = {}
    for part in html.unescape(m.group(1)).split(_META_SEP):
        key, sep, raw = part.partition(": ")
        if not sep:
            continue
        try:
            meta[key.strip()] = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            meta[key.strip()] = raw.strip()
    if not isinstance(meta.get("session_label"), str) or not isinstance(
        meta.get("session_start_unix"), (int, float)
    ):
        return None
    return meta


def ir_periods_from_events(
    events: list[dict[str, Any]], *, close_at_unix: float | None
) -> list[dict[str, Any]]:
    """Pair ``_ir_events.jsonl`` start/end lines into ``IRPeriod`` dicts.

    A trailing unmatched start is the period the crash interrupted; it is
    closed at ``close_at_unix`` (the last sign of life), as the runtime's
    ``_close_open_ir_period`` would have closed it at shutdown.
    """
    periods: list[dict[str, Any]] = []
    open_unix: float | None = None

    def _close(start: float, end: float) -> None:
        periods.append(
            {
                "start": format_wall(start),
                "end": format_wall(end),
                "duration_s": round(end - start, 1),
            }
        )

    for ev in events:
        unix = ev.get("unix")
        if not isinstance(unix, (int, float)):
            continue
        if ev.get("event") == "start":
            open_unix = float(unix)
        elif ev.get("event") == "end" and open_unix is not None:
            _close(open_unix, float(unix))
            open_unix = None
    if open_unix is not None and close_at_unix is not None and close_at_unix > open_unix:
        _close(open_unix, close_at_unix)
    return periods


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Raw dict per line, skipping blank and torn lines (the crash can tear
    the last one)."""
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _label_start_unix(label: str) -> float | None:
    try:
        stamp = datetime.datetime.strptime(label.removeprefix("session_"), "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return stamp.timestamp()


def _write_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def recover_session(session_dir: Path) -> RecoveryResult:
    """Write the missing ``_data.json`` / ``_meta.json`` / ``_hourly.json``.

    ``_data.json`` is the events log's records verbatim, in finalise order
    -- the same list the runtime would have saved. ``_meta.json`` is the
    summary's embedded meta (IR periods taken from ``_ir_events.jsonl``,
    which is complete where the summary's copy may miss the open period),
    plus a ``recovered`` provenance block; without a usable summary it falls
    back to a minimal meta dated from the label, with no frame size.

    ``_data.json`` is written last and every write is atomic, so its
    presence means the recovery finished. Refuses (``FileExistsError``) when
    ``_data.json`` already exists: that is a finalised, possibly enriched
    session.
    """
    label = session_dir.name
    data_path = session_dir / f"{label}_data.json"
    events_path = session_dir / f"{label}_events.jsonl"
    if data_path.exists():
        raise FileExistsError(f"{data_path} already exists; the session was finalised")
    if not events_path.is_file():
        raise FileNotFoundError(f"no {events_path.name} in {session_dir}")

    records = _read_jsonl(events_path)
    ends = [r["time_end_unix"] for r in records if isinstance(r.get("time_end_unix"), (int, float))]
    last_end = max(ends) if ends else None

    summary_path = session_dir / f"{label}_summary.html"
    summary_meta = (
        meta_from_summary_html(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else None
    )
    if summary_meta is not None and summary_meta.get("session_label") == label:
        meta = summary_meta
        meta_source = "summary"
    else:
        meta = {
            "session_label": label,
            "session_start_unix": _label_start_unix(label) or 0.0,
            "frames_processed": 0,
            "pipe_fps": 0.0,
            "avg_infer_ms": 0.0,
            "ir_periods": [],
            "snap_stats": None,
            "frame_size": None,
        }
        meta_source = "label"

    ir_events = _read_jsonl(session_dir / f"{label}_ir_events.jsonl")
    if ir_events:
        # The open period (if any) ends at the last sign of life.
        alive = [float(e["unix"]) for e in ir_events if isinstance(e.get("unix"), (int, float))]
        if last_end is not None:
            alive.append(last_end)
        meta["ir_periods"] = ir_periods_from_events(
            ir_events, close_at_unix=max(alive) if alive else None
        )

    meta["recovered"] = {
        "by": "streettracker recover-session",
        "at": format_wall(datetime.datetime.now().timestamp()),
        "data_from": events_path.name,
        "meta_from": summary_path.name if meta_source == "summary" else "session label",
        "meta_as_of": (
            format_wall(summary_path.stat().st_mtime) if meta_source == "summary" else None
        ),
        "last_track_end": format_wall(last_end) if last_end is not None else None,
    }

    hourly_path = session_dir / f"{label}_hourly.json"
    meta_path = session_dir / f"{label}_meta.json"
    written: list[Path] = []
    if not hourly_path.exists():
        _write_atomic(hourly_path, build_hourly_rollup(list(records), meta["ir_periods"]))
        written.append(hourly_path)
    if not meta_path.exists():
        _write_atomic(meta_path, meta)
        written.append(meta_path)
    _write_atomic(data_path, records)
    written.append(data_path)

    frame_size = meta.get("frame_size")
    return RecoveryResult(
        n_records=len(records),
        meta_source=meta_source,
        frame_size=frame_size if isinstance(frame_size, list) else None,
        last_track_end=meta["recovered"]["last_track_end"],
        written=written,
    )
