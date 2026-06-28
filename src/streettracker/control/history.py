"""Tiny JSON-file persistence for finished jobs / playbooks.

The job + playbook runners are in-memory, so the panel's recent activity would
vanish on restart. This persists each finished item's terminal snapshot to a
capped JSON list so the dashboard's history survives a restart (and a failed
job's copy-paste Claude prompt with it). Writes are atomic (temp + ``os.replace``)
and reads tolerate a missing or corrupt file — persistence must never take the
panel down.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_CAP = 200  # most recent N terminal snapshots kept on disk
MAX_AGE_S = 60 * 3600  # snapshots older than this are dropped (the UI also minimises old runs)


def load(path: Path | None) -> list[dict[str, Any]]:
    """Return the persisted snapshots (oldest-first), or ``[]`` if absent/corrupt."""
    if path is None:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _entry_ts(entry: dict[str, Any]) -> float | None:
    """Best wall-clock timestamp for an entry (terminal time preferred)."""
    for key in ("ended_at", "started_at", "created_at"):
        v = entry.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def prune_old(
    items: list[dict[str, Any]], *, max_age_s: float = MAX_AGE_S, now: float | None = None
) -> list[dict[str, Any]]:
    """Drop entries finished more than ``max_age_s`` ago. Entries without any
    timestamp are kept (can't age them out safely)."""
    cutoff = (time.time() if now is None else now) - max_age_s
    out = []
    for entry in items:
        ts = _entry_ts(entry)
        if ts is None or ts >= cutoff:
            out.append(entry)
    return out


def append(
    path: Path | None,
    entry: dict[str, Any],
    *,
    cap: int = DEFAULT_CAP,
    max_age_s: float = MAX_AGE_S,
) -> None:
    """Append ``entry``, drop snapshots older than ``max_age_s``, trim to ``cap``."""
    if path is None:
        return
    items = prune_old(load(path), max_age_s=max_age_s)
    items.append(entry)
    if len(items) > cap:
        items = items[-cap:]
    _atomic_write(path, items)


def _atomic_write(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
