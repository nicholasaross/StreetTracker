"""User metadata for showcase cars.

A tiny JSON store keyed by canonical plate so a tag follows a car across every
session it appears in. Holds the four user-editable fields -- ``name``,
``owner``, ``notes``, ``favourite`` -- plus a server-set ``updated_at``.

This is the *only* writable state the website owns; everything else
(detections, ANPR, DVSA) is read-only upstream data. Writes are atomic
(temp-file + ``os.replace``) so an interrupted save can't corrupt the store.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# User-editable keys. Anything else in a submitted payload is ignored.
USER_FIELDS = ("name", "owner", "notes", "favourite")

# File name used when only an output root is given.
DEFAULT_FILENAME = "showcase_metadata.json"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_tagged(meta: dict[str, Any] | None) -> bool:
    """True if a metadata entry carries anything a user actually set
    (non-empty name/owner/notes, or favourite). An entry that exists but is
    all-blank counts as untagged."""
    if not meta:
        return False
    if meta.get("favourite"):
        return True
    return any((meta.get(k) or "").strip() for k in ("name", "owner", "notes"))


@dataclass(slots=True)
class MetadataStore:
    """Read/modify/write access to the plate-keyed metadata JSON file."""

    path: Path

    def load(self) -> dict[str, dict[str, Any]]:
        """Return the whole store, or ``{}`` if absent / unreadable."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, plate: str) -> dict[str, Any]:
        """Return one plate's entry (``{}`` if untagged)."""
        return self.load().get(plate, {})

    def set(self, plate: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Merge ``fields`` (only the known USER_FIELDS) into ``plate``'s entry,
        stamp ``updated_at``, persist atomically, and return the new entry.

        Read-modify-write against the on-disk store so concurrent edits to
        *different* plates don't clobber each other."""
        store = self.load()
        entry = dict(store.get(plate, {}))
        entry.update(_clean(fields))
        entry["updated_at"] = _now_iso()
        store[plate] = entry
        self._save(store)
        return entry

    def _save(self, store: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".showcase_meta_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


def _clean(fields: dict[str, Any]) -> dict[str, Any]:
    """Keep only known user fields; coerce types; strip text. Unknown keys
    (and server-managed ones like ``updated_at``) are dropped."""
    out: dict[str, Any] = {}
    for k in USER_FIELDS:
        if k not in fields:
            continue
        if k == "favourite":
            out[k] = bool(fields[k])
        else:
            v = fields[k]
            out[k] = ("" if v is None else str(v)).strip()
    return out
