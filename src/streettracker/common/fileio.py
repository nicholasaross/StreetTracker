"""Shared file I/O: atomic writes, a tolerant JSON read, a torn-tail JSONL reader.

Session artefacts are rewritten in place by the enrichment commands while the
showcase and the control panel read them, and jobs do get killed part-way (the
dev box sleeps, a panel job is cancelled). A plain ``write_text`` truncates the
file first, so a kill mid-write leaves it empty or half-written; for
``_dvsa_labels.json`` that loses lookups that were paid for.

Every writer here goes to a temporary file beside the target and then renames
it over the target, so a reader sees either the old content or the new, never a
partial file. This guards against an interrupted *writer*. It does not
``fsync``, so it is not a defence against power loss.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

# On Windows ``os.replace`` fails with PermissionError while any other process
# has the target open (a reader part-way through it, antivirus, the search
# indexer). Those holds last milliseconds, so retry briefly before giving up:
# 8 attempts with a growing pause is about 1.4 s in the worst case. Elsewhere
# a PermissionError is a real permissions problem, so there is one attempt.
_REPLACE_ATTEMPTS = 8 if os.name == "nt" else 1
_REPLACE_PAUSE_S = 0.05


def _replace(tmp: Path, path: Path) -> None:
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_PAUSE_S * attempt)


def _atomic_write(path: Path, mode: str, write: Callable[[IO[Any]], object]) -> None:
    # A unique name, so two writers to the same target can't share a temp
    # file. It ends in ".tmp", never ".json", so the globs that discover
    # session files don't pick it up.
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, mode, encoding=None if "b" in mode else "utf-8") as fh:
            write(fh)
        _replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` (UTF-8) in one step.

    Newlines are translated as ``Path.write_text`` does, so files keep the
    line endings they have always been written with on each platform. The
    parent directory must exist."""
    _atomic_write(path, "w", lambda fh: fh.write(text))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` in one step."""
    _atomic_write(path, "wb", lambda fh: fh.write(data))


def atomic_write_json(
    path: Path,
    obj: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
) -> None:
    """Replace ``path`` with ``obj`` serialised as JSON, in one step.

    The whole document is serialised before the file is touched, so an object
    that can't be serialised leaves the existing file alone."""
    atomic_write_text(
        path, json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    )


def read_json(path: Path, default: Any = None) -> Any:
    """Parse the JSON file at ``path``, or return ``default`` if it is
    missing, unreadable or not valid JSON.

    For optional sidecars and caches, where an absent or damaged file means
    "nothing there". Use ``json.loads(path.read_text(...))`` where a bad file
    should stop the command."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError: bad JSON or bad UTF-8
        return default


def _log_torn(path: Path) -> Callable[[int, ValueError], None]:
    def warn(line_number: int, exc: ValueError) -> None:
        logger.warning("skipping torn line %d in %s: %s", line_number, path, exc)

    return warn


def iter_jsonl(
    path: Path,
    *,
    on_torn: Callable[[int, ValueError], None] | None = None,
) -> Iterator[Any]:
    """Yield the parsed value of each line of a JSON-lines file.

    Blank lines are skipped. A line that fails to parse is skipped too,
    instead of aborting the read: the logs are appended one fsynced line at a
    time, so a crash or power cut can tear the last line and everything before
    it must stay readable. ``on_torn(line_number, error)`` is called for each
    such line (line numbers start at 1); by default it logs a warning."""
    report = on_torn if on_torn is not None else _log_torn(path)
    # errors="replace": a torn tail can hold bytes that aren't UTF-8 at all.
    # They become U+FFFD, the line fails to parse, and it is skipped like any
    # other torn line, where a strict decode would abort the whole read.
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                report(line_number, exc)
