"""Remember a value computed from files until one of those files changes.

The showcase rebuilds its read models on every refresh, and almost all of the
work is re-reading sessions that closed weeks ago and will never change.
:class:`FileMemo` keeps what was computed from a set of files together with
each file's modification time and size, and recomputes only when one of them
differs (or a file appears or disappears).

It holds values in memory, for the life of the process. Nothing can go stale:
the files' state is checked on every lookup, so a changed file is a miss. The
control panel has its own variant (``control.introspect._by_mtime``), which
also persists to disk and so only takes values that survive a JSON round trip.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")

# One file's state: (modification time in ns, size, file id), or None when it
# is missing.
FileStamp = tuple[int, int, int] | None


def file_stamp(path: Path) -> FileStamp:
    """``path``'s modification time, size and file id, or ``None`` if it can't
    be read.

    The file id (inode) is there for files replaced by an atomic write, which
    is how the enrichment commands write: the replacement is a new file, so it
    is seen as changed even if it has the same size and landed within the same
    clock tick as the one before.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


class FileMemo(Generic[T]):
    """Values keyed by a caller's key, each valid for one state of its files.

    One value is kept per key, so a changed file replaces its entry instead of
    adding to it. Past ``max_entries`` keys the least recently used is dropped.

    The value returned is the one stored, not a copy: treat it as read-only.
    Safe to use from several threads; two threads missing on the same key may
    both compute it, and the later result is kept.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[Hashable, tuple[tuple[FileStamp, ...], T]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable, paths: Iterable[Path], compute: Callable[[], T]) -> T:
        """The value for ``key``: the remembered one if every file in ``paths``
        is as it was when that value was computed, else ``compute()``.

        The files are stamped before ``compute`` runs. If one is rewritten
        while it runs, the value is stored against the older stamp and the
        next lookup recomputes it.
        """
        stamps = tuple(file_stamp(path) for path in paths)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] == stamps:
                self._entries.move_to_end(key)
                return entry[1]
        value = compute()
        with self._lock:
            self._entries[key] = (stamps, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
