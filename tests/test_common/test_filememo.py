"""``common.filememo``: a value is recomputed only when one of its files changes."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from streettracker.common.fileio import atomic_write_text
from streettracker.common.filememo import FileMemo, file_stamp


class _Counter:
    """A compute function that counts its calls and returns the file's text."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self.path.read_text(encoding="utf-8") if self.path.exists() else "<missing>"


def _later(path: Path, seconds: float = 5.0) -> None:
    """Move ``path``'s modification time forward, as a later write would."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(seconds * 1e9)))


def test_file_stamp_of_a_missing_file_is_none(tmp_path: Path) -> None:
    assert file_stamp(tmp_path / "missing.json") is None


def test_file_stamp_changes_with_content_size_and_time(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    first = file_stamp(path)
    assert first is not None and first[1] == 3

    path.write_text("three", encoding="utf-8")
    assert file_stamp(path) != first

    path.write_text("one", encoding="utf-8")
    _later(path)
    assert file_stamp(path) != first  # same size, later time


def test_unchanged_files_are_not_recomputed(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    compute = _Counter(path)
    memo: FileMemo[str] = FileMemo()

    assert memo.get("a", [path], compute) == "one"
    assert memo.get("a", [path], compute) == "one"
    assert memo.get("a", [path], compute) == "one"
    assert compute.calls == 1


def test_a_rewritten_file_is_recomputed_once(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    compute = _Counter(path)
    memo: FileMemo[str] = FileMemo()
    memo.get("a", [path], compute)

    path.write_text("two!", encoding="utf-8")
    assert memo.get("a", [path], compute) == "two!"
    assert memo.get("a", [path], compute) == "two!"
    assert compute.calls == 2
    assert len(memo) == 1  # the new value replaced the old one


def test_an_atomic_rewrite_of_the_same_size_is_noticed(tmp_path: Path) -> None:
    """The enrichment commands replace a file with a new one. That must count
    as a change even when the size is the same and the clock has not moved."""
    path = tmp_path / "a.json"
    atomic_write_text(path, "one")
    before = path.stat()
    compute = _Counter(path)
    memo: FileMemo[str] = FileMemo()
    assert memo.get("a", [path], compute) == "one"

    atomic_write_text(path, "two")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))  # same time, same size
    assert memo.get("a", [path], compute) == "two"
    assert compute.calls == 2


def test_any_one_of_several_files_changing_recomputes(tmp_path: Path) -> None:
    a, b, c = (tmp_path / name for name in ("a.json", "b.json", "c.json"))
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    calls = []
    memo: FileMemo[int] = FileMemo()

    def compute() -> int:
        calls.append(1)
        return len(calls)

    assert memo.get("k", [a, b, c], compute) == 1
    assert memo.get("k", [a, b, c], compute) == 1

    _later(b)
    assert memo.get("k", [a, b, c], compute) == 2

    c.write_text("c", encoding="utf-8")  # a file that was missing appears
    assert memo.get("k", [a, b, c], compute) == 3

    a.unlink()  # and one disappears
    assert memo.get("k", [a, b, c], compute) == 4
    assert memo.get("k", [a, b, c], compute) == 4


def test_a_missing_file_is_a_state_like_any_other(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    compute = _Counter(path)
    memo: FileMemo[str] = FileMemo()
    assert memo.get("a", [path], compute) == "<missing>"
    assert memo.get("a", [path], compute) == "<missing>"
    assert compute.calls == 1


def test_keys_are_independent(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    memo: FileMemo[str] = FileMemo()
    assert memo.get("upper", [path], lambda: "ONE") == "ONE"
    assert memo.get("lower", [path], lambda: "one") == "one"
    assert memo.get("upper", [path], lambda: "recomputed") == "ONE"
    assert len(memo) == 2


def test_a_failed_compute_is_not_remembered(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    memo: FileMemo[str] = FileMemo()

    def failing() -> str:
        raise ValueError("bad file")

    with pytest.raises(ValueError, match="bad file"):
        memo.get("a", [path], failing)
    assert len(memo) == 0
    assert memo.get("a", [path], lambda: "fine") == "fine"


def test_the_least_recently_used_key_is_dropped_past_the_limit(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    memo: FileMemo[str] = FileMemo(max_entries=2)
    memo.get("a", [path], lambda: "a1")
    memo.get("b", [path], lambda: "b1")
    memo.get("a", [path], lambda: "unused")  # touch a, so b is now the oldest
    memo.get("c", [path], lambda: "c1")
    assert len(memo) == 2
    assert memo.get("a", [path], lambda: "a2") == "a1"
    assert memo.get("b", [path], lambda: "b2") == "b2"  # was dropped


def test_a_file_changed_while_computing_is_recomputed_next_time(tmp_path: Path) -> None:
    """The stamp is taken before the value is computed. If the file moves on
    in between, the value may be newer than its stamp, and the next lookup
    must not trust it."""
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    memo: FileMemo[str] = FileMemo()

    def slow_reader() -> str:
        text = path.read_text(encoding="utf-8")
        path.write_text("two!", encoding="utf-8")  # rewritten mid-compute
        return text

    assert memo.get("a", [path], slow_reader) == "one"
    assert memo.get("a", [path], lambda: path.read_text(encoding="utf-8")) == "two!"


def test_clear_forgets_everything(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    compute = _Counter(path)
    memo: FileMemo[str] = FileMemo()
    memo.get("a", [path], compute)
    memo.clear()
    assert len(memo) == 0
    memo.get("a", [path], compute)
    assert compute.calls == 2


def test_lookups_from_several_threads(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text("one", encoding="utf-8")
    memo: FileMemo[str] = FileMemo()
    results: list[str] = []

    def worker() -> None:
        for _ in range(200):
            results.append(memo.get("a", [path], lambda: path.read_text(encoding="utf-8")))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["one"] * 1600
    assert len(memo) == 1
