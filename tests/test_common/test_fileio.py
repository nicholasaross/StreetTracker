"""common/fileio.py: atomic writes, the tolerant JSON read, the JSONL reader."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from streettracker.common import fileio
from streettracker.common.fileio import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    read_json,
)


def test_atomic_write_text_creates_and_replaces(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    atomic_write_text(p, "first")
    atomic_write_text(p, "second")
    assert p.read_text(encoding="utf-8") == "second"
    assert list(tmp_path.iterdir()) == [p]  # no temp file left behind


def test_atomic_write_text_is_utf8(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    atomic_write_text(p, "L→R café")
    assert p.read_bytes() == "L→R café".encode()


def test_atomic_write_bytes(tmp_path: Path) -> None:
    p = tmp_path / "x.jpg"
    atomic_write_bytes(p, b"\xff\xd8\xff\x00\r\n")
    assert p.read_bytes() == b"\xff\xd8\xff\x00\r\n"  # bytes are written untranslated
    assert list(tmp_path.iterdir()) == [p]


def test_atomic_write_json_options(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    atomic_write_json(p, {"b": 1, "a": "é"})
    assert p.read_text(encoding="utf-8") == json.dumps({"b": 1, "a": "é"}, indent=2)

    atomic_write_json(p, {"b": 1, "a": "é"}, indent=None, sort_keys=True, ensure_ascii=False)
    assert p.read_text(encoding="utf-8") == '{"a": "é", "b": 1}'


def test_failed_write_leaves_the_old_file_and_no_temp(tmp_path: Path) -> None:
    p = tmp_path / "labels.json"
    atomic_write_json(p, {"labels": {"AB12CDE": "FORD"}})

    def explode(fh: object) -> None:
        fh.write('{"labels": {"AB12')  # type: ignore[attr-defined]
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError, match="killed mid-write"):
        fileio._atomic_write(p, "w", explode)

    assert json.loads(p.read_text(encoding="utf-8")) == {"labels": {"AB12CDE": "FORD"}}
    assert list(tmp_path.iterdir()) == [p]


def test_unserialisable_object_leaves_the_old_file(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    atomic_write_json(p, {"ok": True})
    with pytest.raises(TypeError):
        atomic_write_json(p, {"bad": object()})
    assert json.loads(p.read_text(encoding="utf-8")) == {"ok": True}
    assert list(tmp_path.iterdir()) == [p]


def test_missing_parent_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        atomic_write_text(tmp_path / "no_such_dir" / "x.json", "{}")


def test_replace_is_retried_while_the_target_is_held_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What Windows does while a reader has the target open: the rename fails
    # with PermissionError until the reader lets go.
    p = tmp_path / "x.json"
    p.write_text("old", encoding="utf-8")
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src: object, dst: object) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "sharing violation")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(fileio, "_REPLACE_ATTEMPTS", 8)  # the Windows setting
    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(fileio.time, "sleep", lambda _s: None)

    atomic_write_text(p, "new")
    assert calls["n"] == 3
    assert p.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.iterdir()) == [p]


def test_replace_gives_up_after_its_attempts_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "x.json"
    p.write_text("old", encoding="utf-8")
    calls = {"n": 0}

    def locked(src: object, dst: object) -> None:
        calls["n"] += 1
        raise PermissionError(13, "sharing violation")

    monkeypatch.setattr(fileio, "_REPLACE_ATTEMPTS", 8)
    monkeypatch.setattr(os, "replace", locked)
    monkeypatch.setattr(fileio.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        atomic_write_text(p, "new")
    assert calls["n"] == 8
    assert p.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [p]


def test_replace_is_not_retried_off_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Elsewhere a PermissionError is a real permissions problem: fail at once.
    calls = {"n": 0}

    def denied(src: object, dst: object) -> None:
        calls["n"] += 1
        raise PermissionError(13, "permission denied")

    monkeypatch.setattr(fileio, "_REPLACE_ATTEMPTS", 1)  # the setting off Windows
    monkeypatch.setattr(os, "replace", denied)
    with pytest.raises(PermissionError):
        atomic_write_text(tmp_path / "x.json", "new")
    assert calls["n"] == 1


def test_read_json(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text('{"a": [1, 2]}', encoding="utf-8")
    assert read_json(p) == {"a": [1, 2]}


@pytest.mark.parametrize(
    "content",
    [b"", b'{"a": ', b"\x00" * 64, b"\xff\xfe not utf-8"],
    ids=["empty", "truncated", "zero-filled", "not-utf8"],
)
def test_read_json_returns_default_for_a_damaged_file(tmp_path: Path, content: bytes) -> None:
    p = tmp_path / "x.json"
    p.write_bytes(content)
    assert read_json(p) is None
    assert read_json(p, {}) == {}


def test_read_json_returns_default_for_a_missing_file_or_a_directory(tmp_path: Path) -> None:
    assert read_json(tmp_path / "absent.json", []) == []
    assert read_json(tmp_path, "fallback") == "fallback"


def test_iter_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"n": 1}\n\n   \n{"n": 2}\n', encoding="utf-8")
    assert list(iter_jsonl(p)) == [{"n": 1}, {"n": 2}]


def test_iter_jsonl_skips_a_torn_tail_and_logs_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"n": 1}\n{"n": 2}\n{"n": 3, "tra', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="streettracker.common.fileio"):
        rows = list(iter_jsonl(p))
    assert rows == [{"n": 1}, {"n": 2}]
    assert [r.getMessage() for r in caplog.records if "torn line 3" in r.getMessage()]


def test_iter_jsonl_reports_torn_lines_to_the_callback(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"n": 1}\nnot json\n{"n": 2}\n{"half', encoding="utf-8")
    torn: list[int] = []
    rows = list(iter_jsonl(p, on_torn=lambda line_number, _exc: torn.append(line_number)))
    assert rows == [{"n": 1}, {"n": 2}]
    assert torn == [2, 4]


def test_iter_jsonl_survives_a_tail_that_is_not_utf8(tmp_path: Path) -> None:
    # After a power cut the last block can hold arbitrary bytes.
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"n": 1}\n{"n": 2}\n\xff\xfe\x00\x00\x9c garbage')
    torn: list[int] = []
    rows = list(iter_jsonl(p, on_torn=lambda line_number, _exc: torn.append(line_number)))
    assert rows == [{"n": 1}, {"n": 2}]
    assert torn == [3]


def test_iter_jsonl_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl(tmp_path / "absent.jsonl"))
