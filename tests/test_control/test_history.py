"""Tests for the on-disk job/playbook history helper."""

from __future__ import annotations

from pathlib import Path

from streettracker.control import history


def test_load_missing_or_none_returns_empty(tmp_path: Path) -> None:
    assert history.load(None) == []
    assert history.load(tmp_path / "nope.json") == []


def test_load_corrupt_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    p.write_text("{not json")
    assert history.load(p) == []
    p.write_text('{"not": "a list"}')  # valid JSON, wrong shape
    assert history.load(p) == []


def test_append_persists_and_caps(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    for i in range(5):
        history.append(p, {"i": i}, cap=3)
    assert [x["i"] for x in history.load(p)] == [2, 3, 4]  # last 3 kept, oldest-first


def test_append_none_path_is_noop() -> None:
    history.append(None, {"i": 1})  # must not raise
