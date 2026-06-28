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


def test_prune_old_drops_aged_keeps_recent_and_undated() -> None:
    now = 1_000_000.0
    items = [
        {"id": "old", "ended_at": now - 25 * 3600},  # >24h -> dropped
        {"id": "fresh", "ended_at": now - 3600},  # 1h -> kept
        {"id": "no-ts"},  # undateable -> kept
        {"id": "by-created", "created_at": now - 30 * 3600},  # only created_at, old -> dropped
    ]
    kept = {x["id"] for x in history.prune_old(items, now=now)}
    assert kept == {"fresh", "no-ts"}


def test_append_prunes_old_entries(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    p.write_text('[{"id": "ancient", "ended_at": 1.0}]')  # epoch-1970, far older than 24h
    history.append(p, {"id": "new", "ended_at": __import__("time").time()})
    ids = [x["id"] for x in history.load(p)]
    assert ids == ["new"]  # the ancient entry was pruned on append
