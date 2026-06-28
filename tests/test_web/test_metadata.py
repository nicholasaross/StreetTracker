"""MetadataStore: round-trip, atomic write, field hygiene, is_tagged()."""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.web.metadata import MetadataStore, is_tagged


def _store(tmp_path: Path) -> MetadataStore:
    return MetadataStore(tmp_path / "showcase_metadata.json")


def test_set_then_get_roundtrip(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.set("AB12CDE", {"name": "My car", "owner": "Shaun", "notes": "blue one", "favourite": True})
    got = s.get("AB12CDE")
    assert got["name"] == "My car"
    assert got["owner"] == "Shaun"
    assert got["notes"] == "blue one"
    assert got["favourite"] is True
    assert "updated_at" in got


def test_save_creates_file_and_is_valid_json(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.set("AB12CDE", {"owner": "Shaun"})
    assert s.path.exists()
    data = json.loads(s.path.read_text(encoding="utf-8"))
    assert data["AB12CDE"]["owner"] == "Shaun"


def test_unknown_keys_ignored(tmp_path: Path) -> None:
    s = _store(tmp_path)
    entry = s.set("AB12CDE", {"name": "x", "junk": "y", "updated_at": "spoof"})
    assert "junk" not in entry
    assert entry["name"] == "x"
    # updated_at is server-managed, not taken from the payload.
    assert entry["updated_at"] != "spoof"


def test_favourite_coerced_to_bool(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.set("AB12CDE", {"favourite": "yes"})["favourite"] is True
    assert s.set("AB12CDE", {"favourite": 0})["favourite"] is False


def test_text_is_stripped(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.set("AB12CDE", {"owner": "  Shaun  "})["owner"] == "Shaun"


def test_partial_update_preserves_other_fields(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.set("AB12CDE", {"name": "My car", "owner": "Me"})
    s.set("AB12CDE", {"owner": "Shaun"})  # only owner this time
    got = s.get("AB12CDE")
    assert got["name"] == "My car"  # preserved
    assert got["owner"] == "Shaun"  # updated


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert _store(tmp_path).load() == {}
    assert _store(tmp_path).get("AB12CDE") == {}


def test_load_corrupt_returns_empty(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.path.write_text("{ not json", encoding="utf-8")
    assert s.load() == {}


def test_two_plates_independent(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.set("AB12CDE", {"owner": "Shaun"})
    s.set("ZZ99ZZA", {"owner": "Me"})
    assert s.get("AB12CDE")["owner"] == "Shaun"
    assert s.get("ZZ99ZZA")["owner"] == "Me"


def test_is_tagged() -> None:
    assert is_tagged(None) is False
    assert is_tagged({}) is False
    assert is_tagged({"name": "", "owner": "", "notes": "", "favourite": False}) is False
    assert is_tagged({"favourite": True}) is True
    assert is_tagged({"owner": "Shaun"}) is True
    assert is_tagged({"notes": "  ", "owner": ""}) is False


def test_make_override_stored_and_stripped(tmp_path: Path) -> None:
    s = _store(tmp_path)
    entry = s.set("AB12CDE", {"make_override": "  DAF  "})
    assert entry["make_override"] == "DAF"


def test_make_hidden_coerced_to_bool(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.set("AB12CDE", {"make_hidden": "yes"})["make_hidden"] is True
    assert s.set("AB12CDE", {"make_hidden": 0})["make_hidden"] is False


def test_is_tagged_includes_make_correction() -> None:
    assert is_tagged({"make_override": "DAF"}) is True
    assert is_tagged({"make_hidden": True}) is True
    assert is_tagged({"make_override": "  "}) is False  # blank override doesn't tag
