"""common/session_files.py: every per-session file name, spelled once."""

from __future__ import annotations

from pathlib import Path

import pytest

from streettracker.common.session_files import SessionFiles
from streettracker.device.runtime import session_paths

# The on-disk names are a contract with every session already recorded and
# pulled, so they are pinned here literally rather than derived.
EXPECTED = {
    "events": "_events.jsonl",
    "ir_events": "_ir_events.jsonl",
    "trajectories": "_trajectories.jsonl",
    "data": "_data.json",
    "meta": "_meta.json",
    "hourly": "_hourly.json",
    "summary_html": "_summary.html",
    "alpr": "_alpr.json",
    "alpr_by_track": "_alpr_by_track.json",
    "static_plates": "_static_plates.json",
    "alpr_labels": "_alpr_labels.json",
    "alpr_scores": "_alpr_scores.json",
    "alpr_report_html": "_alpr_report.html",
    "dvsa_labels": "_dvsa_labels.json",
    "vehicles": "_vehicles.json",
    "vehicle_boxes": "_vehicle_boxes.json",
    "makemodel": "_makemodel.json",
    "makemodel_by_track": "_makemodel_by_track.json",
    "bodytype": "_bodytype.json",
    "bodytype_by_track": "_bodytype_by_track.json",
    "colour": "_colour.json",
    "colour_by_track": "_colour_by_track.json",
    "people": "_people.json",
}


@pytest.mark.parametrize(("attr", "suffix"), sorted(EXPECTED.items()))
def test_each_file_is_the_label_plus_its_suffix(tmp_path: Path, attr: str, suffix: str) -> None:
    session_dir = tmp_path / "session_20261004_095219"
    files = SessionFiles(session_dir)
    assert getattr(files, attr) == session_dir / f"session_20261004_095219{suffix}"


def test_every_path_property_is_pinned() -> None:
    # A new property must be added to EXPECTED too, so its name is reviewed.
    props = {
        name
        for name, value in vars(SessionFiles).items()
        if isinstance(value, property) and name != "label"
    }
    assert props == set(EXPECTED)


def test_label_is_the_directory_name(tmp_path: Path) -> None:
    assert SessionFiles(tmp_path / "session_test").label == "session_test"


def test_building_paths_does_not_touch_the_disk(tmp_path: Path) -> None:
    files = SessionFiles(tmp_path / "never_created")
    assert not files.data.exists()
    assert not files.session_dir.exists()


def test_runtime_session_paths_are_unchanged(tmp_path: Path) -> None:
    # What the device runtime writes must not move: sessions on the Orin and
    # everything already pulled use exactly these names.
    label = "session_20261004_095219"
    base = tmp_path / label
    assert session_paths(tmp_path, label) == {
        "dir": base,
        "events_jsonl": base / f"{label}_events.jsonl",
        "ir_events_jsonl": base / f"{label}_ir_events.jsonl",
        "trajectories_jsonl": base / f"{label}_trajectories.jsonl",
        "data_json": base / f"{label}_data.json",
        "meta_json": base / f"{label}_meta.json",
        "hourly_json": base / f"{label}_hourly.json",
        "summary_html": base / f"{label}_summary.html",
        "heartbeat": base / ".heartbeat",
    }
