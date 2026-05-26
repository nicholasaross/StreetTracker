"""Smoke tests for the `streettracker vehicles` CLI subcommand."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from streettracker.analysis import vehicles as vehicles_module
from streettracker.cli import main as cli_main
from streettracker.common.schema import TrackRecord


def test_top_level_help_lists_vehicles() -> None:
    from streettracker.cli import _HELP

    assert "vehicles" in _HELP


def test_vehicles_help_returns_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        vehicles_module.main(["--help"])
    assert exc.value.code == 0


def test_vehicles_missing_session_dir_returns_2(tmp_path: Path) -> None:
    rc = vehicles_module.main([str(tmp_path / "does_not_exist")])
    assert rc == 2


def test_vehicles_cli_dispatch_writes_output_json(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    """End-to-end: dispatch via the top-level CLI, verify the output
    file lands and headline counts print."""
    session = tmp_path / "session_test"
    session.mkdir()
    (session / "session_test_data.json").write_text(
        json.dumps([asdict(sample_track)])
    )
    (session / "session_test_alpr_by_track.json").write_text(json.dumps({
        "tracks": [{
            "track_id": 42,
            "best_preferred": {
                "track_id": 42, "snap_index": 1,
                "image": "vehicle_42_main_1.jpg",
                "ocr_text": "AB12CDE", "ocr_conf": 0.98, "det_conf": 0.85,
            },
        }],
    }))

    rc = cli_main(["vehicles", str(session)])
    assert rc == 0
    out = session / "session_test_vehicles.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["plate"] == "AB12CDE"
    assert data[0]["n_visits"] == 1
