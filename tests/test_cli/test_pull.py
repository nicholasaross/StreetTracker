"""Tests for ``streettracker pull``.

The pull CLI is mostly an orchestrator around ``ssh`` and ``scp``.
These tests cover the pure helpers (inventory parsing, scp command
construction, byte formatting) and patch ``subprocess.run`` for the
network-touching paths.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from streettracker.cli import pull


def test_human_bytes_scales_through_units() -> None:
    assert pull.human_bytes(0) == "0.0 B"
    assert pull.human_bytes(512) == "512.0 B"
    assert pull.human_bytes(2048) == "2.0 KB"
    assert pull.human_bytes(5 * 1024 * 1024) == "5.0 MB"
    assert pull.human_bytes(3 * 1024**3) == "3.0 GB"


def test_scp_commands_full_pull_uses_recursive_scp(tmp_path: Path) -> None:
    cmds = pull._scp_commands(
        host="orin",
        user="streettracker",
        key="/k/id",
        remote_path="/srv/output/session_x",
        local_parent=tmp_path,
        only_main=False,
    )
    assert len(cmds) == 1
    args = cmds[0]
    assert args[0] == "scp"
    assert "-r" in args
    assert "streettracker@orin:/srv/output/session_x" in args
    assert str(tmp_path) in args


def test_scp_commands_only_main_uses_pattern_list(tmp_path: Path) -> None:
    cmds = pull._scp_commands(
        host="orin",
        user="streettracker",
        key="/k/id",
        remote_path="/srv/output/session_x",
        local_parent=tmp_path,
        only_main=True,
    )
    # one command per pattern; thumbs (<id>.jpg) and HQ (<id>_hq.jpg) excluded
    patterns_seen = [c[-2].split(":")[-1].rsplit("/", 1)[-1] for c in cmds]
    assert "*_main_*.jpg" in patterns_seen
    assert "*.json" in patterns_seen
    assert "*.jsonl" in patterns_seen
    assert "*_summary.html" in patterns_seen
    assert "index.html" in patterns_seen
    assert "*_hq.jpg" not in patterns_seen
    # session subdir was created
    assert (tmp_path / "session_x").is_dir()


def _fake_completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


def test_find_latest_session_returns_remote_basename() -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed("session_20260518_141443\n")
        result = pull.find_latest_session("orin", "u", "/k", "/srv/output")
    assert result == "session_20260518_141443"
    # ssh got the right host/user
    args = mock_run.call_args.args[0]
    assert args[0] == "ssh"
    assert "u@orin" in args


def test_find_latest_session_exits_on_empty_listing() -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed("")
        with pytest.raises(SystemExit):
            pull.find_latest_session("orin", "u", "/k", "/srv/output")


def test_remote_inventory_parses_du_and_find_output() -> None:
    fake_out = (
        "BYTES 1572864\n"
        "FILES 42\n"
        "MAIN 7\n"
        "HQ 14\n"
        "JSONL 1\n"
    )
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(fake_out)
        inv = pull.remote_inventory("orin", "u", "/k", "/srv/output/session_x")
    assert inv.bytes == 1572864
    assert inv.files == 42
    assert inv.main_snaps == 7
    assert inv.hq_crops == 14
    assert inv.jsonl == 1


def test_remote_inventory_tolerates_garbage_lines() -> None:
    """Out-of-order or unknown labels are ignored, not fatal."""
    fake_out = "BYTES 100\nUNKNOWN 9999\nMAIN garbage\nFILES 3\n"
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(fake_out)
        inv = pull.remote_inventory("orin", "u", "/k", "/x")
    assert inv.bytes == 100
    assert inv.files == 3
    assert inv.main_snaps == 0


def test_scp_pull_dry_run_does_not_invoke_subprocess(tmp_path: Path) -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        pull.scp_pull(
            host="orin",
            user="u",
            key="/k",
            remote_path="/srv/output/session_x",
            local_parent=tmp_path,
            only_main=False,
            dry_run=True,
        )
    mock_run.assert_not_called()


def test_scp_pull_runs_scp_when_not_dry_run(tmp_path: Path) -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(returncode=0)
        pull.scp_pull(
            host="orin",
            user="u",
            key="/k",
            remote_path="/srv/output/session_x",
            local_parent=tmp_path,
            only_main=False,
            dry_run=False,
        )
    assert mock_run.call_count == 1
    assert mock_run.call_args.args[0][0] == "scp"


def test_scp_pull_only_main_tolerates_missing_patterns(tmp_path: Path) -> None:
    """With --only-main, a non-zero scp exit (no files matched a pattern)
    is acceptable — sessions sometimes have no main snaps."""
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(returncode=1)
        pull.scp_pull(
            host="orin",
            user="u",
            key="/k",
            remote_path="/srv/output/session_x",
            local_parent=tmp_path,
            only_main=True,
            dry_run=False,
        )


def test_scp_pull_full_pull_exits_on_scp_failure(tmp_path: Path) -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(returncode=1)
        with pytest.raises(SystemExit):
            pull.scp_pull(
                host="orin",
                user="u",
                key="/k",
                remote_path="/srv/output/session_x",
                local_parent=tmp_path,
                only_main=False,
                dry_run=False,
            )


def test_main_returns_1_when_key_missing(tmp_path: Path) -> None:
    fake_key = tmp_path / "does_not_exist"
    rc = pull.main(["--key", str(fake_key), "--session", "session_x"])
    assert rc == 1


def test_main_dry_run_end_to_end(tmp_path: Path) -> None:
    """Full happy-path dry-run: key exists, ssh stubbed, no scp invoked."""
    fake_key = tmp_path / "id"
    fake_key.write_text("not-a-real-key")

    def _ssh_stub(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # remote_inventory; find_latest_session is skipped because we pass --session
        return _fake_completed(
            "BYTES 0\nFILES 0\nMAIN 0\nHQ 0\nJSONL 0\n"
        )

    with patch("streettracker.cli.pull.subprocess.run", side_effect=_ssh_stub):
        rc = pull.main([
            "--key", str(fake_key),
            "--session", "session_test",
            "--target", str(tmp_path / "out"),
            "--dry-run",
        ])
    assert rc == 0
