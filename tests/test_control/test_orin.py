"""Tests for the Orin status probe.

The probe shells out to ``ssh``; these patch ``subprocess.run`` (the same
approach as ``tests/test_cli/test_pull.py``) to exercise the parse + the
non-fatal / timeout degradation without touching the network.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import patch

from streettracker.cli.pull import RemoteInventory
from streettracker.control import orin


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_parse_session_start_valid_and_junk() -> None:
    ts = orin.parse_session_start("session_20200101_000000")
    assert ts is not None and ts > 0
    assert orin.parse_session_start("not_a_session") is None
    assert orin.parse_session_start("session_garbage") is None
    assert orin.parse_session_start(None) is None


def test_live_snapshot_active_session() -> None:
    out = "ACTIVE active\nSESSION session_20200101_000000\nVEH 4210\nPPL 1500\n"
    with patch("streettracker.control.orin.subprocess.run", return_value=_completed(out)):
        st = orin.orin_live_snapshot()
    assert st.reachable is True
    assert st.service_active == "active"
    assert st.live_session == "session_20200101_000000"
    assert st.vehicle_snaps == 4210
    assert st.person_snaps == 1500
    assert st.session_start_unix is not None
    # Active + a past start -> a positive runtime is computed client-of-the-poll.
    assert st.session_runtime_s is not None and st.session_runtime_s > 0


def test_live_snapshot_inactive_has_no_runtime() -> None:
    out = "ACTIVE inactive\nSESSION session_20200101_000000\nVEH 0\nPPL 0\n"
    with patch("streettracker.control.orin.subprocess.run", return_value=_completed(out)):
        st = orin.orin_live_snapshot()
    assert st.reachable is True
    assert st.service_active == "inactive"
    # Runtime is only meaningful while recording -> suppressed when not active.
    assert st.session_runtime_s is None


def test_live_snapshot_unreachable_is_non_fatal() -> None:
    with patch(
        "streettracker.control.orin.subprocess.run",
        return_value=_completed(
            returncode=255, stderr="ssh: connect to host orin port 22: timed out"
        ),
    ):
        st = orin.orin_live_snapshot()
    assert st.reachable is False
    assert st.error and "timed out" in st.error
    assert st.live_session is None


def test_live_snapshot_timeout_is_non_fatal() -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=20)

    with patch("streettracker.control.orin.subprocess.run", side_effect=_raise):
        st = orin.orin_live_snapshot()
    assert st.reachable is False
    assert st.error and "timed out" in st.error


def test_live_snapshot_missing_ssh_binary() -> None:
    with patch("streettracker.control.orin.subprocess.run", side_effect=OSError("no ssh")):
        st = orin.orin_live_snapshot()
    assert st.reachable is False
    assert st.error and "ssh" in st.error


def test_pull_preview_delegates_to_remote_inventory() -> None:
    inv = RemoteInventory(bytes=1_000_000, files=10, main_snaps=8)
    with patch("streettracker.control.orin.remote_inventory", return_value=inv) as m:
        got = orin.pull_preview("session_x", "orin", "u", "/k", "/srv/output")
    assert got.bytes == 1_000_000
    # remote path is parent + session.
    assert m.call_args.args[-1] == "/srv/output/session_x"


def test_estimate_transfer_seconds() -> None:
    # 50 MB at 25 MB/s ~ 2 s.
    assert abs(orin.estimate_transfer_seconds(50 * 1024 * 1024, 25.0) - 2.0) < 0.01
    assert orin.estimate_transfer_seconds(123, 0) == 0.0
