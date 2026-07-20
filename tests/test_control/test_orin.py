"""Tests for the Orin status probe.

The probe shells out to ``ssh``; these patch ``subprocess.run`` (the same
approach as ``tests/test_cli/test_pull.py``) to exercise the parse + the
non-fatal / timeout degradation without touching the network.
"""

from __future__ import annotations

import subprocess
import time
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


# ----------------------------------------------------------------------
# Remote session enumeration.
#
# The panel used to surface only `live_session`, so a CLOSED session that
# was never pulled was invisible on the dashboard. Two sat unpulled for
# days while everything the panel did show got enriched -- one of them
# inside the device's ~7-day snap-prune window (2026-07-20).

_SESS_OUT = (
    "ACTIVE active\n"
    "SESSION session_20200105_120000\n"
    "VEH 10\nPPL 5\n"
    "SESS session_20200105_120000 open 15\n"
    "SESS session_20200104_090000 closed 646\n"
    "SESS session_20200101_000000 closed 0\n"
)


def test_snapshot_lists_sessions_on_the_device() -> None:
    with patch("streettracker.control.orin.subprocess.run", return_value=_completed(_SESS_OUT)):
        st = orin.orin_live_snapshot()
    assert [s.name for s in st.sessions] == [
        "session_20200105_120000",
        "session_20200104_090000",
        "session_20200101_000000",
    ]
    assert [s.main_snaps for s in st.sessions] == [15, 646, 0]
    assert [s.closed for s in st.sessions] == [False, True, True]


def test_only_the_newest_session_is_live_and_only_while_active() -> None:
    with patch("streettracker.control.orin.subprocess.run", return_value=_completed(_SESS_OUT)):
        st = orin.orin_live_snapshot()
    assert [s.is_live for s in st.sessions] == [True, False, False]

    # A stale dir left by a crashed service is not "live".
    stopped = _SESS_OUT.replace("ACTIVE active", "ACTIVE inactive")
    with patch("streettracker.control.orin.subprocess.run", return_value=_completed(stopped)):
        st = orin.orin_live_snapshot()
    assert all(not s.is_live for s in st.sessions)


def test_prune_countdown_tracks_session_age() -> None:
    """Days-left is what makes an unpulled session urgent; a session older
    than the window reports negative, i.e. its snaps are already going."""
    fresh = orin.RemoteSession(name="s", start_unix=time.time() - 86400)
    assert fresh.prune_days_left is not None
    assert 5.9 < fresh.prune_days_left < 6.1

    old = orin.RemoteSession(name="s", start_unix=time.time() - 10 * 86400)
    assert old.prune_days_left is not None and old.prune_days_left < 0

    assert orin.RemoteSession(name="s").prune_days_left is None


def test_probe_is_bounded_to_recent_sessions() -> None:
    """The probe runs every ~45s and costs a `find` per dir; the device
    holds ~60 session dirs, so it must not walk all of them."""
    assert "head -{recent}" in orin._PROBE
    captured: dict[str, Any] = {}

    def _capture(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = args[-1]
        return _completed(_SESS_OUT)

    with patch("streettracker.control.orin.subprocess.run", side_effect=_capture):
        orin.orin_live_snapshot()
    assert f"head -{orin.RECENT_SESSIONS}" in captured["cmd"]


def test_sessions_survive_json_serialisation() -> None:
    with patch("streettracker.control.orin.subprocess.run", return_value=_completed(_SESS_OUT)):
        st = orin.orin_live_snapshot()
    d = st.to_json_dict()
    assert d["sessions"][1]["name"] == "session_20200104_090000"
    assert d["sessions"][1]["main_snaps"] == 646
    assert "prune_days_left" in d["sessions"][1]


def test_snapshot_without_session_lines_still_parses() -> None:
    """Older device / probe mismatch must not break the poller."""
    out = "ACTIVE active\nSESSION session_20200101_000000\nVEH 1\nPPL 2\n"
    with patch("streettracker.control.orin.subprocess.run", return_value=_completed(out)):
        st = orin.orin_live_snapshot()
    assert st.sessions == []
    assert st.vehicle_snaps == 1
