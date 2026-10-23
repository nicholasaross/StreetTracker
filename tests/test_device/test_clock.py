"""Startup wait for the boot's first NTP sync (power-cut clock, 2026-10-23)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from streettracker.device import clock


class _FakeClock:
    """Monotonic + wall clocks that advance only when ``sleep`` is called;
    ``step_at`` jumps the wall clock forward like an NTP step."""

    def __init__(self, *, step_at: float | None = None, step_s: float = 0.0) -> None:
        self.mono = 0.0
        self.wall_t = 1_792_731_404.0  # 06:16:44 BST, the stale restore time
        self.step_at, self.step_s = step_at, step_s

    def sleep(self, s: float) -> None:
        self.mono += s
        self.wall_t += s
        if self.step_at is not None and self.mono >= self.step_at:
            self.wall_t += self.step_s
            self.step_at = None

    def monotonic(self) -> float:
        return self.mono

    def wall(self) -> float:
        return self.wall_t


def _wait(states: list[bool | None], fake: _FakeClock, timeout_s: float = 300.0) -> bool:
    it = iter(states)
    return clock.wait_for_clock_sync(
        timeout_s,
        check=lambda: next(it),
        poll_s=2.0,
        sleep=fake.sleep,
        monotonic=fake.monotonic,
        wall=fake.wall,
    )


def test_already_synced_does_not_wait() -> None:
    fake = _FakeClock()
    assert _wait([True], fake) is True
    assert fake.mono == 0.0  # a restart on a synced device starts at once


def test_waits_for_the_first_sync_and_logs_the_step(caplog: pytest.LogCaptureFixture) -> None:
    fake = _FakeClock(step_at=4.0, step_s=11_265.0)  # +3 h 07 m 45 s, as on 2026-10-23
    with caplog.at_level("INFO"):
        assert _wait([False, False, True], fake) is True
    assert fake.mono == 4.0
    assert "clock stepped +11265 s" in caplog.text


def test_gives_up_after_the_timeout() -> None:
    fake = _FakeClock()
    assert _wait([False] * 100, fake, timeout_s=10.0) is False
    assert fake.mono == 10.0  # bounded: the tracker still starts without internet


def test_no_sync_status_means_nothing_to_wait_for() -> None:
    fake = _FakeClock()
    assert _wait([None], fake) is True
    assert fake.mono == 0.0


def test_zero_timeout_skips_the_check() -> None:
    def _boom() -> bool:
        raise AssertionError("check must not run")

    assert clock.wait_for_clock_sync(0, check=_boom) is True


def _timedatectl(stdout: str) -> Any:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_clock_synced_trusts_the_timesyncd_marker(tmp_path: Path) -> None:
    marker = tmp_path / "synchronized"
    marker.touch()
    with patch("streettracker.device.clock.subprocess.run") as mock_run:
        assert clock.clock_synced(marker) is True
    mock_run.assert_not_called()


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("NTP=yes\nNTPSynchronized=no\n", False),  # waiting for the first sync
        ("NTP=yes\nNTPSynchronized=yes\n", True),  # another NTP client synced it
        ("NTP=no\nNTPSynchronized=no\n", None),  # no NTP client: never will
    ],
)
def test_clock_synced_falls_back_to_timedatectl(
    tmp_path: Path, stdout: str, expected: bool | None
) -> None:
    with patch("streettracker.device.clock.subprocess.run", return_value=_timedatectl(stdout)):
        assert clock.clock_synced(tmp_path / "absent") is expected


def test_clock_synced_without_timedatectl_is_unknown(tmp_path: Path) -> None:
    with patch("streettracker.device.clock.subprocess.run", side_effect=FileNotFoundError):
        assert clock.clock_synced(tmp_path / "absent") is None
