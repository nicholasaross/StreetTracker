"""Wait for the wall clock to be trustworthy before a live session starts.

The Orin keeps no time through a power cut: its RTC comes back unset, and
systemd-timesyncd restores the last time it saved to disk, which can be
hours stale. NTP corrects the clock a minute or two after boot, but by then
``run_session`` has already stamped ``session_start_unix`` and named the
session directory from the stale clock. On 2026-10-23 the service started at
about 09:22 with the clock reading 06:16:44, so the session was named
``session_20261023_061647``, about 3 h early. A name like that can also sort
before the session the outage cut short. Track records were unaffected,
because they are stamped from the live clock and the first one finished
after the sync, but the session label, its meta start and the panel's
runtime were all wrong.

``streettracker run`` therefore waits, with a time limit, for the first
sync of the boot. A restart on an already-synced device doesn't wait at
all. If NTP can't sync (no internet), the tracker starts anyway once the
limit is reached: a misnamed session is better than an unwatched street.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# systemd-timesyncd creates this on the boot's first successful sync (it is
# what systemd-time-wait-sync watches). It lives on /run, so it is gone after
# every reboot and stays put for the rest of the boot.
TIMESYNCD_MARKER = Path("/run/systemd/timesync/synchronized")

DEFAULT_TIMEOUT_S = 300.0


def clock_synced(marker: Path = TIMESYNCD_MARKER) -> bool | None:
    """Whether NTP has set the clock this boot.

    ``True`` once timesyncd's marker exists, or ``timedatectl`` reports the
    kernel clock synchronized (any NTP client). ``False`` while an NTP
    client is enabled but has not synced yet. ``None`` when this host can't
    tell (no ``timedatectl``, or NTP switched off), so there is nothing to
    wait for.
    """
    if marker.exists():
        return True
    try:
        out = subprocess.run(
            ["timedatectl", "show", "-p", "NTP", "-p", "NTPSynchronized"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    if props.get("NTPSynchronized") == "yes":
        return True
    if props.get("NTP") != "yes":
        return None
    return False


def wait_for_clock_sync(
    timeout_s: float = DEFAULT_TIMEOUT_S,
    *,
    check: Callable[[], bool | None] = clock_synced,
    poll_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    wall: Callable[[], float] = time.time,
) -> bool:
    """Block until the clock is synced or ``timeout_s`` passes.

    Returns ``True`` when the clock is trustworthy (synced, or nothing to
    wait for) and ``False`` on timeout. ``timeout_s <= 0`` skips the check.
    Logs how far the clock stepped while waiting, so the journal records
    what the session would otherwise have been misnamed by.
    """
    if timeout_s <= 0:
        return True
    state = check()
    if state is None:
        logger.info("[clock] no NTP sync status on this host; not waiting")
        return True
    if state:
        return True

    t0_mono, t0_wall = monotonic(), wall()
    logger.warning(
        "[clock] NTP has not synced since boot (clock reads %s); waiting up to %.0f s "
        "so the session is not named and dated from a stale clock",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0_wall)),
        timeout_s,
    )
    while True:
        sleep(poll_s)
        state = check()
        waited = monotonic() - t0_mono
        if state is None or state:
            step = (wall() - t0_wall) - waited
            logger.info("[clock] synced after %.0f s; clock stepped %+.0f s", waited, step)
            return True
        if waited >= timeout_s:
            logger.warning(
                "[clock] still not synced after %.0f s; starting anyway (the session "
                "label and start time may be wrong until NTP syncs)",
                waited,
            )
            return False
