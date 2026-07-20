"""Orin device status for the control-panel radiator.

The panel needs to answer, at a glance and without an assistant: *is the
tracker service up, what session is it recording, how long has it run, and how
many 4K snaps has it banked so far* (which in turn drives the "how big / how
long to copy" pull estimate).

This reuses the SSH/inventory plumbing already proven in
:mod:`streettracker.cli.pull` (same host/user/key defaults, same
``find``-based remote enumeration) but wraps it for a *background poller*:

* **Non-fatal.** ``cli.pull._ssh_run`` calls :func:`sys.exit` on SSH failure —
  fine for a one-shot CLI, fatal for a long-lived server. The runner here
  returns an exit code and never exits the process; an unreachable Orin simply
  yields ``reachable=False``.
* **Timeout-bounded.** A hung SSH to a powered-off Orin must not wedge the
  poller, so every call carries ``ConnectTimeout`` + a subprocess timeout.
* **One round-trip.** Service state, live-session name and the snap count come
  back from a single combined remote command rather than three separate SSH
  handshakes.

All functions here are blocking (they shell out to ``ssh``); the async server
calls them via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

# Reuse pull's defaults + pure helpers so the panel and the CLI agree on how to
# reach the device and how to size a transfer.
from streettracker.cli.pull import (
    DEFAULT_HOST,
    DEFAULT_KEY,
    DEFAULT_REMOTE_PARENT,
    DEFAULT_USER,
    RemoteInventory,
    human_bytes,
    remote_inventory,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_KEY",
    "DEFAULT_REMOTE_PARENT",
    "DEFAULT_USER",
    "OrinStatus",
    "count_finalized_tracks",
    "estimate_transfer_seconds",
    "human_bytes",
    "orin_live_snapshot",
    "pull_preview",
    "restart_service",
]

# Default assumed sustained throughput (MB/s) for the Orin -> dev-box copy,
# used only to turn a known byte total into a human ETA on the radiator. The
# real rate varies with the link; surfaced as a CLI knob (``--transfer-mbps``).
DEFAULT_TRANSFER_MBPS = 25.0


# The device prunes 4K snaps after roughly a week; a session's imagery is
# gone after that, JSON aside. Sessions inside this window that aren't on the
# dev box yet are the ones worth shouting about.
SNAP_PRUNE_DAYS = 7.0

# How many of the newest session dirs the probe inspects. Comfortably wider
# than the prune window (a busy day can roll several sessions), while
# keeping the per-poll cost bounded on a device holding ~60 dirs.
RECENT_SESSIONS = 15


@dataclass(slots=True)
class RemoteSession:
    """A session directory on the device, seen from the panel.

    Deliberately cheap to collect: name, closed-ness and snap count come from
    one `ls`/`find` round-trip. Byte size does NOT -- `du` on a 14 GB session
    takes seconds -- so the size/ETA stays on the existing on-demand
    ``pull_preview`` path.
    """

    name: str
    is_live: bool = False
    closed: bool = False  # a _meta.json exists, so the runtime finalised it
    main_snaps: int | None = None
    start_unix: float | None = None

    @property
    def age_days(self) -> float | None:
        if self.start_unix is None:
            return None
        return max(0.0, (time.time() - self.start_unix) / 86400.0)

    @property
    def prune_days_left(self) -> float | None:
        """Days until the device starts dropping this session's 4K snaps."""
        age = self.age_days
        return None if age is None else SNAP_PRUNE_DAYS - age

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["age_days"] = self.age_days
        d["prune_days_left"] = self.prune_days_left
        return d


@dataclass(slots=True)
class OrinStatus:
    """A point-in-time read of the device, safe to serialise to the API.

    ``reachable`` is the master flag: when ``False`` every other field is
    advisory (stale from a prior poll or unknown) and the UI shows "Orin
    unreachable" rather than misleading zeros.
    """

    reachable: bool
    checked_at: float  # unix seconds of this poll
    service_active: str | None = None  # systemd "active" | "inactive" | "failed" | ...
    live_session: str | None = None  # newest session_* dir name on the device
    session_start_unix: float | None = None  # parsed from the session label
    session_runtime_s: float | None = None  # now - session_start (only while active)
    vehicle_snaps: int | None = None  # vehicle 4K snaps banked this session so far
    person_snaps: int | None = None  # person 4K snaps banked this session so far
    # Every session_* dir on the device, newest first. The panel used to see
    # only `live_session`, so CLOSED sessions that were never pulled were
    # invisible -- two of them sat unpulled for days while everything visible
    # got enriched, one of them inside the prune window (2026-07-20).
    sessions: list[RemoteSession] = field(default_factory=list)
    error: str | None = None  # short diagnostic when unreachable

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sessions"] = [s.to_json_dict() for s in self.sessions]
        return d


def _ssh(
    host: str,
    user: str,
    key: str,
    cmd: str,
    *,
    connect_timeout: int = 8,
    timeout: int = 20,
) -> tuple[int, str, str]:
    """Run ``cmd`` on the device, returning ``(returncode, stdout, stderr)``.

    Never raises and never exits: a timeout or a missing ``ssh`` binary maps to
    a non-zero code with the reason in ``stderr`` so callers can degrade to
    ``reachable=False``.
    """
    args = [
        "ssh",
        "-i",
        key,
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        f"{user}@{host}",
        cmd,
    ]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"ssh timed out after {timeout}s"
    except OSError as exc:  # ssh not installed / not on PATH
        return 127, "", f"ssh failed to launch: {exc}"
    return proc.returncode, proc.stdout, proc.stderr.strip()


def parse_session_start(label: str | None) -> float | None:
    """``session_YYYYMMDD_HHMMSS`` -> local-time unix seconds (``None`` on junk).

    The session directory is named for the wall-clock instant the runtime
    created it, so the label alone dates the session without another SSH call.
    """
    if not label or not label.startswith("session_"):
        return None
    stamp = label[len("session_") :]
    try:
        # Interpreted in the dev box's local zone; the Orin and dev box share
        # wall-clock time, so this matches the device's own clock.
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


# Combined one-shot probe. Markers keep parsing trivial and order-independent.
# ``systemctl is-active`` needs no sudo; the ``find | wc -l`` mirrors the
# enumeration ``cli.pull`` already relies on (server-side match, no glob
# expansion -> safe on a 40k-snap live session). Vehicle + person main snaps
# are counted separately so the panel can report both.
_PROBE = (
    'echo "ACTIVE $(systemctl is-active streettracker.service 2>/dev/null)"; '
    "S=$(ls -1 {parent} 2>/dev/null | grep '^session_' | sort | tail -1); "
    'echo "SESSION $S"; '
    'echo "VEH $(find {parent}/"$S" -maxdepth 1 -name \'vehicle_*_main_*.jpg\''
    ' 2>/dev/null | wc -l)"; '
    'echo "PPL $(find {parent}/"$S" -maxdepth 1 -name \'person_*_main_*.jpg\''
    ' 2>/dev/null | wc -l)"; '
    # One line per session dir: name, closed-ness (a _meta.json means the
    # runtime finalised it), and its main-snap count. `find | wc -l` again
    # rather than a glob, so a 40k-snap dir can't blow the argv limit.
    #
    # Capped at the newest RECENT_SESSIONS dirs. This runs on the background
    # poller every ~45 s and each dir costs a `find`; the device holds ~60
    # session dirs, and everything past the prune window has no snaps left to
    # pull anyway, so walking all of them would be pure I/O on the device for
    # rows the panel would discard.
    "for d in $(ls -1 {parent} 2>/dev/null | grep '^session_' | sort -r | head -{recent}); do "
    'c=$([ -f {parent}/"$d"/"$d"_meta.json ] && echo closed || echo open); '
    "n=$(find {parent}/\"$d\" -maxdepth 1 -name '*_main_*.jpg' 2>/dev/null | wc -l); "
    'echo "SESS $d $c $n"; '
    "done"
)


def orin_live_snapshot(
    host: str = DEFAULT_HOST,
    user: str = DEFAULT_USER,
    key: str = DEFAULT_KEY,
    remote_parent: str = DEFAULT_REMOTE_PARENT,
    *,
    timeout: int = 20,
) -> OrinStatus:
    """One SSH round-trip: service state, live session, and its snap count.

    Returns a fully-populated :class:`OrinStatus` on success, or one with
    ``reachable=False`` (plus a short ``error``) when the device can't be
    reached — the poller treats that as a transient, not a crash.
    """
    now = time.time()
    probe = _PROBE.format(parent=remote_parent, recent=RECENT_SESSIONS)
    code, out, err = _ssh(host, user, key, probe, timeout=timeout)
    if code != 0:
        return OrinStatus(reachable=False, checked_at=now, error=err or f"ssh exit {code}")

    active: str | None = None
    session: str | None = None
    veh: int | None = None
    ppl: int | None = None
    sessions: list[RemoteSession] = []
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        tag, val = parts[0], parts[1].strip()
        if tag == "SESS":
            bits = val.split()
            if len(bits) >= 3:
                try:
                    n_snaps: int | None = int(bits[2])
                except ValueError:
                    n_snaps = None
                sessions.append(
                    RemoteSession(
                        name=bits[0],
                        closed=bits[1] == "closed",
                        main_snaps=n_snaps,
                        start_unix=parse_session_start(bits[0]),
                    )
                )
            continue
        if tag == "ACTIVE":
            active = val or None
        elif tag == "SESSION":
            session = val or None
        elif tag in ("VEH", "PPL"):
            try:
                count: int | None = int(val)
            except ValueError:
                count = None
            if tag == "VEH":
                veh = count
            else:
                ppl = count

    # Only the newest dir counts as live, and only while the service is up:
    # a stale dir from a crashed service is closed-in-practice, not recording.
    for s in sessions:
        s.is_live = s.name == session and active == "active"

    start = parse_session_start(session)
    # Runtime is only meaningful while the service is actually recording; a
    # stale dir from a crashed service would otherwise show a growing "runtime".
    runtime = (now - start) if (start is not None and active == "active") else None
    return OrinStatus(
        reachable=True,
        checked_at=now,
        service_active=active,
        live_session=session,
        session_start_unix=start,
        session_runtime_s=runtime,
        vehicle_snaps=veh,
        person_snaps=ppl,
        sessions=sessions,
    )


def pull_preview(
    session: str,
    host: str = DEFAULT_HOST,
    user: str = DEFAULT_USER,
    key: str = DEFAULT_KEY,
    remote_parent: str = DEFAULT_REMOTE_PARENT,
) -> RemoteInventory:
    """Full remote inventory (incl. byte total) for a copy estimate.

    On-demand only — the ``du -sb`` it runs can take a few seconds on a large
    live session, so it is never called from the background poller. Reuses
    :func:`streettracker.cli.pull.remote_inventory` (already non-fatal on SSH
    error: it returns a zeroed inventory rather than exiting).
    """
    remote_path = f"{remote_parent.rstrip('/')}/{session}"
    return remote_inventory(host, user, key, remote_path)


def estimate_transfer_seconds(num_bytes: int, mbps: float = DEFAULT_TRANSFER_MBPS) -> float:
    """Rough ETA (seconds) to copy ``num_bytes`` at ``mbps`` MB/s.

    Deliberately coarse — a single global rate ignoring link variance — but
    enough to answer "minutes or an hour?" before kicking off a pull.
    """
    if mbps <= 0:
        return 0.0
    return num_bytes / (mbps * 1024 * 1024)


def restart_service(
    host: str = DEFAULT_HOST,
    user: str = DEFAULT_USER,
    key: str = DEFAULT_KEY,
    *,
    service: str = "streettracker.service",
    timeout: int = 45,
) -> tuple[bool, str]:
    """Restart the tracker service on the device (finalises the live session and
    starts a new one). Needs the NOPASSWD sudoers entry. Returns ``(ok, detail)``."""
    code, out, err = _ssh(host, user, key, f"sudo -n systemctl restart {service}", timeout=timeout)
    if code == 0:
        return True, "service restarted"
    return False, err or out or f"ssh exit {code}"


def count_finalized_tracks(
    host: str,
    user: str,
    key: str,
    remote_parent: str,
    session: str,
    *,
    timeout: int = 15,
) -> int | None:
    """Lines in a session's ``_events.jsonl`` = finalised tracks (one per line).
    ``None`` if it can't be read."""
    path = f"{remote_parent.rstrip('/')}/{session}/{session}_events.jsonl"
    code, out, _ = _ssh(
        host, user, key, f"wc -l < {shlex.quote(path)} 2>/dev/null", timeout=timeout
    )
    if code != 0:
        return None
    try:
        return int(out.strip().split()[0])
    except (ValueError, IndexError):
        return None
