"""Async Reolink HTTP snapshotter for the StreetTracker runtime.

Reolink's ``/cgi-bin/api.cgi?cmd=Snap`` endpoint returns a full-frame
JPEG of the main stream (typically 4K) regardless of which RTSP stream
the runtime is decoding. That lets us pair the cheap 896x512 sub-stream
inference output with high-resolution stills for downstream ALPR /
make+model / re-id work -- the sub-stream HQ crop is inherently capped
around 250x180.

Architecture
~~~~~~~~~~~~

* Each fire is an ``aiohttp`` GET against a single URL built at
  construction (credentials are URL-encoded; the URL is treated as a
  secret and is not logged).
* :meth:`ReolinkSnapshotter.submit` is *non-async*. It schedules a
  fire-and-forget ``asyncio.Task`` and returns immediately. The
  runtime loop calls it from per-frame decision code and continues
  reading frames; the snap completes on its own.
* A counter caps concurrency (default 2). When at the cap, ``submit``
  records a ``dropped`` stat and returns ``None`` rather than queueing.
  A vehicle has already moved on by the time a queued snap would
  fire, so dropping is strictly better than queueing.
* A periodic :func:`run_cleanup_task` deletes
  ``{session}/{prefix}_<id>_main_<N>.jpg`` files whose mtime is older
  than ``keep_days``. Cheap (stat-only) -- a 24h session has O(thousands)
  files, well under a second.

Mirrors NanoTracker's ``ReolinkSnapshotter`` (threads) translated to
asyncio. URL format, JPEG-magic check, and drop-when-saturated
semantics match. Stats counters are preserved for the dashboard.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import urllib.parse
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


# Reolink's Snap endpoint returns "real" JPEGs starting with the SOI
# marker. On auth failure / busy-camera state it returns an HTML or JSON
# error page; we treat anything not starting with FF D8 FF as a failure.
_JPEG_SOI = b"\xff\xd8\xff"
_MIN_JPEG_BYTES = 100


@dataclasses.dataclass(slots=True)
class SnapshotStats:
    """Counters for the dashboard's "Reolink snaps" panel."""

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    dropped: int = 0


def build_snap_url(
    ip: str, http_port: int, username: str, password: str
) -> str:
    """Build the Reolink ``cmd=Snap`` URL.

    User and password are URL-encoded with ``safe=""`` so any ``&``,
    ``=``, ``+``, ``#``, or whitespace in the credentials survives the
    GET-query embedding intact. (Reolink itself accepts these characters
    in passwords.)
    """
    user_enc = urllib.parse.quote(username, safe="")
    pw_enc = urllib.parse.quote(password, safe="")
    return (
        f"http://{ip}:{http_port}/cgi-bin/api.cgi"
        f"?cmd=Snap&channel=0&user={user_enc}&password={pw_enc}"
    )


@dataclasses.dataclass(slots=True)
class ReolinkSnapshotter:
    """Async fire-and-forget Reolink snapshot client.

    Owns no I/O resources directly: the caller passes in an
    ``aiohttp.ClientSession`` (which they construct and close around
    the lifetime of the runtime). Tests use the same shape, just
    against an ``aiohttp.test_utils.TestServer`` URL.
    """

    url: str
    session: aiohttp.ClientSession
    timeout_s: float = 5.0
    max_concurrent: int = 2

    stats: SnapshotStats = dataclasses.field(default_factory=SnapshotStats)
    _in_flight: int = 0
    _tasks: set[asyncio.Task[bool]] = dataclasses.field(default_factory=set)

    def submit(self, output_path: Path) -> asyncio.Task[bool] | None:
        """Schedule a snap; return the ``Task`` or ``None`` when dropped.

        The task resolves to ``True`` on success (JPEG written to
        ``output_path``) or ``False`` on any failure (HTTP error, body
        not a JPEG, write error). Callers commonly ignore the task --
        result tracking is the runtime loop's job, not the snapshotter's.
        """
        if self._in_flight >= self.max_concurrent:
            self.stats.dropped += 1
            logger.info(
                "[snap] dropped %s (max concurrent=%d)",
                output_path.name,
                self.max_concurrent,
            )
            return None
        self._in_flight += 1
        task = asyncio.create_task(
            self._fire(output_path), name=f"snap-{output_path.name}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[bool]) -> None:
        self._tasks.discard(task)
        self._in_flight = max(0, self._in_flight - 1)

    async def _fire(self, output_path: Path) -> bool:
        self.stats.attempts += 1
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        try:
            async with self.session.get(self.url, timeout=timeout) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP {resp.status}")
                data = await resp.read()
            if len(data) < _MIN_JPEG_BYTES or data[:3] != _JPEG_SOI:
                raise ValueError(
                    f"response is not a JPEG "
                    f"(len={len(data)}, head={data[:16]!r})"
                )
            # Disk write off the event loop -- 4K JPEGs are 1-3 MB and
            # write() through a userspace buffer can take a few ms on
            # the Orin's eMMC under load.
            await asyncio.to_thread(output_path.write_bytes, data)
            self.stats.successes += 1
            return True
        except Exception as exc:
            self.stats.failures += 1
            logger.warning(
                "[snap] failed for %s: %s: %s",
                output_path.name,
                type(exc).__name__,
                exc,
            )
            return False

    async def drain(self, *, timeout: float = 10.0) -> int:
        """Wait for any in-flight snaps to settle, up to ``timeout``.

        Returns the number of tasks that finished in the window. Tasks
        still pending after the window are *not* cancelled -- the
        caller closes the session afterward, which will fail their
        HTTP reads and they'll resolve via the failure path."""
        if not self._tasks:
            return 0
        done, _pending = await asyncio.wait(
            self._tasks, timeout=timeout
        )
        return len(done)


def cleanup_old_snaps(parent_dir: Path, keep_days: int) -> int:
    """Best-effort delete of ``*_main_*.jpg`` under ``parent_dir/<session>/``
    older than ``keep_days``. Returns number of files deleted.

    Sync function -- callers in async contexts wrap it via
    :func:`asyncio.to_thread`. Swallows per-file ``OSError`` so one bad
    path doesn't abort the whole sweep.
    """
    if keep_days <= 0:
        return 0
    cutoff = time.time() - keep_days * 86400.0
    n_deleted = 0
    try:
        for snap_path in parent_dir.glob("*/*_main_*.jpg"):
            try:
                if snap_path.stat().st_mtime < cutoff:
                    snap_path.unlink()
                    n_deleted += 1
            except OSError:
                pass
    except OSError:
        pass
    return n_deleted


async def run_cleanup_task(
    parent_dir: Path,
    *,
    interval_s: float,
    keep_days: int,
) -> None:
    """Periodic cleanup loop.

    Designed to be started as ``asyncio.create_task(run_cleanup_task(...))``
    and cancelled at shutdown. Logs deletion counts; never raises out
    of the loop (transient FS errors are logged and the loop continues).
    """
    if keep_days <= 0:
        logger.info("[snap] cleanup disabled (keep_days <= 0)")
        return
    logger.info(
        "[snap] cleanup task started (parent=%s, interval=%.0fs, keep=%dd)",
        parent_dir, interval_s, keep_days,
    )
    while True:
        try:
            await asyncio.sleep(interval_s)
            deleted = await asyncio.to_thread(
                cleanup_old_snaps, parent_dir, keep_days
            )
            if deleted:
                logger.info("[snap] cleanup deleted %d stale snaps", deleted)
        except asyncio.CancelledError:
            logger.info("[snap] cleanup task cancelled")
            raise
        except Exception as exc:  # pragma: no cover - transient FS errors
            logger.warning("[snap] cleanup error: %s", exc)
