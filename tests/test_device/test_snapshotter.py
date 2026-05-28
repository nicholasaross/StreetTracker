"""Coverage for ``streettracker.device.snapshotter``.

The snapshotter is the only StreetTracker subsystem that talks to a
real Reolink camera over HTTP. Mocking the HTTP layer at the
``aiohttp.ClientSession`` level is awkward (the request/response
context managers are notoriously fiddly to fake), so these tests
instead spin up a tiny in-process ``aiohttp.web`` application that
acts as a fake Reolink and connect a real ``ClientSession`` to it.

That gives realistic coverage of the timeout, status-code, and
content-validation paths without ever touching a network.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from streettracker.device.snapshotter import (
    ReolinkSnapshotter,
    SnapshotStats,
    build_snap_url,
    cleanup_old_snaps,
    run_cleanup_task,
)

# ----------------------------------------------------------------------
# URL builder.


def test_build_snap_url_format() -> None:
    url = build_snap_url("192.168.1.72", 80, "admin", "secret")
    assert url == (
        "http://192.168.1.72:80/cgi-bin/api.cgi?cmd=Snap&channel=0&user=admin&password=secret"
    )


def test_build_snap_url_encodes_special_chars_in_password() -> None:
    """Reolink accepts &, =, +, # in passwords. They must be URL-encoded
    so the GET-query embedding doesn't break."""
    url = build_snap_url("10.0.0.1", 80, "user", "p&w=d+#@!")
    assert "user=user&password=p%26w%3Dd%2B%23%40%21" in url


def test_build_snap_url_encodes_special_chars_in_username() -> None:
    url = build_snap_url("10.0.0.1", 80, "ad min", "p")
    assert "user=ad%20min" in url


# ----------------------------------------------------------------------
# Fake-Reolink fixtures.
#
# Each test parametrises the fake's response, mounts it on an
# in-process aiohttp TestServer, connects a real ClientSession, and
# exercises the snapshotter.


# A small valid JPEG: SOI + APP0 chunk + EOI. Real cameras return
# multi-MB images; this is just enough to pass the "is it a JPEG?"
# header sniff and the ">= MIN_JPEG_BYTES" length gate.
_FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"


async def _start_fake_reolink(handler: Callable) -> tuple[TestServer, str]:
    """Mount ``handler`` at ``/cgi-bin/api.cgi`` on an aiohttp TestServer."""
    app = web.Application()
    app.router.add_get("/cgi-bin/api.cgi", handler)
    server = TestServer(app)
    await server.start_server()
    assert server.host is not None and server.port is not None
    url = f"http://{server.host}:{server.port}/cgi-bin/api.cgi?cmd=Snap"
    return server, url


@pytest.fixture
async def session() -> AsyncIterator[aiohttp.ClientSession]:
    s = aiohttp.ClientSession()
    try:
        yield s
    finally:
        await s.close()


# ----------------------------------------------------------------------
# Happy path: a single submit fires, writes the JPEG, increments stats.


async def test_submit_writes_jpeg_to_disk(tmp_path: Path, session: aiohttp.ClientSession) -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.Response(body=_FAKE_JPEG, content_type="image/jpeg")

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session, max_concurrent=2)
        out = tmp_path / "vehicle_1_main_1.jpg"
        task = snapper.submit(out)
        assert task is not None
        ok = await task
        assert ok is True
        assert out.read_bytes() == _FAKE_JPEG
        assert snapper.stats.attempts == 1
        assert snapper.stats.successes == 1
        assert snapper.stats.failures == 0
        assert snapper.stats.dropped == 0
    finally:
        await server.close()


# ----------------------------------------------------------------------
# Failure paths.


async def test_non_jpeg_response_marked_failure(
    tmp_path: Path, session: aiohttp.ClientSession
) -> None:
    """Reolink returns an HTML/JSON error page on auth failure or busy
    state. We reject anything not starting with the JPEG SOI bytes."""

    async def handler(_req: web.Request) -> web.Response:
        return web.Response(text='{"error":"busy"}', content_type="application/json")

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session)
        task = snapper.submit(tmp_path / "vehicle_1_main_1.jpg")
        assert task is not None
        ok = await task
        assert ok is False
        assert snapper.stats.failures == 1
        assert snapper.stats.successes == 0
        assert not (tmp_path / "vehicle_1_main_1.jpg").exists()
    finally:
        await server.close()


async def test_short_response_marked_failure(
    tmp_path: Path, session: aiohttp.ClientSession
) -> None:
    """Bodies < MIN_JPEG_BYTES are treated as failures even if the SOI
    bytes are present -- a truncated response is not a usable snap."""

    async def handler(_req: web.Request) -> web.Response:
        return web.Response(body=b"\xff\xd8\xff", content_type="image/jpeg")

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session)
        task = snapper.submit(tmp_path / "vehicle_1_main_1.jpg")
        assert task is not None
        assert await task is False
        assert snapper.stats.failures == 1
    finally:
        await server.close()


async def test_http_error_status_marked_failure(
    tmp_path: Path, session: aiohttp.ClientSession
) -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.Response(status=401, text="unauthorized")

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session)
        task = snapper.submit(tmp_path / "vehicle_1_main_1.jpg")
        assert task is not None
        assert await task is False
        assert snapper.stats.failures == 1
    finally:
        await server.close()


async def test_timeout_marked_failure(tmp_path: Path, session: aiohttp.ClientSession) -> None:
    """Slow camera -> aiohttp client timeout -> failure stat."""

    async def handler(_req: web.Request) -> web.Response:
        await asyncio.sleep(0.5)  # longer than timeout_s below
        return web.Response(body=_FAKE_JPEG)

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session, timeout_s=0.1)
        task = snapper.submit(tmp_path / "vehicle_1_main_1.jpg")
        assert task is not None
        assert await task is False
        assert snapper.stats.failures == 1
    finally:
        await server.close()


# ----------------------------------------------------------------------
# Concurrency cap.


async def test_submit_drops_excess_when_at_max_concurrent(
    tmp_path: Path, session: aiohttp.ClientSession
) -> None:
    """When ``max_concurrent`` snaps are in flight, the next ``submit``
    returns ``None`` and increments ``dropped`` -- it does NOT queue."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(_req: web.Request) -> web.Response:
        started.set()
        await release.wait()
        return web.Response(body=_FAKE_JPEG)

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session, max_concurrent=2)
        t1 = snapper.submit(tmp_path / "vehicle_1_main_1.jpg")
        t2 = snapper.submit(tmp_path / "vehicle_2_main_1.jpg")
        # Both should be in flight now.
        assert t1 is not None and t2 is not None
        # Wait for the handler to actually start (so _in_flight bumped).
        await started.wait()
        # Third submit must drop.
        t3 = snapper.submit(tmp_path / "vehicle_3_main_1.jpg")
        assert t3 is None
        assert snapper.stats.dropped == 1
        # Let the handlers finish, then verify the first two succeeded.
        release.set()
        await asyncio.gather(t1, t2)
        assert snapper.stats.successes == 2
        # After tasks complete, a fresh submit should succeed again.
        t4 = snapper.submit(tmp_path / "vehicle_4_main_1.jpg")
        assert t4 is not None
        assert await t4 is True
    finally:
        await server.close()


# ----------------------------------------------------------------------
# drain().


# ----------------------------------------------------------------------
# Latency tracking. The runtime needs per-fire wall-clock latency to
# tune trigger placement post-cutover: shifted upstream of the optimal
# plate-readable position by exactly the measured median latency.


async def test_submit_records_latency_on_success(
    tmp_path: Path, session: aiohttp.ClientSession
) -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.Response(body=_FAKE_JPEG, content_type="image/jpeg")

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session)
        assert len(snapper.stats.latencies_ms) == 0
        task = snapper.submit(tmp_path / "vehicle_1_main_1.jpg")
        assert task is not None
        await task
        assert len(snapper.stats.latencies_ms) == 1
        # Localhost loopback can complete inside Windows'
        # ``time.monotonic()`` resolution tick (~15 ms), so the floor
        # is 0.0 -- not strict positivity. The upper bound just
        # confirms we recorded *something* and not, say, the unix
        # epoch by accident.
        assert 0.0 <= snapper.stats.latencies_ms[0] < 5000.0
    finally:
        await server.close()


async def test_failed_submit_does_not_record_latency(
    tmp_path: Path, session: aiohttp.ClientSession
) -> None:
    """Failure latencies are not interesting for trigger-placement tuning
    (a failed fire produces no usable JPEG), so we don't sample them."""

    async def handler(_req: web.Request) -> web.Response:
        return web.Response(status=500, text="server error")

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session)
        task = snapper.submit(tmp_path / "vehicle_1_main_1.jpg")
        assert task is not None
        await task
        assert snapper.stats.failures == 1
        assert len(snapper.stats.latencies_ms) == 0
    finally:
        await server.close()


def test_latency_summary_empty_returns_count_zero() -> None:
    stats = SnapshotStats()
    assert stats.latency_summary() == {"count": 0}


def test_latency_summary_percentiles_nearest_rank() -> None:
    """Nearest-rank percentiles (no interpolation). 10 samples lets us
    pin the expected indices exactly."""
    stats = SnapshotStats()
    stats.latencies_ms = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    s = stats.latency_summary()
    assert s["count"] == 10
    assert s["min_ms"] == 10.0
    assert s["max_ms"] == 100.0
    # Nearest-rank with int(n*p) indexing: p50 -> index 5 -> 60.0.
    assert s["p50_ms"] == 60.0
    # p90 -> index 9 -> 100.0; p99 -> clamped to last -> 100.0.
    assert s["p90_ms"] == 100.0
    assert s["p99_ms"] == 100.0


def test_latency_summary_rounds_to_one_decimal() -> None:
    stats = SnapshotStats()
    stats.latencies_ms = [123.456, 234.567, 345.678]
    s = stats.latency_summary()
    assert s["p50_ms"] == 234.6
    assert s["min_ms"] == 123.5


def test_latencies_ms_is_bounded_for_long_running_sessions() -> None:
    """The live deployment runs for days without session rotation. The
    latency buffer must cap so a multi-week uptime doesn't grow it
    unboundedly + slow down each idle dashboard regen's percentile sort.
    """
    from streettracker.device.snapshotter import _MAX_LATENCY_SAMPLES

    stats = SnapshotStats()
    # Default-factory'd deque enforces the cap.
    for i in range(_MAX_LATENCY_SAMPLES + 500):
        stats.latencies_ms.append(float(i))
    assert len(stats.latencies_ms) == _MAX_LATENCY_SAMPLES
    # Oldest 500 dropped off the left; newest sample retained.
    assert stats.latencies_ms[0] == 500.0
    assert stats.latencies_ms[-1] == float(_MAX_LATENCY_SAMPLES + 499)
    # Summary still works on a windowed sample.
    s = stats.latency_summary()
    assert s["count"] == _MAX_LATENCY_SAMPLES
    assert s["min_ms"] == 500.0


# ----------------------------------------------------------------------
# drain().


async def test_drain_returns_count_of_completed_tasks(
    tmp_path: Path, session: aiohttp.ClientSession
) -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.Response(body=_FAKE_JPEG)

    server, url = await _start_fake_reolink(handler)
    try:
        snapper = ReolinkSnapshotter(url=url, session=session, max_concurrent=4)
        snapper.submit(tmp_path / "vehicle_1_main_1.jpg")
        snapper.submit(tmp_path / "vehicle_2_main_1.jpg")
        n = await snapper.drain(timeout=2.0)
        assert n == 2
        assert snapper.stats.successes == 2
    finally:
        await server.close()


async def test_drain_with_no_tasks_returns_zero(
    session: aiohttp.ClientSession,
) -> None:
    snapper = ReolinkSnapshotter(url="http://unused/", session=session)
    assert await snapper.drain(timeout=0.1) == 0


# ----------------------------------------------------------------------
# Cleanup of stale files.


def test_cleanup_old_snaps_deletes_only_files_older_than_keep_days(
    tmp_path: Path,
) -> None:
    """``parent_dir/<session>/*_main_*.jpg`` older than ``keep_days``
    are deleted; newer ones, and non-matching filenames, are untouched."""
    sess = tmp_path / "session_20260101_120000"
    sess.mkdir()
    # Old: 10 days ago
    old = sess / "vehicle_1_main_1.jpg"
    old.write_bytes(b"x")
    old_mtime = time.time() - 10 * 86400
    import os

    os.utime(old, (old_mtime, old_mtime))
    # New: just created
    new = sess / "vehicle_2_main_1.jpg"
    new.write_bytes(b"x")
    # Non-matching name in the same dir; must not be touched
    other = sess / "vehicle_3_hq.jpg"
    other.write_bytes(b"x")
    os.utime(other, (old_mtime, old_mtime))  # also old, but doesn't match

    deleted = cleanup_old_snaps(tmp_path, keep_days=7)
    assert deleted == 1
    assert not old.exists()
    assert new.exists()
    assert other.exists()  # not matched by the *_main_*.jpg glob


def test_cleanup_old_snaps_zero_keep_days_is_noop(tmp_path: Path) -> None:
    """``keep_days <= 0`` disables cleanup entirely (defensive guard
    against an operator accidentally setting it to 0 and wiping data)."""
    sess = tmp_path / "session_x"
    sess.mkdir()
    old = sess / "vehicle_1_main_1.jpg"
    old.write_bytes(b"x")
    import os

    os.utime(old, (time.time() - 365 * 86400, time.time() - 365 * 86400))
    deleted = cleanup_old_snaps(tmp_path, keep_days=0)
    assert deleted == 0
    assert old.exists()


def test_cleanup_old_snaps_missing_parent_dir_returns_zero(
    tmp_path: Path,
) -> None:
    """Missing parent dir should not raise; we return 0."""
    deleted = cleanup_old_snaps(tmp_path / "nonexistent", keep_days=7)
    assert deleted == 0


# ----------------------------------------------------------------------
# run_cleanup_task loop.


async def test_run_cleanup_task_invokes_cleanup_periodically(
    tmp_path: Path,
) -> None:
    """The periodic loop wakes up, deletes stale files, and continues
    until cancelled."""
    sess = tmp_path / "sess"
    sess.mkdir()
    stale = sess / "vehicle_1_main_1.jpg"
    stale.write_bytes(b"x")
    import os

    os.utime(stale, (time.time() - 10 * 86400, time.time() - 10 * 86400))

    task = asyncio.create_task(run_cleanup_task(tmp_path, interval_s=0.05, keep_days=1))
    # Wait one interval + a margin so cleanup has run.
    await asyncio.sleep(0.2)
    assert not stale.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_run_cleanup_task_with_keep_days_zero_returns_immediately(
    tmp_path: Path,
) -> None:
    """``keep_days <= 0`` short-circuits the loop without spawning a
    polling task. Verifies the function returns instead of blocking."""
    await asyncio.wait_for(
        run_cleanup_task(tmp_path, interval_s=10.0, keep_days=0),
        timeout=0.5,
    )


# ----------------------------------------------------------------------
# Stats dataclass smoke.


def test_snapshot_stats_starts_at_zero() -> None:
    s = SnapshotStats()
    assert s.attempts == 0
    assert s.successes == 0
    assert s.failures == 0
    assert s.dropped == 0
