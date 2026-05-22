"""Coverage for ``streettracker.device.dashboard``.

Unlike the snapshotter tests (which mock an in-process Reolink), the
dashboard's whole job is to be a real HTTP server, so these tests
bind it to ``0.0.0.0:0`` (kernel-picked ephemeral port), read back the
chosen port via the :class:`Dashboard` handle, and exercise it with a
real ``aiohttp.ClientSession``.

A loopback IP is used in the client URL even though the server binds
``0.0.0.0`` -- requests on Windows CI runners are flaky against the
all-interfaces literal in client mode, while ``127.0.0.1`` always
works.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import pytest

from streettracker.device.dashboard import (
    Dashboard,
    _build_app,
    start_dashboard,
)

# ----------------------------------------------------------------------
# Fixtures.


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    """An empty directory standing in for the session output dir."""
    d = tmp_path / "session_out"
    d.mkdir()
    return d


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    """Plain ``ClientSession`` for hitting the dashboard."""
    async with aiohttp.ClientSession() as s:
        yield s


async def _start_on_ephemeral(directory: Path) -> Dashboard:
    """Start a dashboard on a kernel-picked port; fail loud if bind fails."""
    dash = await start_dashboard(directory, host="127.0.0.1", port=0)
    assert dash is not None, "ephemeral bind should always succeed"
    return dash


# ----------------------------------------------------------------------
# start_dashboard() -- lifecycle.


async def test_start_dashboard_returns_handle_with_resolved_port(
    session_dir: Path,
) -> None:
    dash = await _start_on_ephemeral(session_dir)
    try:
        assert dash.host == "127.0.0.1"
        # Port 0 means kernel picks; the handle exposes the real port.
        assert dash.port > 0
        assert dash.port != 0
    finally:
        await dash.stop()


async def test_dashboard_stop_releases_port(session_dir: Path) -> None:
    """After stop(), the port must be free for a re-bind. If aiohttp
    holds the socket past cleanup() the runtime can't recycle the
    dashboard across config reloads."""
    dash = await _start_on_ephemeral(session_dir)
    port = dash.port
    await dash.stop()

    # Re-bind on the same port should succeed.
    dash2 = await start_dashboard(session_dir, host="127.0.0.1", port=port)
    assert dash2 is not None
    try:
        assert dash2.port == port
    finally:
        await dash2.stop()


# ----------------------------------------------------------------------
# Routing -- index.html present / absent.


async def test_root_serves_index_when_present(
    session_dir: Path, http: aiohttp.ClientSession
) -> None:
    (session_dir / "index.html").write_text(
        "<!doctype html><title>summary</title>OK", encoding="utf-8"
    )
    dash = await _start_on_ephemeral(session_dir)
    try:
        async with http.get(f"http://127.0.0.1:{dash.port}/") as r:
            assert r.status == 200
            body = await r.text()
        assert "summary" in body
        assert "OK" in body
    finally:
        await dash.stop()


async def test_root_serves_friendly_page_when_no_index(
    session_dir: Path, http: aiohttp.ClientSession
) -> None:
    dash = await _start_on_ephemeral(session_dir)
    try:
        async with http.get(f"http://127.0.0.1:{dash.port}/") as r:
            assert r.status == 200
            assert r.content_type == "text/html"
            body = await r.text()
        # The fallback page should be obviously the "warming up" page,
        # not e.g. a stray cached file or a 404.
        assert "No sessions yet" in body
        # And should self-refresh so visitors don't get stuck.
        assert 'http-equiv="refresh"' in body
    finally:
        await dash.stop()


async def test_static_file_served_by_path(session_dir: Path, http: aiohttp.ClientSession) -> None:
    (session_dir / "vehicle_1.jpg").write_bytes(b"\xff\xd8\xff\xe0FAKEJPEG\xff\xd9")
    dash = await _start_on_ephemeral(session_dir)
    try:
        async with http.get(f"http://127.0.0.1:{dash.port}/vehicle_1.jpg") as r:
            assert r.status == 200
            body = await r.read()
        assert body.startswith(b"\xff\xd8\xff")
        assert b"FAKEJPEG" in body
    finally:
        await dash.stop()


async def test_nested_file_served(session_dir: Path, http: aiohttp.ClientSession) -> None:
    (session_dir / "snaps").mkdir()
    (session_dir / "snaps" / "main_1.jpg").write_bytes(b"\xff\xd8\xff\xe0X" * 20)
    dash = await _start_on_ephemeral(session_dir)
    try:
        async with http.get(f"http://127.0.0.1:{dash.port}/snaps/main_1.jpg") as r:
            assert r.status == 200
    finally:
        await dash.stop()


async def test_missing_file_returns_404(session_dir: Path, http: aiohttp.ClientSession) -> None:
    dash = await _start_on_ephemeral(session_dir)
    try:
        async with http.get(f"http://127.0.0.1:{dash.port}/does_not_exist.jpg") as r:
            assert r.status == 404
    finally:
        await dash.stop()


# ----------------------------------------------------------------------
# Security -- aiohttp rejects path traversal automatically; this is
# regression coverage to make sure we don't accidentally turn it off.


async def test_path_traversal_blocked(
    session_dir: Path, http: aiohttp.ClientSession, tmp_path: Path
) -> None:
    # Write a file outside the served directory.
    secret = tmp_path / "secret.txt"
    secret.write_text("nope", encoding="utf-8")

    dash = await _start_on_ephemeral(session_dir)
    try:
        # aiohttp's URL parser intercepts ``..`` segments before they
        # reach the static handler, so the response is a redirect or
        # 404 -- never 200 with the secret's contents.
        async with http.get(
            f"http://127.0.0.1:{dash.port}/../secret.txt",
            allow_redirects=False,
        ) as r:
            assert r.status != 200
            if r.status == 200:
                body = await r.text()
                assert "nope" not in body
    finally:
        await dash.stop()


# ----------------------------------------------------------------------
# Bind failure path -- the runtime treats the dashboard as optional.


async def test_bind_failure_returns_none(session_dir: Path) -> None:
    """If the requested port is already in use, start_dashboard logs
    the failure and returns None instead of raising. The runtime loop
    relies on this to keep going when port 8080 is held by another
    process (e.g. a leftover dashboard from a previous run that
    systemd hasn't yet reaped)."""
    # Grab an ephemeral port and *hold it* so the dashboard can't bind.
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        dash = await start_dashboard(session_dir, host="127.0.0.1", port=port)
        assert dash is None
    finally:
        holder.close()


# ----------------------------------------------------------------------
# _build_app -- in-process coverage that doesn't require a real socket.


async def test_build_app_root_handler_attached(session_dir: Path) -> None:
    """Cheap smoke check that the index route exists, separate from
    the live-socket tests above. Faster to fail if the route table is
    broken."""
    app = _build_app(session_dir)
    # aiohttp's UrlDispatcher exposes resources via iteration.
    resource_paths = {
        getattr(r, "_path", None) or getattr(r, "_prefix", None) for r in app.router.resources()
    }
    assert "/" in resource_paths
