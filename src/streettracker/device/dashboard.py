"""Async HTTP dashboard for the live runtime.

A bare ``aiohttp.web`` static-file server rooted at the session output
directory. Operators point a browser at ``http://<orin>:8080/`` and get
the latest session's summary HTML (via the ``index.html`` redirector
written by ``streettracker.common.summary``). Until the first session
finalizes, ``/`` returns a friendly auto-refreshing waiting page so the
operator can leave the tab open while the runtime warms up.

Design choices
~~~~~~~~~~~~~~

* **Optional component.** ``start_dashboard()`` returns ``None`` on a
  bind failure (port in use, IPv4/IPv6 mismatch, etc.) rather than
  raising. The runtime loop continues without a dashboard; the rest of
  the pipeline -- inference, snapshots, output -- is unaffected.
* **No access log.** ``AppRunner(access_log=None)`` suppresses
  per-request log spam. The systemd journal stays focused on session
  events, not browser refreshes.
* **No path traversal.** ``aiohttp.web.add_static`` rejects ``..``
  segments. The friendly ``/`` handler reads only the literal
  ``index.html`` under the served directory; nothing operator-supplied
  reaches the filesystem.
* **No directory listing.** ``show_index=False``. Subdirectories
  (``snaps/``, ``main/``, etc.) would otherwise be browseable, which
  isn't a security issue but is noise.

Replaces NanoTracker's ``start_http_server`` (nano_tracker.py:989-1037)
-- the original used stdlib ``http.server`` on a daemon thread because
Python 3.6 predates the maturity of ``aiohttp``. StreetTracker's
runtime is asyncio-throughout so the dashboard composes naturally with
the rest of the loop's cleanup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)


# Returned at ``/`` when ``index.html`` hasn't been written yet (first
# session not finalized). 5-second meta refresh so the operator's tab
# updates itself as soon as the first summary lands.
_NO_SESSIONS_HTML = (
    "<!DOCTYPE html><html><head><title>StreetTracker</title>"
    '<meta http-equiv="refresh" content="5">'
    "<style>body{font-family:sans-serif;max-width:40em;margin:4em auto;"
    "color:#444;text-align:center}</style></head><body>"
    "<h1>StreetTracker</h1>"
    "<p>No sessions yet -- the first summary will appear here shortly.</p>"
    "</body></html>"
)


@dataclass(slots=True)
class Dashboard:
    """Handle returned by :func:`start_dashboard`.

    The runtime keeps this object around for the lifetime of the
    session and ``await``s :meth:`stop` during graceful shutdown.
    """

    runner: web.AppRunner
    site: web.TCPSite
    host: str
    port: int  # resolved port (may differ from requested port when port=0)

    async def stop(self) -> None:
        """Stop the site and clean up the runner.

        Idempotent in practice -- aiohttp tolerates ``stop()`` after a
        runner that has already been cleaned up, raising no errors.
        """
        await self.site.stop()
        await self.runner.cleanup()


def _resolved_port(site: web.TCPSite, requested_port: int) -> int:
    """Return the actual port a TCPSite bound to.

    ``site._server`` is the underlying ``asyncio.Server`` (a private
    attribute on aiohttp's side but a stable one across the 3.x line;
    aiohttp's own tests reach for it the same way). When the operator
    requested ``port=0`` the kernel picks an ephemeral port and we want
    to surface that on the :class:`Dashboard` so callers can log it.
    """
    server = site._server
    # ``asyncio.AbstractServer`` (the static type aiohttp annotates) doesn't
    # declare ``sockets``, but the concrete ``asyncio.base_events.Server``
    # instance does. Reach in via getattr so type checkers don't complain.
    sockets = getattr(server, "sockets", None) if server is not None else None
    if not sockets:
        return requested_port
    sockname = sockets[0].getsockname()
    # AF_INET: (host, port); AF_INET6: (host, port, flowinfo, scopeid).
    return int(sockname[1])


def _build_app(directory: Path) -> web.Application:
    """Build the aiohttp Application that serves ``directory``.

    Factored out so tests can drive it through ``aiohttp.test_utils``
    without spinning up a real TCP socket.
    """
    app = web.Application()

    async def _serve_index(_request: web.Request) -> web.StreamResponse:
        index = directory / "index.html"
        if index.is_file():
            return web.FileResponse(index)
        return web.Response(text=_NO_SESSIONS_HTML, content_type="text/html")

    app.router.add_get("/", _serve_index)
    # Static handler covers everything else under the directory.
    # ``show_index=False`` keeps subdirectories opaque; aiohttp blocks
    # ``..`` traversal automatically.
    app.router.add_static("/", str(directory), show_index=False)
    return app


async def start_dashboard(
    directory: Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> Dashboard | None:
    """Start a static-file dashboard serving ``directory``.

    Returns a :class:`Dashboard` handle on success or ``None`` if the
    bind failed (port in use, address invalid, etc.). The runtime
    treats the dashboard as optional and continues without it on
    failure.

    Args:
        directory: session output dir. Must exist when the first
            request comes in; need not exist at this call (we don't
            stat it here, only when the request handler runs).
        host: interface to bind. Default ``0.0.0.0`` matches the
            existing NanoTracker deploy.
        port: TCP port. ``0`` lets the kernel pick an ephemeral port;
            check the returned ``Dashboard.port`` for the actual one.
    """
    app = _build_app(directory)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    try:
        await site.start()
    except OSError as e:
        logger.warning(
            "[dashboard] failed to bind %s:%d -- %s; dashboard disabled",
            host,
            port,
            e,
        )
        await runner.cleanup()
        return None
    bound_port = _resolved_port(site, port)
    logger.info("[dashboard] serving %s at http://%s:%d/", directory, host, bound_port)
    return Dashboard(runner=runner, site=site, host=host, port=bound_port)
