"""``aiohttp`` server for the operator control panel.

Phase 1 is the **read-only decision radiator**: it polls the Orin for live
session status in the background, scans the local ``output/`` + ``runs/`` trees
for session / corpus / model state, and serves a dashboard that answers "what's
the device doing, what have I got locally, and should I retrain / promote?".

Mirrors :mod:`streettracker.web.server` deliberately — same ``_State``-behind-a
-typed-app-key holder, same ``AppRunner``/``TCPSite`` bootstrap, same jinja2
``PackageLoader`` + no-store HTML, same validated image route. The differences
that matter for a control panel:

* **Background pollers.** A ``cleanup_ctx`` task re-reads the Orin (~45 s) and
  rescans local state (~30 s) so the dashboard stays live without per-request
  SSH. Each poll is wrapped so a transient failure (Orin asleep) never crashes
  the server.
* **Access split.** Dashboards bind to the LAN for glanceability, but anything
  that *acts* is localhost-only (:func:`_require_local`). Phase 1 has only one
  such route (``POST /api/refresh``); Phase 2's job/playbook routes will reuse
  the same guard.

CLI: ``streettracker control [--output-root output] [--host 0.0.0.0] [--port 8095]``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import re
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
from aiohttp import web

import streettracker.web
from streettracker.control import introspect, jobs, orin, playbooks

logger = logging.getLogger(__name__)

# Same leaf-JPEG guard as the showcase image route.
_SAFE_IMAGE = re.compile(r"^[A-Za-z0-9_]+\.jpg$")
_SAFE_SESSION = re.compile(r"^session_[0-9_]+$")

# Peers we treat as "this machine" for control actions. Covers IPv4/IPv6
# loopback and the IPv4-mapped-IPv6 form aiohttp may report.
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}

DEFAULT_PORT = 8095


@dataclass
class _State:
    """Mutable, in-memory radiator state behind one app key.

    ``orin`` is refreshed by the background poller; ``local`` by the local
    rescan. Both are plain JSON-able dicts/dataclasses so the API can hand them
    straight to ``web.json_response``.
    """

    output_root: Path
    runs_dir: Path
    model_path: Path | None
    orin_host: str
    orin_user: str
    orin_key: str
    remote_parent: str
    transfer_mbps: float

    orin: orin.OrinStatus | None = None
    local: dict[str, Any] = field(default_factory=dict)
    local_refreshed_at: float = 0.0

    def refresh_local(self, *, allow_torch: bool = True) -> None:
        self.local = introspect.local_snapshot(
            self.output_root, self.runs_dir, self.model_path, allow_torch=allow_torch
        )
        self.local_refreshed_at = time.time()
        # Persist whatever this scan newly learned. A no-op when nothing
        # changed, so the steady-state poll costs nothing extra; the point is
        # that the NEXT panel start doesn't re-glob 400k files.
        introspect.save_cache()

    def refresh_orin(self) -> None:
        self.orin = orin.orin_live_snapshot(
            self.orin_host, self.orin_user, self.orin_key, self.remote_parent
        )

    def status_payload(self) -> dict[str, Any]:
        return {
            "orin": self.orin.to_json_dict() if self.orin else None,
            "local": self.local,
            "local_refreshed_at": self.local_refreshed_at,
            "server_time": time.time(),
            "transfer_mbps": self.transfer_mbps,
        }


STATE: web.AppKey[_State] = web.AppKey("state", _State)
JINJA: web.AppKey[jinja2.Environment] = web.AppKey("jinja", jinja2.Environment)
BRAND_SVGS: web.AppKey[dict[str, str]] = web.AppKey("brand_svgs", dict)
RUNNER: web.AppKey[jobs.JobRunner] = web.AppKey("runner", jobs.JobRunner)
PLAYBOOKS: web.AppKey[playbooks.PlaybookRunner] = web.AppKey("playbooks", playbooks.PlaybookRunner)

# Subcommands the panel may launch as jobs — the documented operational
# processes. The submit route refuses anything outside this set.
ALLOWED_JOB_KINDS = frozenset(
    {
        "pull",
        "alpr-run",
        "dvsa-label",
        "dvsa-apply",
        "vehicles",
        "people",
        "makemodel",
        "bodytype",
        "makemodel-build-uk",
        "makemodel-train-uk",
        "export-engine",
    }
)

# Bundled brand marks live with the showcase package; inline them into the
# dashboard so the corpus make-distribution chart can show real logos with no
# extra requests (same approach the showcase stats page uses).
_BRANDS_DIR = Path(streettracker.web.__file__).resolve().parent / "static" / "brands"


def _load_brand_svgs() -> dict[str, str]:
    out: dict[str, str] = {}
    if _BRANDS_DIR.is_dir():
        for p in sorted(_BRANDS_DIR.glob("*.svg")):
            with contextlib.suppress(OSError):
                out[p.stem] = p.read_text(encoding="utf-8")
    return out


# ----------------------------------------------------------------------
# Template env + filters
# ----------------------------------------------------------------------


def _make_jinja() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("streettracker.control", "templates"),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    return env


def _render(request: web.Request, template: str, **ctx: Any) -> web.Response:
    tmpl = request.app[JINJA].get_template(template)
    return web.Response(
        text=tmpl.render(**ctx),
        content_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


# ----------------------------------------------------------------------
# Access helpers
# ----------------------------------------------------------------------


def _is_local(request: web.Request) -> bool:
    """True when the request originates from this machine.

    Control actions are gated on this so a LAN viewer can watch the radiator
    but cannot restart the camera service or promote a model.
    """
    return (request.remote or "") in _LOOPBACK


def _require_local(request: web.Request) -> None:
    if not _is_local(request):
        raise web.HTTPForbidden(text="control actions are restricted to localhost")


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


def _jobs_payload(request: web.Request) -> list[dict[str, Any]]:
    return request.app[RUNNER].snapshots()


def _playbooks_payload(request: web.Request) -> list[dict[str, Any]]:
    return request.app[PLAYBOOKS].snapshots()


def _page_status(request: web.Request) -> dict[str, Any]:
    status = request.app[STATE].status_payload()
    status["jobs"] = _jobs_payload(request)
    status["playbooks"] = _playbooks_payload(request)
    return status


async def _dashboard(request: web.Request) -> web.Response:
    return _render(
        request,
        "dashboard.html",
        status=_page_status(request),
        is_local=_is_local(request),
        brand_svgs=request.app[BRAND_SVGS],
        active_page="dashboard",
    )


async def _training(request: web.Request) -> web.Response:
    return _render(
        request,
        "training.html",
        status=_page_status(request),
        is_local=_is_local(request),
        brand_svgs=request.app[BRAND_SVGS],
        active_page="training",
    )


async def _api_status(request: web.Request) -> web.Response:
    payload = _page_status(request)
    payload["is_local"] = _is_local(request)
    return web.json_response(payload)


async def _api_jobs(request: web.Request) -> web.Response:
    return web.json_response(_jobs_payload(request))


async def _api_job(request: web.Request) -> web.Response:
    job = request.app[RUNNER].get(request.match_info["id"])
    if job is None:
        raise web.HTTPNotFound(text="unknown job")
    return web.json_response(job.snapshot())


async def _api_submit_job(request: web.Request) -> web.Response:
    """Launch a documented process as a job (localhost-only)."""
    _require_local(request)
    try:
        body = await request.json()
    except ValueError:  # includes json.JSONDecodeError
        raise web.HTTPBadRequest(text="body must be JSON") from None
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="body must be a JSON object")
    kind = body.get("kind")
    if kind not in ALLOWED_JOB_KINDS:
        raise web.HTTPBadRequest(text=f"unknown or disallowed job kind: {kind!r}")
    args = body.get("args") or []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise web.HTTPBadRequest(text="args must be a list of strings")
    label = str(body.get("label") or "")
    job = request.app[RUNNER].submit(jobs.JobSpec(kind=kind, args=args, label=label))
    return web.json_response(job.snapshot())


async def _api_cancel_job(request: web.Request) -> web.Response:
    _require_local(request)
    ok = await request.app[RUNNER].cancel(request.match_info["id"])
    if not ok:
        raise web.HTTPNotFound(text="unknown job")
    return web.json_response({"cancelled": request.match_info["id"]})


async def _api_playbooks(request: web.Request) -> web.Response:
    return web.json_response(_playbooks_payload(request))


async def _api_playbook(request: web.Request) -> web.Response:
    runner = request.app[PLAYBOOKS]
    pb = runner.get(request.match_info["id"])
    if pb is None:
        raise web.HTTPNotFound(text="unknown playbook")
    return web.json_response(runner.snapshot(pb))


async def _api_submit_playbook(request: web.Request) -> web.Response:
    """Launch a multi-step playbook (localhost-only; destructive ones need confirm)."""
    _require_local(request)
    try:
        body = await request.json()
    except ValueError:
        raise web.HTTPBadRequest(text="body must be JSON") from None
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="body must be a JSON object")
    name = body.get("name")
    if not isinstance(name, str):
        raise web.HTTPBadRequest(text="name must be a string")
    meta = playbooks.PLAYBOOKS.get(name)
    if meta is None:
        raise web.HTTPBadRequest(text=f"unknown playbook: {name!r}")
    if meta["destructive"] and body.get("confirm") is not True:
        raise web.HTTPBadRequest(text=f"'{name}' is destructive — resubmit with confirm=true")

    state = request.app[STATE]
    ctx = playbooks.PlaybookContext(
        output_root=state.output_root,
        runs_dir=state.runs_dir,
        model_path=state.model_path or introspect.production_model_path(),
        orin_host=state.orin_host,
        orin_user=state.orin_user,
        orin_key=state.orin_key,
        remote_parent=state.remote_parent,
        target=str(state.output_root),
    )
    kwargs: dict[str, str] = {}
    if name == "enrich":
        session = body.get("session")
        if (
            not isinstance(session, str)
            or not _SAFE_SESSION.match(session)
            or not (state.output_root / session).is_dir()
        ):
            raise web.HTTPBadRequest(text="enrich requires a valid local session")
        kwargs["session"] = session
    elif name == "roll":
        # Resolve the live session over SSH; refuse if the device is down/idle.
        snap = await asyncio.to_thread(
            orin.orin_live_snapshot,
            state.orin_host,
            state.orin_user,
            state.orin_key,
            state.remote_parent,
        )
        if not snap.reachable or snap.service_active != "active" or not snap.live_session:
            raise web.HTTPBadRequest(text="roll needs the device reachable and actively recording")
        kwargs["old_session"] = snap.live_session
    elif name == "promote":
        run = body.get("run")
        if run is not None and not isinstance(run, str):
            raise web.HTTPBadRequest(text="run must be a string")
        if run:
            kwargs["run_name"] = run

    try:
        label, steps = playbooks.build_playbook(name, ctx, **kwargs)
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from None
    runner = request.app[PLAYBOOKS]
    pb = runner.submit(name, steps, label=label)
    return web.json_response(runner.snapshot(pb))


async def _api_cancel_playbook(request: web.Request) -> web.Response:
    _require_local(request)
    ok = await request.app[PLAYBOOKS].cancel(request.match_info["id"])
    if not ok:
        raise web.HTTPNotFound(text="unknown playbook")
    return web.json_response({"cancelled": request.match_info["id"]})


async def _api_orin(request: web.Request) -> web.Response:
    state = request.app[STATE]
    return web.json_response(state.orin.to_json_dict() if state.orin else None)


async def _api_sessions(request: web.Request) -> web.Response:
    return web.json_response(request.app[STATE].local.get("sessions", []))


async def _api_corpus(request: web.Request) -> web.Response:
    return web.json_response(request.app[STATE].local.get("corpus"))


async def _api_model(request: web.Request) -> web.Response:
    return web.json_response(request.app[STATE].local.get("model"))


async def _api_pull_preview(request: web.Request) -> web.Response:
    """On-demand remote inventory + copy estimate for one session.

    Read-only (a remote ``du``/``find``) so it is LAN-allowed, but it can take
    a few seconds, hence on-demand rather than part of the poll.
    """
    session = request.match_info["session"]
    if not _SAFE_SESSION.match(session):
        raise web.HTTPBadRequest(text="bad session label")
    state = request.app[STATE]
    inv = await asyncio.to_thread(
        orin.pull_preview,
        session,
        state.orin_host,
        state.orin_user,
        state.orin_key,
        state.remote_parent,
    )
    est = orin.estimate_transfer_seconds(inv.bytes, state.transfer_mbps)
    return web.json_response(
        {
            "session": session,
            "bytes": inv.bytes,
            "human_bytes": orin.human_bytes(inv.bytes),
            "files": inv.files,
            "main_snaps": inv.main_snaps,
            "vehicle_snaps": inv.vehicle_main_snaps,
            "person_snaps": max(0, inv.main_snaps - inv.vehicle_main_snaps),
            "est_seconds": est,
            "transfer_mbps": state.transfer_mbps,
        }
    )


async def _api_refresh(request: web.Request) -> web.Response:
    """Force an immediate Orin + local rescan (localhost-only)."""
    _require_local(request)
    state = request.app[STATE]
    await asyncio.to_thread(state.refresh_orin)
    await asyncio.to_thread(state.refresh_local)
    return web.json_response(_page_status(request))


async def _serve_image(request: web.Request) -> web.StreamResponse:
    session = request.match_info["session"]
    filename = request.match_info["filename"]
    state = request.app[STATE]
    if not _SAFE_SESSION.match(session) or not _SAFE_IMAGE.match(filename):
        raise web.HTTPNotFound()
    base = (state.output_root / session).resolve()
    path = (base / filename).resolve()
    if base.parent != state.output_root.resolve() or base != path.parent or not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


# ----------------------------------------------------------------------
# Background pollers
# ----------------------------------------------------------------------


async def _poll_loop(name: str, fn: Any, interval: float) -> None:
    """Run blocking ``fn`` (in a thread) every ``interval`` s, forever.

    Each tick is isolated: an exception is logged and the loop continues, so a
    flaky Orin connection or a half-written JSON never kills the poller.
    """
    while True:
        try:
            await asyncio.to_thread(fn)
        except Exception:  # keep the radiator alive through any poll error
            logger.exception("[control] %s poll failed", name)
        await asyncio.sleep(interval)


def _make_background(poll_orin: float, poll_local: float) -> Any:
    async def _background(app: web.Application) -> AsyncIterator[None]:
        state = app[STATE]
        tasks = [
            asyncio.create_task(_poll_loop("orin", state.refresh_orin, poll_orin)),
            asyncio.create_task(_poll_loop("local", state.refresh_local, poll_local)),
        ]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    return _background


# ----------------------------------------------------------------------
# App assembly
# ----------------------------------------------------------------------


def build_app(
    output_root: Path,
    *,
    runs_dir: Path = Path("runs"),
    model_path: Path | None = None,
    orin_host: str = orin.DEFAULT_HOST,
    orin_user: str = orin.DEFAULT_USER,
    orin_key: str = orin.DEFAULT_KEY,
    remote_parent: str = orin.DEFAULT_REMOTE_PARENT,
    transfer_mbps: float = orin.DEFAULT_TRANSFER_MBPS,
    poll_orin: float = 45.0,
    poll_local: float = 30.0,
    background: bool = True,
) -> web.Application:
    """Build the control-panel :class:`aiohttp.web.Application`.

    Does an eager *local* scan so the first request paints immediately and
    ``main()`` can report counts; the Orin probe + the torch-backed model read
    are left to the background poller (``background=False`` in tests skips it).
    """
    state = _State(
        output_root=output_root,
        runs_dir=runs_dir,
        model_path=model_path,
        orin_host=orin_host,
        orin_user=orin_user,
        orin_key=orin_key,
        remote_parent=remote_parent,
        transfer_mbps=transfer_mbps,
    )
    # Restore the previous run's read cache BEFORE the eager scan, or startup
    # re-globs every session directory and re-parses every data.json. That cost
    # is O(files under output/) and lands exactly when you restart the panel
    # after a big pull -- cold OS cache, disk still busy with the enrich.
    introspect.load_cache(output_root)

    # allow_torch=False keeps startup off the torch import path; the background
    # rescan fills model metadata in moments.
    state.refresh_local(allow_torch=not background)

    app = web.Application()
    app[STATE] = state
    app[JINJA] = _make_jinja()
    app[BRAND_SVGS] = _load_brand_svgs()
    app[RUNNER] = jobs.JobRunner(cwd=Path.cwd(), history_path=output_root / "control_jobs.json")
    app[PLAYBOOKS] = playbooks.PlaybookRunner(
        app[RUNNER], history_path=output_root / "control_playbooks.json"
    )

    app.router.add_get("/", _dashboard)
    app.router.add_get("/training", _training)
    app.router.add_get("/api/status", _api_status)
    app.router.add_get("/api/orin", _api_orin)
    app.router.add_get("/api/sessions", _api_sessions)
    app.router.add_get("/api/corpus", _api_corpus)
    app.router.add_get("/api/model", _api_model)
    app.router.add_get("/api/pull-preview/{session}", _api_pull_preview)
    app.router.add_post("/api/refresh", _api_refresh)
    app.router.add_get("/api/jobs", _api_jobs)
    app.router.add_post("/api/jobs", _api_submit_job)
    app.router.add_get("/api/jobs/{id}", _api_job)
    app.router.add_post("/api/jobs/{id}/cancel", _api_cancel_job)
    app.router.add_get("/api/playbooks", _api_playbooks)
    app.router.add_post("/api/playbooks", _api_submit_playbook)
    app.router.add_get("/api/playbooks/{id}", _api_playbook)
    app.router.add_post("/api/playbooks/{id}/cancel", _api_cancel_playbook)
    app.router.add_get("/images/{session}/{filename}", _serve_image)

    if background:
        app.cleanup_ctx.append(_make_background(poll_orin, poll_local))
    return app


async def _serve(args: argparse.Namespace) -> None:
    app = build_app(
        args.output_root,
        runs_dir=args.runs_dir,
        model_path=args.model,
        orin_host=args.orin_host,
        orin_user=args.orin_user,
        orin_key=args.orin_key,
        remote_parent=args.remote_parent,
        transfer_mbps=args.transfer_mbps,
        poll_orin=args.poll_orin,
        poll_local=args.poll_local,
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    totals = app[STATE].local.get("totals", {})
    print(
        f"[control] {totals.get('n_sessions', 0)} local sessions, "
        f"{totals.get('n_snaps', 0)} snaps; Orin {args.orin_user}@{args.orin_host}"
    )
    print(f"[control] serving http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    if args.host not in {"127.0.0.1", "localhost"}:
        print("[control] dashboards are LAN-visible; control actions stay localhost-only.")
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="streettracker control",
        description="Operator control panel: live radiator + process control (Phase 1: radiator).",
    )
    ap.add_argument("--output-root", type=Path, default=Path("output"), help="session_* root")
    ap.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="training runs/corpora root"
    )
    ap.add_argument(
        "--model", type=Path, default=None, help="production model path (default: packaged)"
    )
    ap.add_argument(
        "--host", default="0.0.0.0", help="bind interface (default 0.0.0.0 = LAN-visible radiator)"
    )
    ap.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"bind port (default {DEFAULT_PORT})"
    )
    ap.add_argument("--orin-host", default=orin.DEFAULT_HOST, help="Orin SSH host/alias")
    ap.add_argument("--orin-user", default=orin.DEFAULT_USER, help="Orin SSH user")
    ap.add_argument("--orin-key", default=orin.DEFAULT_KEY, help="Orin SSH private key")
    ap.add_argument("--remote-parent", default=orin.DEFAULT_REMOTE_PARENT, help="device output dir")
    ap.add_argument(
        "--transfer-mbps",
        type=float,
        default=orin.DEFAULT_TRANSFER_MBPS,
        help="assumed copy throughput (MB/s) for pull ETAs",
    )
    ap.add_argument("--poll-orin", type=float, default=45.0, help="Orin poll interval (s)")
    ap.add_argument("--poll-local", type=float, default=30.0, help="local rescan interval (s)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.output_root.is_dir():
        print(f"[control] not a directory: {args.output_root}", file=sys.stderr)
        return 2
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        print("\n[control] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
