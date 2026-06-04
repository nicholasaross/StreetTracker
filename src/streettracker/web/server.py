"""``aiohttp`` server for the vehicle showcase site.

Serves a cross-session gallery of the enriched (plate-read + DVSA-labelled)
cars, featuring the regularly appearing ones, plus a per-car metadata editor.

Runs on the dev box, off-device -- contrast :mod:`streettracker.device.dashboard`
(one live session's static summary on the Orin). Reuses that module's
``AppRunner`` / ``TCPSite`` plumbing pattern.

Design:

* **Aggregate once, cache in memory.** :func:`build_showcase` reads ~14
  sessions (~1-2 s); the result is cached in a :class:`_State` holder on the
  app. ``POST /api/refresh`` re-aggregates *in place* (so we never mutate the
  app mapping after start, which aiohttp deprecates) to pick up newly pulled
  sessions.
* **Metadata merged per request.** The (small) plate-keyed metadata file is
  read fresh each request and merged onto the cached cars, so edits are
  immediate without re-aggregating.
* **Local-only by default.** Binds ``127.0.0.1`` -- the page shows plate data
  and personal tags. ``--host 0.0.0.0`` opts into LAN exposure.
* **Templates via jinja2 PackageLoader** (autoescaped); images served from the
  output root through a validated route.

CLI: ``streettracker showcase [--output-root output] [--host H] [--port P]``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
from aiohttp import web

from streettracker.web.aggregate import ShowcaseCar, build_showcase, discover_sessions
from streettracker.web.metadata import DEFAULT_FILENAME, MetadataStore, is_tagged

# A snap filename is ``<prefix>_<id>[_main_<n>|_hq].jpg`` -- word chars + a
# single ``.jpg``. No path separators, no ``..``, nothing but a leaf JPEG.
_SAFE_IMAGE = re.compile(r"^[A-Za-z0-9_]+\.jpg$")


@dataclass
class _State:
    """Mutable, in-memory showcase state.

    Held behind a single app key so ``/api/refresh`` can rebuild it *in place*
    -- reassigning ``app[...]`` after the app has started is deprecated.
    """

    output_root: Path
    cars: list[ShowcaseCar] = field(default_factory=list)
    cars_by_plate: dict[str, ShowcaseCar] = field(default_factory=dict)
    sessions: set[str] = field(default_factory=set)

    def reaggregate(self) -> None:
        self.cars = build_showcase(self.output_root)
        self.cars_by_plate = {c.plate: c for c in self.cars}
        self.sessions = {d.name for d in discover_sessions(self.output_root)}

    @property
    def n_regulars(self) -> int:
        return sum(1 for c in self.cars if c.kind == "different-day")


# Typed app keys (aiohttp's recommended alternative to bare-string keys).
STATE: web.AppKey[_State] = web.AppKey("state", _State)
META_STORE: web.AppKey[MetadataStore] = web.AppKey("meta_store", MetadataStore)
JINJA: web.AppKey[jinja2.Environment] = web.AppKey("jinja", jinja2.Environment)


# ---------------------------------------------------------------------------
# Template filters
# ---------------------------------------------------------------------------


def _fmt_dt(iso: str) -> str:
    """ISO timestamp -> human 'Tue 26 May 2026, 13:21' (passthrough on junk)."""
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%a %d %b %Y, %H:%M")
    except ValueError:
        return iso


def _join_counts(counts: dict[str, int]) -> str:
    """{'right to left': 3, 'left to right': 1} -> 'right to left ×3, left to right ×1'."""
    if not counts:
        return "—"
    return ", ".join(f"{k} ×{v}" for k, v in counts.items())


def _make_jinja() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("streettracker.web", "templates"),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    env.filters["fmt_dt"] = _fmt_dt
    env.filters["join_counts"] = _join_counts
    return env


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _meta_view(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a stored metadata entry into the shape templates/JSON expect."""
    return {
        "name": (raw.get("name") or ""),
        "owner": (raw.get("owner") or ""),
        "notes": (raw.get("notes") or ""),
        "favourite": bool(raw.get("favourite", False)),
        "updated_at": (raw.get("updated_at") or ""),
    }


def _merge(car: ShowcaseCar, store: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = store.get(car.plate, {})
    d = car.to_json_dict()
    d["meta"] = _meta_view(raw)
    d["tagged"] = is_tagged(raw)
    return d


def _merge_all(cars: list[ShowcaseCar], store: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [_merge(c, store) for c in cars]


def _stats(cars: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(cars),
        "regulars": sum(1 for c in cars if c["kind"] == "different-day"),
        "same_day": sum(1 for c in cars if c["kind"] == "same-day"),
        "one_off": sum(1 for c in cars if c["kind"] == "one-off"),
        "labelled": sum(1 for c in cars if c["make"]),
        "tagged": sum(1 for c in cars if c["tagged"]),
    }


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


def _render(request: web.Request, template: str, **ctx: Any) -> web.Response:
    tmpl = request.app[JINJA].get_template(template)
    return web.Response(text=tmpl.render(**ctx), content_type="text/html")


async def _gallery(request: web.Request) -> web.Response:
    store = request.app[META_STORE].load()
    cars = _merge_all(request.app[STATE].cars, store)
    regulars = [c for c in cars if c["kind"] == "different-day"]
    return _render(request, "gallery.html", cars=cars, regulars=regulars, stats=_stats(cars))


async def _car_page(request: web.Request) -> web.Response:
    plate = request.match_info["plate"]
    car = request.app[STATE].cars_by_plate.get(plate)
    if car is None:
        raise web.HTTPNotFound(text=f"unknown plate: {plate}")
    store = request.app[META_STORE].load()
    return _render(request, "car.html", car=_merge(car, store))


async def _api_cars(request: web.Request) -> web.Response:
    store = request.app[META_STORE].load()
    return web.json_response(_merge_all(request.app[STATE].cars, store))


async def _api_car(request: web.Request) -> web.Response:
    plate = request.match_info["plate"]
    car = request.app[STATE].cars_by_plate.get(plate)
    if car is None:
        raise web.HTTPNotFound(text=f"unknown plate: {plate}")
    store = request.app[META_STORE].load()
    return web.json_response(_merge(car, store))


async def _api_set_metadata(request: web.Request) -> web.Response:
    plate = request.match_info["plate"]
    if plate not in request.app[STATE].cars_by_plate:
        raise web.HTTPNotFound(text=f"unknown plate: {plate}")
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON") from None
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="body must be a JSON object")
    entry = request.app[META_STORE].set(plate, payload)
    return web.json_response(
        {"plate": plate, "meta": _meta_view(entry), "tagged": is_tagged(entry)}
    )


async def _api_refresh(request: web.Request) -> web.Response:
    request.app[STATE].reaggregate()
    return web.json_response({"cars": len(request.app[STATE].cars)})


async def _serve_image(request: web.Request) -> web.StreamResponse:
    session = request.match_info["session"]
    filename = request.match_info["filename"]
    state = request.app[STATE]
    if session not in state.sessions or not _SAFE_IMAGE.match(filename):
        raise web.HTTPNotFound()
    base = (state.output_root / session).resolve()
    path = (base / filename).resolve()
    # Defence in depth on top of the regex + single-segment route matching.
    if base != path.parent or not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------


def build_app(output_root: Path, *, metadata_path: Path | None = None) -> web.Application:
    """Build the showcase :class:`aiohttp.web.Application`.

    Aggregates the cars eagerly so the first request is fast and ``main()`` can
    report the count. Factored out so tests can drive it with
    ``aiohttp.test_utils`` without a real socket.
    """
    state = _State(output_root=output_root)
    state.reaggregate()

    app = web.Application()
    app[STATE] = state
    app[META_STORE] = MetadataStore(
        metadata_path if metadata_path is not None else output_root / DEFAULT_FILENAME
    )
    app[JINJA] = _make_jinja()

    app.router.add_get("/", _gallery)
    app.router.add_get("/car/{plate}", _car_page)
    app.router.add_get("/api/cars", _api_cars)
    app.router.add_get("/api/cars/{plate}", _api_car)
    app.router.add_put("/api/cars/{plate}/metadata", _api_set_metadata)
    app.router.add_post("/api/refresh", _api_refresh)
    app.router.add_get("/images/{session}/{filename}", _serve_image)
    return app


async def _serve(output_root: Path, host: str, port: int, metadata_path: Path | None) -> None:
    app = build_app(output_root, metadata_path=metadata_path)
    state = app[STATE]
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(
        f"[showcase] {len(state.cars)} identified cars ({state.n_regulars} regulars) "
        f"from {len(state.sessions)} sessions"
    )
    print(f"[showcase] serving http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="streettracker showcase",
        description="Local website showcasing enriched + recurring vehicles.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Directory holding the session_* output dirs (default: output).",
    )
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind interface (default 127.0.0.1; use 0.0.0.0 for LAN).",
    )
    ap.add_argument("--port", type=int, default=8090, help="Bind port (default 8090).")
    ap.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=f"User-metadata JSON path (default: <output-root>/{DEFAULT_FILENAME}).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.output_root.is_dir():
        print(f"[showcase] not a directory: {args.output_root}", file=sys.stderr)
        return 2
    try:
        asyncio.run(_serve(args.output_root, args.host, args.port, args.metadata))
    except KeyboardInterrupt:
        print("\n[showcase] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
