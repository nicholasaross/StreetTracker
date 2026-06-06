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
import contextlib
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
from streettracker.web.stats import (
    DEFAULT_ROAD_AXIS_PX,
    Stats,
    build_stats,
    m_per_px_from_road_length,
)

# A snap filename is ``<prefix>_<id>[_main_<n>|_hq].jpg`` -- word chars + a
# single ``.jpg``. No path separators, no ``..``, nothing but a leaf JPEG.
_SAFE_IMAGE = re.compile(r"^[A-Za-z0-9_]+\.jpg$")

# A brand slug. Lowercase alphanumeric, optional single hyphens. Used by
# /brand/{slug}.svg; the regex stops a slug carrying anything resembling a
# path traversal or scheme component before it ever hits the filesystem.
_BRAND_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_BRAND_MAX_SLUG = 32
# Directory of bundled SVG brand marks. Ships with the package; slugs are
# typically the Simple Icons name (CC0) or a custom hand-drawn match for
# brands SI doesn't cover (e.g. Mercedes-Benz, Land Rover).
_BRANDS_DIR = Path(__file__).resolve().parent / "static" / "brands"


@dataclass
class _State:
    """Mutable, in-memory showcase state.

    Held behind a single app key so ``/api/refresh`` can rebuild it *in place*
    -- reassigning ``app[...]`` after the app has started is deprecated.
    """

    output_root: Path
    m_per_px: float | None = None
    cars: list[ShowcaseCar] = field(default_factory=list)
    cars_by_plate: dict[str, ShowcaseCar] = field(default_factory=dict)
    sessions: set[str] = field(default_factory=set)
    stats: Stats | None = None

    def reaggregate(self) -> None:
        self.cars = build_showcase(self.output_root)
        self.cars_by_plate = {c.plate: c for c in self.cars}
        self.sessions = {d.name for d in discover_sessions(self.output_root)}
        self.stats = build_stats(self.output_root, m_per_px=self.m_per_px)

    @property
    def n_regulars(self) -> int:
        return sum(1 for c in self.cars if c.kind == "different-day")


# Typed app keys (aiohttp's recommended alternative to bare-string keys).
STATE: web.AppKey[_State] = web.AppKey("state", _State)
META_STORE: web.AppKey[MetadataStore] = web.AppKey("meta_store", MetadataStore)
JINJA: web.AppKey[jinja2.Environment] = web.AppKey("jinja", jinja2.Environment)
BRAND_SVGS: web.AppKey[dict[str, str]] = web.AppKey("brand_svgs", dict)


def _load_brand_svgs() -> dict[str, str]:
    """Return ``{slug: svg_markup}`` for every bundled brand SVG.

    Read once at app startup and inlined into the stats page so the brand
    marks are part of the HTML response itself. No image-load events, no
    separate HTTP request, no browser cache or extension between the user
    and the logo -- the SVG markup ships with the page. Source files live
    under :data:`_BRANDS_DIR` and are version-controlled."""
    out: dict[str, str] = {}
    if _BRANDS_DIR.is_dir():
        for p in sorted(_BRANDS_DIR.glob("*.svg")):
            with contextlib.suppress(OSError):
                out[p.stem] = p.read_text(encoding="utf-8")
    return out


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
    # ``no-store`` keeps browsers from serving a stale HTML page across server
    # restarts -- a real wart after a code change, since the embedded JS (e.g.
    # MAKE_SLUG, makeIcon URLs) would otherwise come from a prior version.
    # The static SVGs / images served by other routes keep their own cache
    # headers; this only suppresses HTML caching.
    return web.Response(
        text=tmpl.render(**ctx),
        content_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


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


async def _stats_page(request: web.Request) -> web.Response:
    stats = request.app[STATE].stats
    return _render(
        request,
        "stats.html",
        stats=stats.to_json_dict() if stats else {},
        brand_svgs=request.app[BRAND_SVGS],
    )


async def _api_stats(request: web.Request) -> web.Response:
    stats = request.app[STATE].stats
    return web.json_response(stats.to_json_dict() if stats else {})


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


async def _brand_svg(request: web.Request) -> web.StreamResponse:
    """Serve a make's brand mark from the bundled SVG set.

    The whole set lives under :data:`_BRANDS_DIR` and ships with the package
    (Simple Icons SVGs for the brands SI covers, hand-drawn monochrome SVGs
    for the gaps -- Mercedes-Benz, Land Rover, Alfa Romeo, Lexus, Jaguar,
    Cupra). Brands with no bundled file 404; the template's monogram
    fallback handles those.

    No network, no caching layer: just static-file serving. The slug regex +
    length cap + resolve-and-compare are belt-and-braces against any
    path-traversal attempt via a hostile slug.
    """
    slug = request.match_info["slug"]
    if len(slug) > _BRAND_MAX_SLUG or not _BRAND_SLUG.match(slug):
        raise web.HTTPNotFound()
    path = (_BRANDS_DIR / f"{slug}.svg").resolve()
    if path.parent != _BRANDS_DIR or not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(
        path,
        headers={
            "Content-Type": "image/svg+xml",
            # Bundled SVGs are write-once per release; long cache is fine.
            "Cache-Control": "public, max-age=604800",
        },
    )


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------


def build_app(
    output_root: Path,
    *,
    metadata_path: Path | None = None,
    m_per_px: float | None = None,
) -> web.Application:
    """Build the showcase :class:`aiohttp.web.Application`.

    Aggregates the cars + stats eagerly so the first request is fast and
    ``main()`` can report counts. Factored out so tests can drive it with
    ``aiohttp.test_utils`` without a real socket. ``m_per_px`` calibrates
    speed to mph (``None`` -> px/s).
    """
    state = _State(output_root=output_root, m_per_px=m_per_px)
    state.reaggregate()

    app = web.Application()
    app[STATE] = state
    app[META_STORE] = MetadataStore(
        metadata_path if metadata_path is not None else output_root / DEFAULT_FILENAME
    )
    app[JINJA] = _make_jinja()
    app[BRAND_SVGS] = _load_brand_svgs()

    app.router.add_get("/", _gallery)
    app.router.add_get("/stats", _stats_page)
    app.router.add_get("/car/{plate}", _car_page)
    app.router.add_get("/api/cars", _api_cars)
    app.router.add_get("/api/cars/{plate}", _api_car)
    app.router.add_put("/api/cars/{plate}/metadata", _api_set_metadata)
    app.router.add_get("/api/stats", _api_stats)
    app.router.add_post("/api/refresh", _api_refresh)
    app.router.add_get("/images/{session}/{filename}", _serve_image)
    app.router.add_get("/brand/{slug}.svg", _brand_svg)
    return app


def load_m_per_px(
    cli_m_per_px: float | None = None,
    cli_road_length_m: float | None = None,
    *,
    config_path: Path = Path("configs/showcase.json"),
) -> float | None:
    """Resolve the speed calibration (metres-per-pixel) for mph display.

    Precedence: ``--m-per-px`` > ``--road-length-m`` > ``configs/showcase.json``
    (``m_per_px`` or ``road_length_m``). ``None`` everywhere -> speeds stay px/s.
    """
    if cli_m_per_px is not None:
        return cli_m_per_px
    if cli_road_length_m is not None:
        return m_per_px_from_road_length(cli_road_length_m)
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(cfg, dict):
            if cfg.get("m_per_px") is not None:
                return float(cfg["m_per_px"])
            if cfg.get("road_length_m") is not None:
                return m_per_px_from_road_length(float(cfg["road_length_m"]))
    return None


async def _serve(
    output_root: Path,
    host: str,
    port: int,
    metadata_path: Path | None,
    m_per_px: float | None,
) -> None:
    app = build_app(output_root, metadata_path=metadata_path, m_per_px=m_per_px)
    state = app[STATE]
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    speed_mode = "mph" if m_per_px is not None else "px/s (uncalibrated)"
    print(
        f"[showcase] {len(state.cars)} identified cars ({state.n_regulars} regulars) "
        f"from {len(state.sessions)} sessions; speed in {speed_mode}"
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
    ap.add_argument(
        "--m-per-px",
        type=float,
        default=None,
        help=(
            "Speed calibration: metres per inference-frame pixel, so the stats "
            "page can show mph. Overrides configs/showcase.json. Unset -> px/s."
        ),
    )
    ap.add_argument(
        "--road-length-m",
        type=float,
        default=None,
        help=(
            "Convenience calibration: real length (m) of the visible road; "
            f"m_per_px = road_length_m / {int(DEFAULT_ROAD_AXIS_PX)} (the traced "
            "road's travel-axis pixel length). Ignored if --m-per-px is given."
        ),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.output_root.is_dir():
        print(f"[showcase] not a directory: {args.output_root}", file=sys.stderr)
        return 2
    m_per_px = load_m_per_px(args.m_per_px, args.road_length_m)
    try:
        asyncio.run(_serve(args.output_root, args.host, args.port, args.metadata, m_per_px))
    except KeyboardInterrupt:
        print("\n[showcase] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
