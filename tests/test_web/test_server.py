"""HTTP surface of the showcase server, driven in-process with aiohttp's
``test_utils`` (no real socket; matches asyncio_mode=auto in pyproject).

Covers: the JSON API shape, metadata PUT persistence + validation, image
serving + its security guards (unknown session, non-jpg / data-file exposure),
and that the HTML pages render.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from streettracker.common.schema import TrackRecord
from streettracker.web.aggregate import ShowcaseCar
from streettracker.web.server import _apply_make_override, _gather_meta, build_app

SESSION = "session_20260526_120000"
_FAKE_JPEG = b"\xff\xd8\xff\xe0FAKEJPEG\xff\xd9"


def _track(tid: int) -> TrackRecord:
    start = "2026-05-26T13:00:00+01:00"
    return TrackRecord(
        track_id=tid,
        class_id=2,
        class_name="car",
        time_start=start,
        time_end=start,
        time_start_unix=1_700_000_000.0,
        time_end_unix=1_700_000_006.0,
        time_start_s=0.0,
        time_end_s=6.0,
        duration_visible=6.0,
        direction="left to right",
        speed_px_s=100.0,
        color="white",
        lane="middle",
        avg_confidence=0.9,
        displacement_px=500.0,
        net_displacement_px=480.0,
        num_detections=30,
        asset_prefix="vehicle",
        main_snaps=[1],
    )


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    root = tmp_path / "output"
    d = root / SESSION
    d.mkdir(parents=True)
    (d / f"{SESSION}_data.json").write_text(json.dumps([asdict(_track(1))]))
    (d / f"{SESSION}_alpr_by_track.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "track_id": 1,
                        "best_preferred": {
                            "track_id": 1,
                            "snap_index": 1,
                            "image": "vehicle_1_main_1.jpg",
                            "ocr_text": "AB12CDE",
                            "ocr_conf": 0.97,
                            "det_conf": 0.85,
                        },
                    }
                ]
            }
        )
    )
    (d / f"{SESSION}_dvsa_labels.json").write_text(
        json.dumps(
            {
                "labels": {
                    "AB12CDE": {
                        "make": "FORD",
                        "model": "FOCUS",
                        "year": 2018,
                        "primary_colour": "Blue",
                        "fuel_type": "Petrol",
                        "engine_size_cc": 1499,
                    }
                }
            }
        )
    )
    for fn in ("vehicle_1.jpg", "vehicle_1_hq.jpg", "vehicle_1_main_1.jpg"):
        (d / fn).write_bytes(_FAKE_JPEG)
    return root


@pytest.fixture
async def client(output_root: Path) -> AsyncIterator[TestClient]:
    app = build_app(output_root)
    async with TestClient(TestServer(app)) as c:
        yield c


# ----------------------------------------------------------------------
# JSON API


async def test_api_cars_lists_identified_car(client: TestClient) -> None:
    r = await client.get("/api/cars")
    assert r.status == 200
    cars = await r.json()
    assert len(cars) == 1
    c = cars[0]
    assert c["plate"] == "AB12CDE"
    assert c["make"] == "FORD"
    assert c["tagged"] is False
    assert c["meta"]["owner"] == ""


async def test_api_car_detail_and_404(client: TestClient) -> None:
    assert (await client.get("/api/cars/AB12CDE")).status == 200
    assert (await client.get("/api/cars/NOPLATE")).status == 404


async def test_put_metadata_persists(client: TestClient, output_root: Path) -> None:
    r = await client.put(
        "/api/cars/AB12CDE/metadata",
        json={"owner": "Shaun", "name": "My car", "favourite": True},
    )
    assert r.status == 200
    body = await r.json()
    assert body["meta"]["owner"] == "Shaun"
    assert body["tagged"] is True

    # Reflected on the next read...
    again = await (await client.get("/api/cars/AB12CDE")).json()
    assert again["meta"]["owner"] == "Shaun"
    assert again["tagged"] is True

    # ...and durably on disk.
    store = json.loads((output_root / "showcase_metadata.json").read_text())
    assert store["AB12CDE"]["owner"] == "Shaun"
    assert store["AB12CDE"]["favourite"] is True


async def test_put_make_override_wins_over_dvsa(client: TestClient) -> None:
    # AB12CDE has DVSA make FORD; an operator override must win on the card.
    r = await client.put("/api/cars/AB12CDE/metadata", json={"make_override": "DAF"})
    assert r.status == 200
    car = await (await client.get("/api/cars/AB12CDE")).json()
    assert car["make"] == "DAF"
    assert car["make_model_source"] == "manual"
    assert car["model"] is None and car["year"] is None
    assert car["meta"]["make_override"] == "DAF"
    assert car["tagged"] is True


async def test_put_make_hidden_suppresses_make(client: TestClient) -> None:
    r = await client.put("/api/cars/AB12CDE/metadata", json={"make_hidden": True})
    assert r.status == 200
    car = await (await client.get("/api/cars/AB12CDE")).json()
    assert car["make"] is None
    assert car["make_model_source"] == "hidden"
    assert car["meta"]["make_hidden"] is True


def test_apply_make_override_noop_keeps_make() -> None:
    d = {"make": "AUDI", "model": None, "year": None, "make_model_source": "dvsa"}
    _apply_make_override(d, {})
    assert d["make"] == "AUDI" and d["make_model_source"] == "dvsa"


def test_apply_make_override_hide_wins_over_override() -> None:
    d = {"make": "PORSCHE", "model": "911", "year": 2017, "make_model_source": "cnn"}
    _apply_make_override(d, {"make_hidden": True, "make_override": "DAF"})
    assert d["make"] is None and d["make_model_source"] == "hidden"


def test_gather_meta_follows_ocr_variant() -> None:
    # A tag set on a misread variant (SK69NJZ) surfaces on the canonical card
    # (SK59NJZ), while the canonical's own field (favourite) still wins.
    car = ShowcaseCar(
        plate="SK59NJZ",
        make=None,
        model=None,
        year=None,
        make_model_source=None,
        primary_colour=None,
        fuel_type=None,
        engine_size_cc=None,
        n_visits=1,
        n_sessions=1,
        n_dates=1,
        kind="one-off",
        first_seen="",
        last_seen="",
        plate_variants=["SK69NJZ"],
    )
    store = {
        "SK59NJZ": {"favourite": True},
        "SK69NJZ": {"name": "Our car", "owner": "Louise"},
    }
    raw = _gather_meta(car, store)
    assert raw["owner"] == "Louise"
    assert raw["name"] == "Our car"
    assert raw["favourite"] is True


async def test_put_metadata_unknown_plate_404(client: TestClient) -> None:
    r = await client.put("/api/cars/NOPLATE/metadata", json={"owner": "x"})
    assert r.status == 404


async def test_put_metadata_rejects_bad_json(client: TestClient) -> None:
    r = await client.put(
        "/api/cars/AB12CDE/metadata", data="{not json", headers={"Content-Type": "application/json"}
    )
    assert r.status == 400


async def test_put_metadata_rejects_non_object(client: TestClient) -> None:
    r = await client.put("/api/cars/AB12CDE/metadata", json=[1, 2, 3])
    assert r.status == 400


# ----------------------------------------------------------------------
# Image serving + guards


async def test_image_served(client: TestClient) -> None:
    r = await client.get(f"/images/{SESSION}/vehicle_1_hq.jpg")
    assert r.status == 200
    assert (await r.read()).startswith(b"\xff\xd8\xff")


async def test_image_unknown_session_404(client: TestClient) -> None:
    r = await client.get("/images/session_does_not_exist/vehicle_1_hq.jpg")
    assert r.status == 404


async def test_image_non_jpg_not_exposed(client: TestClient) -> None:
    # The data.json IS in the served dir; the .jpg-only regex must block it.
    r = await client.get(f"/images/{SESSION}/{SESSION}_data.json")
    assert r.status == 404


async def test_image_missing_jpg_404(client: TestClient) -> None:
    r = await client.get(f"/images/{SESSION}/vehicle_999.jpg")
    assert r.status == 404


# ----------------------------------------------------------------------
# HTML pages + refresh


async def test_gallery_renders(client: TestClient) -> None:
    r = await client.get("/")
    assert r.status == 200
    html = await r.text()
    assert "AB12CDE" in html
    assert "identified cars" in html


async def test_car_page_renders_and_404(client: TestClient) -> None:
    r = await client.get("/car/AB12CDE")
    assert r.status == 200
    assert "FORD" in await r.text()
    assert (await client.get("/car/NOPLATE")).status == 404


async def test_schedule_page_renders(client: TestClient) -> None:
    r = await client.get("/schedule")
    assert r.status == 200
    html = await r.text()
    # The chart script always renders; the timetable panel only when there are
    # enough recurring waves (the tiny test fixture shows the empty state).
    assert "renderScheduleChart" in html
    assert "class / session schedule" in html


async def test_refresh(client: TestClient) -> None:
    r = await client.post("/api/refresh")
    assert r.status == 200
    assert (await r.json())["cars"] == 1


# ----------------------------------------------------------------------
# Bundled brand SVGs (/brand/{slug}.svg)


async def test_brand_svg_ford_bundled(client: TestClient) -> None:
    # Bundled with the package -- ships in src/streettracker/web/static/brands/.
    r = await client.get("/brand/ford.svg")
    assert r.status == 200
    assert r.content_type == "image/svg+xml"
    body = await r.read()
    assert body.lstrip().startswith(b"<svg")


async def test_brand_svg_mercedes_bundled(client: TestClient) -> None:
    # Hand-drawn fill-in for a brand Simple Icons doesn't cover. Same route.
    r = await client.get("/brand/mercedes.svg")
    assert r.status == 200
    assert r.content_type == "image/svg+xml"


async def test_brand_svg_unknown_make_404(client: TestClient) -> None:
    # Valid-shape slug, but no bundled SVG -- the template's monogram fallback
    # handles this; the route just 404s without surprises.
    r = await client.get("/brand/madeupcarbrand.svg")
    assert r.status == 404


async def test_brand_svg_invalid_slug_404(client: TestClient) -> None:
    # Bad shapes: rejected by the slug regex before any filesystem lookup.
    for bad in (
        "Ford.svg",  # uppercase letter
        "ford..svg",  # double dot
        "_ford.svg",  # leading underscore
        "ford .svg",  # space (URL-encoded)
        "a" * 33 + ".svg",  # too long (> _BRAND_MAX_SLUG)
    ):
        r = await client.get(f"/brand/{bad}")
        assert r.status == 404, f"{bad!r} should not be served"


async def test_brand_svg_path_traversal_rejected(client: TestClient) -> None:
    # Even if a hostile path-traversal segment somehow reached the handler,
    # the slug regex rejects anything with a slash or dot.
    r = await client.get("/brand/..%2Fetc.svg")
    assert r.status == 404
