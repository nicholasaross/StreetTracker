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
from streettracker.web.server import build_app

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


async def test_refresh(client: TestClient) -> None:
    r = await client.post("/api/refresh")
    assert r.status == 200
    assert (await r.json())["cars"] == 1
