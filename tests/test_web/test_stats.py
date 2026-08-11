"""build_stats() + the /stats HTTP surface, on synthetic sessions.

Covers bucketing (by camera-local time_start, incl. a session spanning two
dates), direction counts, day-of-week, 15-min buckets, the weekday×hour heatmap,
fastest-track selection (ordering, glitch filter, plate resolution, image URLs),
speed averages, the px/s↔mph calibration, and make/colour distributions.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from streettracker.common.schema import TrackRecord
from streettracker.web.server import build_app
from streettracker.web.stats import _DOW, MPH_PER_M_S, build_stats

_FAKE_JPEG = b"\xff\xd8\xff\xe0X\xff\xd9"


def _track(
    tid: int,
    *,
    date: str = "2026-05-26",
    hhmmss: str = "09:00:00",
    direction: str = "left to right",
    speed: float = 100.0,
    color: str = "white",
    ndet: int = 20,
    cls: str = "car",
) -> TrackRecord:
    start = f"{date}T{hhmmss}+01:00"
    unix = datetime.fromisoformat(start).timestamp()
    return TrackRecord(
        track_id=tid,
        class_id=2 if cls == "car" else 0,
        class_name=cls,
        time_start=start,
        time_end=start,
        time_start_unix=unix,
        time_end_unix=unix + 5,
        time_start_s=0.0,
        time_end_s=5.0,
        duration_visible=5.0,
        direction=direction,
        speed_px_s=speed,
        color=color,
        lane="middle",
        avg_confidence=0.9,
        displacement_px=speed * 5,
        net_displacement_px=speed * 5,
        num_detections=ndet,
        asset_prefix="vehicle" if cls == "car" else "person",
        main_snaps=[1],
    )


def _mk_session(
    root: Path,
    name: str,
    tracks: list[TrackRecord],
    *,
    dvsa: dict | None = None,
    alpr: dict | None = None,
    images: bool = True,
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / f"{name}_data.json").write_text(json.dumps([asdict(t) for t in tracks]))
    if dvsa is not None:
        (d / f"{name}_dvsa_labels.json").write_text(json.dumps(dvsa))
    if alpr is not None:
        (d / f"{name}_alpr_by_track.json").write_text(json.dumps(alpr))
    if images:
        for t in tracks:
            (d / f"vehicle_{t.track_id}_main_1.jpg").write_bytes(_FAKE_JPEG)
            (d / f"vehicle_{t.track_id}_hq.jpg").write_bytes(_FAKE_JPEG)
            (d / f"vehicle_{t.track_id}.jpg").write_bytes(_FAKE_JPEG)
    return d


def _alpr(*reads: tuple[int, str, float]) -> dict:
    return {
        "tracks": [
            {
                "track_id": tid,
                "best_preferred": {
                    "track_id": tid,
                    "snap_index": 1,
                    "image": f"vehicle_{tid}_main_1.jpg",
                    "ocr_text": plate,
                    "ocr_conf": conf,
                    "det_conf": 0.85,
                },
            }
            for tid, plate, conf in reads
        ]
    }


# ----------------------------------------------------------------------
# Bucketing


def test_daily_direction_counts(tmp_path: Path) -> None:
    _mk_session(
        tmp_path,
        "session_20260526_090000",
        [
            _track(1, date="2026-05-26", direction="left to right"),
            _track(2, date="2026-05-26", direction="right to left"),
            _track(3, date="2026-05-26", direction="left to right"),
        ],
    )
    s = build_stats(tmp_path)
    assert s.overall["total_journeys"] == 3
    day = next(x for x in s.daily if x["date"] == "2026-05-26")
    assert (day["l2r"], day["r2l"], day["total"]) == (2, 1, 3)
    assert s.overall["pct_l2r"] == 67 and s.overall["pct_r2l"] == 33


def test_session_spanning_two_dates_splits(tmp_path: Path) -> None:
    # One session dir whose tracks fall on two calendar dates by time_start.
    _mk_session(
        tmp_path,
        "session_20260601_173815",
        [
            _track(1, date="2026-06-01", hhmmss="23:50:00"),
            _track(2, date="2026-06-02", hhmmss="00:10:00"),
        ],
    )
    s = build_stats(tmp_path)
    dates = {x["date"] for x in s.daily}
    assert dates == {"2026-06-01", "2026-06-02"}


def test_dow_and_heatmap(tmp_path: Path) -> None:
    date = "2026-05-26"
    wd = datetime.fromisoformat(f"{date}T12:00:00+01:00").weekday()
    _mk_session(
        tmp_path,
        "session_20260526_120000",
        [
            _track(1, date=date, hhmmss="12:05:00"),
            _track(2, date=date, hhmmss="12:40:00"),
        ],
    )
    s = build_stats(tmp_path)
    assert s.dow[_DOW[wd]]["total"] == 2
    assert s.heatmap[wd][12] == 2


def test_15min_buckets(tmp_path: Path) -> None:
    _mk_session(
        tmp_path,
        "session_20260526_090000",
        [
            _track(1, hhmmss="09:07:00", direction="left to right"),  # bucket 36
            _track(2, hhmmss="09:22:00", direction="right to left"),  # bucket 37
        ],
    )
    s = build_stats(tmp_path)
    buckets = s.by_day_15min["2026-05-26"]
    assert buckets[36]["l2r"] == 1 and buckets[36]["r2l"] == 0
    assert buckets[37]["r2l"] == 1


def test_persons_excluded(tmp_path: Path) -> None:
    _mk_session(
        tmp_path,
        "session_20260526_090000",
        [
            _track(1, cls="car"),
            _track(2, cls="person"),
        ],
    )
    assert build_stats(tmp_path).overall["total_journeys"] == 1


# ----------------------------------------------------------------------
# Speed + fastest


def test_speed_avg_by_direction_pxs(tmp_path: Path) -> None:
    _mk_session(
        tmp_path,
        "session_20260526_090000",
        [
            _track(1, direction="left to right", speed=100.0),
            _track(2, direction="left to right", speed=200.0),
            _track(3, direction="right to left", speed=60.0),
        ],
    )
    s = build_stats(tmp_path)
    assert s.speed_unit == "px/s"
    assert s.speed["avg_l2r"] == 150  # (100+200)/2 rounded
    assert s.speed["avg_r2l"] == 60


def test_mph_calibration(tmp_path: Path) -> None:
    _mk_session(
        tmp_path,
        "session_20260526_090000",
        [
            _track(1, direction="left to right", speed=100.0),
        ],
    )
    s = build_stats(tmp_path, m_per_px=0.05)
    assert s.speed_unit == "mph"
    expected = round(100.0 * 0.05 * MPH_PER_M_S, 1)  # ~11.2
    assert s.speed["avg_l2r"] == expected
    assert s.speed["fastest"][0]["speed"] == expected
    assert s.speed["fastest"][0]["unit"] == "mph"


def test_fastest_ordering_glitch_filter_and_plate(tmp_path: Path) -> None:
    _mk_session(
        tmp_path,
        "session_20260526_090000",
        [
            _track(1, speed=120.0, ndet=20),  # plated, solid
            _track(2, speed=999.0, ndet=3),  # glitch: too few detections -> excluded
            _track(3, speed=80.0, ndet=20),
        ],
        alpr=_alpr((1, "AB12CDE", 0.97)),
    )
    s = build_stats(tmp_path)
    fast = s.speed["fastest"]
    assert [f["track_id"] for f in fast] == [1, 3]  # glitch 2 excluded, sorted desc
    assert fast[0]["plate"] == "AB12CDE"  # canonical plate resolved
    assert fast[0]["thumb"] and fast[0]["full"]  # existence-checked image urls
    assert fast[1]["plate"] is None  # track 3 had no read


def test_fastest_non_canonical_plate_not_linked(tmp_path: Path) -> None:
    _mk_session(
        tmp_path,
        "session_20260526_090000",
        [_track(1, speed=120.0)],
        alpr=_alpr((1, "GARBAGE9", 0.99)),
    )
    assert build_stats(tmp_path).speed["fastest"][0]["plate"] is None


# ----------------------------------------------------------------------
# Make / colour


def test_make_dedupe_across_sessions(tmp_path: Path) -> None:
    dv = {"labels": {"AB12CDE": {"make": "FORD"}, "LA68EWY": {"make": "FIAT"}}}
    _mk_session(tmp_path, "session_20260526_090000", [_track(1)], dvsa=dv)
    # Same FORD plate again in a later session -> counted once; add a VW.
    _mk_session(
        tmp_path,
        "session_20260527_090000",
        [_track(2, date="2026-05-27")],
        dvsa={"labels": {"AB12CDE": {"make": "FORD"}, "GL74JYW": {"make": "VOLKSWAGEN"}}},
    )
    makes = dict(build_stats(tmp_path).makes)
    assert makes == {"FORD": 1, "FIAT": 1, "VOLKSWAGEN": 1}


def test_colour_distribution(tmp_path: Path) -> None:
    _mk_session(
        tmp_path,
        "session_20260526_090000",
        [
            _track(1, color="white"),
            _track(2, color="white"),
            _track(3, color="black"),
        ],
    )
    colours = dict(build_stats(tmp_path).colours)
    assert colours == {"white": 2, "black": 1}


def test_colour_mix_dvsa_preferred_cnn_then_hsv(tmp_path: Path) -> None:
    """Colour mix precedence: DVSA register colour (plated) > colour CNN
    sidecar > the low-res HSV `color` field for tracks nothing else covers."""
    d = _mk_session(
        tmp_path,
        "session_col",
        [
            _track(1, color="black"),  # HSV black, but DVSA says Blue -> blue
            _track(2, color="black"),  # HSV black, but CNN says silver -> silver
            _track(3, color="green"),  # no DVSA, no CNN -> HSV green
        ],
        dvsa={"labels": {"AB12CDE": {"primary_colour": "Blue", "track_ids": [1]}}},
    )
    (d / "session_col_colour_by_track.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {"track_id": 1, "colour": "grey", "conf": 0.9},  # ignored: DVSA wins
                    {"track_id": 2, "colour": "silver", "conf": 0.9},  # used: no DVSA
                ]
            }
        )
    )
    colours = dict(build_stats(tmp_path).colours)
    assert colours == {"blue": 1, "silver": 1, "green": 1}


def test_body_type_mix_dvsa_preferred_cnn_fallback(tmp_path: Path) -> None:
    """The body-type mix uses the DVSA-model-derived class where a plate is
    known (reliable) and the CNN sidecar only for tracks DVSA didn't cover."""
    d = _mk_session(
        tmp_path,
        "session_bt",
        [_track(1), _track(2), _track(3)],
        dvsa={
            "labels": {
                # track 1: DVSA FOCUS -> hatchback (wins over any CNN guess).
                "AB12CDE": {"make": "FORD", "model": "FOCUS", "track_ids": [1]},
            }
        },
    )
    (d / "session_bt_bodytype_by_track.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {"track_id": 1, "body_type": "van", "conf": 0.9},  # ignored: DVSA wins
                    {"track_id": 2, "body_type": "suv", "conf": 0.9},  # used: no DVSA
                    {"track_id": 3, "body_type": None, "conf": 0.2},  # unconfident: skipped
                ]
            }
        )
    )
    mix = dict(build_stats(tmp_path).bodytypes)
    assert mix == {"hatchback": 1, "suv": 1}  # track 3 has no usable body type


def test_empty_root(tmp_path: Path) -> None:
    s = build_stats(tmp_path)
    assert s.overall["total_journeys"] == 0
    assert s.daily == [] and s.makes == [] and s.speed["fastest"] == []
    assert s.bodytypes == []


# ----------------------------------------------------------------------
# HTTP surface


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    root = tmp_path / "output"
    root.mkdir()
    _mk_session(
        root,
        "session_20260526_090000",
        [
            _track(1, direction="left to right", speed=150.0),
            _track(2, direction="right to left", speed=90.0),
        ],
        dvsa={"labels": {"AB12CDE": {"make": "FORD"}}},
        alpr=_alpr((1, "AB12CDE", 0.97)),
    )
    return root


@pytest.fixture
async def client(output_root: Path) -> AsyncIterator[TestClient]:
    async with TestClient(TestServer(build_app(output_root))) as c:
        yield c


async def test_stats_page_renders(client: TestClient) -> None:
    r = await client.get("/stats")
    assert r.status == 200
    html = await r.text()
    assert "vehicle journeys" in html
    assert "const STATS" in html  # embedded data for the charts


async def test_api_stats_shape(client: TestClient) -> None:
    r = await client.get("/api/stats")
    assert r.status == 200
    data = await r.json()
    assert data["overall"]["total_journeys"] == 2
    assert {"daily", "dow", "by_day_15min", "heatmap", "speed", "makes", "colours"} <= data.keys()
    assert data["speed_unit"] == "px/s"


async def test_api_stats_mph_when_calibrated(output_root: Path) -> None:
    app = build_app(output_root, m_per_px=0.05)
    async with TestClient(TestServer(app)) as c:
        data = await (await c.get("/api/stats")).json()
    assert data["speed_unit"] == "mph"


async def test_gallery_has_stats_nav(client: TestClient) -> None:
    html = await (await client.get("/")).text()
    assert 'href="/stats"' in html
    assert 'href="/people"' in html


async def test_people_page_renders(client: TestClient) -> None:
    r = await client.get("/people")
    assert r.status == 200
    html = await r.text()
    assert "const STATS" in html  # embedded data for the people charts
    assert "renderPeople" in html


async def test_people_moved_off_stats_page(client: TestClient) -> None:
    """The People charts now live on /people; /stats should link out to them
    rather than carry the block itself."""
    html = await (await client.get("/stats")).text()
    assert "renderPeople" not in html
    assert 'href="/people"' in html


# ----------------------------------------------------------------------
# People (person-track aggregates).


def test_people_block_aggregates_person_tracks(tmp_path: Path) -> None:
    """Person tracks feed the people block (counts, direction split,
    heatmap, dwell) and stay OUT of the vehicle journey totals."""
    from dataclasses import replace

    tracks = [
        # 2026-05-26 is a Tuesday.
        replace(_track(1, hhmmss="09:05:00", cls="person"), duration_visible=5.0),
        replace(
            _track(2, hhmmss="09:20:00", direction="right to left", cls="person"),
            duration_visible=12.0,
        ),
        replace(
            _track(3, hhmmss="18:00:00", direction="right to left", cls="person"),
            duration_visible=70.0,
        ),
        _track(4, hhmmss="09:00:00", cls="car"),
    ]
    _mk_session(tmp_path, "session_p", tracks)

    s = build_stats(tmp_path)

    assert s.overall["total_journeys"] == 1  # people excluded from car totals
    p = s.people
    assert p["total"] == 3
    assert p["n_days"] == 1
    assert p["per_day_mean"] == 3.0
    assert p["pct_l2r"] == 33
    assert p["pct_r2l"] == 67
    assert p["busiest_hour"] == {"hour": 9, "total": 2}
    assert p["dow"]["Tue"] == {"l2r": 1, "r2l": 2, "total": 3}
    assert p["heatmap"][1][9] == 2  # Tue 09:00 bucket
    assert p["heatmap"][1][18] == 1
    assert p["dwell"]["median_s"] == 12.0
    assert p["dwell"]["p90_s"] == 70.0
    hist = p["dwell"]["hist"]
    assert hist[0]["n"] == 1  # 5s  -> [0, 10)
    assert hist[1]["n"] == 1  # 12s -> [10, 20)
    assert hist[-1]["lo"] == 60.0  # open-ended 60+ bucket
    assert hist[-1]["hi"] is None
    assert hist[-1]["n"] == 1  # 70s


def test_class_suspect_tracks_excluded_from_people_and_cars(tmp_path: Path) -> None:
    """The runtime's kinematics guardrail (car-shaped "person" bboxes)
    keeps flagged tracks out of both the people block and car totals."""
    from dataclasses import replace

    tracks = [
        replace(_track(1, cls="person"), class_suspect=True),  # flagged
        _track(2, cls="person"),
        _track(3, cls="car"),
    ]
    _mk_session(tmp_path, "session_s", tracks)

    s = build_stats(tmp_path)

    assert s.people["total"] == 1  # suspect excluded
    assert s.overall["total_journeys"] == 1  # and not re-counted as a car


def test_people_block_empty_without_person_tracks(tmp_path: Path) -> None:
    _mk_session(tmp_path, "session_c", [_track(1, cls="car")])

    s = build_stats(tmp_path)

    assert s.people["total"] == 0
    assert s.people["dwell"]["hist"] == []
    assert s.people["busiest_hour"] is None


def _people_json(walkers: int, joggers: int, cyclists: int, dog_walkers: int) -> dict:
    n = walkers + joggers + cyclists
    return {
        "session": "x",
        "params": {"m_per_px": 0.05, "jogger_min_m_s": 2.5, "min_overlap_s": 3.0},
        "summary": {
            "n_person_tracks": n,
            "walkers": walkers,
            "joggers": joggers,
            "cyclists": cyclists,
            "dog_walkers": dog_walkers,
            "n_dog_tracks": dog_walkers,
            "n_dogs_paired": dog_walkers,
            "n_bicycle_tracks": cyclists,
            "n_bicycles_paired": cyclists,
        },
        "people": [],
    }


def test_people_kinds_roll_up_across_sessions(tmp_path: Path) -> None:
    """_people.json summaries sum into people["kinds"] with percentages;
    sessions without the sidecar contribute nothing (not an error)."""
    d1 = _mk_session(tmp_path, "session_a", [_track(1, cls="person")])
    (d1 / "session_a_people.json").write_text(json.dumps(_people_json(6, 3, 1, 2)))
    d2 = _mk_session(tmp_path, "session_b", [_track(2, cls="person")])
    (d2 / "session_b_people.json").write_text(json.dumps(_people_json(4, 1, 0, 0)))
    _mk_session(tmp_path, "session_c", [_track(3, cls="person")])  # no sidecar

    k = build_stats(tmp_path).people["kinds"]

    assert k["classified"] == 15
    assert k["walkers"] == 10 and k["joggers"] == 4 and k["cyclists"] == 1
    assert k["dog_walks"] == 2
    assert k["pct_walkers"] == 67 and k["pct_joggers"] == 27 and k["pct_cyclists"] == 7
    # Sidecars without a "walks" key (pre-dedup) fall back to 1:1 tracks.
    assert k["walks"] == 15


def test_people_kinds_zero_without_sidecars(tmp_path: Path) -> None:
    _mk_session(tmp_path, "session_p", [_track(1, cls="person")])

    k = build_stats(tmp_path).people["kinds"]

    assert k["classified"] == 0
    assert k["pct_walkers"] == 0  # no division by zero


def _person_row(
    tid: int,
    date: str,
    hhmm: str,
    *,
    kind: str = "walker",
    dog: bool = False,
    direction: str = "left to right",
    rt: int | None = None,
    start_unix: float = 0.0,
) -> dict:
    return {
        "track_id": tid,
        "walk_id": tid,
        "time_start": f"{date}T{hhmm}:00+01:00",
        "time_start_unix": start_unix,
        "time_end_unix": start_unix + 20.0,
        "direction": direction,
        "num_detections": 50,
        "kind": kind,
        "dog_walker": dog,
        "round_trip_id": rt,
    }


def _mk_people_session(
    root: Path, name: str, rows: list[dict], *, chance: int | None = None
) -> None:
    d = _mk_session(root, name, [_track(1, cls="person")])
    doc = _people_json(len(rows), 0, 0, 0)
    doc["people"] = rows
    if chance is not None:
        doc["summary"]["round_trips_chance"] = chance
    (d / f"{name}_people.json").write_text(json.dumps(doc))


def test_round_trips_pool_with_away_median(tmp_path: Path) -> None:
    """Paired rows (shared round_trip_id) count once, and the away time
    is the gap between the outbound walk's end and the return's start."""
    _mk_people_session(
        tmp_path,
        "session_rt",
        [
            _person_row(1, "2026-07-01", "08:00", rt=1, start_unix=1000.0),
            _person_row(
                2, "2026-07-01", "08:10", rt=1, direction="right to left", start_unix=1620.0
            ),
        ],
    )
    k = build_stats(tmp_path).people["kinds"]
    assert k["round_trips"] == 1
    assert k["away_min_median"] == 10.0  # (1620 - 1020) / 60
    assert k["round_trips_est"] is None  # no chance control in the sidecar


def test_round_trips_chance_corrected_estimate(tmp_path: Path) -> None:
    """When sidecars carry the time-shift control, the likely-genuine
    count is raw pairs minus the chance floor (clamped at zero)."""
    _mk_people_session(
        tmp_path,
        "session_rt2",
        [
            _person_row(1, "2026-07-01", "08:00", rt=1, start_unix=1000.0),
            _person_row(
                2, "2026-07-01", "08:10", rt=1, direction="right to left", start_unix=1620.0
            ),
        ],
        chance=1,
    )
    k = build_stats(tmp_path).people["kinds"]
    assert k["round_trips"] == 1
    assert k["round_trips_est"] == 0


def test_habitual_dog_walk_slot(tmp_path: Path) -> None:
    """Dog walks recurring near one time across >=3 dates form a slot;
    a one-off at another time does not."""
    _mk_people_session(
        tmp_path,
        "session_h",
        [
            _person_row(1, "2026-07-01", "07:50", dog=True, start_unix=1.0),
            _person_row(2, "2026-07-02", "07:55", dog=True, start_unix=100.0),
            _person_row(3, "2026-07-03", "08:00", dog=True, start_unix=200.0),
            _person_row(4, "2026-07-03", "15:00", dog=True, start_unix=300.0),
        ],
    )
    hab = build_stats(tmp_path).people["habitual"]
    assert len(hab) == 1
    s = hab[0]
    assert s["kind"] == "dog walk" and s["dkey"] == "l2r"
    assert s["time"] == "07:55"  # median of 07:50 / 07:55 / 08:00
    assert s["days_seen"] == 3 and s["days_covered"] == 3
    assert s["n_walks"] == 3


def test_habitual_needs_min_days(tmp_path: Path) -> None:
    _mk_people_session(
        tmp_path,
        "session_h2",
        [
            _person_row(1, "2026-07-01", "07:50", dog=True, start_unix=1.0),
            _person_row(2, "2026-07-02", "07:55", dog=True, start_unix=100.0),
        ],
    )
    assert build_stats(tmp_path).people["habitual"] == []


def test_habitual_rejects_traffic_waves(tmp_path: Path) -> None:
    """A window averaging >1.6 walks/day is a rush-hour wave, not an
    individual habit -- consumed without producing a slot."""
    rows = []
    tid = 0
    for day in ("2026-07-01", "2026-07-02", "2026-07-03"):
        for hhmm in ("08:00", "08:10", "08:20"):  # 3 joggers/day, same window
            tid += 1
            rows.append(_person_row(tid, day, hhmm, kind="jogger", start_unix=float(tid * 1000)))
    _mk_people_session(tmp_path, "session_h4", rows)
    assert build_stats(tmp_path).people["habitual"] == []


def test_habitual_skips_plain_walkers(tmp_path: Path) -> None:
    """Walkers are too dense to mine -- same-time walks on many dates
    still produce no slot (they aren't fed to the miner at all)."""
    rows = [_person_row(i, f"2026-07-{i:02d}", "07:50", start_unix=float(i)) for i in range(1, 6)]
    _mk_people_session(tmp_path, "session_h3", rows)
    assert build_stats(tmp_path).people["habitual"] == []


async def test_api_stats_includes_people(client: TestClient) -> None:
    data = await (await client.get("/api/stats")).json()
    assert "people" in data
    assert {"total", "dow", "heatmap", "dwell", "habitual"} <= data["people"].keys()
