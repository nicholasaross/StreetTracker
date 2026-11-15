"""Tests for drawing the plate-read audit set (review E1.2)."""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from streettracker.analysis.alpr.audit import (
    SampleSizes,
    SessionPool,
    allocate,
    allocate_by_domain,
    clean_label,
    crop_path,
    draw,
    draw_audit_set,
    light_band,
    read_group,
    session_pool,
    skip_reason,
    solar_elevation_deg,
    support_band,
)
from streettracker.analysis.alpr.gate import PlateGate

GATE = PlateGate(mode="combined", supported_conf=0.8, min_support=2)


def _unix(*when: int) -> float:
    return datetime(*when, tzinfo=timezone.utc).timestamp()


# ----------------------------------------------------------------------
# Light


def test_solar_elevation_at_greenwich_midsummer() -> None:
    # Noon: 90 - latitude + the sun's declination (23.44 at the solstice).
    assert solar_elevation_deg(_unix(2026, 6, 21, 12), 51.48, 0.0) == pytest.approx(61.9, abs=0.3)
    # Midnight: the sun is as far below the northern horizon as it gets.
    assert solar_elevation_deg(_unix(2026, 6, 21, 0), 51.48, 0.0) == pytest.approx(-15.1, abs=0.3)


def test_solar_elevation_follows_longitude_and_season() -> None:
    noon = _unix(2026, 3, 20, 12)
    assert solar_elevation_deg(noon, 0.0, 0.0) > 87.0  # equinox, on the equator
    assert solar_elevation_deg(noon, 0.0, 180.0) < -87.0  # the far side of the earth
    # Midwinter noon in the north of England is barely "day".
    assert 10.0 < solar_elevation_deg(_unix(2026, 12, 21, 12), 55.0, -1.6) < 12.5


def test_light_bands() -> None:
    assert [light_band(e) for e in (40.0, 6.0, 5.9, 0.0, -6.0, -6.1, -30.0)] == [
        "day",
        "day",
        "dusk",
        "dusk",
        "dusk",
        "dark",
        "dark",
    ]


# ----------------------------------------------------------------------
# How a read was reached

NEAR = (100.0, 100.0, 300.0, 250.0)  # nearest the hint
FAR = (600.0, 100.0, 800.0, 250.0)
HINT = (120, 110, 280, 240)


def _path(plate: tuple[int, int, int, int], **kw: object) -> str:
    kwargs = {"hint": HINT, "road_polygon_frac": None, **kw}
    return crop_path(plate, [FAR, NEAR], (1000, 600), **kwargs)[0]  # type: ignore[arg-type]


def test_crop_path_names_the_candidate_that_holds_the_plate() -> None:
    assert crop_path(
        (180, 200, 240, 220), [FAR, NEAR], (1000, 600), hint=HINT, road_polygon_frac=None
    ) == (
        "rank0",
        NEAR,
    )
    assert _path((680, 200, 740, 220)) == "rank1"
    assert _path((400, 500, 460, 520)) == "unmatched"


def test_crop_path_counts_the_detector_padding() -> None:
    # 20 px left of the nearest box: inside its 30 px crop padding.
    assert _path((60, 200, 100, 220)) == "rank0"


def test_crop_path_falls_back_to_the_hint_window() -> None:
    hint = (400, 400, 520, 520)  # the car has left its hint; a plate is still read there
    plate = (440, 450, 500, 470)
    assert crop_path(plate, [FAR], (1000, 600), hint=hint, road_polygon_frac=None)[0] == "hint_crop"


def test_crop_path_takes_the_bottom_retry_from_the_read() -> None:
    path, box = crop_path(
        (680, 200, 740, 220),
        [FAR, NEAR],
        (1000, 600),
        hint=HINT,
        road_polygon_frac=None,
        bottom_retry=True,
    )
    assert (path, box) == ("bottom_retry", FAR)


def test_crop_path_leaves_out_vehicles_off_the_road() -> None:
    # The road is the left half: FAR is off it, so it is no candidate at all.
    road = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
    assert _path((680, 200, 740, 220), road_polygon_frac=road) == "unmatched"
    assert _path((180, 200, 240, 220), road_polygon_frac=road) == "rank0"


def test_read_groups_and_support_bands() -> None:
    assert read_group("rank0", True) == "rank0_pass"
    assert read_group("rank0", False) == "rank0_fail"
    assert read_group("rank1", True) == "rank1"
    assert {read_group(p, True) for p in ("hint_crop", "bottom_retry", "unmatched")} == {"fallback"}
    assert [support_band(n) for n in (0, 1, 2, 4, 5, 40)] == ["1", "1", "2-4", "2-4", "5+", "5+"]


# ----------------------------------------------------------------------
# Allocation and drawing


def test_allocate_gives_small_strata_a_floor_and_never_overdraws() -> None:
    sizes = {"big": 10_000, "mid": 400, "small": 4, "tiny": 2, "empty": 0}
    got = allocate(sizes, 40, power=0.5, floor=3)
    assert sum(got.values()) == 40
    assert got["empty"] == 0 and got["tiny"] == 2 and got["small"] >= 3
    assert got["big"] > got["mid"] > got["small"]
    assert all(got[k] <= sizes[k] for k in sizes)
    # Square-root allocation: far less lopsided than the populations.
    assert got["big"] / got["mid"] < 10


def test_allocate_power_zero_is_equal_and_power_one_proportional() -> None:
    sizes = {"a": 900, "b": 90, "c": 10}
    assert allocate(sizes, 30, power=0.0, floor=0) == {"a": 10, "b": 10, "c": 10}
    assert allocate(sizes, 100, power=1.0, floor=0) == {"a": 90, "b": 9, "c": 1}


def test_allocate_copes_with_too_few_items_and_too_few_reads() -> None:
    assert allocate({"a": 50, "b": 50, "c": 50}, 2, floor=3) in (
        {"a": 1, "b": 1, "c": 0},
        {"a": 1, "b": 0, "c": 1},
        {"a": 0, "b": 1, "c": 1},
    )
    assert allocate({"a": 3, "b": 2}, 100) == {"a": 3, "b": 2}  # all there is
    assert allocate({}, 10) == {}
    assert allocate({"a": 5}, 0) == {"a": 0}


def test_allocate_by_domain_gives_each_domain_a_floor_then_square_root_shares() -> None:
    sizes = {"x|1": 5000, "x|2": 50, "y|1": 40, "y|2": 10, "z|1": 4}
    got = allocate_by_domain(sizes, 60, lambda s: s.split("|")[0], floor=2, domain_floor=10)
    assert sum(got.values()) == 60
    assert got["y|1"] + got["y|2"] == 10  # its floor; the rest goes to the big domain
    assert got["z|1"] == 4  # all a small domain holds
    assert got["x|1"] + got["x|2"] == 46 and got["x|2"] >= 2


def _cands(stratum: str, n: int) -> list[dict]:
    return [{"key": f"{stratum}/{i}", "stratum": stratum} for i in range(n)]


def test_draw_is_repeatable_and_respects_the_allocation() -> None:
    candidates = _cands("a", 30) + _cands("b", 5)
    allocation = {"a": 4, "b": 9}
    first = draw(candidates, allocation, random.Random(7))
    again = draw(list(reversed(candidates)), allocation, random.Random(7))
    assert [c["key"] for c in first] == [c["key"] for c in again]  # input order doesn't matter
    assert Counter(c["stratum"] for c in first) == {"a": 4, "b": 5}
    assert draw(candidates, allocation, random.Random(8)) != first


def test_draw_passes_over_unusable_candidates() -> None:
    candidates = _cands("a", 10)
    got = draw(candidates, {"a": 4}, random.Random(1), usable=lambda c: int(c["key"][-1]) % 2 == 0)
    assert len(got) == 4 and all(int(c["key"][-1]) % 2 == 0 for c in got)


# ----------------------------------------------------------------------
# A session's candidates

LABEL = "session_20260621_100000"
SNAP_SIZE = (1792, 1024)  # twice the sub-stream, so boxes scale by 2


def _read(tid: int, snap: int, text: str, conf: float, bbox: list[int], **extra: object) -> dict:
    return {
        "image": f"vehicle_{tid}_main_{snap}.jpg",
        "track_id": tid,
        "snap_index": snap,
        "pipeline": "preferred",
        "det_bbox": bbox,
        "det_conf": 0.9,
        "ocr_text": text,
        "ocr_conf": conf,
        "canonical_uk_shape": True,
        **extra,
    }


def _best(read: dict, n_agree: int = 0) -> dict:
    keys = ("track_id", "snap_index", "image", "ocr_text", "ocr_conf", "canonical_uk_shape")
    return {**{k: read[k] for k in keys}, "n_agree": n_agree}


def _car(tid: int, direction: str, when: float, snaps: list[int], sub_box: list[int]) -> dict:
    return {
        "track_id": tid,
        "class_name": "car",
        "direction": direction,
        "time_start_unix": when,
        "time_start": datetime.fromtimestamp(when, timezone.utc).isoformat(),
        "main_snaps": snaps,
        "main_snap_bboxes": [sub_box for _ in snaps],
    }


@pytest.fixture
def session(tmp_path: Path) -> Path:
    sd = tmp_path / LABEL
    sd.mkdir()
    noon, night = _unix(2026, 6, 21, 12), _unix(2026, 6, 21, 0)
    car_box = [100, 100, 200, 180]  # sub-stream; 200,200-400,360 in the snap
    records = [
        _car(1, "right to left", noon, [1, 2], car_box),
        _car(2, "left to right", noon, [1], car_box),
        _car(3, "left to right", night, [1, 2], car_box),
        _car(4, "right to left", noon, [1], car_box),
        _car(6, "right to left", noon, [], car_box),  # never snapped
        {**_car(7, "right to left", noon, [1], car_box), "class_suspect": True},
        {**_car(8, "left to right", noon, [1], car_box), "class_name": "person"},
        {**_car(9, "stationary", noon, [1], car_box)},
    ]
    on_car = [280, 300, 340, 320]  # inside the tracked car's box
    on_other = [1280, 300, 1340, 320]  # inside a second vehicle further off
    reads = [
        _read(1, 1, "AB12CDE", 0.97, on_car, plate_colour="white"),
        _read(1, 2, "AB12CDE", 0.90, on_car, plate_colour="white"),
        _read(2, 1, "XY65ZZZ", 0.85, on_other, plate_colour="yellow"),
        # Track 3: the confident read is another car's (white) front plate.
        _read(3, 1, "FR09ONT", 0.99, on_other, plate_colour="white", colour_suspect=True),
        _read(3, 2, "RE11ARX", 0.60, on_car, plate_colour="yellow"),
        # Track 4: its only read is a parked car's plate.
        _read(4, 1, "PA12KED", 0.95, on_other, static_suspect=True),
        _read(7, 1, "SU55ECT", 0.99, on_car),
        {**_read(1, 3, "JUNK", 0.99, on_car), "canonical_uk_shape": False},
    ]
    rollup = {
        "tracks": [
            {"track_id": 1, "best_preferred": _best(reads[0], n_agree=1)},
            {"track_id": 2, "best_preferred": _best(reads[2])},
            {"track_id": 3, "best_preferred": _best(reads[4])},
        ]
    }
    boxes = {r["image"]: [[200, 200, 400, 360, 0.9], [1200, 200, 1400, 360, 0.8]] for r in reads}
    files = {
        "_data.json": records,
        "_meta.json": {"frame_size": [896, 512]},
        "_alpr.json": reads,
        "_alpr_by_track.json": rollup,
        "_static_plates.json": {
            "crop_mode": "fullframe",
            "ocr_conf": "min_char",
            "plate_colour": "hsv_v1",
        },
        "_vehicle_boxes.json": {"params": {}, "boxes": boxes},
        "_dvsa_labels.json": {
            "labels": {
                "AB12CDE": {
                    "plate": "AB12CDE",
                    "track_ids": [1],
                    "make": "FORD",
                    "model": "FIESTA",
                    "year": 2012,
                    "primary_colour": "Blue",
                },
                # A label whose track's best read no longer passes the gate.
                "XY65ZZZ": {"plate": "XY65ZZZ", "track_ids": [2], "make": "KIA", "model": "CEED"},
                "NO11MAK": {"plate": "NO11MAK", "track_ids": [1], "make": None},
            }
        },
    }
    for suffix, content in files.items():
        (sd / f"{LABEL}{suffix}").write_text(json.dumps(content))
    Image.new("RGB", SNAP_SIZE).save(sd / "vehicle_1_main_1.jpg")
    return sd


def _pool(session: Path, **kw: object) -> SessionPool:
    kwargs = {
        "gate": GATE,
        "support": {"AB12CDE": 6, "XY65ZZZ": 1},
        "road_polygon_frac": None,
        "latitude": 51.48,
        "longitude": 0.0,
        **kw,
    }
    pool = session_pool(session, **kwargs)  # type: ignore[arg-type]
    assert pool is not None
    return pool


def test_skip_reason(session: Path) -> None:
    assert skip_reason(session) is None
    stamps = session / f"{LABEL}_static_plates.json"
    stamps.write_text(json.dumps({"crop_mode": "hint", "ocr_conf": "min_char"}))
    assert "fullframe" in (skip_reason(session) or "")
    stamps.write_text(json.dumps({"crop_mode": "fullframe"}))
    assert "alpr-rescore" in (skip_reason(session) or "")
    stamps.write_text(json.dumps({"crop_mode": "fullframe", "ocr_conf": "min_char"}))
    assert "alpr-colour" in (skip_reason(session) or "")
    (session / f"{LABEL}_vehicle_boxes.json").unlink()
    assert skip_reason(session) == f"no {LABEL}_vehicle_boxes.json"


def test_pool_counts_cars_by_direction_and_light(session: Path) -> None:
    pool = _pool(session)
    # Tracks 1, 4 and 6; not the suspect, the person or the stationary one.
    assert pool.cars == {"right to left|day": 3, "left to right|day": 1, "left to right|dark": 1}
    assert pool.cars_with_snaps["right to left|day"] == 2


def test_pool_best_reads_carry_their_stratum_and_gate(session: Path) -> None:
    best = {c["track_id"]: c for c in _pool(session).best}
    assert set(best) == {1, 2, 3}  # track 7 is class_suspect

    one = best[1]
    assert one["key"] == f"{LABEL}/vehicle_1_main_1.jpg"
    assert one["stratum"] == "best|right to left|day|rank0_pass"
    assert (one["path"], one["vehicle_box"]) == ("rank0", [200, 200, 400, 360])
    assert one["hint"] == [200, 200, 400, 360] and one["image_size"] == list(SNAP_SIZE)
    assert (one["n_agree"], one["support"], one["gate_pass"]) == (1, 6, True)
    assert one["track_best"] == {
        "ocr_text": "AB12CDE",
        "ocr_conf": 0.97,
        "canonical_uk_shape": True,
    }

    two = best[2]  # plate on the second-nearest vehicle; a lone read fails the combined gate
    assert two["stratum"] == "best|left to right|day|rank1"
    assert (two["gate_pass"], two["support"]) == (False, 1)

    assert best[3]["stratum"] == "best|left to right|dark|rank0_fail"  # 0.60 < 0.80


def test_pool_rejected_reads_are_the_ones_that_would_have_been_best(session: Path) -> None:
    pool = _pool(session)
    (colour,) = pool.colour_rejected
    assert (colour["track_id"], colour["ocr_text"]) == (3, "FR09ONT")
    assert colour["stratum"] == "colour_rejected|left to right"
    assert colour["track_best"]["ocr_text"] == "RE11ARX"  # what the track fell back to

    (static,) = pool.static_rejected
    assert (static["track_id"], static["stratum"]) == (4, "static_rejected|right to left")
    assert static["track_best"] is None


def test_pool_skips_a_flagged_read_the_kept_read_already_beats(session: Path) -> None:
    alpr_path = session / f"{LABEL}_alpr.json"
    reads = json.loads(alpr_path.read_text())
    next(r for r in reads if r["ocr_text"] == "FR09ONT")["ocr_conf"] = 0.5  # below the kept 0.60
    alpr_path.write_text(json.dumps(reads))
    assert _pool(session).colour_rejected == []


def test_pool_dvsa_candidates_are_gate_passing_reads_of_the_labelled_plate(session: Path) -> None:
    (dvsa,) = _pool(session).dvsa
    assert (dvsa["sample"], dvsa["stratum"], dvsa["track_id"]) == ("dvsa", "dvsa|5+", 1)
    assert dvsa["dvsa"] == {
        "plate": "AB12CDE",
        "make": "FORD",
        "model": "FIESTA",
        "year": 2012,
        "primary_colour": "Blue",
        "colour_class": "blue",
        "body_type": "hatchback",
        "support": 6,
        "support_band": "5+",
    }


def test_pool_is_none_when_no_snap_opens(session: Path) -> None:
    (session / "vehicle_1_main_1.jpg").unlink()
    assert (
        session_pool(
            session, gate=GATE, support={}, road_polygon_frac=None, latitude=51.48, longitude=0.0
        )
        is None
    )


# ----------------------------------------------------------------------
# The whole draw


def _item(sample: str, stratum: str, i: int, **extra: object) -> dict:
    direction, light = "right to left", "day"
    if sample == "best":
        _s, direction, light, _g = stratum.split("|")
    return {
        "key": f"{stratum}/{i}",
        "sample": sample,
        "stratum": stratum,
        "direction": direction,
        "light": light,
        "gate_pass": stratum.endswith("pass"),
        **extra,
    }


def _big_pool() -> SessionPool:
    pool = SessionPool("s")
    for stratum, n in {
        "best|right to left|day|rank0_pass": 400,
        "best|right to left|day|rank0_fail": 60,
        "best|right to left|day|rank1": 12,
        "best|left to right|dark|rank0_pass": 30,
        "best|left to right|dark|fallback": 2,
    }.items():
        pool.best += [_item("best", stratum, i) for i in range(n)]
    pool.colour_rejected = [
        _item("colour_rejected", "colour_rejected|left to right", i) for i in range(40)
    ]
    pool.static_rejected = [
        _item("static_rejected", "static_rejected|right to left", i) for i in range(3)
    ]
    for plate in range(30):  # every plate seen on three tracks
        band = ("1", "2-4", "5+")[plate % 3]
        pool.dvsa += [
            _item("dvsa", f"dvsa|{band}", plate * 10 + t, dvsa={"plate": f"P{plate}"})
            for t in range(3)
        ]
    pool.cars.update({"right to left|day": 700, "left to right|dark": 90})
    pool.cars_with_snaps.update({"right to left|day": 650, "left to right|dark": 80})
    return pool


def test_draw_audit_set_records_populations_and_sample_sizes() -> None:
    sizes = SampleSizes(best=40, colour_rejected=10, static_rejected=10, dvsa=12)
    got = draw_audit_set([_big_pool()], sizes, random.Random(3))

    by_sample = Counter(c["sample"] for c in got.items)
    assert by_sample == {"best": 40, "colour_rejected": 10, "static_rejected": 3, "dvsa": 12}
    assert len({c["key"] for c in got.items}) == len(got.items)

    assert got.strata["best|right to left|day|rank0_pass"]["population"] == 400
    assert got.strata["static_rejected|right to left"] == {"population": 3, "sampled": 3}
    assert got.strata["dvsa|1"] == {"population": 10, "sampled": 4}  # plates, not tracks
    for stratum, info in got.strata.items():
        assert info["sampled"] == sum(1 for c in got.items if c["stratum"] == stratum)

    # The small domain gets its floor of ten; the rest go to the big one.
    best = Counter((c["direction"], c["light"]) for c in got.items if c["sample"] == "best")
    assert best == {("right to left", "day"): 30, ("left to right", "dark"): 10}

    assert got.domains["right to left|day"] == {
        "cars": 700,
        "cars_with_snaps": 650,
        "best": 472,
        "gated": 400,
    }
    assert got.domains["left to right|day"]["cars"] == 0


def test_draw_audit_set_takes_one_read_per_dvsa_car() -> None:
    got = draw_audit_set([_big_pool()], SampleSizes(0, 0, 0, 30), random.Random(0))
    plates = [c["dvsa"]["plate"] for c in got.items]
    assert len(plates) == 30 and len(set(plates)) == 30


def test_draw_audit_set_is_repeatable_and_shuffled() -> None:
    sizes = SampleSizes(best=40, colour_rejected=10, static_rejected=3, dvsa=12)
    first = draw_audit_set([_big_pool()], sizes, random.Random(5))
    again = draw_audit_set([_big_pool()], sizes, random.Random(5))
    assert [c["key"] for c in first.items] == [c["key"] for c in again.items]
    # Samples are interleaved, so a part-labelled set covers all of them.
    assert len({c["sample"] for c in first.items[:20]}) >= 3


# ----------------------------------------------------------------------
# Labels


def test_clean_label_normalises_the_plate_text() -> None:
    got = clean_label(
        {"attribution": "tracked", "text": " ab12 c.e ", "text_first": "ab12cde", "note": "  hi "},
        dvsa=False,
    )
    assert got == {
        "attribution": "tracked",
        "text": "AB12C.E",
        "text_first": "AB12CDE",
        "reconciled": None,
        "note": "hi",
    }


def test_clean_label_allows_an_illegible_plate_and_drops_text_for_a_non_plate() -> None:
    assert clean_label({"attribution": "parked", "text": None}, dvsa=False)["text"] is None
    assert clean_label({"attribution": "parked", "text": "  "}, dvsa=False)["text"] is None
    got = clean_label(
        {"attribution": "not_plate", "text": "TAXI", "reconciled": "kept"}, dvsa=False
    )
    assert (got["text"], got["reconciled"]) == (None, None)


def test_clean_label_dvsa_answers() -> None:
    answers = {"make": "yes", "colour": "no", "body": "unsure"}
    got = clean_label({"attribution": "tracked", "text": "AB12CDE", "dvsa": answers}, dvsa=True)
    assert got["dvsa"] == answers
    assert "dvsa" not in clean_label({"attribution": "tracked", "dvsa": answers}, dvsa=False)


@pytest.mark.parametrize(
    ("payload", "dvsa", "message"),
    [
        ({"attribution": "maybe"}, False, "attribution"),
        ({"attribution": "tracked", "text": "A"}, False, "2 to 8"),
        ({"attribution": "tracked", "text": "AB12CDEFG"}, False, "2 to 8"),
        ({"attribution": "tracked", "text": 12}, False, "text"),
        ({"attribution": "tracked", "reconciled": "dunno"}, False, "reconciled"),
        ({"attribution": "tracked", "note": 3}, False, "note"),
        ({"attribution": "tracked"}, True, "dvsa answers"),
        ({"attribution": "tracked", "dvsa": {"make": "yes", "colour": "yes"}}, True, "body"),
    ],
)
def test_clean_label_rejects(payload: dict, dvsa: bool, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        clean_label(payload, dvsa=dvsa)
