"""Vehicle locator for the CNN heads: plate anchoring, trajectory fallback,
the per-session vehicle-box cache, and checkpoint-driven crop settings.

The full-frame YOLO detector is injected as a stub, so this runs without
ultralytics / torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from streettracker.analysis.vehicle_locator import (
    SnapVehicleLocator,
    VehicleBoxCache,
    anchor_plate_bbox,
    plate_anchored_box,
    plate_matches,
    resolve_crop_settings,
)

# Snap is 400x300; the sub-stream is 640x360, so hints scale x*0.625, y*0.833.
_W, _H = 400, 300


class _StubDetector:
    """Returns fixed boxes (x1, y1, x2, y2, conf) and counts calls."""

    def __init__(self, boxes: list[tuple[float, float, float, float, float]]) -> None:
        self.boxes = boxes
        self.calls = 0

    def __call__(self, image: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        self.calls += 1
        return list(self.boxes)


def _session(tmp_path: Path, *, alpr: list[dict] | None = None) -> Path:
    """Track 1 with snaps 1 and 2. The stale fire-time hint for both sits
    at (40,30)-(125,150) in snap px -- the car has since moved right."""
    sess = tmp_path / "session_s"
    sess.mkdir()
    for n in (1, 2):
        Image.new("RGB", (_W, _H), (90, 90, 90)).save(sess / f"vehicle_1_main_{n}.jpg")
    sess.joinpath("session_s_data.json").write_text(
        json.dumps(
            [
                {
                    "track_id": 1,
                    "main_snaps": [1, 2],
                    "main_snap_bboxes": [[64, 36, 200, 180], [64, 36, 200, 180]],
                }
            ]
        )
    )
    sess.joinpath("session_s_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    if alpr is not None:
        sess.joinpath("session_s_alpr.json").write_text(json.dumps(alpr))
    return sess


def _read(n: int, text: str, bbox: list[int], **kw: object) -> dict:
    rec = {
        "track_id": 1,
        "snap_index": n,
        "det_bbox": bbox,
        "ocr_text": text,
        "ocr_conf": 0.95,
        "canonical_uk_shape": True,
        "static_suspect": False,
    }
    rec.update(kw)
    return rec


# The tracked car (moved right of its hint), a parked car near the hint,
# and a small far-away car.
_CAR = (200.0, 100.0, 320.0, 200.0, 0.9)
_PARKED = (20.0, 40.0, 110.0, 140.0, 0.8)
_TINY = (350.0, 10.0, 370.0, 30.0, 0.5)
_IMG = np.zeros((_H, _W, 3), dtype=np.uint8)


# ----------------------------------------------------------------------
# Pure helpers.


def test_plate_matches_exact_variant_and_length() -> None:
    assert plate_matches("AB12CDE", "AB12CDE")
    assert plate_matches("ab12 cde", "AB12CDE")
    assert plate_matches("AB12CDF", "AB12CDE")  # one-char OCR variant
    assert not plate_matches("AB12CD", "AB12CDE")  # different length
    assert not plate_matches("XY99ZZZ", "AB12CDE")
    assert not plate_matches(None, "AB12CDE")


def test_anchor_skips_static_and_picks_best_conf() -> None:
    recs = [
        _read(1, "AB12CDE", [0, 0, 10, 5], static_suspect=True),
        _read(1, "AB12CDE", [10, 10, 20, 15], ocr_conf=0.7),
        _read(1, "AB12CDE", [30, 30, 40, 35], ocr_conf=0.99),
        _read(1, "ZZ99ZZZ", [50, 50, 60, 55], ocr_conf=1.0),  # a different car
    ]
    assert anchor_plate_bbox(recs, "AB12CDE") == (30.0, 30.0, 40.0, 35.0)
    # Inference anchor (no plate): any canonical read -- the best overall.
    assert anchor_plate_bbox(recs, None) == (50.0, 50.0, 60.0, 55.0)
    noncanon = [_read(1, "AB1", [1, 1, 2, 2], canonical_uk_shape=False)]
    assert anchor_plate_bbox(noncanon, None) is None


def test_plate_anchored_box_takes_tightest_container() -> None:
    big = (0.0, 0.0, 400.0, 300.0)
    car = (200.0, 100.0, 320.0, 200.0)
    assert plate_anchored_box([big, car], (250.0, 180.0, 280.0, 190.0)) == car
    assert plate_anchored_box([car], (10.0, 10.0, 20.0, 15.0)) is None


def test_resolve_crop_settings() -> None:
    # Legacy checkpoint (no crop metadata): unchanged hint path + legacy pad.
    assert resolve_crop_settings({}, "auto", None, 0.25) == ("hint", 0.25)
    # Plate-anchored corpus: fullframe inference at the recorded pad.
    meta = {"crop_mode": "plate", "crop_pad_frac": 0.1}
    assert resolve_crop_settings(meta, "auto", None, 0.25) == ("fullframe", 0.1)
    # Explicit CLI choices win.
    assert resolve_crop_settings(meta, "hint", 0.3, 0.25) == ("hint", 0.3)


# ----------------------------------------------------------------------
# Vehicle-box cache.


def test_cache_detects_once_and_persists(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    det = _StubDetector([_CAR])
    cache = VehicleBoxCache(sess, detector=det)
    assert cache.boxes("a.jpg", _IMG) == [_CAR[:4]]
    assert cache.boxes("a.jpg", _IMG) == [_CAR[:4]]
    assert det.calls == 1
    cache.save()

    det2 = _StubDetector([_PARKED])
    reloaded = VehicleBoxCache(sess, detector=det2)
    assert reloaded.boxes("a.jpg", _IMG) == [_CAR[:4]]  # served from disk
    assert det2.calls == 0


def test_cache_ignores_file_from_other_detector_params(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    cache = VehicleBoxCache(sess, detector=_StubDetector([_CAR]), imgsz=1920)
    cache.boxes("a.jpg", _IMG)
    cache.save()
    det = _StubDetector([_PARKED])
    other = VehicleBoxCache(sess, detector=det, imgsz=1280)
    assert other.boxes("a.jpg", _IMG) == [_PARKED[:4]]
    assert det.calls == 1


# ----------------------------------------------------------------------
# Locator modes.


def _locator(sess: Path, mode: str, boxes: list) -> SnapVehicleLocator:
    cache = VehicleBoxCache(sess, detector=_StubDetector(boxes))
    return SnapVehicleLocator(sess, mode, cache=cache)


def test_hint_mode_returns_the_stale_hint(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    loc = SnapVehicleLocator(sess, "hint")
    box, src = loc.locate(sess / "vehicle_1_main_1.jpg", 1, 1, None)
    assert (box, src) == ((40, 30, 125, 150), "hint")


def test_plate_mode_crops_the_car_holding_its_plate(tmp_path: Path) -> None:
    sess = _session(tmp_path, alpr=[_read(1, "AB12CDE", [250, 180, 280, 190])])
    loc = _locator(sess, "plate", [_CAR, _PARKED, _TINY])
    box, src = loc.locate(sess / "vehicle_1_main_1.jpg", 1, 1, _IMG, plate="AB12CDE")
    # The moved car, not the parked one sitting under the stale hint.
    assert (box, src) == ((200, 100, 320, 200), "plate")
    # Snap 2 has no read of the plate -> skipped, never guessed.
    assert loc.locate(sess / "vehicle_1_main_2.jpg", 1, 2, _IMG, plate="AB12CDE") == (None, None)
    assert loc.has_plate_read(1, 1, "AB12CDE") and not loc.has_plate_read(1, 2, "AB12CDE")


def test_plate_mode_requires_a_plate(tmp_path: Path) -> None:
    sess = _session(tmp_path, alpr=[])
    loc = _locator(sess, "plate", [_CAR])
    with pytest.raises(ValueError, match="plate"):
        loc.locate(sess / "vehicle_1_main_1.jpg", 1, 1, _IMG)


def test_fullframe_prefers_plate_anchor_then_trajectory(tmp_path: Path) -> None:
    sess = _session(tmp_path, alpr=[_read(1, "AB12CDE", [250, 180, 280, 190])])
    loc = _locator(sess, "fullframe", [_CAR, _PARKED, _TINY])
    # Snap 1 carries a read: plate-anchored.
    assert loc.locate(sess / "vehicle_1_main_1.jpg", 1, 1, _IMG) == ((200, 100, 320, 200), "plate")
    # Snap 2 has none: nearest plate-sized vehicle to the stale hint (the
    # tiny far car is below the 45 px minimum height). Here that is the
    # PARKED car -- the trajectory rule's known failure mode when the
    # tracked car has moved further than a neighbour sits from the hint;
    # .claude/fullframe_crop_spotcheck.py measures how often it bites.
    box, src = loc.locate(sess / "vehicle_1_main_2.jpg", 1, 2, _IMG)
    assert src == "trajectory"
    assert box == (20, 40, 110, 140)
    assert dict(loc.source_counts) == {"plate": 1, "trajectory": 1}


def test_fullframe_skips_unreadable_image_and_missing_hint(tmp_path: Path) -> None:
    sess = _session(tmp_path, alpr=[])
    loc = _locator(sess, "fullframe", [_CAR])
    assert loc.locate(sess / "vehicle_1_main_1.jpg", 1, 1, None) == (None, None)
    # Track 9 has no recorded bbox: no anchor for "which car" -> skip.
    assert loc.locate(sess / "vehicle_9_main_1.jpg", 9, 1, _IMG) == (None, None)


def test_close_persists_boxes(tmp_path: Path) -> None:
    sess = _session(tmp_path, alpr=[])
    loc = _locator(sess, "fullframe", [_CAR])
    loc.locate(sess / "vehicle_1_main_1.jpg", 1, 1, _IMG)
    loc.close()
    data = json.loads((sess / "session_s_vehicle_boxes.json").read_text())
    assert list(data["boxes"]) == ["vehicle_1_main_1.jpg"]


# ----------------------------------------------------------------------
# Stationary vehicles. Five tracks pass a van stopped at the far end, a
# minute apart. Each track's stale hint sits nearer the van than the car.

_VAN = (300.0, 60.0, 380.0, 130.0, 0.9)
_HINT_SUB = [320, 60, 448, 156]  # (200, 50, 280, 130) in snap px: beside the van
_ON_VAN_SUB = [480, 72, 608, 156]  # (300, 60, 380, 130) in snap px: the van itself
_VAN_BOX = (300, 60, 380, 130)


def _mover(track: int) -> tuple[float, float, float, float, float]:
    """The car a track followed: somewhere different in every track's snap."""
    x = 10.0 + 12.0 * track
    return (x, 120.0, x + 120.0, 220.0, 0.9)


def _mover_box(track: int) -> tuple[int, int, int, int]:
    x = 10 + 12 * track
    return (x, 120, x + 120, 220)


class _PerSnapDetector:
    """Hands out one list of boxes per snap, in the order snaps arrive."""

    def __init__(self, per_snap: list[list[tuple[float, float, float, float, float]]]) -> None:
        self._per_snap = iter(per_snap)

    def __call__(self, image: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        return list(next(self._per_snap))


def _street(tmp_path: Path, *, records: dict[int, dict] | None = None) -> Path:
    """Tracks 1-5, one snap each, fired a minute apart. ``records`` overrides
    fields of a track's record."""
    sess = tmp_path / "session_s"
    sess.mkdir()
    data = []
    for track in range(1, 6):
        Image.new("RGB", (_W, _H), (90, 90, 90)).save(sess / f"vehicle_{track}_main_1.jpg")
        record = {
            "track_id": track,
            "main_snaps": [1],
            "main_snap_bboxes": [_HINT_SUB],
            "main_snap_fire_unix": [1000.0 + 60.0 * track],
        }
        record.update((records or {}).get(track, {}))
        data.append(record)
    sess.joinpath("session_s_data.json").write_text(json.dumps(data))
    sess.joinpath("session_s_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    return sess


def _snap_path(sess: Path, track: int) -> Path:
    return sess / f"vehicle_{track}_main_1.jpg"


def _warm_cache(sess: Path, per_snap: list[list]) -> None:
    """Put every snap through the detector, as an earlier command would have."""
    cache = VehicleBoxCache(sess, detector=_PerSnapDetector(per_snap))
    for track in range(1, 6):
        cache.boxes(f"vehicle_{track}_main_1.jpg", _IMG)
    cache.save()


def _van_and_mover() -> list[list]:
    return [[_VAN, _mover(track)] for track in range(1, 6)]


def test_a_stationary_vehicle_is_not_a_passing_tracks_crop(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    _warm_cache(sess, _van_and_mover())
    loc = SnapVehicleLocator(sess, "fullframe")
    # The van is nearer the stale hint, but it was there before track 3 came
    # and after it had gone.
    assert loc.locate(_snap_path(sess, 3), 3, 1, _IMG) == (_mover_box(3), "trajectory")
    # Nothing shows the van before track 1 or after track 5, so for all the
    # locator knows it is their car.
    assert loc.locate(_snap_path(sess, 1), 1, 1, _IMG) == (_VAN_BOX, "trajectory")
    assert loc.locate(_snap_path(sess, 5), 5, 1, _IMG) == (_VAN_BOX, "trajectory")


def test_the_guard_can_be_turned_off(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    _warm_cache(sess, _van_and_mover())
    loc = SnapVehicleLocator(sess, "fullframe", stationary_guard=False)
    assert loc.locate(_snap_path(sess, 3), 3, 1, _IMG) == (_VAN_BOX, "trajectory")


def test_a_snap_showing_only_the_stationary_vehicle_gets_no_crop(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    per_snap = _van_and_mover()
    per_snap[2] = [_VAN]  # track 3's car was not detected
    _warm_cache(sess, per_snap)
    loc = SnapVehicleLocator(sess, "fullframe")
    assert loc.locate(_snap_path(sess, 3), 3, 1, _IMG) == (None, None)
    assert dict(loc.source_counts) == {"none": 1}


@pytest.mark.parametrize(
    "record",
    [
        {"main_snap_bboxes": [_ON_VAN_SUB]},  # on it when the snap fired
        {"main_snap_bboxes_done": [_ON_VAN_SUB]},  # on it when the snap landed
    ],
    ids=["fire-time box", "completion-time box"],
)
def test_a_track_on_the_stationary_vehicle_keeps_it(tmp_path: Path, record: dict) -> None:
    sess = _street(tmp_path, records={3: record})
    _warm_cache(sess, [[_VAN]] * 5)
    loc = SnapVehicleLocator(sess, "fullframe")
    assert loc.locate(_snap_path(sess, 3), 3, 1, _IMG) == (_VAN_BOX, "trajectory")
    # Track 2 only passed it.
    assert loc.locate(_snap_path(sess, 2), 2, 1, _IMG) == (None, None)


def test_a_plate_is_not_anchored_to_a_stationary_vehicle(tmp_path: Path) -> None:
    # The passing car's plate sits inside the stopped van's box too (the car
    # is in front of it), and the van's box is the tighter of the two.
    sess = _street(tmp_path)
    in_front = [(100.0 + 30.0 * track, 40.0, 399.0, 250.0, 0.9) for track in range(1, 6)]
    _warm_cache(sess, [[_VAN, car] for car in in_front])
    read = _read(1, "AB12CDE", [330, 100, 360, 110], track_id=3)
    sess.joinpath("session_s_alpr.json").write_text(json.dumps([read]))

    guarded = SnapVehicleLocator(sess, "fullframe")
    assert guarded.locate(_snap_path(sess, 3), 3, 1, _IMG) == ((190, 40, 399, 250), "plate")
    unguarded = SnapVehicleLocator(sess, "fullframe", stationary_guard=False)
    assert unguarded.locate(_snap_path(sess, 3), 3, 1, _IMG) == (_VAN_BOX, "plate")
    # The mode that takes the car's own plate is guarded too.
    by_plate = SnapVehicleLocator(sess, "plate")
    box, src = by_plate.locate(_snap_path(sess, 3), 3, 1, _IMG, plate="AB12CDE")
    assert (box, src) == ((190, 40, 399, 250), "plate")


def _behind_van(track: int) -> tuple[float, float, float, float, float]:
    """A passing car further up the road, a third or more of it behind the van."""
    x = 250.0 + 8.0 * track
    return (x, 30.0, x + 80.0, 110.0, 0.9)


def test_a_box_hidden_behind_a_stationary_vehicle_is_not_a_crop(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    _warm_cache(sess, [[_VAN, _behind_van(track)] for track in range(1, 6)])
    # The right box, but a crop of it would show the van across the car.
    assert SnapVehicleLocator(sess, "fullframe").locate(_snap_path(sess, 3), 3, 1, _IMG) == (
        None,
        None,
    )
    unguarded = SnapVehicleLocator(sess, "fullframe", stationary_guard=False)
    assert unguarded.locate(_snap_path(sess, 3), 3, 1, _IMG)[1] == "trajectory"


def test_a_plate_read_keeps_a_box_that_is_partly_hidden(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    _warm_cache(sess, [[_VAN, _behind_van(track)] for track in range(1, 6)])
    # The car's plate was read above the van: the car is in view.
    read = _read(1, "AB12CDE", [285, 45, 300, 52], track_id=3)
    sess.joinpath("session_s_alpr.json").write_text(json.dumps([read]))
    loc = SnapVehicleLocator(sess, "fullframe")
    assert loc.locate(_snap_path(sess, 3), 3, 1, _IMG) == ((274, 30, 354, 110), "plate")


def test_settle_finds_the_crops_a_first_pass_got_wrong(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    cache = VehicleBoxCache(sess, detector=_PerSnapDetector(_van_and_mover()))
    loc = SnapVehicleLocator(sess, "fullframe", cache=cache)
    # First pass over a session: when a snap is located, the later snaps
    # have not been through the detector, so nothing is known to be stopped.
    for track in range(1, 6):
        assert loc.locate(_snap_path(sess, track), track, 1, _IMG) == (_VAN_BOX, "trajectory")

    assert loc.settle() == [(2, 1), (3, 1), (4, 1)]
    for track in (2, 3, 4):
        assert loc.locate(_snap_path(sess, track), track, 1, _IMG) == (
            _mover_box(track),
            "trajectory",
        )
    assert loc.settle() == []  # final
    # Each snap is counted once, however often it was located.
    assert dict(loc.source_counts) == {"trajectory": 5}

    written = json.loads((sess / "session_s_stationary_vehicles.json").read_text())
    assert written["n_snaps"] == 5
    assert (written["n_crops_refused"], written["n_left_without_crop"]) == (3, 0)
    assert (loc.n_refused, loc.n_left_without) == (3, 0)
    assert written["vehicles"] == [
        {
            "box": [300, 60, 380, 130],
            "first": "1970-01-01T00:18:40+00:00",
            "last": "1970-01-01T00:20:40+00:00",
            "n_crops": 3,
            "n_tracks": 3,
        }
    ]
    assert written["rule"]["same_box_iou"] == 0.9


def test_settle_changes_nothing_once_the_cache_is_full(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    _warm_cache(sess, _van_and_mover())
    loc = SnapVehicleLocator(sess, "fullframe")
    for track in range(1, 6):
        loc.locate(_snap_path(sess, track), track, 1, _IMG)
    assert loc.settle() == []
    # What the guard did is still counted and written.
    assert (loc.n_refused, loc.n_left_without) == (3, 0)
    written = json.loads((sess / "session_s_stationary_vehicles.json").read_text())
    assert [v["n_tracks"] for v in written["vehicles"]] == [3]


def test_settle_counts_the_snaps_left_with_no_crop(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    per_snap = _van_and_mover()
    per_snap[2] = [_VAN]
    _warm_cache(sess, per_snap)
    loc = SnapVehicleLocator(sess, "fullframe")
    for track in range(1, 6):
        loc.locate(_snap_path(sess, track), track, 1, _IMG)
    loc.settle()
    assert (loc.n_refused, loc.n_left_without) == (3, 1)


def test_hint_mode_has_no_guard(tmp_path: Path) -> None:
    sess = _street(tmp_path)
    loc = SnapVehicleLocator(sess, "hint")
    loc.locate(_snap_path(sess, 3), 3, 1, None)
    assert loc.settle() == []
    assert not (sess / "session_s_stationary_vehicles.json").exists()


def test_a_session_without_snap_times_uses_each_tracks_start(tmp_path: Path) -> None:
    # Recorded before 2026-10-04: no main_snap_fire_unix.
    old = {
        t: {"main_snap_fire_unix": None, "time_start_unix": 1000.0 + 60.0 * t} for t in range(1, 6)
    }
    sess = _street(tmp_path, records=old)
    _warm_cache(sess, _van_and_mover())
    loc = SnapVehicleLocator(sess, "fullframe")
    assert loc.locate(_snap_path(sess, 3), 3, 1, _IMG) == (_mover_box(3), "trajectory")
