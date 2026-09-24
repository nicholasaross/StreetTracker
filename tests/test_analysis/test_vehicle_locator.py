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
