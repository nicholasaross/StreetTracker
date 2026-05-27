"""PipelineRunner: stub detector + recognizer so we don't need torch / EasyOCR.

Covers the imread / detect-miss / detect+recognize / crop-save / error paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from streettracker.analysis.alpr.base import PlateDetection, PlateRead
from streettracker.analysis.alpr.runner import PipelineRunner

cv2 = pytest.importorskip("cv2")


class _FakeDetector:
    name = "fake-det"

    def __init__(self, detection: PlateDetection | None) -> None:
        self._d = detection
        self.last_bbox_hint: tuple[int, int, int, int] | None = None

    def detect(self, image, *, bbox_hint=None):  # noqa: ANN001 — runtime test stub
        self.last_bbox_hint = bbox_hint
        return self._d


class _FakeRecognizer:
    name = "fake-ocr"

    def __init__(self, read: PlateRead | None) -> None:
        self._r = read

    def recognize(self, crop_bgr):  # noqa: ANN001
        return self._r


class _ExplodingDetector:
    name = "boom-det"

    def detect(self, image, *, bbox_hint=None):  # noqa: ANN001
        del bbox_hint
        raise RuntimeError("kaboom")


@pytest.fixture
def vehicle_snap(tmp_path: Path) -> Path:
    """A 200x200 BGR JPEG on disk — stands in for a real main snap."""
    p = tmp_path / "vehicle_1_main_1.jpg"
    img = np.full((200, 200, 3), 64, dtype=np.uint8)
    img[80:120, 40:160] = (200, 200, 200)  # fake "plate" stripe
    cv2.imwrite(str(p), img)
    return p


def test_run_imread_failure_records_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jpg"
    runner = PipelineRunner(
        "fake", _FakeDetector(detection=None), _FakeRecognizer(read=None)
    )
    res = runner.run(
        image_path=missing, track_id=1, snap_index=1,
        class_name="vehicle", crop_out_dir=tmp_path / "crops",
    )
    assert res.error == "imread_failed"
    assert res.detection is None
    assert res.crop_path is None


def test_run_no_detection_returns_no_error(vehicle_snap: Path, tmp_path: Path) -> None:
    runner = PipelineRunner(
        "fake", _FakeDetector(detection=None), _FakeRecognizer(read=None)
    )
    res = runner.run(
        image_path=vehicle_snap, track_id=1, snap_index=1,
        class_name="vehicle", crop_out_dir=tmp_path / "crops",
    )
    assert res.error is None
    assert res.detection is None
    assert res.read is None
    # NOTE: early-returns inside the `with Timer() as t:` block capture
    # t.ms before __exit__ runs, so pipeline_ms is 0.0 here. Only the
    # final fall-through return (detect + recognize success) records a
    # nonzero ms — see test_run_detection_and_read_persists_crop.
    assert res.pipeline_ms == 0.0


def test_run_detection_and_read_persists_crop(
    vehicle_snap: Path, tmp_path: Path
) -> None:
    det = PlateDetection(bbox=(40, 80, 160, 120), det_confidence=0.92)
    read = PlateRead(text="ABC123", ocr_confidence=0.88, raw_text="ABC 123")
    crops = tmp_path / "crops"
    runner = PipelineRunner("fake", _FakeDetector(det), _FakeRecognizer(read))
    res = runner.run(
        image_path=vehicle_snap, track_id=42, snap_index=2,
        class_name="vehicle", crop_out_dir=crops,
    )
    assert res.error is None
    assert res.detection is det
    assert res.read is read
    assert res.crop_path is not None
    crop_file = Path(res.crop_path)
    assert crop_file.exists()
    # Crop landed under the named pipeline subdir
    assert crop_file.parent == crops
    # And the file is a valid JPEG (cv2 can re-read it)
    re_read = cv2.imread(str(crop_file))
    assert re_read is not None and re_read.size > 0


def test_run_detector_exception_surfaces_into_error_field(
    vehicle_snap: Path, tmp_path: Path
) -> None:
    runner = PipelineRunner(
        "boom", _ExplodingDetector(), _FakeRecognizer(read=None)
    )
    res = runner.run(
        image_path=vehicle_snap, track_id=1, snap_index=1,
        class_name="vehicle", crop_out_dir=tmp_path / "crops",
    )
    assert res.error is not None
    assert "RuntimeError" in res.error
    assert "kaboom" in res.error


def test_run_ghost_rects_zero_image_before_detect(tmp_path: Path) -> None:
    """ghost_rects zero-fills the image rect before the detector sees
    it. Verified by giving the detector a stub that records the
    pixel value at a point inside the mask: if the mask is applied,
    the pixel reads as 0; otherwise it carries the original fill
    value. Mask coords are interpreted in the source_size coord
    system and scaled to the actual image dimensions."""
    # 384x216 image filled with 200 (so a 0-masked region is clearly
    # distinguishable). Save as JPEG and let cv2 re-read it.
    p = tmp_path / "vehicle_1_main_1.jpg"
    img = np.full((216, 384, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(p), img)

    seen_pixel: dict[str, int] = {}

    class _PeekDetector:
        name = "peek"

        def detect(self, image, *, bbox_hint=None):  # noqa: ANN001
            del bbox_hint
            # Mask rect at source_size [3840, 2160] is [750, 700,
            # 960, 860]; scaled to 384x216 (sx=sy=0.1) it becomes
            # [75, 70, 96, 86]. (78, 85) is inside that rect.
            seen_pixel["centre"] = int(image[78, 85, 0])
            return None

    runner = PipelineRunner("fake", _PeekDetector(), _FakeRecognizer(read=None))
    runner.run(
        image_path=p, track_id=1, snap_index=1,
        class_name="vehicle", crop_out_dir=tmp_path / "crops",
        ghost_rects=[(750, 700, 960, 860)],
        ghost_source_size=(3840, 2160),
    )

    assert seen_pixel["centre"] == 0


def test_run_ghost_rects_without_source_size_uses_raw_coords(
    tmp_path: Path,
) -> None:
    """When ghost_source_size is None, rects are applied at raw pixel
    coords (no scaling)."""
    p = tmp_path / "vehicle_1_main_1.jpg"
    img = np.full((216, 384, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(p), img)

    seen_pixel: dict[str, int] = {}

    class _PeekDetector:
        name = "peek"

        def detect(self, image, *, bbox_hint=None):  # noqa: ANN001
            del bbox_hint
            # Rect (50, 50, 150, 100) in raw pixel coords; (75, 75)
            # is inside.
            seen_pixel["centre"] = int(image[75, 75, 0])
            return None

    runner = PipelineRunner("fake", _PeekDetector(), _FakeRecognizer(read=None))
    runner.run(
        image_path=p, track_id=1, snap_index=1,
        class_name="vehicle", crop_out_dir=tmp_path / "crops",
        ghost_rects=[(50, 50, 150, 100)],
        ghost_source_size=None,
    )

    assert seen_pixel["centre"] == 0


def test_to_json_round_trip_has_expected_fields(
    vehicle_snap: Path, tmp_path: Path
) -> None:
    det = PlateDetection(bbox=(40, 80, 160, 120), det_confidence=0.92)
    read = PlateRead(text="ABC123", ocr_confidence=0.88, raw_text="ABC 123")
    runner = PipelineRunner("fake", _FakeDetector(det), _FakeRecognizer(read))
    res = runner.run(
        image_path=vehicle_snap, track_id=42, snap_index=2,
        class_name="vehicle", crop_out_dir=tmp_path / "crops",
    )
    j = res.to_json()
    assert j["image"] == "vehicle_1_main_1.jpg"
    assert j["pipeline"] == "fake"
    assert j["det_bbox"] == [40, 80, 160, 120]
    assert j["ocr_text"] == "ABC123"
    assert j["ocr_raw"] == "ABC 123"
    assert j["error"] is None
