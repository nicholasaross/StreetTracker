"""PreCropDetector: vehicle-bbox pre-crop wrapper around a plate detector.

Stubs ultralytics + the underlying plate detector so the tests don't
require torch / onnx weights. The PreCropDetector module is itself a
thin coordinate-projection wrapper; that's what we test here.
"""

from __future__ import annotations

import numpy as np

from streettracker.analysis.alpr.base import PlateDetection
from streettracker.analysis.alpr.precrop import PreCropDetector


class _FakePlateDetector:
    """Tracks the crop it was called with so tests can assert pre-crop
    actually happened and the projection arithmetic is right."""

    name = "fake-plate-det"

    def __init__(self, detection: PlateDetection | None) -> None:
        self._d = detection
        self.last_input_shape: tuple[int, int] | None = None
        self.calls = 0

    def detect(
        self,
        image: np.ndarray,
        *,
        bbox_hint: tuple[int, int, int, int] | None = None,
    ) -> PlateDetection | None:
        del bbox_hint  # this stub is the underlying detector, not the wrapper
        self.calls += 1
        self.last_input_shape = image.shape[:2]
        return self._d


class _FakeBoxes:
    def __init__(self, xy: np.ndarray) -> None:
        self.xyxy = _Tensor(xy)

    def __len__(self) -> int:
        return len(self.xyxy.cpu().numpy())


class _Tensor:
    """Mimics the ``.cpu().numpy()`` chain ultralytics returns."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeResult:
    def __init__(self, boxes_xyxy: np.ndarray) -> None:
        self.boxes = _FakeBoxes(boxes_xyxy)


class _FakeYOLO:
    """Returns pre-configured vehicle bboxes regardless of input."""

    def __init__(self, vehicle_bboxes: np.ndarray) -> None:
        self._vehicle_bboxes = vehicle_bboxes

    def predict(self, *args, **kwargs):  # noqa: ANN002, ANN003 — stub
        return [_FakeResult(self._vehicle_bboxes)]


def _make_image(h: int = 2160, w: int = 3840) -> np.ndarray:
    """A blank 4K BGR frame -- the PreCropDetector doesn't care about
    content, only shape (for clamping the crop coordinates)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _inject_fake_yolo(detector: PreCropDetector, bboxes: np.ndarray) -> None:
    """Bypass ultralytics by pre-installing a fake YOLO instance."""
    detector._yolo = _FakeYOLO(bboxes)  # noqa: SLF001 -- test wiring


def test_pre_crop_no_vehicle_falls_back_to_full_image() -> None:
    """If the vehicle detector finds nothing, run the plate detector
    on the full original image (preserves prior behaviour for ambiguous
    frames -- parked-car-only scenes, partial occlusion, etc.)."""
    plate = _FakePlateDetector(detection=PlateDetection(bbox=(10, 20, 80, 60), det_confidence=0.5))
    pre = PreCropDetector(plate_detector=plate)
    _inject_fake_yolo(pre, np.zeros((0, 4), dtype=np.float32))  # no vehicles

    img = _make_image()
    out = pre.detect(img)

    assert out is not None
    # The plate bbox should be untouched -- no crop projection happened.
    assert out.bbox == (10, 20, 80, 60)
    assert out.det_confidence == 0.5
    # Plate detector ran exactly once, on the full image.
    assert plate.calls == 1
    assert plate.last_input_shape == img.shape[:2]


def test_pre_crop_projects_plate_bbox_back_to_original_coords() -> None:
    """A plate at (50, 30, 150, 70) inside a vehicle crop starting at
    (1000, 800) in the original 4K image must come out at
    (1050, 830, 1150, 870) in original-image coordinates so the
    downstream OCR-crop step (which slices the original) frames the
    plate correctly."""
    plate = _FakePlateDetector(detection=PlateDetection(bbox=(50, 30, 150, 70), det_confidence=0.9))
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.0)  # zero pad simplifies the assertion
    # Vehicle bbox at (1000, 800) -> (2000, 1600). With pad_frac=0 the
    # crop origin is exactly (1000, 800).
    _inject_fake_yolo(pre, np.array([[1000.0, 800.0, 2000.0, 1600.0]]))

    out = pre.detect(_make_image())

    assert out is not None
    assert out.bbox == (1050, 830, 1150, 870)
    assert out.det_confidence == 0.9
    assert plate.calls == 1
    # The plate detector saw the cropped vehicle region (1000x800),
    # not the original 4K frame.
    assert plate.last_input_shape == (800, 1000)


def test_pre_crop_iterates_vehicles_largest_first() -> None:
    """If the largest vehicle's crop yields no plate, the next-largest
    is tried. Lets the wrapper recover frames where a tracked car is
    partly off-screen but a parked car alongside it has a clean plate."""
    # Configure the plate detector to miss on the first call, hit on the second.
    class _ToggleDetector:
        name = "toggle"

        def __init__(self) -> None:
            self.calls = 0
            self.shapes: list[tuple[int, int]] = []

        def detect(
            self,
            image: np.ndarray,
            *,
            bbox_hint: tuple[int, int, int, int] | None = None,
        ) -> PlateDetection | None:
            del bbox_hint
            self.calls += 1
            self.shapes.append(image.shape[:2])
            if self.calls == 1:
                return None
            return PlateDetection(bbox=(5, 5, 25, 15), det_confidence=0.7)

    plate = _ToggleDetector()
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.0)
    # Two vehicles: a large one at (1000,800)-(2500,1800) area=1.5M,
    # and a smaller one at (100,100)-(300,200) area=20k.
    _inject_fake_yolo(
        pre,
        np.array([
            [100.0, 100.0, 300.0, 200.0],
            [1000.0, 800.0, 2500.0, 1800.0],
        ]),
    )

    out = pre.detect(_make_image())

    assert out is not None
    assert plate.calls == 2
    # Largest first: shape (1000, 1500) -> then (100, 200).
    assert plate.shapes[0] == (1000, 1500)
    assert plate.shapes[1] == (100, 200)
    # Plate found in the second (smaller) crop at (5,5,25,15) -> projected
    # back to the smaller vehicle bbox origin (100,100).
    assert out.bbox == (105, 105, 125, 115)


def test_pre_crop_padding_adds_margin_around_bbox() -> None:
    """pad_frac=0.10 on a 1000x800 vehicle bbox should expand to
    1200x960 (100 px on each side x, 80 px y) before the crop. The
    cap is set high to isolate fractional behaviour from the cap."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(0, 0, 50, 20), det_confidence=0.5)
    )
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.10, pad_max_px=999)
    _inject_fake_yolo(pre, np.array([[1000.0, 800.0, 2000.0, 1600.0]]))

    pre.detect(_make_image())

    # 1000 px wide bbox + 10% padding each side = 1200 px wide crop.
    # 800 px tall bbox + 10% padding each side = 960 px tall crop.
    assert plate.last_input_shape == (960, 1200)


def test_pre_crop_padding_capped_for_wide_bboxes() -> None:
    """pad_max_px caps per-side padding regardless of pad_frac. A
    1000x800 bbox with pad_frac=0.15 would naturally pad 150/120 px,
    but with pad_max_px=30 the actual padding is 30 px each side.
    This prevents wide BotSORT bboxes from spilling into neighbouring
    cars (Step 10 of CLAUDE.md ANPR tuning loop)."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(0, 0, 50, 20), det_confidence=0.5)
    )
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.15, pad_max_px=30)
    _inject_fake_yolo(pre, np.array([[1000.0, 800.0, 2000.0, 1600.0]]))

    pre.detect(_make_image())

    # 1000 wide + 30 each side = 1060.  800 tall + 30 each side = 860.
    assert plate.last_input_shape == (860, 1060)


def test_pre_crop_clamps_padded_crop_to_image_bounds() -> None:
    """Padding past the image edge must clamp rather than producing
    negative indices."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(0, 0, 50, 20), det_confidence=0.5)
    )
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.50, pad_max_px=999)
    # Vehicle bbox right at the top-left corner -- padding would push
    # the crop to negative coords without the clamp.
    _inject_fake_yolo(pre, np.array([[0.0, 0.0, 100.0, 100.0]]))

    pre.detect(_make_image(h=500, w=500))

    # Clamp at 0 on the leading edges; pad of 50 px on the trailing
    # edges (100 + 50 padding = 150).
    assert plate.last_input_shape == (150, 150)


def test_pre_crop_all_vehicles_miss_falls_back_to_full_image() -> None:
    """If every vehicle crop is fed to the plate detector and none
    yield a plate, fall back to running the plate detector on the
    full image one last time."""
    plate = _FakePlateDetector(detection=None)
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.0)
    _inject_fake_yolo(pre, np.array([[100.0, 100.0, 300.0, 200.0]]))

    out = pre.detect(_make_image())

    # One call on the vehicle crop, one final call on the full image.
    assert plate.calls == 2
    assert out is None  # plate detector returns None on both


def test_pre_crop_with_hint_skips_yolo_entirely() -> None:
    """When a bbox_hint is provided (typically piped from the
    TrackRecord's main_snap_bboxes), the YOLO vehicle-detection step
    is bypassed and the plate detector runs directly on the hinted
    crop. This is the path that eliminates parked-car aliasing."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(50, 30, 150, 70), det_confidence=0.9)
    )
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.0)
    # Deliberately DO NOT inject a fake YOLO -- if the wrapper tried
    # to load ultralytics here the test would either ImportError or
    # download a model from the network. The hint path must bypass
    # _ensure_loaded entirely.

    out = pre.detect(
        _make_image(), bbox_hint=(1000, 800, 2000, 1600),
    )

    assert out is not None
    # Plate at (50, 30) inside crop starting at (1000, 800) -> (1050, 830).
    assert out.bbox == (1050, 830, 1150, 870)
    assert out.det_confidence == 0.9
    assert plate.calls == 1
    assert plate.last_input_shape == (800, 1000)


def test_pre_crop_hint_applies_padding() -> None:
    """The bbox_hint path applies the same pad_frac padding as the
    YOLO path so both routes feed the plate detector consistent
    crop margins."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(0, 0, 30, 10), det_confidence=0.5)
    )
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.10, pad_max_px=999)

    pre.detect(_make_image(), bbox_hint=(1000, 800, 2000, 1600))

    # 1000-px-wide hint + 10% pad each side -> 1200x960 crop.
    assert plate.last_input_shape == (960, 1200)


def test_pre_crop_hint_clamps_to_image_bounds() -> None:
    """A hint near the image edge with padding overflowing the frame
    is clamped, not skipped."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(0, 0, 30, 10), det_confidence=0.5)
    )
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.50, pad_max_px=999)

    pre.detect(_make_image(h=500, w=500), bbox_hint=(0, 0, 100, 100))

    # 100x100 hint clamped at the origin, +50px pad on the trailing edges.
    assert plate.last_input_shape == (150, 150)


def test_pre_crop_degenerate_hint_falls_back_to_full_image() -> None:
    """A hint with zero or negative area (after clamping past the
    image edge) is treated as missing and the plate detector runs on
    the original image. Defensive against sub-stream bboxes that
    drifted off-frame before fire-time."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(1, 1, 5, 3), det_confidence=0.3)
    )
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.0)

    out = pre.detect(_make_image(h=500, w=500), bbox_hint=(600, 600, 700, 700))

    assert out is not None
    assert plate.calls == 1
    # Should have seen the full 500x500 frame, not the (clamped-to-zero) crop.
    assert plate.last_input_shape == (500, 500)


def test_pre_crop_name_decorates_underlying_detector_name() -> None:
    plate = _FakePlateDetector(detection=None)
    pre = PreCropDetector(plate_detector=plate)
    assert pre.name == "precrop-fake-plate-det"


# ----------------------------------------------------------------------
# Vehicle stage inside a motion-window hint (Step 16).


def test_window_hint_vehicle_stage_projects_through_both_offsets() -> None:
    """With ``vehicle_stage_in_hint``, the wide window hint is vehicle-
    detected first and the plate detector runs on the TIGHT vehicle crop
    (restoring plate pixels); the plate bbox projects back through the
    vehicle-crop offset AND the window offset."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(10, 20, 90, 50), det_confidence=0.7)
    )
    pre = PreCropDetector(
        plate_detector=plate, pad_frac=0.0, vehicle_stage_in_hint=True
    )
    # Vehicle found at (300,100)-(700,300) RELATIVE to the window crop.
    _inject_fake_yolo(pre, np.array([[300.0, 100.0, 700.0, 300.0]]))

    out = pre.detect(_make_image(), bbox_hint=(1000, 500, 3000, 900))

    assert out is not None
    # Plate detector saw the tight 400x200 vehicle crop, not the 2000px window.
    assert plate.last_input_shape == (200, 400)
    # bbox = plate (10,20) + vehicle offset (300,100) + window offset (1000,500).
    assert out.bbox == (1310, 620, 1390, 650)


def test_window_hint_vehicle_stage_falls_back_to_window_when_no_vehicle() -> None:
    """COCO vehicle detector finding nothing inside the window must not
    lose the read entirely -- fall back to plate-detecting the window."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(5, 5, 50, 25), det_confidence=0.4)
    )
    pre = PreCropDetector(
        plate_detector=plate, pad_frac=0.0, vehicle_stage_in_hint=True
    )
    _inject_fake_yolo(pre, np.empty((0, 4)))

    out = pre.detect(_make_image(), bbox_hint=(1000, 500, 3000, 900))

    assert out is not None
    # Fell back to the whole 2000x400 window crop.
    assert plate.last_input_shape == (400, 2000)
    assert out.bbox == (1005, 505, 1050, 525)


def test_legacy_hint_path_unchanged_without_flag() -> None:
    """Default construction (flag off) keeps the v1 behaviour: plate
    detector runs directly on the hinted crop, no vehicle stage."""
    plate = _FakePlateDetector(
        detection=PlateDetection(bbox=(1, 1, 9, 5), det_confidence=0.9)
    )
    pre = PreCropDetector(plate_detector=plate, pad_frac=0.0)
    # No fake YOLO injected: the vehicle stage would crash if invoked.

    out = pre.detect(_make_image(), bbox_hint=(100, 100, 300, 200))

    assert out is not None
    assert plate.last_input_shape == (100, 200)
