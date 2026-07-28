"""TrajectoryCropDetector: full-frame vehicle detection + trajectory-
anchored candidate crops (the productionised E1 crop path).

Stubs ultralytics + the underlying plate detector, mirroring
test_precrop.py -- the wrapper is candidate-selection + coordinate-
projection logic, which is what's tested here.
"""

from __future__ import annotations

import numpy as np

from streettracker.analysis.alpr.base import PlateDetection
from streettracker.analysis.alpr.fullframe import TrajectoryCropDetector


class _RecordingPlateDetector:
    """Returns a configured detection per call index; records the crop
    shapes it was shown so candidate order and geometry are assertable."""

    name = "fake-plate-det"

    def __init__(self, detections: list[PlateDetection | None]) -> None:
        self._dets = detections
        self.input_shapes: list[tuple[int, int]] = []

    def detect(self, image: np.ndarray, *, bbox_hint=None) -> PlateDetection | None:
        del bbox_hint
        self.input_shapes.append(image.shape[:2])
        i = len(self.input_shapes) - 1
        return self._dets[i] if i < len(self._dets) else None


class _Tensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeBoxes:
    def __init__(self, xy: np.ndarray) -> None:
        self.xyxy = _Tensor(xy)


class _FakeResult:
    def __init__(self, xy: np.ndarray) -> None:
        self.boxes = _FakeBoxes(xy)


class _FakeYOLO:
    def __init__(self, vehicle_bboxes: np.ndarray) -> None:
        self._boxes = vehicle_bboxes

    def predict(self, *args, **kwargs):  # noqa: ANN002, ANN003 -- stub
        return [_FakeResult(self._boxes)]


# A polygon covering the left half of the frame (fractional coords).
LEFT_HALF = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]


def _make(det: TrajectoryCropDetector, bboxes: list[list[float]]) -> None:
    det._yolo = _FakeYOLO(np.array(bboxes, dtype=np.float32))  # noqa: SLF001


def _img(h: int = 1000, w: int = 2000) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_hint_ranks_nearest_vehicle_first() -> None:
    plate = _RecordingPlateDetector([PlateDetection(bbox=(10, 10, 60, 30), det_confidence=0.9)])
    det = TrajectoryCropDetector(plate, pad_px=0, max_candidates=1)
    # Two vehicles; the hint sits on the second (at x~1400).
    _make(det, [[100, 500, 400, 700], [1300, 500, 1600, 700]])

    out = det.detect(_img(), bbox_hint=(1350, 520, 1550, 680))

    assert out is not None
    # Crop was the second vehicle's 300x200 box.
    assert plate.input_shapes == [(200, 300)]
    # Plate bbox projected by the crop origin (1300, 500).
    assert out.bbox == (1310, 510, 1360, 530)


def test_no_hint_prefers_largest() -> None:
    plate = _RecordingPlateDetector([PlateDetection(bbox=(0, 0, 10, 5), det_confidence=0.5)])
    det = TrajectoryCropDetector(plate, pad_px=0, max_candidates=1)
    _make(det, [[0, 0, 100, 100], [500, 200, 1100, 650]])  # second is far larger

    out = det.detect(_img())

    assert out is not None
    assert plate.input_shapes == [(450, 600)]


def test_off_road_vehicle_rejected_by_polygon() -> None:
    # Both vehicles same size; only the left one is inside the polygon
    # (left half of a 2000px-wide frame => x < 1000).
    plate = _RecordingPlateDetector([PlateDetection(bbox=(5, 5, 50, 25), det_confidence=0.8)])
    det = TrajectoryCropDetector(plate, road_polygon_frac=LEFT_HALF, pad_px=0, max_candidates=2)
    _make(det, [[1400, 500, 1700, 700], [200, 500, 500, 700]])
    # Hint sits on the OFF-road vehicle -- it must still lose.
    out = det.detect(_img(), bbox_hint=(1400, 500, 1700, 700))

    assert out is not None
    assert plate.input_shapes == [(200, 300)]
    assert out.bbox == (205, 505, 250, 525)


def test_small_vehicles_ignored() -> None:
    plate = _RecordingPlateDetector([None])
    det = TrajectoryCropDetector(plate, pad_px=0)
    _make(det, [[100, 100, 200, 140]])  # 40px tall < default 45 minimum

    out = det.detect(_img())

    # Candidate filtered out; fallback (no hint) = full-image detect.
    assert out is None
    assert plate.input_shapes == [(1000, 2000)]


def test_best_confidence_wins_across_candidates() -> None:
    plate = _RecordingPlateDetector(
        [
            PlateDetection(bbox=(1, 1, 20, 10), det_confidence=0.55),
            PlateDetection(bbox=(2, 2, 30, 12), det_confidence=0.85),
        ]
    )
    det = TrajectoryCropDetector(plate, pad_px=0, max_candidates=2)
    _make(det, [[100, 500, 400, 700], [1300, 500, 1600, 700]])

    out = det.detect(_img(), bbox_hint=(100, 500, 400, 700))

    assert out is not None
    assert out.det_confidence == 0.85
    # Projected by the SECOND candidate's origin (1300, 500).
    assert out.bbox == (1302, 502, 1330, 512)


def test_no_vehicles_falls_back_to_hint_crop() -> None:
    plate = _RecordingPlateDetector([PlateDetection(bbox=(3, 4, 33, 14), det_confidence=0.6)])
    det = TrajectoryCropDetector(plate, pad_px=0)
    _make(det, [])

    out = det.detect(_img(), bbox_hint=(700, 300, 900, 400))

    assert out is not None
    # Fallback crop was exactly the hint (pad 0): 100x200.
    assert plate.input_shapes == [(100, 200)]
    assert out.bbox == (703, 304, 733, 314)


def test_no_vehicles_no_hint_falls_back_to_full_image() -> None:
    plate = _RecordingPlateDetector([None])
    det = TrajectoryCropDetector(plate)
    _make(det, [])

    assert det.detect(_img()) is None
    assert plate.input_shapes == [(1000, 2000)]
