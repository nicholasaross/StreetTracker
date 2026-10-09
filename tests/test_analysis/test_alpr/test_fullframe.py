"""TrajectoryCropDetector: full-frame vehicle detection + trajectory-
anchored candidate crops (the productionised E1 crop path).

Stubs ultralytics + the underlying plate detector, mirroring
test_precrop.py -- the wrapper is candidate-selection + coordinate-
projection logic, which is what's tested here.
"""

from __future__ import annotations

import numpy as np

from streettracker.analysis.alpr.base import PlateDetection
from streettracker.analysis.alpr.fullframe import TrajectoryCropDetector, rank_vehicle_candidates


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


# A horizontal road band: y in [0.4, 0.8] of the frame (400-800 px of 1000).
ROAD_BAND = [(0.0, 0.4), (1.0, 0.4), (1.0, 0.8), (0.0, 0.8)]


# A van in the far zone: its box centre (y=350) sits above the road, over
# the houses, but its bottom edge (y=600) is on the tarmac.
TALL_VAN = (300.0, 100.0, 700.0, 600.0)
# A car parked at the kerb: centre (y=350) over the pavement, bottom edge
# (y=420) just inside the road band.
KERB_CAR = (1000.0, 280.0, 1300.0, 420.0)
# A car on the road: centre (y=600) inside the band.
ROAD_CAR = (1500.0, 500.0, 1800.0, 700.0)


def test_on_road_point_centre_vs_bottom() -> None:
    boxes = [TALL_VAN, KERB_CAR, ROAD_CAR, (1300.0, 600.0, 1600.0, 900.0)]

    def ranked(point: str) -> list:
        return rank_vehicle_candidates(
            boxes, 2000, 1000, bbox_hint=None, road_polygon_frac=ROAD_BAND, on_road_point=point
        )

    # Default (centre): the van and the kerb car are out; the foreground
    # box is in (its centre, y=750, is inside the band).
    assert set(ranked("centre")) == {ROAD_CAR, (1300.0, 600.0, 1600.0, 900.0)}
    # Bottom: wheels on the road -- the foreground box (bottom y=900) is out.
    assert set(ranked("bottom")) == {TALL_VAN, KERB_CAR, ROAD_CAR}


# A stale hint overlapping the van (IoU 0.23) whose own crop misses the plate.
VAN_HINT = (500, 150, 900, 450)


def test_tall_vehicle_found_by_bottom_retry() -> None:
    # Centre rule: no candidates. Hint crop: no plate. Retry: the van.
    plate = _RecordingPlateDetector([None, PlateDetection(bbox=(5, 5, 50, 25), det_confidence=0.8)])
    det = TrajectoryCropDetector(plate, road_polygon_frac=ROAD_BAND, pad_px=0)
    _make(det, [list(TALL_VAN)])

    out = det.detect(_img(), bbox_hint=VAN_HINT)

    assert out is not None
    assert plate.input_shapes == [(300, 400), (500, 400)]
    assert out.bbox == (305, 105, 350, 125)
    assert out.bottom_retry is True  # so the static filter won't learn from it


def test_bottom_retry_ignores_vehicles_away_from_the_hint() -> None:
    # The kerb car has its wheels on the road but doesn't overlap the
    # hint: a parked car, not the tracked one, so the retry skips it.
    plate = _RecordingPlateDetector([None, PlateDetection(bbox=(5, 5, 50, 25), det_confidence=0.8)])
    det = TrajectoryCropDetector(plate, road_polygon_frac=ROAD_BAND, pad_px=0)
    _make(det, [list(KERB_CAR)])

    assert det.detect(_img(), bbox_hint=(100, 450, 300, 650)) is None
    assert plate.input_shapes == [(200, 200)]  # the hint crop only


def test_bottom_retry_never_displaces_a_centre_rule_plate() -> None:
    # The kerb car sits nearest the hint, but only the centre rule's
    # candidate is tried because it yields a plate.
    plate = _RecordingPlateDetector([PlateDetection(bbox=(1, 1, 20, 10), det_confidence=0.6)])
    det = TrajectoryCropDetector(plate, road_polygon_frac=ROAD_BAND, pad_px=0)
    _make(det, [list(KERB_CAR), list(ROAD_CAR)])

    out = det.detect(_img(), bbox_hint=(1000, 280, 1300, 420))

    assert out is not None
    assert plate.input_shapes == [(200, 300)]  # ROAD_CAR only
    assert out.bbox == (1501, 501, 1520, 510)
    assert out.bottom_retry is False


def test_bottom_retry_skips_vehicles_already_tried() -> None:
    # ROAD_CAR (centre rule) and the hint crop find nothing; the retry
    # tries only the van, not ROAD_CAR again. (Overlap check off, so it
    # can't be what keeps ROAD_CAR out.)
    plate = _RecordingPlateDetector([None, None, None])
    det = TrajectoryCropDetector(
        plate, road_polygon_frac=ROAD_BAND, pad_px=0, retry_min_hint_iou=0.0
    )
    _make(det, [list(ROAD_CAR), list(TALL_VAN)])

    out = det.detect(_img(), bbox_hint=(1200, 800, 1300, 900))

    assert out is None
    assert plate.input_shapes == [(200, 300), (100, 100), (500, 400)]


def test_no_bottom_retry_without_polygon() -> None:
    plate = _RecordingPlateDetector([None, None])
    det = TrajectoryCropDetector(plate, pad_px=0)
    _make(det, [list(TALL_VAN)])

    assert det.detect(_img(), bbox_hint=(1200, 100, 1500, 300)) is None
    # The van was a candidate already (no polygon), then the hint crop.
    assert plate.input_shapes == [(500, 400), (200, 300)]


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
