"""Full-frame trajectory-matched crop targeting (the "E1 crop path").

The hint-window crop strategy (fire/done bbox union + lookahead) was
falsified on 2026-07-28: completion-time bboxes sit a median ~510 4K px
from where the car actually is in the saved image, so hint crops
routinely frame empty road, the wrong car, or background parked plates.
An offline re-score of three soaks with the approach below lifted the
R->L per-car canonical rate from 7-10 % to 75-93 % on the SAME images
(see ``.claude/rl_rescore_e1.py`` and the 2026-07-27 evidence dossier).

Strategy per snap:

1. detect vehicles on the FULL frame (ultralytics, COCO car/bus/truck);
2. keep candidates that are on the road (operator polygon) and big
   enough to plausibly carry a readable plate;
3. rank by distance to the per-snap bbox hint -- the hint is stale but
   it is still the best available anchor for *which* vehicle is the
   tracked one -- and try the plate detector on the best few;
4. return the highest-confidence plate detection, projected back to
   full-image coordinates.

Parked cars inside the polygon can still win a candidate slot; the
static-plate filter (``staticfilter.py``) removes their reads at the
rollup stage, exactly as it does for hint crops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from streettracker.analysis.alpr.base import PlateDetection

if TYPE_CHECKING:
    import numpy as np

# Vehicles smaller than this (px height in the snap) can't carry a
# readable plate at this scene's resolution -- the canonical-read p25
# plate width is ~79 px, which sits on cars well above this height.
DEFAULT_MIN_VEHICLE_H_PX = 45

# Padding around the vehicle bbox before plate detection, matching the
# hint path's cap so detector context stays comparable.
DEFAULT_PAD_PX = 30

# How many ranked candidates get a plate-detection attempt. The tracked
# car is almost always rank 0 or 1; more just adds background risk.
DEFAULT_MAX_CANDIDATES = 2

# Full-frame inference size for the vehicle stage. 4512-wide snaps at
# 1920 keep even far-zone cars (>=45 px) comfortably detectable.
DEFAULT_VEHICLE_IMGSZ = 1920

_VEHICLE_COCO_CLASSES = (2, 5, 7)  # car, bus, truck


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


class TrajectoryCropDetector:
    """Wraps a plate detector with full-frame vehicle detection and
    trajectory-anchored candidate selection.

    ``road_polygon_frac`` is the operator-traced road outline as
    fractional ``(x, y)`` vertices (resolution-independent, the same
    coordinates ``triggers_proposal.json`` carries). ``None`` disables
    the on-road filter.
    """

    def __init__(
        self,
        plate_detector: Any,
        vehicle_model: str = "yolov8m.pt",
        road_polygon_frac: list[tuple[float, float]] | None = None,
        *,
        min_vehicle_h_px: int = DEFAULT_MIN_VEHICLE_H_PX,
        pad_px: int = DEFAULT_PAD_PX,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        vehicle_conf: float = 0.2,
        vehicle_imgsz: int = DEFAULT_VEHICLE_IMGSZ,
    ) -> None:
        self._plate_detector = plate_detector
        self._vehicle_model_spec = vehicle_model
        self._poly_frac = road_polygon_frac
        self._min_h = min_vehicle_h_px
        self._pad = pad_px
        self._max_candidates = max_candidates
        self._vehicle_conf = vehicle_conf
        self._imgsz = vehicle_imgsz
        self._yolo: Any = None
        self.name = f"fullframe-{getattr(plate_detector, 'name', 'unknown')}"

    def _ensure_loaded(self) -> None:
        if self._yolo is not None:
            return
        from ultralytics import YOLO

        self._yolo = YOLO(self._vehicle_model_spec)

    def detect(
        self,
        image: np.ndarray,
        *,
        bbox_hint: tuple[int, int, int, int] | None = None,
    ) -> PlateDetection | None:
        h, w = image.shape[:2]
        self._ensure_loaded()

        result = self._yolo.predict(
            image,
            classes=list(_VEHICLE_COCO_CLASSES),
            conf=self._vehicle_conf,
            imgsz=self._imgsz,
            verbose=False,
        )[0]
        xy = result.boxes.xyxy.cpu().numpy()

        poly = [(px * w, py * h) for px, py in self._poly_frac] if self._poly_frac else None
        hint_c = (
            ((bbox_hint[0] + bbox_hint[2]) / 2.0, (bbox_hint[1] + bbox_hint[3]) / 2.0)
            if bbox_hint is not None
            else None
        )

        candidates: list[tuple[float, tuple[float, float, float, float]]] = []
        for row in xy:
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            if y2 - y1 < self._min_h:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if poly is not None and not _point_in_polygon(cx, cy, poly):
                continue
            if hint_c is not None:
                rank = ((cx - hint_c[0]) ** 2 + (cy - hint_c[1]) ** 2) ** 0.5
            else:
                rank = -(x2 - x1) * (y2 - y1)  # no anchor: biggest first
            candidates.append((rank, (x1, y1, x2, y2)))
        candidates.sort(key=lambda c: c[0])

        best: PlateDetection | None = None
        for _rank, (x1, y1, x2, y2) in candidates[: self._max_candidates]:
            cx1 = max(0, int(x1 - self._pad))
            cy1 = max(0, int(y1 - self._pad))
            cx2 = min(w, int(x2 + self._pad))
            cy2 = min(h, int(y2 + self._pad))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            det = self._plate_detector.detect(image[cy1:cy2, cx1:cx2])
            if det is None:
                continue
            px1, py1, px2, py2 = det.bbox
            projected = PlateDetection(
                bbox=(px1 + cx1, py1 + cy1, px2 + cx1, py2 + cy1),
                det_confidence=det.det_confidence,
            )
            if best is None or projected.det_confidence > best.det_confidence:
                best = projected
        if best is not None:
            return best

        # No on-road vehicle yielded a plate. Fall back to the legacy
        # crop so degenerate frames keep their old behaviour: the hint
        # crop when we have one, else the full image.
        if bbox_hint is not None:
            hx1 = max(0, int(bbox_hint[0] - self._pad))
            hy1 = max(0, int(bbox_hint[1] - self._pad))
            hx2 = min(w, int(bbox_hint[2] + self._pad))
            hy2 = min(h, int(bbox_hint[3] + self._pad))
            if hx2 > hx1 and hy2 > hy1:
                det = self._plate_detector.detect(image[hy1:hy2, hx1:hx2])
                if det is not None:
                    px1, py1, px2, py2 = det.bbox
                    return PlateDetection(
                        bbox=(px1 + hx1, py1 + hy1, px2 + hx1, py2 + hy1),
                        det_confidence=det.det_confidence,
                    )
                return None
        return self._plate_detector.detect(image)
