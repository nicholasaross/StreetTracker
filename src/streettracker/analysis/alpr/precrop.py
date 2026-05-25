"""Vehicle pre-crop wrapper for plate detectors.

Diagnostic study on 2026-05-25 (see CLAUDE.md ANPR tuning loop, "B1
diagnostic") found that the dominant failure mode for the 137 / 387
cars in the soak that yielded no OCR read was **plate-detector miss
on every snap, on both pipelines** -- never an OCR miss. The plates
were visible to a human eye in the 4K image, but after the detectors'
internal downscale (default `yolo-v9-t-384` resamples 4K -> 384 input)
a 100-px-wide plate becomes ~10 px wide, below the detector's
minimum-object capability.

The fix is a vehicle pre-crop pass: use a coarse vehicle detector
(ultralytics YOLOv8n, COCO classes 2 / 5 / 7 = car / bus / truck) to
find the car bbox in the 4K image, crop with light padding, then run
the underlying plate detector on the crop. The plate now occupies
10-30 % of the detector's input instead of 0.3 %.

Empirical smoke test on a 15-track sample of no-read cars:

  detector variant                          | full image | pre-cropped
  ------------------------------------------|------------|------------
  yolo-v9-t-384 (current production default) | 0  / 15    | (untested)
  yolo-v9-t-640                              | 4  / 15    | 15 / 15
  yolo-v9-s-608                              | 2  / 15    | (untested)
  bespoke YOLOv8m (the trained plate detector) | 0  / 15  | 14 / 15

The wrapper is independent of the underlying detector. It implements
the same ``detect(image) -> PlateDetection | None`` protocol so
:class:`streettracker.analysis.alpr.runner.PipelineRunner` consumes
it unchanged. Bboxes are projected back to original-image coordinates
so the downstream OCR-crop step in the runner still operates on the
original image without modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from streettracker.analysis.alpr.base import PlateDetection

if TYPE_CHECKING:
    import numpy as np

# COCO classes that count as a vehicle for pre-crop purposes.
# 2 = car, 5 = bus, 7 = truck. Bicycle (1) and motorcycle (3) excluded
# -- their bboxes are too small/skinny for plate detection to benefit.
_VEHICLE_COCO_CLASSES = (2, 5, 7)


class PreCropDetector:
    """Wrap a plate detector with a vehicle pre-crop step.

    Lazy-loads the underlying ultralytics YOLO model on first
    ``detect()`` call so importing this module is cheap. Falls back
    to running the underlying plate detector on the full image if
    no vehicle is detected.

    ``pad_frac`` controls how much padding (as a fraction of the
    bbox side length) is added around the vehicle bbox before
    cropping. 0.15 gives enough margin for plates that hang slightly
    over the bumper without blowing up the crop size.

    ``vehicle_conf`` is the YOLO confidence threshold for the
    vehicle stage. Set generously (0.25) since we'd rather over-crop
    than skip a real vehicle and fall back to full-image detection.
    """

    name: str

    def __init__(
        self,
        plate_detector,
        vehicle_model: str | Path = "yolov8n.pt",
        vehicle_conf: float = 0.25,
        pad_frac: float = 0.15,
    ) -> None:
        self._plate_detector = plate_detector
        self._vehicle_model_spec = str(vehicle_model)
        self._vehicle_conf = vehicle_conf
        self._pad_frac = pad_frac
        self._yolo = None
        self.name = f"precrop-{getattr(plate_detector, 'name', 'unknown')}"

    def _ensure_loaded(self) -> None:
        if self._yolo is not None:
            return
        # Deferred import so non-alpr code paths don't pay for ultralytics.
        from ultralytics import YOLO

        self._yolo = YOLO(self._vehicle_model_spec)

    def detect(self, image: np.ndarray) -> PlateDetection | None:
        import numpy as np

        self._ensure_loaded()
        h, w = image.shape[:2]

        # Vehicle detection. ``verbose=False`` suppresses the per-frame
        # ultralytics stdout banner.
        result = self._yolo.predict(
            image,
            classes=list(_VEHICLE_COCO_CLASSES),
            conf=self._vehicle_conf,
            verbose=False,
        )[0]
        boxes = result.boxes
        if len(boxes) == 0:
            # No vehicle found at all. Fall back to running the plate
            # detector on the full image -- preserves the prior
            # behaviour for ambiguous frames (parked-car-only,
            # partial occlusion, etc.).
            return self._plate_detector.detect(image)

        # Iterate vehicle bboxes from largest to smallest area until we
        # get a plate detection. Most images have one car; this
        # multi-vehicle loop matters for frames with a parked car in
        # the foreground + the tracked car distant.
        xy = boxes.xyxy.cpu().numpy()
        areas = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
        order = np.argsort(-areas)
        for i in order:
            x1, y1, x2, y2 = xy[int(i)]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = bw * self._pad_frac, bh * self._pad_frac
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(w, int(x2 + pad_x))
            cy2 = min(h, int(y2 + pad_y))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crop = image[cy1:cy2, cx1:cx2]

            det = self._plate_detector.detect(crop)
            if det is None:
                continue

            # Project the plate bbox from crop-relative back into
            # original-image coordinates so PipelineRunner's
            # crop_with_padding (which slices the original) still
            # frames the plate correctly.
            px1, py1, px2, py2 = det.bbox
            return PlateDetection(
                bbox=(px1 + cx1, py1 + cy1, px2 + cx1, py2 + cy1),
                det_confidence=det.det_confidence,
            )

        # All vehicle crops exhausted with no plate found. Final
        # fallback: the full image. Rare in practice but defensible.
        return self._plate_detector.detect(image)
