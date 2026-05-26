"""``PipelineRunner`` composes a ``Detector`` + ``Recognizer`` into a uniform
``Pipeline``.

This is the seam that lets the ablation config (bespoke detector +
fast-plate-ocr OCR) exist as a one-liner without duplicating the load /
crop / save / time scaffolding.

Ported from NanoTracker's ``alpr/pipelines/runner.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from streettracker.analysis.alpr.base import (
    PlateDetection,
    PlateRead,
    PlateResult,
    Timer,
    atomic_write_bytes,
    crop_with_padding,
)

if TYPE_CHECKING:
    import numpy as np


class Detector(Protocol):
    name: str

    def detect(
        self,
        image: np.ndarray,
        *,
        bbox_hint: tuple[int, int, int, int] | None = None,
    ) -> PlateDetection | None:
        """Detect a plate. Optional ``bbox_hint`` gives the
        tracked-vehicle bbox in *image*'s pixel coords; detectors that
        know how to use it (e.g. :class:`PreCropDetector`) skip their
        own vehicle-detection step. Detectors that don't care must
        accept the kwarg and ignore it.
        """
        ...


class Recognizer(Protocol):
    name: str

    def recognize(self, crop_bgr: np.ndarray) -> PlateRead | None: ...


class PipelineRunner:
    """Detector + Recognizer + I/O scaffolding under a single name."""

    def __init__(self, name: str, detector: Detector, recognizer: Recognizer) -> None:
        self.name = name
        self._detector = detector
        self._recognizer = recognizer

    def run(
        self,
        image_path: Path,
        track_id: int,
        snap_index: int,
        class_name: str,
        crop_out_dir: Path,
        *,
        bbox_hint: tuple[int, int, int, int] | None = None,
        ghost_rects: list[tuple[int, int, int, int]] | None = None,
        ghost_source_size: tuple[int, int] | None = None,
    ) -> PlateResult:
        import cv2  # deferred — pipelines can import this module without cv2

        image_name = image_path.name
        with Timer() as t:
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    return PlateResult(
                        image_path=str(image_path), image_name=image_name,
                        track_id=track_id, snap_index=snap_index, class_name=class_name,
                        pipeline=self.name, detection=None, read=None, crop_path=None,
                        pipeline_ms=0.0, error="imread_failed",
                    )

                # Zero out parked-car / no-go regions before any detection
                # runs. Rects are in the operator-supplied source_size
                # coord system; scale to the actual snap dims (a Reolink
                # firmware change could move from 4512x2512 to a different
                # resolution without invalidating the mask). When
                # source_size is unknown, assume the rects are already in
                # snap pixel coords.
                if ghost_rects:
                    h_img, w_img = image.shape[:2]
                    if ghost_source_size:
                        src_w, src_h = ghost_source_size
                        sx = w_img / src_w if src_w > 0 else 1.0
                        sy = h_img / src_h if src_h > 0 else 1.0
                    else:
                        sx = sy = 1.0
                    for x1, y1, x2, y2 in ghost_rects:
                        rx1 = max(0, min(int(x1 * sx), w_img))
                        ry1 = max(0, min(int(y1 * sy), h_img))
                        rx2 = max(0, min(int(x2 * sx), w_img))
                        ry2 = max(0, min(int(y2 * sy), h_img))
                        if rx2 > rx1 and ry2 > ry1:
                            image[ry1:ry2, rx1:rx2] = 0

                detection = self._detector.detect(image, bbox_hint=bbox_hint)
                if detection is None:
                    return PlateResult(
                        image_path=str(image_path), image_name=image_name,
                        track_id=track_id, snap_index=snap_index, class_name=class_name,
                        pipeline=self.name, detection=None, read=None, crop_path=None,
                        pipeline_ms=t.ms, error=None,
                    )

                crop = crop_with_padding(image, detection.bbox, pad_frac=0.10)
                crop_path = _save_crop(crop, crop_out_dir, image_name)
                read = self._recognizer.recognize(crop)
            except Exception as e:  # noqa: BLE001 — surfaced into PlateResult.error
                return PlateResult(
                    image_path=str(image_path), image_name=image_name,
                    track_id=track_id, snap_index=snap_index, class_name=class_name,
                    pipeline=self.name, detection=None, read=None, crop_path=None,
                    pipeline_ms=t.ms, error=f"{type(e).__name__}: {e}",
                )

        return PlateResult(
            image_path=str(image_path), image_name=image_name,
            track_id=track_id, snap_index=snap_index, class_name=class_name,
            pipeline=self.name, detection=detection, read=read, crop_path=crop_path,
            pipeline_ms=t.ms, error=None,
        )


def _save_crop(crop_bgr: np.ndarray, out_dir: Path, image_name: str) -> str:
    import cv2

    if crop_bgr.size == 0:
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / image_name
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    atomic_write_bytes(out_path, buf.tobytes())
    return str(out_path)
