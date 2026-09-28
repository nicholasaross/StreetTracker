"""Preferred pipeline: ``open-image-models`` (ONNX plate detection) +
``fast-plate-ocr`` (ONNX recognition).

Both are by ankandrew (https://github.com/ankandrew) and share a
lightweight ONNX-only posture. No torch dep — onnxruntime backs both.

Ported from NanoTracker's ``alpr/pipelines/preferred.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from streettracker.analysis.alpr.base import (
    PlateDetection,
    PlateRead,
    normalize_plate_text,
)

if TYPE_CHECKING:
    import numpy as np


def _gpu_first_providers() -> list[str]:
    """ONNX execution providers, CUDA-first, with TensorRT excluded.

    onnxruntime-gpu advertises ``TensorrtExecutionProvider``, so a
    session created with the default provider list tries TensorRT first,
    fails (the dev box has no TensorRT plugin libs), and onnxruntime
    prints a multi-line ``*** EP Error ***`` banner before falling back
    to CUDA on *every* alpr-run. It's harmless -- the run still lands on
    the GPU -- but it reads like a crash and tripped the control panel's
    issue detector. Requesting only ``[CUDA, CPU]`` (filtered to what's
    actually compiled in) keeps GPU acceleration, drops the banner, and
    still degrades cleanly to CPU on a CPU-only host (e.g. the Orin,
    which has no GPU onnxruntime and never runs ALPR anyway).
    """
    import onnxruntime as ort  # type: ignore[import-untyped]

    available = set(ort.get_available_providers())
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]


class OpenImageModelsDetector:
    name = "oim"

    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        det_conf: float = 0.25,
    ) -> None:
        from open_image_models import LicensePlateDetector

        self._detector = LicensePlateDetector(
            detection_model=detector_model, providers=_gpu_first_providers()
        )
        self._det_conf = det_conf

    def detect(
        self,
        image: np.ndarray,
        *,
        bbox_hint: tuple[int, int, int, int] | None = None,
    ) -> PlateDetection | None:
        # Accept ``bbox_hint`` for protocol compliance with the
        # PreCrop-aware Detector interface; this detector ignores it
        # (the wrapper consults it).
        del bbox_hint
        detections = self._detector.predict(image)
        if not detections:
            return None
        best: PlateDetection | None = None
        for d in detections:
            bbox, conf = _extract_bbox_conf(d)
            if conf is None or conf < self._det_conf:
                continue
            if best is None or conf > best.det_confidence:
                best = PlateDetection(bbox=bbox, det_confidence=conf)
        return best


class FastPlateOcrRecognizer:
    name = "fast-plate-ocr"

    def __init__(
        self,
        ocr_model: str = "global-plates-mobile-vit-v2-model",
    ) -> None:
        # Class name shifted between versions; try the current name first, then legacy.
        try:
            from fast_plate_ocr import (
                LicensePlateRecognizer as _Recognizer,  # type: ignore[attr-defined]
            )
        except ImportError:
            from fast_plate_ocr import (
                ONNXPlateRecognizer as _Recognizer,  # type: ignore[attr-defined,no-redef]
            )
        # Pin providers (CUDA-first, no TensorRT) as for the detector.
        # fast-plate-ocr is pinned >=0.3, whose LicensePlateRecognizer
        # takes ``providers``; the legacy ONNXPlateRecognizer branch
        # above only trips on much older builds the pin excludes.
        self._recognizer = _Recognizer(ocr_model, providers=_gpu_first_providers())
        # Fail at start-up, not hours into a run, if this fast-plate-ocr
        # build returns confidences in a shape _unpack_ocr_output can't
        # interpret (it raises rather than guessing).
        import numpy as np

        _unpack_ocr_output(
            self._recognizer.run(np.zeros((64, 128), np.uint8), return_confidence=True)
        )

    def recognize(self, crop_bgr: np.ndarray) -> PlateRead | None:
        import cv2

        if crop_bgr.size == 0:
            return None
        # fast-plate-ocr's global model is single-channel: per its
        # docstring, in-memory ndarrays "are assumed to already use the
        # expected color mode", so a 3-channel BGR crop raises an
        # ONNXRuntime shape error.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        out = self._recognizer.run(gray, return_confidence=True)
        raw, conf, char_probs = _unpack_ocr_output(out)
        if not raw:
            return None
        return PlateRead(
            text=normalize_plate_text(raw),
            ocr_confidence=conf,
            raw_text=raw,
            char_probs=char_probs,
        )


def _extract_bbox_conf(
    detection: object,
) -> tuple[tuple[int, int, int, int], float | None]:
    conf = getattr(detection, "confidence", None)
    if conf is None:
        conf = getattr(detection, "score", None)
    conf = float(conf) if conf is not None else None

    bb = (
        getattr(detection, "bounding_box", None)
        or getattr(detection, "bbox", None)
        or getattr(detection, "box", None)
    )
    if bb is not None and all(hasattr(bb, attr) for attr in ("x1", "y1", "x2", "y2")):
        bbox = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2))
    elif isinstance(detection, (tuple, list)) and len(detection) >= 4:
        bbox = tuple(int(v) for v in detection[:4])  # type: ignore[assignment]
    else:
        bbox = (0, 0, 0, 0)
    return bbox, conf


def _unpack_ocr_output(out: object) -> tuple[str, float, list[float] | None]:
    """Decoded text, read confidence and per-character probabilities.

    The confidence is the probability of the read's LEAST certain
    character (the min over the decoded characters). A plate is only as
    right as its worst character, and one uncertain character is exactly
    the one-slip misread that can resolve to a different real car on the
    DVSA register. Padding slots past the decoded text are left out: they
    are "no character here" predictions and say nothing about the plate.

    fast-plate-ocr 1.1.x returns ``list[PlatePrediction]`` whose
    ``char_probs`` is already the per-slot max probability, shape
    ``(max_plate_slots,)``. Until 2026-09-28 this function took a second
    max over that vector, so ``ocr_conf`` was the single most confident
    slot (~1.0) for every read, garbage included, and every downstream
    ``conf >= 0.9`` gate passed everything. A ``char_probs`` of any other
    shape now raises instead of being coerced.

    Older outputs are still accepted: ``(list[str], probs)`` with
    ``probs`` of shape ``(N, max_plate_slots)``, and text-only outputs
    (confidence 0.0, no per-character probabilities).
    """
    import numpy as np

    if isinstance(out, str):
        return out, 0.0, None
    if isinstance(out, list) and out:
        first = out[0]
        # v1.1.x: PlatePrediction dataclass
        plate = getattr(first, "plate", None)
        if plate is not None:
            text = str(plate)
            char_probs = getattr(first, "char_probs", None)
            if char_probs is None:
                return text, 0.0, None
            conf, probs = _char_confidence(text, char_probs)
            return text, conf, probs
        if isinstance(first, str):
            return first, 0.0, None
        if isinstance(first, tuple) and len(first) == 2:
            return str(first[0]), float(first[1]), None
    if isinstance(out, tuple) and len(out) == 2:
        texts, conf_obj = out
        first_text = (texts[0] if texts else "") if isinstance(texts, list) else texts
        text = str(first_text)
        if hasattr(conf_obj, "__iter__"):
            arr = np.asarray(conf_obj, dtype=float)
            if arr.ndim == 2:  # (N, max_plate_slots): this image's row
                arr = arr[0]
            conf, probs = _char_confidence(text, arr)
            return text, conf, probs
        return text, float(conf_obj), None  # type: ignore[arg-type]
    return "", 0.0, None


def _char_confidence(text: str, per_slot: object) -> tuple[float, list[float]]:
    """``(min, probs)`` over the per-slot probabilities of ``text``'s
    characters. ``per_slot`` must be 1-D, one probability per slot, with
    the decoded text occupying the leading slots."""
    import numpy as np

    arr = np.asarray(per_slot, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"expected per-slot OCR probabilities of shape (slots,), got {arr.shape}; "
            f"this fast-plate-ocr version's output format is not supported"
        )
    if len(text) > arr.shape[0]:
        raise ValueError(f"OCR text {text!r} is longer than its {arr.shape[0]} probability slots")
    probs = [round(float(p), 4) for p in arr[: len(text)]]
    return (min(probs) if probs else 0.0), probs
