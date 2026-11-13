"""The preferred pipeline's detector and recogniser, with the ONNX libraries
stubbed.

``open-image-models``, ``fast-plate-ocr`` and ``onnxruntime`` come with the
``alpr`` extra, which CI doesn't install. Each test puts a stand-in module in
``sys.modules`` (the code imports them inside its functions), so what runs
here is our own wrapper: which providers we ask for, which detection we keep,
what we feed the OCR model and how we turn its output into a ``PlateRead``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from streettracker.analysis.alpr.preferred import (
    FastPlateOcrRecognizer,
    OpenImageModelsDetector,
    _char_confidence,
    _extract_bbox_conf,
    _gpu_first_providers,
    _unpack_ocr_output,
)

cv2 = pytest.importorskip("cv2")

_ALL_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def _install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    """Put a stand-in module called ``name`` in ``sys.modules``."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _onnxruntime(monkeypatch: pytest.MonkeyPatch, available: list[str]) -> None:
    _install(monkeypatch, "onnxruntime", get_available_providers=lambda: list(available))


# ----------------------------------------------------------------------
# Execution providers.


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        # onnxruntime-gpu on the dev box: TensorRT is advertised but unusable,
        # and asking for it prints an error banner on every run.
        (_ALL_PROVIDERS, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        # Whatever order the library lists them in, CUDA is asked for first.
        (
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (["CPUExecutionProvider"], ["CPUExecutionProvider"]),
        (["TensorrtExecutionProvider"], []),
    ],
    ids=["gpu build", "cuda first", "cpu only", "tensorrt alone"],
)
def test_providers_are_cuda_then_cpu_and_never_tensorrt(
    available: list[str], expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _onnxruntime(monkeypatch, available)
    assert _gpu_first_providers() == expected


# ----------------------------------------------------------------------
# Detector.


def _box(x1: int, y1: int, x2: int, y2: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


def _hit(conf: float, box: tuple[int, int, int, int]) -> types.SimpleNamespace:
    """A detection shaped like open-image-models' ``DetectionResult``."""
    return types.SimpleNamespace(confidence=conf, bounding_box=_box(*box))


class _FakeLicensePlateDetector:
    """Stands in for ``open_image_models.LicensePlateDetector``."""

    created: list[_FakeLicensePlateDetector] = []

    def __init__(self, detection_model: str, providers: list[str]) -> None:
        self.detection_model = detection_model
        self.providers = providers
        self.returns: list[Any] = []
        self.seen: list[np.ndarray] = []
        type(self).created.append(self)

    def predict(self, image: np.ndarray) -> list[Any]:
        self.seen.append(image)
        return list(self.returns)


@pytest.fixture
def oim(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLicensePlateDetector]:
    _onnxruntime(monkeypatch, _ALL_PROVIDERS)
    monkeypatch.setattr(_FakeLicensePlateDetector, "created", [])
    _install(monkeypatch, "open_image_models", LicensePlateDetector=_FakeLicensePlateDetector)
    return _FakeLicensePlateDetector


def _detector(
    oim: type[_FakeLicensePlateDetector], returns: list[Any], **kwargs: Any
) -> tuple[OpenImageModelsDetector, _FakeLicensePlateDetector]:
    detector = OpenImageModelsDetector(**kwargs)
    (library,) = oim.created
    library.returns = returns
    return detector, library


_IMAGE = np.zeros((40, 80, 3), dtype=np.uint8)


def test_detector_loads_the_named_model_on_cuda_then_cpu(
    oim: type[_FakeLicensePlateDetector],
) -> None:
    OpenImageModelsDetector("yolo-v9-t-640-license-plate-end2end")
    (library,) = oim.created
    assert library.detection_model == "yolo-v9-t-640-license-plate-end2end"
    assert library.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_detector_keeps_the_most_confident_plate(oim: type[_FakeLicensePlateDetector]) -> None:
    detector, library = _detector(
        oim,
        [_hit(0.40, (1, 2, 3, 4)), _hit(0.90, (10, 20, 30, 40)), _hit(0.60, (5, 6, 7, 8))],
    )
    found = detector.detect(_IMAGE)
    assert found is not None
    assert found.bbox == (10, 20, 30, 40)
    assert found.det_confidence == pytest.approx(0.90)
    assert library.seen[0] is _IMAGE  # the image goes to the model untouched


def test_detector_drops_detections_below_its_threshold(
    oim: type[_FakeLicensePlateDetector],
) -> None:
    detector, _ = _detector(oim, [_hit(0.24, (1, 2, 3, 4))])  # default threshold 0.25
    assert detector.detect(_IMAGE) is None

    detector, _ = _detector_again(oim, [_hit(0.25, (1, 2, 3, 4))])
    found = detector.detect(_IMAGE)
    assert found is not None and found.bbox == (1, 2, 3, 4)  # at the threshold counts

    detector, _ = _detector_again(oim, [_hit(0.5, (1, 2, 3, 4))], det_conf=0.6)
    assert detector.detect(_IMAGE) is None


def _detector_again(
    oim: type[_FakeLicensePlateDetector], returns: list[Any], **kwargs: Any
) -> tuple[OpenImageModelsDetector, _FakeLicensePlateDetector]:
    oim.created.clear()
    return _detector(oim, returns, **kwargs)


def test_detector_returns_none_when_nothing_is_found(oim: type[_FakeLicensePlateDetector]) -> None:
    detector, _ = _detector(oim, [])
    assert detector.detect(_IMAGE) is None


def test_detector_accepts_and_ignores_a_bbox_hint(oim: type[_FakeLicensePlateDetector]) -> None:
    # The crop-targeting wrappers use the hint; this detector must take the
    # argument (the Detector protocol) and search the whole image it is given.
    detector, library = _detector(oim, [_hit(0.8, (1, 2, 3, 4))])
    found = detector.detect(_IMAGE, bbox_hint=(0, 0, 10, 10))
    assert found is not None and found.bbox == (1, 2, 3, 4)
    assert library.seen[0] is _IMAGE


def test_detector_skips_a_detection_with_no_confidence(
    oim: type[_FakeLicensePlateDetector],
) -> None:
    detector, _ = _detector(oim, [(1, 2, 3, 4), _hit(0.3, (5, 6, 7, 8))])
    found = detector.detect(_IMAGE)
    assert found is not None and found.bbox == (5, 6, 7, 8)


# ----------------------------------------------------------------------
# Detection shapes across library versions.


def test_extract_reads_confidence_and_bounding_box() -> None:
    assert _extract_bbox_conf(_hit(0.7, (1, 2, 3, 4))) == ((1, 2, 3, 4), 0.7)


def test_extract_accepts_the_older_field_names() -> None:
    scored = types.SimpleNamespace(score=0.6, bbox=_box(1, 2, 3, 4))
    assert _extract_bbox_conf(scored) == ((1, 2, 3, 4), 0.6)
    boxed = types.SimpleNamespace(confidence=0.5, box=_box(5, 6, 7, 8))
    assert _extract_bbox_conf(boxed) == ((5, 6, 7, 8), 0.5)


def test_extract_truncates_float_coordinates() -> None:
    hit = types.SimpleNamespace(confidence=0.9, bounding_box=_box(1.9, 2.2, 3.7, 4.1))  # type: ignore[arg-type]
    assert _extract_bbox_conf(hit) == ((1, 2, 3, 4), 0.9)


def test_extract_from_a_plain_sequence_has_no_confidence() -> None:
    assert _extract_bbox_conf((1.0, 2.0, 3.0, 4.0, 0.9)) == ((1, 2, 3, 4), None)
    assert _extract_bbox_conf([5, 6, 7, 8]) == ((5, 6, 7, 8), None)


def test_extract_falls_back_to_an_empty_box() -> None:
    assert _extract_bbox_conf(types.SimpleNamespace(confidence=0.4)) == ((0, 0, 0, 0), 0.4)
    half = types.SimpleNamespace(confidence=0.4, bounding_box=types.SimpleNamespace(x1=1, y1=2))
    assert _extract_bbox_conf(half) == ((0, 0, 0, 0), 0.4)
    assert _extract_bbox_conf(object()) == ((0, 0, 0, 0), None)


# ----------------------------------------------------------------------
# Recogniser.


@dataclass(frozen=True, slots=True)
class _Prediction:
    """The fields of fast-plate-ocr's ``PlatePrediction`` that we read."""

    plate: str
    char_probs: np.ndarray | None = None


def _read(plate: str, per_slot: list[float]) -> list[_Prediction]:
    return [_Prediction(plate, np.asarray(per_slot, dtype=np.float32))]


class _FakeOcr:
    """Stands in for ``fast_plate_ocr.LicensePlateRecognizer``. The first call
    is the recogniser's start-up self-check; later calls return ``returns``."""

    created: list[_FakeOcr] = []
    self_check_output: Any = None

    def __init__(self, model: str, providers: list[str]) -> None:
        self.model = model
        self.providers = providers
        self.returns: Any = _read("", [1.0] * 9)
        self.calls: list[tuple[np.ndarray, bool]] = []
        type(self).created.append(self)

    def run(self, image: np.ndarray, return_confidence: bool = False) -> Any:
        self.calls.append((image, return_confidence))
        if len(self.calls) == 1:
            check = type(self).self_check_output
            return check if check is not None else _read("", [1.0] * 9)
        return self.returns


@pytest.fixture
def ocr(monkeypatch: pytest.MonkeyPatch) -> type[_FakeOcr]:
    _onnxruntime(monkeypatch, _ALL_PROVIDERS)
    monkeypatch.setattr(_FakeOcr, "created", [])
    monkeypatch.setattr(_FakeOcr, "self_check_output", None)
    _install(monkeypatch, "fast_plate_ocr", LicensePlateRecognizer=_FakeOcr)
    return _FakeOcr


def _recognizer(ocr: type[_FakeOcr], returns: Any) -> tuple[FastPlateOcrRecognizer, _FakeOcr]:
    recognizer = FastPlateOcrRecognizer()
    (library,) = ocr.created
    library.returns = returns
    return recognizer, library


def test_recognizer_loads_the_named_model_on_cuda_then_cpu(ocr: type[_FakeOcr]) -> None:
    FastPlateOcrRecognizer("global-plates-mobile-vit-v2-model")
    (library,) = ocr.created
    assert library.model == "global-plates-mobile-vit-v2-model"
    assert library.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_recognizer_checks_the_output_format_at_start_up(ocr: type[_FakeOcr]) -> None:
    FastPlateOcrRecognizer()
    (library,) = ocr.created
    ((image, return_confidence),) = library.calls
    assert image.shape == (64, 128) and image.dtype == np.uint8  # one grayscale frame
    assert return_confidence is True


def test_recognizer_refuses_to_start_on_an_unknown_output_format(
    ocr: type[_FakeOcr], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that returns per-slot, per-symbol probabilities must fail when
    the recogniser is built, not hours into a run with a wrong confidence."""
    wrong = [_Prediction("", np.full((9, 37), 0.5, dtype=np.float32))]
    monkeypatch.setattr(_FakeOcr, "self_check_output", wrong)
    with pytest.raises(ValueError, match="not supported"):
        FastPlateOcrRecognizer()


def test_recognizer_falls_back_to_the_legacy_class_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _onnxruntime(monkeypatch, _ALL_PROVIDERS)
    monkeypatch.setattr(_FakeOcr, "created", [])
    monkeypatch.setattr(_FakeOcr, "self_check_output", None)
    _install(monkeypatch, "fast_plate_ocr", ONNXPlateRecognizer=_FakeOcr)
    FastPlateOcrRecognizer("some-model")
    assert [library.model for library in _FakeOcr.created] == ["some-model"]


def test_recognizer_reads_a_colour_crop_as_grayscale(ocr: type[_FakeOcr]) -> None:
    recognizer, library = _recognizer(
        ocr, _read("AB12CDE", [0.99, 0.98, 0.97, 0.50, 0.96, 0.95, 0.94, 1.0, 1.0])
    )
    crop = np.zeros((20, 60, 3), dtype=np.uint8)
    crop[:, :, 1] = 200  # pure green: gray = 0.587 * 200
    read = recognizer.recognize(crop)

    image, return_confidence = library.calls[-1]
    # The global model takes one channel; a 3-channel array would be a shape
    # error inside ONNX Runtime.
    assert image.shape == (20, 60)
    assert int(image[0, 0]) == pytest.approx(117, abs=1)
    assert return_confidence is True

    assert read is not None
    assert read.text == "AB12CDE"
    assert read.raw_text == "AB12CDE"
    assert read.ocr_confidence == pytest.approx(0.50, abs=1e-4)  # the weakest character
    assert read.char_probs == pytest.approx([0.99, 0.98, 0.97, 0.50, 0.96, 0.95, 0.94], abs=1e-4)


def test_recognizer_passes_a_grayscale_crop_through(ocr: type[_FakeOcr]) -> None:
    recognizer, library = _recognizer(ocr, _read("AB12CDE", [0.9] * 9))
    crop = np.full((20, 60), 128, dtype=np.uint8)
    assert recognizer.recognize(crop) is not None
    assert library.calls[-1][0] is crop


def test_recognizer_normalises_the_text_and_keeps_the_raw_read(ocr: type[_FakeOcr]) -> None:
    recognizer, _ = _recognizer(ocr, _read("ab12-cde", [0.9] * 9))
    read = recognizer.recognize(np.zeros((20, 60, 3), dtype=np.uint8))
    assert read is not None
    assert read.text == "AB12CDE"
    assert read.raw_text == "ab12-cde"
    # One probability per raw character, the separator included.
    assert read.char_probs is not None and len(read.char_probs) == 8


def test_recognizer_returns_none_for_an_empty_crop(ocr: type[_FakeOcr]) -> None:
    recognizer, library = _recognizer(ocr, _read("AB12CDE", [0.9] * 9))
    assert recognizer.recognize(np.zeros((0, 60, 3), dtype=np.uint8)) is None
    assert len(library.calls) == 1  # only the start-up check: the model was not run


def test_recognizer_returns_none_when_the_model_reads_nothing(ocr: type[_FakeOcr]) -> None:
    recognizer, _ = _recognizer(ocr, _read("", [1.0] * 9))
    assert recognizer.recognize(np.zeros((20, 60, 3), dtype=np.uint8)) is None


# ----------------------------------------------------------------------
# Output shapes not covered in test_preferred.py.


def test_unpack_a_list_of_text_confidence_pairs() -> None:
    assert _unpack_ocr_output([("AB12CDE", 0.7)]) == ("AB12CDE", 0.7, None)


def test_unpack_an_unknown_output_reads_as_nothing() -> None:
    assert _unpack_ocr_output(None) == ("", 0.0, None)
    assert _unpack_ocr_output([object()]) == ("", 0.0, None)


def test_char_confidence_of_no_text_is_zero() -> None:
    assert _char_confidence("", np.array([0.9, 0.8])) == (0.0, [])


def test_char_confidence_rounds_to_four_places() -> None:
    conf, probs = _char_confidence("AB", [0.123456, 0.987654, 0.5])
    assert probs == [0.1235, 0.9877]
    assert conf == 0.1235
