"""OCR-output unpacking in ``streettracker.analysis.alpr.preferred``.

fast-plate-ocr isn't installed on CI, so these drive
``_unpack_ocr_output`` with a stand-in for its 1.1.0 ``PlatePrediction``:
same fields, and ``char_probs`` in the shape the library returns --
already the per-slot max probability, ``(max_plate_slots,)``, pad slots
included (``core/process.py``: ``char_probs = np.max(predictions,
axis=-1)``). The global model has 9 slots and pads with ``_``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from streettracker.analysis.alpr.preferred import _unpack_ocr_output


@dataclass(frozen=True, slots=True)
class PlatePrediction:
    """Field-for-field copy of ``fast_plate_ocr.core.types.PlatePrediction``."""

    plate: str
    char_probs: np.ndarray | None = None
    region: str | None = None
    region_prob: float | None = None


def _pred(plate: str, per_slot: list[float]) -> list[PlatePrediction]:
    return [PlatePrediction(plate=plate, char_probs=np.asarray(per_slot, dtype=np.float32))]


class TestFastPlateOcr11:
    def test_confidence_is_the_weakest_decoded_character(self) -> None:
        # 7 characters + 2 near-certain pad slots; one character at 0.35.
        out = _pred("AB12CDE", [0.99, 0.35, 0.99, 0.98, 0.97, 0.99, 0.96, 1.0, 1.0])
        text, conf, probs = _unpack_ocr_output(out)
        assert text == "AB12CDE"
        assert conf == pytest.approx(0.35, abs=1e-4)
        assert probs is not None and len(probs) == 7  # pad slots excluded

    def test_garbage_read_no_longer_scores_near_one(self) -> None:
        # Regression for the 2026-09-28 bug: the old code returned the MAX
        # over slots (a pad slot at ~1.0) -- 0.96 for this smeared misread,
        # measured on the real model -- so every conf >= 0.9 gate passed it.
        out = _pred("AE00012", [0.96, 0.17, 0.23, 0.31, 0.42, 0.25, 0.21, 0.96, 0.95])
        _text, conf, _probs = _unpack_ocr_output(out)
        assert conf < 0.2

    def test_clean_read_keeps_a_high_confidence(self) -> None:
        out = _pred("AB12CDE", [0.98, 0.98, 0.97, 0.96, 0.96, 0.96, 0.94, 0.96, 0.96])
        _text, conf, probs = _unpack_ocr_output(out)
        assert conf == pytest.approx(0.94, abs=1e-4)
        assert probs == pytest.approx([0.98, 0.98, 0.97, 0.96, 0.96, 0.96, 0.94], abs=1e-4)

    def test_pad_slots_never_lower_the_confidence(self) -> None:
        # A shaky pad slot past the text says nothing about the characters.
        out = _pred("AB12CD", [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.40, 0.30, 0.20])
        _text, conf, probs = _unpack_ocr_output(out)
        assert conf == pytest.approx(0.99, abs=1e-4)
        assert probs is not None and len(probs) == 6

    def test_empty_plate(self) -> None:
        text, conf, probs = _unpack_ocr_output(_pred("", [1.0] * 9))
        assert (text, conf, probs) == ("", 0.0, [])

    def test_no_char_probs(self) -> None:
        assert _unpack_ocr_output([PlatePrediction(plate="AB12CDE")]) == ("AB12CDE", 0.0, None)

    def test_unexpected_shape_raises_instead_of_guessing(self) -> None:
        # (slots, vocab) -- what the old code assumed -- is not what 1.1.0
        # returns; an unknown shape must fail loudly, not be coerced.
        probs = np.full((9, 37), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            _unpack_ocr_output([PlatePrediction(plate="AB12CDE", char_probs=probs)])

    def test_text_longer_than_slots_raises(self) -> None:
        with pytest.raises(ValueError, match="longer"):
            _unpack_ocr_output(_pred("AB12CDEFG", [0.9] * 7))


class TestLegacyShapes:
    def test_tuple_with_per_image_slot_rows(self) -> None:
        # Pre-1.0 API: (list[str], probs) with probs shaped (N, slots).
        probs = np.array([[0.9, 0.8, 0.95, 1.0, 1.0]])
        text, conf, char_probs = _unpack_ocr_output((["AB1"], probs))
        assert text == "AB1"
        assert conf == pytest.approx(0.8)
        assert char_probs == pytest.approx([0.9, 0.8, 0.95])

    def test_tuple_with_scalar_confidence(self) -> None:
        assert _unpack_ocr_output(("AB12CDE", 0.7)) == ("AB12CDE", 0.7, None)

    def test_text_only_outputs(self) -> None:
        assert _unpack_ocr_output("AB12CDE") == ("AB12CDE", 0.0, None)
        assert _unpack_ocr_output(["AB12CDE"]) == ("AB12CDE", 0.0, None)
        assert _unpack_ocr_output([]) == ("", 0.0, None)
