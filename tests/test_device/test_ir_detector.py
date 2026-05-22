"""Coverage for ``streettracker.device.ir_detector``.

The detector is pure-Python plus numpy; no fixtures, no I/O. Frames
here are tiny synthetic arrays just large enough to exercise the
stride sampler.
"""

from __future__ import annotations

import numpy as np
import pytest

from streettracker.common.schema import IRPeriod
from streettracker.device.ir_detector import IRDetector, is_ir_frame

# ----------------------------------------------------------------------
# Synthetic-frame helpers.

# 64x64 keeps stride sampling sane: at stride=16 there are 4x4=16
# sampled pixels, enough for "max" reductions to behave.
_H, _W = 64, 64

# Base unix timestamp used by all per-frame test schedules. Windows'
# datetime.fromtimestamp() rejects values close to the epoch, so tests
# build wall_times as _T0 + offset_seconds rather than starting at 0.0.
# Pinned to 2026-05-22T12:00:00 UTC for determinism in the formatted
# output that does land in IRPeriod fields.
_T0 = 1_779_792_000.0


def _ir_frame(value: int = 120) -> np.ndarray:
    """Uniform-grayscale BGR frame: R=G=B=value (perfect IR)."""
    return np.full((_H, _W, 3), value, dtype=np.uint8)


def _day_frame() -> np.ndarray:
    """Saturated-colour BGR frame, definitely not IR."""
    f = np.zeros((_H, _W, 3), dtype=np.uint8)
    f[:, :, 0] = 0  # B
    f[:, :, 1] = 128  # G
    f[:, :, 2] = 255  # R
    return f


# ----------------------------------------------------------------------
# is_ir_frame()


def test_is_ir_frame_true_on_uniform_gray() -> None:
    assert is_ir_frame(_ir_frame(80)) is True
    assert is_ir_frame(_ir_frame(0)) is True
    assert is_ir_frame(_ir_frame(255)) is True


def test_is_ir_frame_false_on_saturated_color() -> None:
    assert is_ir_frame(_day_frame()) is False


def test_is_ir_frame_false_when_one_channel_drifts_past_threshold() -> None:
    f = _ir_frame(100)
    f[0, 0, 2] = 100 + 9  # default threshold is 8
    assert is_ir_frame(f) is False


def test_is_ir_frame_true_when_drift_within_threshold() -> None:
    f = _ir_frame(100)
    f[0, 0, 2] = 100 + 8  # exactly at threshold
    assert is_ir_frame(f) is True


def test_is_ir_frame_threshold_parameter_loosens_check() -> None:
    f = _ir_frame(100)
    f[0, 0, 2] = 100 + 20  # past default 8
    assert is_ir_frame(f) is False
    assert is_ir_frame(f, channel_diff_threshold=25) is True


def test_is_ir_frame_stride_skips_offending_pixel() -> None:
    f = _ir_frame(100)
    # Drift only at pixel (1, 1) — stride=16 starts at (0, 0) so this
    # pixel is *not* sampled and the frame still reads as IR.
    f[1, 1, 2] = 200
    assert is_ir_frame(f, sample_stride=16) is True
    # Stride=1 samples every pixel including (1, 1) — now sees it.
    assert is_ir_frame(f, sample_stride=1) is False


def test_is_ir_frame_rejects_non_3channel_inputs() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        is_ir_frame(np.zeros((_H, _W), dtype=np.uint8))
    with pytest.raises(ValueError, match="HxWx3"):
        is_ir_frame(np.zeros((_H, _W, 4), dtype=np.uint8))


# ----------------------------------------------------------------------
# IRDetector — initial state + warm-up.


def test_fresh_detector_starts_in_day_mode() -> None:
    det = IRDetector()
    assert det.in_ir_mode is False


def test_no_transition_during_warmup_even_if_all_ir() -> None:
    """A detector that opens its eyes mid-night spends one window in
    day mode before flipping. This guards against a startup glitch
    where the camera reports IR before YOLO has loaded."""
    det = IRDetector(hysteresis_frames=5)
    for i in range(4):
        assert det.update(_ir_frame(), wall_time=_T0 + float(i)) is None
        assert det.in_ir_mode is False
    # 5th frame fills the history; transition fires.
    assert det.update(_ir_frame(), wall_time=_T0 + 4.0) is None
    assert det.in_ir_mode is True


# ----------------------------------------------------------------------
# Hysteresis — transitions only fire on consecutive readings.


def test_day_to_ir_requires_full_window() -> None:
    det = IRDetector(hysteresis_frames=5)
    # 4 IR + 1 day: not all IR, stay in day.
    for i in range(4):
        det.update(_ir_frame(), wall_time=_T0 + float(i))
    det.update(_day_frame(), wall_time=_T0 + 4.0)
    assert det.in_ir_mode is False
    # 5 IR in a row: flip.
    for i in range(5, 10):
        det.update(_ir_frame(), wall_time=_T0 + float(i))
    assert det.in_ir_mode is True


def test_hysteresis_flap_protection_in_day_mode() -> None:
    """29 IR + 1 day still stays in day mode (the canonical test from
    NanoTracker's regression notes)."""
    det = IRDetector(hysteresis_frames=30)
    for i in range(29):
        det.update(_ir_frame(), wall_time=_T0 + float(i))
    det.update(_day_frame(), wall_time=_T0 + 29.0)
    assert det.in_ir_mode is False


def test_ir_to_day_requires_full_window() -> None:
    det = IRDetector(hysteresis_frames=5)
    # Drive into IR.
    for i in range(5):
        det.update(_ir_frame(), wall_time=_T0 + float(i))
    assert det.in_ir_mode is True
    # 4 day + 1 IR: not none-IR, stay in IR.
    for i in range(5, 9):
        det.update(_day_frame(), wall_time=_T0 + float(i))
    det.update(_ir_frame(), wall_time=_T0 + 9.0)
    assert det.in_ir_mode is True
    # 5 day in a row: flip back.
    for i in range(10, 15):
        det.update(_day_frame(), wall_time=_T0 + float(i))
    assert det.in_ir_mode is False


# ----------------------------------------------------------------------
# IRPeriod emission semantics.


def test_no_period_emitted_on_day_to_ir_transition() -> None:
    """Only the closing transition emits the period. Entering IR is
    detected via the in_ir_mode property by the runtime."""
    det = IRDetector(hysteresis_frames=3)
    for i in range(2):
        assert det.update(_ir_frame(), wall_time=_T0 + float(i)) is None
    # 3rd frame closes the window and flips to IR — no period yet.
    result = det.update(_ir_frame(), wall_time=_T0 + 2.0)
    assert result is None
    assert det.in_ir_mode is True


def test_period_emitted_on_ir_to_day_with_correct_bounds() -> None:
    det = IRDetector(hysteresis_frames=3)
    # Enter IR at wall_time=2.0 (the frame that closes the IR window).
    for i in range(3):
        det.update(_ir_frame(), wall_time=_T0 + float(i))
    assert det.in_ir_mode is True
    # Leave IR at wall_time=12.0 (the frame that closes the day window).
    for i in range(10, 12):
        assert det.update(_day_frame(), wall_time=_T0 + float(i)) is None
    period = det.update(_day_frame(), wall_time=_T0 + 12.0)
    assert isinstance(period, IRPeriod)
    assert period.duration_s == pytest.approx(10.0)
    # ISO timestamps — the exact prefix depends on the local timezone,
    # but the formats should at least be self-consistent strings.
    assert isinstance(period.start, str) and "T" in period.start
    assert isinstance(period.end, str) and "T" in period.end


def test_detector_resets_after_period_emission() -> None:
    """After a full day -> IR -> day cycle, the detector should be
    ready to detect the next IR period without leaking state."""
    det = IRDetector(hysteresis_frames=3)
    for i in range(3):
        det.update(_ir_frame(), wall_time=_T0 + float(i))
    for i in range(10, 13):
        det.update(_day_frame(), wall_time=_T0 + float(i))
    assert det.in_ir_mode is False
    # Second IR period: should fire again.
    for i in range(20, 23):
        det.update(_ir_frame(), wall_time=_T0 + float(i))
    assert det.in_ir_mode is True
    period = None
    for i in range(30, 33):
        period = det.update(_day_frame(), wall_time=_T0 + float(i))
    assert isinstance(period, IRPeriod)
    assert period.duration_s == pytest.approx(10.0)


def test_update_propagates_invalid_frame_shape() -> None:
    det = IRDetector()
    with pytest.raises(ValueError, match="HxWx3"):
        det.update(np.zeros((_H, _W), dtype=np.uint8), wall_time=_T0)
