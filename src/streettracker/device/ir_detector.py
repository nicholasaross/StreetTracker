"""IR / night-mode detector for the live runtime.

Reolink (and most outdoor IP cameras) switches to IR LEDs at night,
producing a monochrome image where R, G, and B are equal up to sensor
noise. YOLO trained on colour images detects much less reliably under
IR, and colour voting becomes meaningless. Rather than waste inference
budget and pollute the session JSON with low-quality IR-period entries,
the runtime detects this state per frame and skips ``yolo.infer()``
entirely while in IR mode. Decode keeps running so the RTSP buffer
drains and we can detect the day-mode transition.

The detector itself is stateless on a per-frame basis aside from a
small rolling history; it doesn't touch tracks or output. The runtime
loop is responsible for:

* skipping inference while :attr:`IRDetector.in_ir_mode` is ``True``,
* flushing active tracks on the day -> IR transition (so the boundary
  is a clean cut in the output), and
* appending the closed :class:`~streettracker.common.schema.IRPeriod`
  returned by :meth:`IRDetector.update` to ``SessionMeta.ir_periods``.

Ported from NanoTracker's ``is_ir_frame`` (nano_tracker.py:295-314) and
the hysteresis state machine at nano_tracker.py:1419-1444.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from streettracker.common.output import format_wall
from streettracker.common.schema import IRPeriod

if TYPE_CHECKING:
    import numpy as np


# Defaults match NanoTracker. The threshold may need calibration if the
# camera outputs tinted IR rather than pure mono (some Reolinks have a
# faint greenish cast); raise it to be more tolerant.
_DEFAULT_CHANNEL_DIFF_THRESHOLD = 8
_DEFAULT_SAMPLE_STRIDE = 16
_DEFAULT_HYSTERESIS_FRAMES = 30


def is_ir_frame(
    frame: np.ndarray,
    *,
    channel_diff_threshold: int = _DEFAULT_CHANNEL_DIFF_THRESHOLD,
    sample_stride: int = _DEFAULT_SAMPLE_STRIDE,
) -> bool:
    """Return ``True`` if ``frame`` looks like monochrome IR (R ~= G ~= B).

    Stride-samples the frame for speed -- the whole check is
    sub-millisecond on a Jetson Orin at 1080p. Channel layout (BGR vs
    RGB) doesn't matter: we look at the differences between adjacent
    channel pairs, which are zero on a true monochrome frame regardless
    of axis order.

    Args:
        frame: ``H x W x 3`` uint8 image (OpenCV BGR is the typical
            input). Must be 3-channel; 2D / 4-channel inputs raise
            ``ValueError`` to fail fast on a misconfigured pipeline
            rather than silently classifying every frame as IR.
        channel_diff_threshold: maximum per-pixel ``|c_i - c_j|``
            across the strided sample for the frame to count as IR.
        sample_stride: take every Nth row and column. Larger strides
            are faster but more likely to miss small coloured patches
            (e.g. a single brake-light pixel) on otherwise IR frames.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"is_ir_frame expects HxWx3 image, got shape {frame.shape!r}")
    import numpy as np  # local import keeps module import cheap

    s = frame[::sample_stride, ::sample_stride].astype(np.int16)
    diff_01 = int(np.abs(s[:, :, 1] - s[:, :, 0]).max())
    diff_12 = int(np.abs(s[:, :, 2] - s[:, :, 1]).max())
    return diff_01 <= channel_diff_threshold and diff_12 <= channel_diff_threshold


@dataclass(slots=True)
class IRDetector:
    """Stateful IR/day-mode detector with hysteresis.

    Hysteresis prevents single-frame flips around dusk / dawn: a state
    change requires ``hysteresis_frames`` consecutive readings in the
    new state. ``29 IR + 1 day`` stays in day mode; ``30 IR`` flips to
    IR. Same in the reverse direction.

    Construction sets day mode. The first hysteresis-window worth of
    frames is treated as a warm-up period -- no transition can fire
    until at least ``hysteresis_frames`` updates have been seen, so a
    detector that opens its eyes mid-night still spends one window in
    day mode before transitioning.

    :meth:`update` returns a closed :class:`IRPeriod` on the IR -> day
    transition (the closing event) and ``None`` otherwise. To detect
    the day -> IR transition (so the runtime can flush active tracks),
    callers should diff :attr:`in_ir_mode` across calls themselves.
    """

    channel_diff_threshold: int = _DEFAULT_CHANNEL_DIFF_THRESHOLD
    sample_stride: int = _DEFAULT_SAMPLE_STRIDE
    hysteresis_frames: int = _DEFAULT_HYSTERESIS_FRAMES

    _in_ir_mode: bool = False
    _ir_period_start_unix: float | None = None
    _history: list[bool] = field(default_factory=list)

    @property
    def in_ir_mode(self) -> bool:
        """``True`` while inference should be skipped."""
        return self._in_ir_mode

    def update(self, frame: np.ndarray, wall_time: float) -> IRPeriod | None:
        """Feed a frame; return a closed :class:`IRPeriod` on day-resume.

        Args:
            frame: ``H x W x 3`` uint8 image. Forwarded to
                :func:`is_ir_frame`.
            wall_time: unix timestamp (``time.time()``) at frame
                capture. Used as both the start of an IR period (on
                the day -> IR transition) and its end (on IR -> day).
        """
        ir_now = is_ir_frame(
            frame,
            channel_diff_threshold=self.channel_diff_threshold,
            sample_stride=self.sample_stride,
        )
        self._history.append(ir_now)
        if len(self._history) > self.hysteresis_frames:
            self._history.pop(0)

        if len(self._history) < self.hysteresis_frames:
            return None

        if not self._in_ir_mode and all(self._history):
            self._in_ir_mode = True
            self._ir_period_start_unix = wall_time
            return None

        if self._in_ir_mode and not any(self._history):
            assert self._ir_period_start_unix is not None
            start_unix = self._ir_period_start_unix
            self._in_ir_mode = False
            self._ir_period_start_unix = None
            return IRPeriod(
                start=format_wall(start_unix),
                end=format_wall(wall_time),
                duration_s=round(wall_time - start_unix, 1),
            )

        return None
