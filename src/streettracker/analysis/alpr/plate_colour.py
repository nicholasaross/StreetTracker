"""Plate background colour from a saved plate crop (review E1.3).

UK plates are white at the front and yellow at the back. On this scene a
right-to-left car shows the camera its front plate and a left-to-right car its
rear (visually confirmed in the 27.3 h soak and again on 2026-10-04 contact
sheets), so a read whose plate colour contradicts its track's direction
belongs to another car -- an oncoming or parked one -- or the track's direction
is wrong. Nothing else checks that a plate is on the tracked car (finding R2):
the DVSA register can't, because a misattributed plate is a real plate.

The classifier looks at the bright pixels (the plate background; glyphs are
dark) in the middle of the crop, away from the detector box's margin and the
GB/EV band on the left. Measured on confident daytime reads (2026-10-04, three
sessions): yellow plates sit at hue 15-25 (OpenCV 0-180 scale) with median
saturation ~120, washing out to ~35-70 in bright sun; white plates come out
slightly blue (hue ~100) at saturation 20-60, or near-grey. A crop with no
colour at all is an IR (black-and-white) frame, where both plates look white.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The direction whose cars show the camera their FRONT (white) plate. Scene
# geometry, not a property of UK plates: re-check if the camera moves.
FRONT_PLATE_DIRECTION = "right to left"

_YELLOW_HUE = (8.0, 40.0)
_YELLOW_MIN_SAT = 35.0
_WHITE_MAX_SAT = 35.0  # below this the hue is noise: call it white
_BLUISH_WHITE_HUE = (75.0, 135.0)
_BLUISH_WHITE_MAX_SAT = 90.0
_MONO_MAX_CHANNEL_DIFF = 8  # same rule as device.ir_detector.is_ir_frame


@dataclass(frozen=True, slots=True)
class PlateColour:
    """``label`` is "yellow", "white", "mono" (IR frame) or "unsure"."""

    label: str
    hue: float
    sat: float
    val: float


def classify_plate_colour(crop: np.ndarray) -> PlateColour:
    """Classify a BGR plate crop's background colour."""
    import cv2

    if crop.ndim != 3 or crop.shape[2] != 3 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return PlateColour("unsure", 0.0, 0.0, 0.0)
    if int(np.abs(np.diff(crop.astype(np.int16), axis=2)).max()) <= _MONO_MAX_CHANNEL_DIFF:
        return PlateColour("mono", 0.0, 0.0, 0.0)
    h, w = crop.shape[:2]
    mid = crop[int(h * 0.2) : max(int(h * 0.8), int(h * 0.2) + 1), int(w * 0.15) : int(w * 0.9)]
    hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    bright = hsv[hsv[:, 2] >= np.percentile(hsv[:, 2], 60)]
    hue, sat, val = (float(np.median(bright[:, i])) for i in range(3))
    if _YELLOW_HUE[0] <= hue <= _YELLOW_HUE[1] and sat >= _YELLOW_MIN_SAT:
        label = "yellow"
    elif sat < _WHITE_MAX_SAT or (
        _BLUISH_WHITE_HUE[0] <= hue <= _BLUISH_WHITE_HUE[1] and sat < _BLUISH_WHITE_MAX_SAT
    ):
        label = "white"
    else:
        label = "unsure"
    return PlateColour(label, round(hue, 1), round(sat, 1), round(val, 1))


def expected_plate_colour(direction: str | None) -> str | None:
    """The plate colour a track moving in ``direction`` should show."""
    if direction == FRONT_PLATE_DIRECTION:
        return "white"
    if direction in ("left to right", "right to left"):
        return "yellow"
    return None


def colour_consistent(label: str, direction: str | None) -> bool | None:
    """``True``/``False`` when both are known; ``None`` for mono/unsure crops
    or an unknown direction."""
    expected = expected_plate_colour(direction)
    if expected is None or label not in ("yellow", "white"):
        return None
    return label == expected
