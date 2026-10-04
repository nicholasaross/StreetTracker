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

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Provenance stamp written into ``<session>_static_plates.json`` once a
# session's reads carry ``plate_colour`` / ``colour_suspect`` (alpr-run since
# 2026-10-04, or ``alpr-colour`` for older sessions).
PLATE_COLOUR_METHOD = "hsv_v1"

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


def _crop_path(record: dict[str, Any], session_dir: Path) -> Path | None:
    """``alpr_crops/<pipeline>/<image>`` in the session (portable if the
    session moved), else the recorded path."""
    name, pipeline = record.get("image"), record.get("pipeline")
    if name and pipeline:
        p = session_dir / "alpr_crops" / str(pipeline) / str(name)
        if p.is_file():
            return p
    stored = record.get("crop_path")
    if stored and Path(str(stored).replace("\\", "/")).is_file():
        return Path(str(stored).replace("\\", "/"))
    return None


def mark_colour_suspects(
    records: list[dict[str, Any]],
    session_dir: Path,
    direction_by_track: dict[int, str],
) -> Counter[str]:
    """Annotate the preferred pipeline's reads in place.

    Every read with text and a saved crop gets ``plate_colour`` (the label)
    and ``colour_suspect`` (``True`` when the colour contradicts the track's
    direction). The by-track rollup then skips suspects exactly as it skips
    ``static_suspect`` reads, so the track's best read falls back to one whose
    plate is on the tracked car (or the track goes unread). Re-running is
    safe: earlier annotations are replaced. Returns label counts plus
    ``suspect`` and ``no_crop``.
    """
    import cv2

    stats: Counter[str] = Counter()
    for r in records:
        r.pop("plate_colour", None)
        r.pop("colour_suspect", None)
        if r.get("pipeline") != "preferred" or not r.get("ocr_text"):
            continue
        path = _crop_path(r, session_dir)
        img = cv2.imread(str(path)) if path is not None else None
        if img is None:
            stats["no_crop"] += 1
            continue
        label = classify_plate_colour(img).label
        r["plate_colour"] = label
        stats[label] += 1
        try:
            direction = direction_by_track.get(int(r["track_id"]))
        except (KeyError, TypeError, ValueError):
            direction = None
        if colour_consistent(label, direction) is False:
            r["colour_suspect"] = True
            stats["suspect"] += 1
    return stats
