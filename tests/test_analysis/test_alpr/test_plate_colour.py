"""Tests for the plate background-colour check (review E1.3)."""

from __future__ import annotations

import numpy as np

from streettracker.analysis.alpr.plate_colour import (
    FRONT_PLATE_DIRECTION,
    classify_plate_colour,
    colour_consistent,
    expected_plate_colour,
)


def _plate(bgr: tuple[int, int, int], *, h: int = 46, w: int = 136) -> np.ndarray:
    """A plate-sized crop: background colour, dark glyph bars, a dark margin."""
    img = np.zeros((h, w, 3), np.uint8)
    img[4:-4, 4:-4] = bgr
    for x in range(20, w - 20, 16):  # glyphs: dark, ~30 % of the plate
        img[12:-12, x : x + 6] = (25, 25, 25)
    return img


def test_yellow_rear_plate() -> None:
    c = classify_plate_colour(_plate((40, 200, 235)))  # BGR yellow
    assert c.label == "yellow"
    assert 15 <= c.hue <= 35


def test_washed_out_yellow_still_yellow() -> None:
    assert classify_plate_colour(_plate((150, 205, 220))).label == "yellow"


def test_bluish_white_front_plate() -> None:
    c = classify_plate_colour(_plate((235, 215, 200)))  # camera white balance: slight blue
    assert c.label == "white"
    assert 75 <= c.hue <= 135


def test_grey_white_front_plate() -> None:
    # Low saturation (hue meaningless) but some chroma, so not an IR frame.
    assert classify_plate_colour(_plate((195, 205, 215))).label == "white"


def test_monochrome_ir_frame() -> None:
    assert classify_plate_colour(_plate((180, 180, 180))).label == "mono"


def test_strong_other_colour_is_unsure() -> None:
    assert classify_plate_colour(_plate((60, 60, 220))).label == "unsure"  # red


def test_degenerate_crop_is_unsure() -> None:
    assert classify_plate_colour(np.zeros((2, 2, 3), np.uint8)).label == "unsure"


def test_expected_colour_follows_scene_direction() -> None:
    assert FRONT_PLATE_DIRECTION == "right to left"
    assert expected_plate_colour("right to left") == "white"
    assert expected_plate_colour("left to right") == "yellow"
    assert expected_plate_colour(None) is None


def test_colour_consistent() -> None:
    assert colour_consistent("yellow", "left to right") is True
    assert colour_consistent("white", "left to right") is False
    assert colour_consistent("white", "right to left") is True
    assert colour_consistent("mono", "right to left") is None
    assert colour_consistent("unsure", "left to right") is None
    assert colour_consistent("yellow", None) is None
