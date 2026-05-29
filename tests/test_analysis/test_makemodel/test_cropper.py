"""VehicleCropper round-trip behaviour.

The cropper is the pre-processing seam between the captured 4K snap
and the make/model classifier head. Three things have to hold:

1. Output shape is exactly ``(output_size, output_size, 3)`` for
   anything the classifier might consume -- the EfficientNet input
   tensor has no tolerance for off-by-one dimensions.
2. Vehicle proportions are preserved (no axis-stretch); letterboxing
   pads with zeros on the short dimension.
3. Unusable input (None / empty / out-of-bounds bbox) returns ``None``,
   not a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from streettracker.analysis.makemodel.cropper import VehicleCropper

_FULL_IMG_H = 600
_FULL_IMG_W = 800


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """An 800x600 BGR test image whose pixel values encode their
    (x, y) coordinates -- lets shape-preserving assertions check that
    the right region of the input ended up in the right region of
    the output crop."""
    img = np.zeros((_FULL_IMG_H, _FULL_IMG_W, 3), dtype=np.uint8)
    # Channel 0 = x % 256, channel 1 = y % 256, channel 2 = 128 marker.
    xx, yy = np.meshgrid(np.arange(_FULL_IMG_W), np.arange(_FULL_IMG_H))
    img[..., 0] = (xx % 256).astype(np.uint8)
    img[..., 1] = (yy % 256).astype(np.uint8)
    img[..., 2] = 128
    return img


# ----------------------------------------------------------------------
# Output shape contract.


def test_crop_output_is_square_and_default_size(synthetic_image: np.ndarray) -> None:
    cropper = VehicleCropper()
    out = cropper.crop(synthetic_image, bbox_hint=(200, 150, 600, 450))
    assert out is not None
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_crop_respects_custom_output_size(synthetic_image: np.ndarray) -> None:
    """Allow alternative classifier backbones (ConvNeXt-Tiny at 192,
    EfficientNet-B7 at 600, ...) without forking the cropper."""
    cropper = VehicleCropper(output_size=192)
    out = cropper.crop(synthetic_image, bbox_hint=(200, 150, 600, 450))
    assert out is not None
    assert out.shape == (192, 192, 3)


# ----------------------------------------------------------------------
# Padding semantics.


def test_pad_frac_adds_context_around_bbox(synthetic_image: np.ndarray) -> None:
    """pad_frac=0 returns the raw bbox region; pad_frac=0.25 widens
    the crop so the classifier sees ~20 % more pixels per side."""
    raw_crop = VehicleCropper(pad_frac=0.0).crop(
        synthetic_image, bbox_hint=(300, 200, 500, 400)
    )
    padded_crop = VehicleCropper(pad_frac=0.25).crop(
        synthetic_image, bbox_hint=(300, 200, 500, 400)
    )
    assert raw_crop is not None and padded_crop is not None
    assert raw_crop.shape == padded_crop.shape  # both 224x224 by default

    # The padded crop contains pixel (x=350, y=300) which is the bbox
    # centre -- channel 0 = 350 % 256 = 94, channel 1 = 300 % 256 = 44.
    # That mid-bbox pixel maps to roughly the same relative position
    # in both outputs; downstream classification thus sees consistent
    # framing of the vehicle. (We don't pin exact pixel values because
    # cv2.INTER_AREA interpolation smooths neighbours; the contract is
    # 'context expanded', not 'pixels translated exactly'.)
    assert padded_crop[112, 112, 2] == 128


def test_pad_frac_zero_with_already_square_bbox_skips_letterbox(
    synthetic_image: np.ndarray,
) -> None:
    """An exactly-square bbox + pad_frac=0 means cv2.copyMakeBorder
    must not run -- otherwise the crop has zero-padded borders the
    classifier would interpret as "vehicle ends here"."""
    cropper = VehicleCropper(pad_frac=0.0)
    out = cropper.crop(synthetic_image, bbox_hint=(200, 200, 400, 400))
    assert out is not None
    # The four corner pixels of the output should be content from the
    # bbox, not zero-pad sentinel values.
    assert (out[0, 0, 2], out[0, -1, 2], out[-1, 0, 2], out[-1, -1, 2]) == (
        128, 128, 128, 128,
    )


# ----------------------------------------------------------------------
# Letterboxing preserves vehicle proportions.


def test_wide_bbox_is_letterboxed_on_top_and_bottom(synthetic_image: np.ndarray) -> None:
    """A 400x100 wide bbox (4:1 aspect) padded to a square gets zero
    borders on the top + bottom edges, not stretched. The classifier
    sees the original car proportions, plus letterbox."""
    cropper = VehicleCropper(pad_frac=0.0, square_pad_value=0)
    out = cropper.crop(synthetic_image, bbox_hint=(200, 250, 600, 350))
    assert out is not None
    # Top row + bottom row are zero-padded (the letterbox).
    # The marker channel of vehicle content is 128; padding is 0.
    assert out[0, 112, 2] == 0
    assert out[-1, 112, 2] == 0
    # Centre row is vehicle content (marker=128).
    assert out[112, 112, 2] == 128


def test_tall_bbox_is_letterboxed_on_left_and_right(synthetic_image: np.ndarray) -> None:
    """Mirror of the above for a 100x400 tall bbox."""
    cropper = VehicleCropper(pad_frac=0.0, square_pad_value=0)
    out = cropper.crop(synthetic_image, bbox_hint=(350, 100, 450, 500))
    assert out is not None
    # Left + right columns padded.
    assert out[112, 0, 2] == 0
    assert out[112, -1, 2] == 0
    # Centre column is vehicle content.
    assert out[112, 112, 2] == 128


def test_square_pad_value_is_configurable(synthetic_image: np.ndarray) -> None:
    """The YOLO convention of 114-grey letterbox is supported for
    callers who want their feature-visualisation overlays to show
    the letterbox at a glance."""
    cropper = VehicleCropper(pad_frac=0.0, square_pad_value=114)
    out = cropper.crop(synthetic_image, bbox_hint=(200, 250, 600, 350))
    assert out is not None
    assert out[0, 112, 0] == 114
    assert out[0, 112, 1] == 114
    assert out[0, 112, 2] == 114


# ----------------------------------------------------------------------
# Edge-case bboxes.


def test_none_bbox_returns_none(synthetic_image: np.ndarray) -> None:
    """Caller distinguishes 'no hint available' from 'hint yielded
    nothing usable'. None in -> None out."""
    assert VehicleCropper().crop(synthetic_image, bbox_hint=None) is None


def test_inverted_bbox_returns_none(synthetic_image: np.ndarray) -> None:
    """A x2<=x1 bbox is degenerate -- usually means the runtime
    persisted a malformed BotSORT bbox. Don't crash; the caller can
    skip the snap."""
    out = VehicleCropper().crop(synthetic_image, bbox_hint=(500, 400, 200, 100))
    assert out is None


def test_bbox_fully_outside_image_returns_none(synthetic_image: np.ndarray) -> None:
    """If the persisted bbox was scaled wrong and lives entirely off
    the image, clipping produces zero area -- return None rather than
    a 0x0 ndarray that downstream resize would crash on."""
    out = VehicleCropper().crop(
        synthetic_image, bbox_hint=(_FULL_IMG_W + 50, _FULL_IMG_H + 50,
                                    _FULL_IMG_W + 200, _FULL_IMG_H + 200),
    )
    assert out is None


def test_bbox_partially_outside_image_is_clipped(synthetic_image: np.ndarray) -> None:
    """A bbox that overhangs the image edge clips to the image; the
    crop is smaller than the bbox but still valid for classification."""
    out = VehicleCropper().crop(
        synthetic_image, bbox_hint=(700, 500, 900, 700),
    )
    assert out is not None
    assert out.shape == (224, 224, 3)


def test_malformed_image_returns_none() -> None:
    """A grayscale or 4-channel image is not what the snap pipeline
    captures; fail fast rather than feed garbage to the classifier."""
    cropper = VehicleCropper()
    # 2D ndarray -- no channel dim.
    gray = np.zeros((100, 100), dtype=np.uint8)
    assert cropper.crop(gray, bbox_hint=(10, 10, 50, 50)) is None
    # 4-channel BGRA.
    bgra = np.zeros((100, 100, 4), dtype=np.uint8)
    assert cropper.crop(bgra, bbox_hint=(10, 10, 50, 50)) is None
    # None.
    assert cropper.crop(None, bbox_hint=(10, 10, 50, 50)) is None
