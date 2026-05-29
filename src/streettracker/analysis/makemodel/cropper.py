"""Bbox-hint vehicle cropper for the make/model classifier.

The ALPR pipeline ships with :class:`streettracker.analysis.alpr.precrop.PreCropDetector`,
which crops *tight* around the tracked vehicle bbox -- ``pad_frac=0.15,
pad_max_px=30`` -- to keep neighbouring cars OUT of the plate
detector's field of view (the FD61PVX failure mode, see Step 10 of
the ANPR tuning loop in CLAUDE.md).

For make/model classification we want the *opposite* framing:

* **More context** -- the classifier identifies a car from its grille,
  headlight shape, wheel-arch curvature, and roof line. Cropping too
  tight on the bbox loses bumper / lower-body cues that distinguish,
  say, a Ford Focus from a Vauxhall Astra at the same camera angle.
* **No pixel cap** -- the plate-detector cap exists to stop a wide
  near-camera bbox from spilling into adjacent cars; for the
  classifier we *do* want the full vehicle even if that means a
  wider crop near the camera.
* **Square aspect** -- the EfficientNet-B0 / B4 backbones expect
  224x224 input. We letterbox with zero borders rather than stretch,
  so vehicle proportions stay correct.

Two intentional non-features:

* No detection step. This module takes the bbox as given (typically
  from ``TrackRecord.main_snap_bboxes``); a missing or unusable bbox
  returns ``None`` and the caller decides what to do (skip the snap,
  fall back to a coarse YOLO car-detector, etc.).
* No model inference. The cropper is the pre-processing seam; the
  classifier head consumes its output.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Defaults tuned for EfficientNet-B0 / B4 at 224 input. The pad_frac of
# 0.25 lifts ~20 % of vehicle context outside the BotSORT bbox into the
# classifier's view without dragging neighbouring cars in at typical
# camera distances on this scene (verified by eye on the 25.8h re-soak
# snaps post-canonical-filter).
_DEFAULT_PAD_FRAC = 0.25
_DEFAULT_OUTPUT_SIZE = 224


@dataclasses.dataclass(slots=True)
class VehicleCropper:
    """Bbox-hint cropper sized for a classifier backbone.

    ``pad_frac`` is the fraction of ``max(bbox_w, bbox_h)`` added to
    each side of the bbox before cropping. 0.25 is the default and
    works on this camera at typical distances; 0.35 also tested
    visually and was within noise.

    ``output_size`` is the square edge the cropped region resizes to.
    224 matches EfficientNet-B0 / -B4 expectations; the planned
    ConvNeXt-Tiny ablation would set 192 instead.

    ``square_pad_value`` is the constant border colour used when the
    padded bbox isn't square. 0 (black) matches ImageNet-normalisation
    expectations and won't be confused with image content. Set to a
    grey shade (e.g. 114, the YOLO convention) if downstream feature
    visualisations need the letterbox visible.
    """

    pad_frac: float = _DEFAULT_PAD_FRAC
    output_size: int = _DEFAULT_OUTPUT_SIZE
    square_pad_value: int = 0

    def crop(
        self,
        image_bgr: np.ndarray,
        bbox_hint: tuple[int, int, int, int] | None,
    ) -> np.ndarray | None:
        """Crop ``image_bgr`` around ``bbox_hint`` for classifier input.

        Args:
            image_bgr: ``H x W x 3`` uint8 BGR image. Typically the
                full 4K Reolink snap, but any image with the same
                bbox coordinate system works.
            bbox_hint: ``(x1, y1, x2, y2)`` in ``image_bgr``'s pixel
                coordinates. ``None`` returns ``None`` immediately
                (caller distinguishes "no hint" from "hint produced
                an empty crop").

        Returns:
            ``output_size x output_size x 3`` uint8 BGR ndarray, or
            ``None`` when the bbox is unusable (degenerate area after
            clipping, or zero-pixel intersection with the image).
        """
        import cv2
        import numpy as np

        if bbox_hint is None:
            return None
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            return None

        h_img, w_img = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_hint
        if x2 <= x1 or y2 <= y1:
            return None

        # Pad the bbox by pad_frac * max(w, h). max() so a wide-thin or
        # tall-thin bbox still receives proportional padding on the
        # short dimension -- prevents the letterbox from dominating the
        # final image.
        bw, bh = (x2 - x1), (y2 - y1)
        pad = int(round(self.pad_frac * max(bw, bh)))
        cx1 = max(0, int(x1) - pad)
        cy1 = max(0, int(y1) - pad)
        cx2 = min(w_img, int(x2) + pad)
        cy2 = min(h_img, int(y2) + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = image_bgr[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            return None

        # Letterbox to square. Padding on the shorter side preserves
        # vehicle proportions for the classifier; stretching to a
        # non-square aspect would distort grille / headlight geometry,
        # which the classifier explicitly relies on.
        if cw > ch:
            top = (cw - ch) // 2
            bottom = cw - ch - top
            squared = cv2.copyMakeBorder(
                crop, top, bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(self.square_pad_value,) * 3,
            )
        elif ch > cw:
            left = (ch - cw) // 2
            right = ch - cw - left
            squared = cv2.copyMakeBorder(
                crop, 0, 0, left, right,
                cv2.BORDER_CONSTANT, value=(self.square_pad_value,) * 3,
            )
        else:
            squared = crop

        # cv2.INTER_AREA is the documented "good for downscale" choice
        # and is what torchvision's transforms.Resize uses by default
        # for shrinking. The vast majority of our crops are larger
        # than 224 (typical 400-800 px tracked-vehicle bbox at 4K),
        # so this is the operative path.
        resized = cv2.resize(
            squared, (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA,
        )
        # cv2.resize returns the same dtype as input; ensure uint8 for
        # downstream Tensor() conversion (torchvision expects uint8 or
        # float in [0, 1]; uint8 is what the runtime captures.)
        return np.asarray(resized, dtype=np.uint8)
