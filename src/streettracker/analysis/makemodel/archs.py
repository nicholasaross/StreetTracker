"""Canonical make/model backbone arch names -- deliberately torch-free.

These names live apart from :mod:`streettracker.analysis.makemodel.model`
(which imports ``torch``/``torchvision`` at module load) so that config,
CLI ``choices=``, and tests can validate a ``--backbone`` value against
the supported set without dragging the ML stack in -- which CI does not
install (the Jetson torch wheel is outside the base deps).

``model.py`` re-exports both names and asserts its ``_BACKBONES`` mapping
(name -> torchvision constructor) covers exactly ``SUPPORTED_ARCHS``, so
this stays the single source of truth.
"""

from __future__ import annotations

# EfficientNet compound-scaled backbones. B0@224, B4@380, B5@456,
# B6@528, B7@600 -- a higher-resolution crop set wants the matching
# backbone, not just a bigger input fed to B0. B6/B7 added 2026-07-07
# (43.8k-crop corpus; B5 won at every corpus growth so far, so capacity
# above it may now pay). No AMP in the trainer: on the 10 GB 3080, B6
# @528 wants --batch-size 4 (B5 @456 runs at 8; 64 OOMs).
SUPPORTED_ARCHS = (
    "efficientnet_b0",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
)

DEFAULT_ARCH = "efficientnet_b0"
