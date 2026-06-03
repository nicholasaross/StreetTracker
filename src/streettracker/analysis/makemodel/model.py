"""Multi-head make/model classifier (EfficientNet-B0 backbone).

Phase 2 of the make/model plan (the CompCars CNN prong). The DVSA-first
prong (PRs #41/#42/#43) ground-truths make/model/year for the readable
~25-30 % of cars via the MOT API; this classifier covers the
*unreadable* majority -- it works from pixels alone, no plate needed.
Predictions it produces carry ``make_model_source = "cnn"``, distinct
from the ``"dvsa"`` ground truth.

Design decisions (``docs/makemodel_design.md`` §4) already locked in:

* **EfficientNet-B0 backbone**, ImageNet-pretrained, fine-tuned on
  CompCars' surveillance-nature subset. ~5M params -- fits the batch
  ``alpr-run``-style off-device pass and trains in ~1h/epoch on the
  local RTX 3080. Bigger backbones (B3, ConvNeXt-Tiny) buy ~3-5 %
  accuracy for ~3x compute; not worth it while the bottleneck is
  dataset domain-match, not backbone capacity.
* **Multi-head.** A coarse ``make`` head and a fine ``model`` head
  share the backbone's pooled 1280-d feature vector. Make is
  recoverable from logo / grille shape (high accuracy expected);
  model is harder and benefits from sharing make's features. A single
  flat (make x model) head would waste capacity on the impossible
  combinations.

The head set is **parametrised** (``head_sizes``), not hard-coded:

* The CompCars *surveillance* subset labels make + model only -- it
  carries **no year**. So this net trains a ``make`` + ``model`` head;
  there is deliberately no ``year`` head despite §4 listing one as
  optional. Year for CNN-classified cars stays ``None`` (the DVSA
  prong supplies exact year for the plate-readable subset). Adding a
  year head later -- e.g. after a fine-tune on the year-labelled
  web-nature subset -- is a one-line change to ``head_sizes``, no
  surgery on this module.

This module is a leaf: it imports torch at module scope and is only
pulled in by the training / inference entry points, never by the
``analysis.makemodel`` package init or any device-runtime path, so the
import cost is paid only when you are actually doing ML.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import (  # type: ignore[import-untyped]  # torchvision ships no py.typed
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_b5,
)

# Bumped whenever the on-disk checkpoint layout changes incompatibly.
# load_checkpoint refuses formats it does not understand rather than
# silently mis-loading a future schema.
CHECKPOINT_FORMAT = 1
DEFAULT_ARCH = "efficientnet_b0"

# Backbone name -> (constructor, ImageNet weights). EfficientNet's
# compound scaling pairs a larger backbone with a higher native input
# resolution -- B0@224, B4@380, B5@456 -- so a higher-resolution crop
# set wants the matching backbone, not just a bigger input fed to B0.
# feature_dim is read off each backbone's own classifier, so the heads
# adapt (B0=1280, B4=1792, B5=2048) with no per-arch wiring.
_BACKBONES = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    "efficientnet_b5": (efficientnet_b5, EfficientNet_B5_Weights.IMAGENET1K_V1),
}
SUPPORTED_ARCHS = tuple(_BACKBONES)


class MakeModelNet(nn.Module):
    """EfficientNet-B0 with parallel per-attribute classification heads.

    ``head_sizes`` maps a head name to its class count, e.g.
    ``{"make": 75, "model": 281}``. :meth:`forward` returns a dict of
    raw logits keyed by the same names, so the trainer can apply a
    per-head loss (and weight the heads independently) and the
    inference path can argmax / softmax each head on its own.

    ``dropout`` is the rate applied before each head's linear layer;
    0.2 matches torchvision's EfficientNet-B0 ImageNet classifier.

    ``pretrained`` loads ImageNet-1k backbone weights (a torchvision
    hub download on first use). Set ``False`` for from-checkpoint
    reconstruction and for offline unit tests -- the conv stack still
    builds, just randomly initialised.

    ``arch`` selects the EfficientNet backbone: ``"efficientnet_b0"``
    (default, 224 px) or the compound-scaled ``"efficientnet_b4"`` /
    ``"efficientnet_b5"`` for the higher-resolution UK crop sets. The
    head widths follow the backbone's feature dim automatically.
    """

    def __init__(
        self,
        head_sizes: Mapping[str, int],
        *,
        dropout: float = 0.2,
        pretrained: bool = True,
        arch: str = DEFAULT_ARCH,
    ) -> None:
        super().__init__()
        if not head_sizes:
            raise ValueError("head_sizes must contain at least one head")
        bad = {name: n for name, n in head_sizes.items() if n <= 0}
        if bad:
            raise ValueError(f"head class counts must be positive, got {bad}")
        if arch not in _BACKBONES:
            raise ValueError(f"unsupported backbone arch {arch!r}; known: {sorted(_BACKBONES)}")

        constructor, weights_enum = _BACKBONES[arch]
        weights = weights_enum if pretrained else None
        backbone = constructor(weights=weights)
        self.arch = arch

        # efficientnet_b0.classifier == Sequential(Dropout, Linear(1280, 1000)).
        # Read the feature width off the ImageNet head, then strip it:
        # backbone(x) thereafter returns the pooled 1280-d feature vector
        # (features -> avgpool -> flatten all run before .classifier).
        classifier = backbone.classifier
        assert isinstance(classifier, nn.Sequential)  # noqa: S101 - torchvision contract
        final = classifier[-1]
        assert isinstance(final, nn.Linear)  # noqa: S101
        self.feature_dim: int = final.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Preserve insertion order so checkpoint round-trips and the
        # trainer's per-head loss list stay aligned with the heads.
        self.head_sizes: dict[str, int] = dict(head_sizes)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(self.feature_dim, n_classes),
                )
                for name, n_classes in self.head_sizes.items()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}


def save_checkpoint(
    model: MakeModelNet,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Serialise ``model`` plus the architecture metadata needed to
    reconstruct it.

    ``metadata`` is an opaque dict the trainer fills with the
    label maps (head name -> {class index -> "Make Model" string}),
    the training config, val metrics, etc. It is stored verbatim and
    returned by :func:`load_checkpoint`. Keep it to plain
    JSON-compatible primitives so the checkpoint loads under torch's
    ``weights_only`` safe unpickler.
    """
    ckpt: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "arch": model.arch,
        "head_sizes": dict(model.head_sizes),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[MakeModelNet, dict[str, Any]]:
    """Reconstruct a :class:`MakeModelNet` and its metadata from disk.

    The backbone is built *without* ImageNet weights (``pretrained=
    False``) since the saved ``state_dict`` immediately overwrites
    them -- so loading is offline and skips the hub download. Returns
    the model (already moved to ``map_location`` and in ``eval`` mode)
    and the verbatim metadata dict.
    """
    # weights_only=True: our checkpoint is tensors + plain primitives,
    # so it loads under the safe unpickler and we never exec arbitrary
    # pickled classes from a model file.
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    fmt = ckpt.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {fmt!r} (this build understands {CHECKPOINT_FORMAT})"
        )
    arch = ckpt.get("arch")
    if arch not in _BACKBONES:
        raise ValueError(
            f"checkpoint arch {arch!r} is not a supported backbone {sorted(_BACKBONES)}"
        )

    model = MakeModelNet(ckpt["head_sizes"], pretrained=False, arch=arch)
    model.load_state_dict(ckpt["state_dict"])
    model.to(map_location)
    model.eval()
    return model, dict(ckpt.get("metadata", {}))
