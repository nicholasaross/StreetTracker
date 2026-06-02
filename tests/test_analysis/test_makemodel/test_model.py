"""MakeModelNet construction, forward contract, and checkpoint I/O.

The make/model classifier is the universal (plate-free) prong of the
make/model plan. Before any training data exists, four things have to
hold so the training + inference scaffolding can be built against a
stable seam:

1. ``forward`` returns one logit tensor per head, named and shaped to
   match ``head_sizes`` -- the trainer's per-head loss and the
   inference argmax both key off these names.
2. Degenerate head specs (empty / non-positive class counts) fail loud
   at construction, not as a cryptic shape error mid-training.
3. A checkpoint round-trips exactly: same weights, same head config,
   same opaque metadata (the label maps the inference CLI needs to turn
   a class index back into "Ford Focus").
4. The whole thing runs on CPU with ``pretrained=False`` -- so CI never
   reaches for the torchvision weight hub.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from streettracker.analysis.makemodel.model import (
    CHECKPOINT_FORMAT,
    MakeModelNet,
    load_checkpoint,
    save_checkpoint,
)

# Tiny class counts keep the random-init backbone fast on CPU. The
# backbone is identical regardless of head width, so head size has no
# bearing on what these tests exercise.
_HEADS = {"make": 7, "model": 13}


def _tiny_net() -> MakeModelNet:
    # pretrained=False: random-init conv stack, no hub download. Offline
    # and fast enough for unit tests.
    return MakeModelNet(_HEADS, pretrained=False)


# ----------------------------------------------------------------------
# Forward contract.


def test_forward_returns_one_logit_tensor_per_head() -> None:
    net = _tiny_net().eval()
    out = net(torch.randn(2, 3, 224, 224))
    assert set(out) == {"make", "model"}
    assert out["make"].shape == (2, 7)
    assert out["model"].shape == (2, 13)


def test_head_order_is_preserved() -> None:
    """ModuleDict iteration order must match the head_sizes insertion
    order so the trainer's loss-weight list lines up with the heads."""
    net = MakeModelNet({"model": 13, "make": 7}, pretrained=False)
    assert list(net.heads.keys()) == ["model", "make"]


def test_feature_dim_is_efficientnet_b0_width() -> None:
    """EfficientNet-B0 pools to a 1280-d feature vector; the heads are
    Linear(1280, n). A regression here means the backbone strip went
    wrong."""
    assert _tiny_net().feature_dim == 1280


def test_supports_single_head() -> None:
    """A make-only configuration is valid -- it's the fallback if the
    model head underperforms the ship bar."""
    net = MakeModelNet({"make": 7}, pretrained=False).eval()
    out = net(torch.randn(1, 3, 224, 224))
    assert set(out) == {"make"}
    assert out["make"].shape == (1, 7)


# ----------------------------------------------------------------------
# Construction guards.


def test_empty_heads_rejected() -> None:
    with pytest.raises(ValueError, match="at least one head"):
        MakeModelNet({}, pretrained=False)


def test_nonpositive_head_size_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        MakeModelNet({"make": 7, "model": 0}, pretrained=False)


# ----------------------------------------------------------------------
# Checkpoint round-trip.


def test_checkpoint_roundtrip_preserves_weights_and_metadata(tmp_path: Path) -> None:
    net = _tiny_net().eval()
    # The label maps are exactly what the inference CLI needs to render
    # a prediction; they ride along in the opaque metadata dict.
    metadata = {
        "class_maps": {
            "make": {"0": "Audi", "1": "BMW"},
            "model": {"0": "A3", "1": "3 Series"},
        },
        "val_top1": {"make": 0.88, "model": 0.62},
    }
    ckpt_path = tmp_path / "mm.pt"
    save_checkpoint(net, ckpt_path, metadata=metadata)

    loaded, meta = load_checkpoint(ckpt_path)
    assert loaded.head_sizes == _HEADS
    assert meta == metadata

    # Identical weights -> identical outputs for the same input.
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        before = net(x)
        after = loaded(x)
    for head in _HEADS:
        assert torch.allclose(before[head], after[head], atol=1e-6)


def test_checkpoint_without_metadata_loads_empty_dict(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "mm.pt"
    save_checkpoint(_tiny_net(), ckpt_path)
    _, meta = load_checkpoint(ckpt_path)
    assert meta == {}


def test_load_rejects_unknown_format(tmp_path: Path) -> None:
    """A checkpoint from a future, incompatible layout must fail loud
    rather than silently mis-map weights."""
    ckpt_path = tmp_path / "future.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT + 99,
            "arch": "efficientnet_b0",
            "head_sizes": _HEADS,
            "state_dict": {},
            "metadata": {},
        },
        ckpt_path,
    )
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(ckpt_path)


def test_load_rejects_arch_mismatch(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "wrong_arch.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "arch": "convnext_tiny",
            "head_sizes": _HEADS,
            "state_dict": {},
            "metadata": {},
        },
        ckpt_path,
    )
    with pytest.raises(ValueError, match="arch"):
        load_checkpoint(ckpt_path)
