"""Training loop: loss weighting, top-k metric, and an offline
end-to-end fine-tune over a synthetic sv_data tree.

The end-to-end paths run with ``pretrained=False`` + ``force_cpu=True``
so they never touch the torchvision weight hub or a GPU -- fast and
CI-safe. They still exercise the real orchestration: DataLoader,
multi-head loss, AMP scaler (disabled on CPU), checkpointing, and the
``streettracker makemodel-train`` CLI entry point.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import savemat

from streettracker.analysis.makemodel.model import MakeModelNet, load_checkpoint
from streettracker.analysis.makemodel.training import (
    TrainConfig,
    evaluate,
    inverse_freq_weights,
    main,
    topk_correct,
    train,
    train_one_epoch,
)

_ROWS = [
    ("Audi", "Audi Q5", 5),
    ("BMW", "BMW 3 Series", 10),
    ("Audi", "Audi A4", 20),
]


def _build_sv_tree(root: Path) -> Path:
    """Tiny sv_data tree: 4 train + 2 test images across model ids 1..3,
    with a real savemat label file."""
    img = root / "image"
    rows = {
        "train_surveillance.txt": ["1/a.jpg", "2/b.jpg", "3/c.jpg", "1/d.jpg"],
        "test_surveillance.txt": ["2/e.jpg", "3/f.jpg"],
    }
    for split, lines in rows.items():
        for rel in lines:
            p = img / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 30), (90, 90, 90)).save(p, "JPEG")
        (root / split).write_text("\n".join(lines) + "\n")
    cells = np.empty((3, 3), dtype=object)
    for i, (make, model, web_id) in enumerate(_ROWS):
        cells[i, 0] = np.array([make])
        cells[i, 1] = np.array([model])
        cells[i, 2] = np.array([[web_id]])
    savemat(root / "sv_make_model_name.mat", {"sv_make_model_name": cells})
    return root


def _tiny_loader(n_batches: int = 2) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """In-memory stand-in for a DataLoader: (images, label-dict) batches."""
    return [
        (
            torch.randn(2, 3, 224, 224),
            {"make": torch.tensor([0, 1]), "model": torch.tensor([0, 2])},
        )
        for _ in range(n_batches)
    ]


# ----------------------------------------------------------------------
# Loss weighting.


def test_inverse_freq_weights_upweights_rare_class() -> None:
    # counts: class0 rare (10), class1 common (90).
    w = inverse_freq_weights([10, 90])
    assert w[0] > w[1]
    # sklearn-'balanced' formula: total / (n_classes * count).
    assert torch.allclose(w, torch.tensor([5.0, 100.0 / 180.0]), atol=1e-4)
    # The per-sample average weight is normalised to 1.0.
    sample_weighted_mean = (10 * w[0] + 90 * w[1]) / 100
    assert math.isclose(float(sample_weighted_mean), 1.0, abs_tol=1e-5)


def test_inverse_freq_weights_zero_count_is_safe() -> None:
    w = inverse_freq_weights([0, 50])
    assert torch.isfinite(w).all()


# ----------------------------------------------------------------------
# Top-k metric.


def test_topk_correct_counts_hits() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0, 0.0],  # pred 1
            [0.9, 0.1, 0.0, 0.0],  # pred 0
            [0.0, 0.0, 0.0, 1.0],  # pred 3
        ]
    )
    target = torch.tensor([1, 1, 3])
    hits = topk_correct(logits, target, (1, 5))
    assert hits[1] == 2  # rows 0 and 2 correct at top-1
    # top-5 is capped to n_classes=4; every target lands within top-4.
    assert hits[5] == 3


# ----------------------------------------------------------------------
# Train / eval loops (in-memory loader, CPU).


def test_train_one_epoch_returns_finite_loss() -> None:
    model = MakeModelNet({"make": 2, "model": 3}, pretrained=False)
    losses = {h: torch.nn.CrossEntropyLoss() for h in ("make", "model")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loss = train_one_epoch(
        model, _tiny_loader(2), optimizer, losses, torch.device("cpu"), scaler, amp=False
    )
    assert isinstance(loss, float)
    assert math.isfinite(loss)


def test_evaluate_reports_per_head_accuracy_in_unit_range() -> None:
    model = MakeModelNet({"make": 2, "model": 3}, pretrained=False)
    metrics = evaluate(model, _tiny_loader(2), torch.device("cpu"))
    assert set(metrics) == {"make", "model"}
    for head in ("make", "model"):
        assert set(metrics[head]) == {1, 5}
        for acc in metrics[head].values():
            assert 0.0 <= acc <= 1.0


# ----------------------------------------------------------------------
# End-to-end orchestration (offline, CPU).


def test_train_smoke_writes_loadable_checkpoint(tmp_path: Path) -> None:
    sv_root = _build_sv_tree(tmp_path / "sv_data")
    out = tmp_path / "run"
    config = TrainConfig(
        epochs=1,
        batch_size=2,
        num_workers=0,
        pretrained=False,
        amp=False,
        force_cpu=True,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    summary = train(sv_root, out, config)

    ckpt = out / "best.pt"
    assert ckpt.exists()
    assert (out / "history.json").exists()
    assert summary["best_epoch"] == 1

    # The checkpoint reconstructs and carries the label maps for inference.
    model, meta = load_checkpoint(ckpt)
    assert isinstance(model, MakeModelNet)
    assert model.head_sizes == {"make": 2, "model": 3}
    assert meta["labels"]["make_names"] == ["Audi", "BMW"]
    assert meta["epoch"] == 1


def test_cli_main_smoke(tmp_path: Path) -> None:
    sv_root = _build_sv_tree(tmp_path / "sv_data")
    out = tmp_path / "run"
    rc = main(
        [
            str(sv_root),
            "--out",
            str(out),
            "--smoke",
            "--batch-size",
            "2",
            "--no-pretrained",
            "--cpu",
        ]
    )
    assert rc == 0
    assert (out / "best.pt").exists()


def test_cli_main_rejects_missing_dir(tmp_path: Path) -> None:
    rc = main([str(tmp_path / "nope"), "--smoke"])
    assert rc == 1
