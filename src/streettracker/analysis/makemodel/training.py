"""Fine-tune the make/model classifier on CompCars surveillance.

Implements design doc §5.5: AdamW (lr 1e-4, weight-decay 1e-2) with
cosine LR decay, batch 64, mixed precision, per-head cross-entropy
summed across the make + model heads, with **inverse-frequency class
weighting** so the long tail of rare makes isn't drowned out. Early
stopping watches val *model* top-1 (the harder, binding head).

CLI::

    streettracker makemodel-train <sv_data_dir> --out runs/mm1
        [--epochs 20] [--batch-size 64] [--lr 1e-4] [--num-workers 4]
        [--smoke]                     # 1 epoch, 3 train + 3 val batches
        [--limit-train-batches N] [--limit-val-batches N]
        [--no-amp] [--no-class-weighting] [--no-pretrained] [--cpu]

The best checkpoint (by val model top-1) is written to
``<out>/best.pt`` via
:func:`streettracker.analysis.makemodel.model.save_checkpoint`, with the
:class:`~streettracker.analysis.makemodel.dataset.SurveillanceLabels`
maps + the run config + per-epoch metrics folded into its metadata so
the inference CLI can reconstruct everything from the file alone.

This is an off-device dev-box task (trains on the local RTX 3080); it is
never imported by the Orin runtime.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from streettracker.analysis.makemodel.dataset import build_datasets
from streettracker.analysis.makemodel.model import MakeModelNet, save_checkpoint

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.utils.data import DataLoader

    from streettracker.analysis.makemodel.dataset import CompCarsSurveillance

_HEADS = ("make", "model")
_TOPKS = (1, 5)
# Early stopping + checkpoint selection watch this head's top-1: model is
# the harder objective and the binding ship-bar target (§7).
_MONITOR_HEAD = "model"


@dataclasses.dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyperparameters for a fine-tune run (defaults = design doc §5.5)."""

    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    input_size: int = 224
    num_workers: int = 4
    amp: bool = True
    dropout: float = 0.2
    pretrained: bool = True
    class_weighting: bool = True
    patience: int = 5
    seed: int = 0
    force_cpu: bool = False
    # Freeze the backbone and train only the head(s) -- a linear probe.
    # Strong regulariser on small data (the UK make set), where a full
    # fine-tune memorises the few train cars.
    freeze_backbone: bool = False
    # Cap batches per epoch for smoke / sanity runs; None = full epoch.
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None


def inverse_freq_weights(counts: list[int]) -> torch.Tensor:
    """Per-class weights ∝ 1/frequency (the sklearn-'balanced' formula
    ``total / (n_classes * count)``), so the *per-sample* average weight
    is 1 and the loss scale stays comparable to the unweighted case.

    A class with zero samples would otherwise divide by zero; clamp the
    denominator at 1. (In the surveillance train split all 281 models +
    68 makes are present, so this is just defensive.)
    """
    counts_t = torch.tensor(counts, dtype=torch.float32)
    total = counts_t.sum()
    n = len(counts)
    return total / (counts_t.clamp(min=1.0) * n)


def topk_correct(
    logits: torch.Tensor, target: torch.Tensor, ks: Iterable[int] = _TOPKS
) -> dict[int, int]:
    """Count, per k, how many rows have the true label in their top-k."""
    ks = tuple(ks)
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)  # (B, maxk) indices, high->low
    hit = pred.eq(target.view(-1, 1))  # (B, maxk) bool
    out: dict[int, int] = {}
    for k in ks:
        kk = min(k, maxk)
        out[k] = int(hit[:, :kk].any(dim=1).sum().item())
    return out


def _build_head_losses(
    train_ds: CompCarsSurveillance, *, class_weighting: bool, device: torch.device
) -> dict[str, nn.Module]:
    counts = train_ds.class_counts() if class_weighting else None
    losses: dict[str, nn.Module] = {}
    for head in _HEADS:
        weight = inverse_freq_weights(counts[head]).to(device) if counts is not None else None
        losses[head] = nn.CrossEntropyLoss(weight=weight)
    return losses


def train_one_epoch(
    model: MakeModelNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    head_losses: dict[str, nn.Module],
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    amp: bool,
    limit_batches: int | None = None,
) -> float:
    """One training pass; returns mean total loss over the run batches."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for i, (images, targets) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        # Heads come from head_losses, so this loop serves any head set
        # (CompCars make+model, or a make-only UK classifier).
        targets = {h: targets[h].to(device, non_blocking=True) for h in head_losses}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model(images)
            loss = torch.stack([head_losses[h](out[h], targets[h]) for h in head_losses]).sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: MakeModelNet,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = False,
    limit_batches: int | None = None,
) -> dict[str, dict[int, float]]:
    """Per-head top-1 / top-5 accuracy over the val set (or a capped
    prefix). Returns ``{"make": {1: acc, 5: acc}, "model": {...}}``."""
    model.eval()
    heads = tuple(model.head_sizes)  # head-agnostic: works for make-only too
    correct: dict[str, dict[int, int]] = {h: dict.fromkeys(_TOPKS, 0) for h in heads}
    total = 0
    for i, (images, targets) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model(images)
        for head in heads:
            tgt = targets[head].to(device, non_blocking=True)
            batch_hits = topk_correct(out[head], tgt, _TOPKS)
            for k in _TOPKS:
                correct[head][k] += batch_hits[k]
        total += images.shape[0]

    total = max(total, 1)
    return {h: {k: correct[h][k] / total for k in _TOPKS} for h in heads}


def train(sv_root: str | Path, out_dir: str | Path, config: TrainConfig) -> dict:
    """Run a full fine-tune; checkpoint the best epoch by val model top-1.

    Returns a summary dict (best epoch, best metrics, checkpoint path,
    per-epoch history).
    """
    torch.manual_seed(config.seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not config.force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    # AMP only pays off on CUDA; silently disable on CPU.
    amp = config.amp and device.type == "cuda"

    train_ds, val_ds, labels = build_datasets(sv_root, input_size=config.input_size)
    pin = device.type == "cuda"
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
    )

    model = MakeModelNet(
        labels.head_sizes, dropout=config.dropout, pretrained=config.pretrained
    ).to(device)
    head_losses = _build_head_losses(
        train_ds, class_weighting=config.class_weighting, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    ckpt_path = out_path / "best.pt"
    history: list[dict] = []
    best_top1 = -1.0
    best_epoch = -1
    epochs_since_best = 0

    def _summary() -> dict:
        return {
            "best_epoch": best_epoch,
            "best_model_top1": round(best_top1, 4),
            "checkpoint": str(ckpt_path) if best_epoch > 0 else None,
            "history": history,
        }

    print(
        f"[makemodel-train] device={device.type} amp={amp} "
        f"train={len(train_ds)} val={len(val_ds)} "
        f"makes={labels.n_makes} models={labels.n_models}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            head_losses,
            device,
            scaler,
            amp=amp,
            limit_batches=config.limit_train_batches,
        )
        scheduler.step()
        metrics = evaluate(
            model,
            val_loader,
            device,
            amp=amp,
            limit_batches=config.limit_val_batches,
        )
        dt = time.time() - t0
        monitor = metrics[_MONITOR_HEAD][1]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "seconds": round(dt, 1),
            "val": {h: {f"top{k}": round(metrics[h][k], 4) for k in _TOPKS} for h in _HEADS},
        }
        history.append(row)
        print(
            f"[makemodel-train] epoch {epoch}/{config.epochs} "
            f"loss={train_loss:.4f} "
            f"make@1={metrics['make'][1]:.3f} "
            f"model@1={metrics['model'][1]:.3f} model@5={metrics['model'][5]:.3f} "
            f"({dt:.0f}s)",
            flush=True,
        )

        if monitor > best_top1:
            best_top1 = monitor
            best_epoch = epoch
            epochs_since_best = 0
            save_checkpoint(
                model,
                ckpt_path,
                metadata={
                    "labels": labels.to_metadata(),
                    "config": dataclasses.asdict(config),
                    "epoch": epoch,
                    "val_metrics": row["val"],
                },
            )
        else:
            epochs_since_best += 1

        # Rewrite the run summary every epoch so a long background run is
        # tail-able (history.json) rather than opaque until the end.
        (out_path / "history.json").write_text(json.dumps(_summary(), indent=2))

        if epochs_since_best >= config.patience:
            print(
                f"[makemodel-train] early stop: no val {_MONITOR_HEAD}@1 "
                f"improvement in {config.patience} epochs",
                flush=True,
            )
            break

    summary = _summary()
    (out_path / "history.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[makemodel-train] done: best epoch {best_epoch} model@1={best_top1:.3f} -> {ckpt_path}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``streettracker makemodel-train <sv_data_dir>``."""
    import argparse

    # slots=True means TrainConfig.<field> is a slot descriptor, not the
    # default value -- read defaults off a default instance instead.
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(prog="streettracker makemodel-train")
    parser.add_argument("sv_root", type=Path, help="CompCars sv_data directory")
    parser.add_argument("--out", type=Path, default=Path("runs/makemodel"), help="output dir")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument(
        "--no-class-weighting", action="store_true", help="disable inverse-freq weighting"
    )
    parser.add_argument(
        "--no-pretrained", action="store_true", help="skip ImageNet init (debug/tests)"
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 epoch, 3 train + 3 val batches -- verify the loop end-to-end",
    )
    args = parser.parse_args(argv)

    if not args.sv_root.is_dir():
        print(f"[makemodel-train] not a directory: {args.sv_root}")
        return 1

    limit_train = args.limit_train_batches
    limit_val = args.limit_val_batches
    epochs = args.epochs
    if args.smoke:
        epochs = 1
        limit_train = limit_train or 3
        limit_val = limit_val or 3

    config = TrainConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        patience=args.patience,
        amp=not args.no_amp,
        class_weighting=not args.no_class_weighting,
        pretrained=not args.no_pretrained,
        force_cpu=args.cpu,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
    )
    train(args.sv_root, args.out, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
