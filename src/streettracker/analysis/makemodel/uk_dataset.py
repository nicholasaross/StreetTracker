"""StreetTracker-native make dataset, auto-labelled by DVSA plate lookups.

After the CompCars CNN failed UK validation (domain gap, 2026-06-02 --
~1% make accuracy on real cars), the fix is to train on *this* scene
instead of a foreign dataset. DVSA make/model from a readable plate
ground-truths that car's crops, so we can build a right-domain training
set straight from the sessions we already capture:

  DVSA label (plate -> make)  x  the car's 4K snaps + BotSORT bboxes
      -> make-labelled crops at the real camera angle / resolution.

Crops are made with the SAME :class:`VehicleCropper` the inference path
uses, so there's no train/inference framing skew (the CompCars failure
mode). Splits are **by car** (plate), never by crop -- a car's frames
are near-duplicates, so mixing them across train/val inflates val.

Make-only for now: model-level labels are too sparse per class. The
trainer reuses :func:`...training.train_one_epoch` / ``evaluate`` (made
head-agnostic) with a single ``make`` head.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from streettracker.analysis.makemodel.cropper import VehicleCropper
from streettracker.analysis.makemodel.dataset import build_eval_transform, build_train_transform
from streettracker.analysis.makemodel.model import (
    SUPPORTED_ARCHS,
    MakeModelNet,
    save_checkpoint,
)
from streettracker.analysis.makemodel.training import (
    TrainConfig,
    evaluate,
    inverse_freq_weights,
    train_one_epoch,
)
from streettracker.analysis.snap_assets import (
    discover_vehicle_snaps,
    load_bbox_index,
    resolve_bbox_hint,
)

# DVSA sometimes returns two spellings for one marque (e.g. both "MERCEDES"
# and "MERCEDES-BENZ"), which would otherwise split a make across two classes
# and dilute its training signal. Fold known synonyms onto one canonical label.
_MAKE_SYNONYMS = {
    "MERCEDES": "MERCEDES-BENZ",
}


def normalize_make(make: str | None) -> str:
    """DVSA make -> class label. DVSA names are already canonical UK
    strings ("FORD", "MERCEDES-BENZ", "LAND ROVER"); upper/strip, collapse
    internal whitespace, then fold known synonyms (see ``_MAKE_SYNONYMS``)
    for stable folder/label names."""
    if not make:
        return ""
    name = " ".join(make.upper().split())
    return _MAKE_SYNONYMS.get(name, name)


def _load_session_labels(session_dir: Path) -> dict[str, dict[str, Any]]:
    """``plate -> {make, track_ids}`` from a session's DVSA labels, or {}."""
    matches = list(session_dir.glob("*_dvsa_labels.json"))
    if not matches:
        return {}
    try:
        return json.loads(matches[0].read_text()).get("labels", {})
    except (OSError, json.JSONDecodeError):
        return {}


@dataclasses.dataclass(slots=True)
class _Cand:
    """A candidate snap for a car, ranked by bbox area before cropping."""

    area: int
    sess_name: str
    tid: int
    snap_index: int
    hint: tuple[int, int, int, int]
    src_path: str


def extract_crops(
    sessions: list[Path],
    out_dir: str | Path,
    *,
    min_cars_per_make: int = 5,
    pad_frac: float = 0.25,
    max_per_car: int | None = None,
    select_top_by_area: int | None = None,
    output_size: int = 224,
) -> dict[str, Any]:
    """Crop DVSA-labelled cars from their snaps into ``out_dir/<MAKE>/``.

    Keeps only makes with at least ``min_cars_per_make`` distinct cars
    (rare makes can't be learned or validated). Writes ``manifest.json``
    (``{"makes": [...], "samples": [{"path", "make", "car"}]}``) for the
    by-car split. ``car`` is the plate, so a vehicle recurring across
    sessions stays one car (no cross-split leakage).

    ``select_top_by_area`` keeps only each car's N largest-bbox snaps
    (closest pass = most pixels/detail at 224) instead of all of them --
    the best-view lever for when averaging over distant/oblique snaps
    dilutes the signal. ``None`` keeps all (``max_per_car`` still caps).

    ``output_size`` is the square edge each crop is resized to (224 by
    default). Raise to 384 to keep more of the 4K source detail -- only
    useful while source bboxes exceed it (typically 400-800 px here), so
    re-extract from the snaps rather than upscaling an existing 224 set.
    """
    import cv2  # type: ignore[import-untyped]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: gather labelled cars + per-make car counts.
    car_make: dict[str, str] = {}
    car_tracks: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for sess in sessions:
        for plate, v in _load_session_labels(sess).items():
            mk = normalize_make(v.get("make"))
            if not mk:
                continue
            car_make[plate] = mk
            for tid in v.get("track_ids", []):
                car_tracks[plate].append((sess, int(tid)))
    make_car_counts = Counter(car_make.values())
    kept = {m for m, n in make_car_counts.items() if n >= min_cars_per_make}

    # Pass 2: crop the snaps of cars in kept makes. Index target tids per
    # session so we discover each session's snaps only once.
    sess_tid_plate: dict[Path, dict[int, str]] = defaultdict(dict)
    for plate, tracks in car_tracks.items():
        if car_make[plate] not in kept:
            continue
        for sess, tid in tracks:
            sess_tid_plate[sess][tid] = plate

    # Gather candidate snaps (with bbox area, no decode yet), then per car
    # keep the top-N largest-bbox snaps before cropping -- cv2.imread only
    # runs for kept snaps.
    candidates: dict[str, list[_Cand]] = defaultdict(list)
    for sess, tid_plate in sess_tid_plate.items():
        bbox_index, sub_size = load_bbox_index(sess)
        for path, tid, snap_index, _cls in discover_vehicle_snaps(sess):
            car = tid_plate.get(tid)
            if car is None:
                continue
            hint = resolve_bbox_hint(path, tid, snap_index, bbox_index, sub_size)
            if hint is None:
                continue
            x1, y1, x2, y2 = hint
            area = max(0, x2 - x1) * max(0, y2 - y1)
            candidates[car].append(_Cand(area, sess.name, tid, snap_index, hint, str(path)))

    cropper = VehicleCropper(pad_frac=pad_frac, output_size=output_size)
    samples: list[dict[str, str]] = []
    cap = select_top_by_area if select_top_by_area is not None else max_per_car
    for car, cands in candidates.items():
        cands.sort(key=lambda c: -c.area)  # largest bbox (closest) first
        mk = car_make[car]
        (out_dir / mk).mkdir(exist_ok=True)
        for cand in cands[:cap] if cap is not None else cands:
            image = cv2.imread(cand.src_path)
            if image is None:
                continue
            crop = cropper.crop(image, cand.hint)
            if crop is None:
                continue
            rel = f"{mk}/{car}_{cand.sess_name}_{cand.tid}_{cand.snap_index}.jpg"
            cv2.imwrite(str(out_dir / rel), crop)
            samples.append({"path": rel, "make": mk, "car": car})

    manifest = {
        "makes": sorted(kept),
        "min_cars_per_make": min_cars_per_make,
        "samples": samples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "n_makes": len(kept),
        "n_cars": sum(1 for m in car_make.values() if m in kept),
        "n_crops": len(samples),
        "make_car_counts": {m: make_car_counts[m] for m in sorted(kept)},
    }


@dataclasses.dataclass(slots=True)
class _Sample:
    path: str
    make_index: int
    car: str


class UKMakeDataset(Dataset):
    """One by-car split of an extracted UK make dataset.

    ``make_names`` is read from the manifest unless injected (the val
    dataset must share the train dataset's class order). The by-car
    split is deterministic + stratified by make, so every make with >=2
    cars contributes at least one val car.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        make_names: tuple[str, ...] | None = None,
        transform: Any = None,
        val_frac: float = 0.2,
        seed: int = 0,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self._root = Path(dataset_dir)
        manifest = json.loads((self._root / "manifest.json").read_text())
        self.make_names = make_names if make_names is not None else tuple(manifest["makes"])
        idx = {m: i for i, m in enumerate(self.make_names)}
        self._transform = transform

        car_make = {s["car"]: s["make"] for s in manifest["samples"]}
        cars_by_make: dict[str, list[str]] = defaultdict(list)
        for car, mk in car_make.items():
            cars_by_make[mk].append(car)

        val_cars: set[str] = set()
        for mk, cars in cars_by_make.items():
            cars = sorted(cars)
            # str seed -> stable across runs (not salted like hash()).
            random.Random(f"{seed}:{mk}").shuffle(cars)
            n_val = max(1, round(val_frac * len(cars))) if len(cars) > 1 else 0
            val_cars.update(cars[:n_val])

        want_val = split == "val"
        self._samples = [
            _Sample(s["path"], idx[s["make"]], s["car"])
            for s in manifest["samples"]
            if s["make"] in idx and ((s["car"] in val_cars) == want_val)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, int]]:
        from PIL import Image

        s = self._samples[index]
        image = Image.open(self._root / s.path).convert("RGB")
        out = self._transform(image) if self._transform is not None else image
        return out, {"make": s.make_index}

    def class_counts(self) -> dict[str, list[int]]:
        counts = [0] * len(self.make_names)
        for s in self._samples:
            counts[s.make_index] += 1
        return {"make": counts}


def train_uk_make(dataset_dir: str | Path, out_dir: str | Path, config: TrainConfig) -> dict:
    """Fine-tune a make-only EfficientNet-B0 on the extracted UK crops.

    Reuses the (head-agnostic) train/eval loop with a single ``make``
    head + inverse-frequency class weighting. Checkpoints the best epoch
    by val make@1 to ``<out>/best.pt``; returns a summary dict.
    """
    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.force_cpu else "cpu")
    amp = config.amp and device.type == "cuda"

    train_ds = UKMakeDataset(
        dataset_dir, "train", transform=build_train_transform(config.input_size), seed=config.seed
    )
    make_names = train_ds.make_names
    val_ds = UKMakeDataset(
        dataset_dir,
        "val",
        make_names=make_names,
        transform=build_eval_transform(config.input_size),
        seed=config.seed,
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin,
        drop_last=len(train_ds) >= config.batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=pin
    )

    model = MakeModelNet(
        {"make": len(make_names)},
        dropout=config.dropout,
        pretrained=config.pretrained,
        arch=config.backbone,
    ).to(device)
    if config.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    weight = None
    if config.class_weighting:
        weight = inverse_freq_weights(train_ds.class_counts()["make"]).to(device)
    head_losses: dict[str, nn.Module] = {"make": nn.CrossEntropyLoss(weight=weight)}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    print(
        f"[makemodel-uk] device={device.type} amp={amp} makes={len(make_names)} "
        f"train_crops={len(train_ds)} val_crops={len(val_ds)}",
        flush=True,
    )

    ckpt_path = out_path / "best.pt"
    history: list[dict] = []
    best, best_epoch, since = -1.0, -1, 0
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
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
            model, val_loader, device, amp=amp, limit_batches=config.limit_val_batches
        )
        m1, m5 = metrics["make"][1], metrics["make"][5]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(loss, 4),
                "val_make_top1": round(m1, 4),
                "val_make_top5": round(m5, 4),
            }
        )
        print(
            f"[makemodel-uk] epoch {epoch}/{config.epochs} loss={loss:.4f} "
            f"make@1={m1:.3f} make@5={m5:.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if m1 > best:
            best, best_epoch, since = m1, epoch, 0
            save_checkpoint(
                model,
                ckpt_path,
                metadata={
                    "kind": "uk_make",
                    "make_names": list(make_names),
                    "input_size": config.input_size,
                    "epoch": epoch,
                    "val_make_top1": round(m1, 4),
                },
            )
        else:
            since += 1
            if since >= config.patience:
                print(f"[makemodel-uk] early stop after {epoch} epochs", flush=True)
                break

    summary = {
        "best_epoch": best_epoch,
        "best_val_make_top1": round(best, 4),
        "makes": list(make_names),
        "history": history,
    }
    (out_path / "history.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[makemodel-uk] done: best make@1={best:.3f} (epoch {best_epoch}) -> {ckpt_path}",
        flush=True,
    )
    return summary


def _discover_labelled_sessions() -> list[Path]:
    """All ``output/session_*`` dirs that carry a DVSA labels file."""
    import glob

    return sorted({Path(g).parent for g in glob.glob("output/session_*/*_dvsa_labels.json")})


def build_main(argv: list[str] | None = None) -> int:
    """CLI: ``streettracker makemodel-build-uk <out_dir>`` -- extract the
    DVSA-labelled crop dataset (auto-discovers labelled sessions)."""
    import argparse

    parser = argparse.ArgumentParser(prog="streettracker makemodel-build-uk")
    parser.add_argument("out_dir", type=Path, help="output crop-dataset dir")
    parser.add_argument(
        "--sessions",
        type=Path,
        nargs="*",
        default=None,
        help="session dirs (default: auto-discover output/session_*/ with DVSA labels)",
    )
    parser.add_argument("--min-cars-per-make", type=int, default=5)
    parser.add_argument("--pad-frac", type=float, default=0.1)
    parser.add_argument("--max-per-car", type=int, default=None)
    parser.add_argument(
        "--top-by-area",
        type=int,
        default=None,
        help="keep only each car's N largest-bbox (closest) snaps -- best-view selection",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=224,
        help="square crop edge (px); 384 keeps more 4K detail than the 224 default",
    )
    args = parser.parse_args(argv)

    sessions = list(args.sessions) if args.sessions else _discover_labelled_sessions()
    if not sessions:
        print("[makemodel-build-uk] no DVSA-labelled sessions found")
        return 1
    print(f"[makemodel-build-uk] {len(sessions)} session(s) -> {args.out_dir}")
    stats = extract_crops(
        sessions,
        args.out_dir,
        min_cars_per_make=args.min_cars_per_make,
        pad_frac=args.pad_frac,
        max_per_car=args.max_per_car,
        select_top_by_area=args.top_by_area,
        output_size=args.output_size,
    )
    print(
        f"[makemodel-build-uk] {stats['n_makes']} makes, {stats['n_cars']} cars, "
        f"{stats['n_crops']} crops"
    )
    for mk, n in sorted(stats["make_car_counts"].items(), key=lambda x: -x[1]):
        print(f"  {mk:<16}{n}")
    return 0


def train_main(argv: list[str] | None = None) -> int:
    """CLI: ``streettracker makemodel-train-uk <crops_dir>`` -- train the
    make classifier on an extracted dataset. A real entry point, so
    ``--num-workers > 0`` works (unlike a stdin heredoc)."""
    import argparse

    defaults = TrainConfig()
    parser = argparse.ArgumentParser(prog="streettracker makemodel-train-uk")
    parser.add_argument("crops_dir", type=Path, help="makemodel-build-uk output dir")
    parser.add_argument("--out", type=Path, default=Path("runs/uk_make"))
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument(
        "--input-size",
        type=int,
        default=defaults.input_size,
        help="square model input resolution (px); 384 = higher-detail crops vs 224 default",
    )
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_ARCHS,
        default=defaults.backbone,
        help="EfficientNet backbone; b4 (380px) / b5 (456px) are compound-scaled for hi-res crops",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", help="linear-probe (frozen backbone)"
    )
    parser.add_argument("--no-class-weighting", action="store_true")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="skip ImageNet init (debug/tests)"
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    if not (args.crops_dir / "manifest.json").exists():
        print(
            f"[makemodel-train-uk] no manifest.json in {args.crops_dir} "
            f"-- run makemodel-build-uk first"
        )
        return 1
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        dropout=args.dropout,
        patience=args.patience,
        input_size=args.input_size,
        backbone=args.backbone,
        freeze_backbone=args.freeze_backbone,
        class_weighting=not args.no_class_weighting,
        pretrained=not args.no_pretrained,
        force_cpu=args.cpu,
    )
    train_uk_make(args.crops_dir, args.out, config)
    return 0
