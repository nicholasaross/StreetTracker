"""Run the make/model classifier over a closed session's 4K snaps.

The off-device, batch counterpart to ``makemodel-train`` (design doc
§3). Mirrors ``alpr-run``: read the session's ``*_main_*.jpg`` snaps +
the per-snap BotSORT bboxes, crop each to the *tracked* vehicle (so we
classify the same car the plate detector targets, not whichever is
largest in frame), run the fine-tuned EfficientNet-B0, and write::

    <session>_makemodel.json           # per-image top-k candidates
    <session>_makemodel_by_track.json  # per-track confidence-weighted best

CLI::

    streettracker makemodel <session_dir>
        [--model PATH]            # default: analysis/makemodel/models/makemodel_b0.pt
        [--top-k 5] [--conf-threshold 0.4] [--pad-frac 0.25]
        [--limit N] [--cpu]

Predictions carry ``make_model_source = "cnn"`` once a separate apply
pass folds them onto ``data.json`` (mirrors ``dvsa-apply``); that
write-back is intentionally NOT done here.

Preprocessing note: the snap is cropped via
:class:`~streettracker.analysis.makemodel.cropper.VehicleCropper`
(context pad + square letterbox to 224) then ImageNet-normalised. The
training eval transform used Resize+CenterCrop (no letterbox), so the
black letterbox borders are a mild train/inference framing skew -- a
known refinement to evaluate during the UK validation, not yet tuned.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from streettracker.analysis.alpr.base import atomic_write_text
from streettracker.analysis.makemodel.cropper import VehicleCropper
from streettracker.analysis.makemodel.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SurveillanceLabels,
)
from streettracker.analysis.makemodel.model import load_checkpoint
from streettracker.analysis.snap_assets import (
    discover_vehicle_snaps,
    load_bbox_index,
    resolve_bbox_hint,
)

if TYPE_CHECKING:
    import numpy as np

# Conventional bundled-checkpoint location (gitignored via *.pt), mirroring
# alpr's DEFAULT_BESPOKE_MODEL. Copy a trained best.pt here, or pass --model.
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "makemodel_b0.pt"

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


@dataclasses.dataclass(slots=True)
class Candidate:
    """One ranked (make, model, year, conf) prediction for a snap."""

    make: str
    model: str
    year: int | None
    conf: float


class MakeModelClassifier:
    """Loaded checkpoint + cropper + preprocessing, ready to classify
    a 4K snap given the tracked vehicle's bbox.

    ``top_k`` candidates are taken from the **model** head (the fine,
    281-class one); each candidate's make is its model's make
    (``make_of_model``), so make/model are always mutually consistent.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        top_k: int = 5,
        pad_frac: float = 0.25,
    ) -> None:
        self.device = torch.device(device)
        self.model, meta = load_checkpoint(checkpoint, map_location=self.device)
        self.labels = SurveillanceLabels.from_metadata(meta["labels"])
        self.top_k = top_k
        self._cropper = VehicleCropper(pad_frac=pad_frac)
        self._mean = _MEAN.to(self.device)
        self._std = _STD.to(self.device)

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        import numpy as np

        # cropper outputs 224x224 BGR uint8; -> RGB CHW float, ImageNet norm.
        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        tensor = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self._mean) / self._std

    @torch.no_grad()
    def classify(
        self, image_bgr: np.ndarray, bbox_hint: tuple[int, int, int, int] | None
    ) -> list[Candidate]:
        """Top-k (make, model, year, conf) for the hinted vehicle.

        Returns ``[]`` when there's no usable bbox (no hint, or a
        degenerate crop) -- make/model needs the car localised; the
        full 4K frame isn't a single-vehicle crop.
        """
        crop = self._cropper.crop(image_bgr, bbox_hint)
        if crop is None:
            return []
        tensor = self._preprocess(crop).unsqueeze(0)
        probs = torch.softmax(self.model(tensor)["model"], dim=1)[0]
        k = min(self.top_k, probs.shape[0])
        confs, idxs = probs.topk(k)
        out: list[Candidate] = []
        for conf, idx in zip(confs.tolist(), idxs.tolist(), strict=True):
            make = self.labels.make_names[self.labels.make_of_model[idx]]
            out.append(Candidate(make, self.labels.model_names[idx], None, float(conf)))
        return out


def aggregate_by_track(
    per_image_rows: list[dict[str, Any]], conf_threshold: float
) -> list[dict[str, Any]]:
    """Confidence-weighted vote of each snap's top-1 into a per-track best.

    For each track, sum the top-1 confidence per (make, model) across its
    snaps and take the argmax. ``conf`` is the winning class's best single
    read; ``n_high_conf_reads`` counts snaps whose top-1 cleared the
    threshold. If no read clears it, make/model/year are emitted as
    ``None`` (low-confidence track) while ``conf`` is kept for debugging.
    """
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        if row["top_k"]:
            by_track[row["track_id"]].append(row)

    tracks: list[dict[str, Any]] = []
    for tid in sorted(by_track):
        rows = by_track[tid]
        weight: dict[tuple[str, str, int | None], float] = defaultdict(float)
        best_conf: dict[tuple[str, str, int | None], float] = defaultdict(float)
        n_high = 0
        for row in rows:
            top = row["top_k"][0]
            key = (top["make"], top["model"], top["year"])
            weight[key] += top["conf"]
            best_conf[key] = max(best_conf[key], top["conf"])
            if top["conf"] >= conf_threshold:
                n_high += 1
        best_key = max(weight, key=lambda k: weight[k])
        make, model, year = best_key
        conf = best_conf[best_key]
        confident = conf >= conf_threshold
        tracks.append(
            {
                "track_id": tid,
                "make": make if confident else None,
                "model": model if confident else None,
                "year": year if confident else None,
                "conf": round(conf, 4),
                "n_high_conf_reads": n_high,
                "n_reads": len(rows),
            }
        )
    return tracks


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``streettracker makemodel <session_dir>``."""
    import argparse

    parser = argparse.ArgumentParser(prog="streettracker makemodel")
    parser.add_argument("session_dir", type=Path, help="closed session directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="checkpoint .pt")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--pad-frac", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0, help="first N snaps only (0=all)")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA present")
    args = parser.parse_args(argv)

    if not args.session_dir.is_dir():
        print(f"[makemodel] not a directory: {args.session_dir}")
        return 2
    if not args.model.exists():
        print(
            f"[makemodel] checkpoint not found: {args.model}\n"
            f"            train one (`streettracker makemodel-train ...`) and copy its "
            f"best.pt here, or pass --model <path>."
        )
        return 1

    import cv2  # type: ignore[import-untyped]  # opencv ships no py.typed

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    clf = MakeModelClassifier(args.model, device=device, top_k=args.top_k, pad_frac=args.pad_frac)

    snaps = discover_vehicle_snaps(args.session_dir)
    if not snaps:
        print(f"[makemodel] no vehicle_*_main_*.jpg snaps in {args.session_dir}")
        return 1
    if args.limit > 0:
        snaps = snaps[: args.limit]
    bbox_index, sub_size = load_bbox_index(args.session_dir)
    print(
        f"[makemodel] {len(snaps)} snaps; {len(bbox_index)} per-snap bboxes; "
        f"device={device}; {clf.labels.n_makes} makes / {clf.labels.n_models} models"
    )

    per_image: list[dict[str, Any]] = []
    n_no_hint = 0
    for i, (path, tid, snap_index, _cls) in enumerate(snaps, 1):
        hint = resolve_bbox_hint(path, tid, snap_index, bbox_index, sub_size)
        if hint is None:
            n_no_hint += 1
        image = cv2.imread(str(path))
        cands = clf.classify(image, hint) if image is not None else []
        per_image.append(
            {
                "track_id": tid,
                "snap_index": snap_index,
                "image": path.name,
                "has_bbox": hint is not None,
                "top_k": [dataclasses.asdict(c) for c in cands],
            }
        )
        if i % 100 == 0 or i == len(snaps):
            print(f"  {i}/{len(snaps)} classified")

    by_track = aggregate_by_track(per_image, args.conf_threshold)

    label = args.session_dir.name
    img_path = args.session_dir / f"{label}_makemodel.json"
    track_path = args.session_dir / f"{label}_makemodel_by_track.json"
    atomic_write_text(img_path, json.dumps(per_image, indent=2))
    atomic_write_text(track_path, json.dumps({"tracks": by_track}, indent=2))

    n_conf = sum(1 for t in by_track if t["make"] is not None)
    print(f"[makemodel] wrote {img_path}")
    print(f"[makemodel] wrote {track_path}")
    print(
        f"[makemodel] {len(by_track)} tracks classified, {n_conf} above conf "
        f"{args.conf_threshold} ({n_no_hint} snaps had no bbox hint)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
