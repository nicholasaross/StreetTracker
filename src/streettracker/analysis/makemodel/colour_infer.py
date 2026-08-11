"""Run the learned colour classifier over a closed session's 4K snaps.

The off-device batch counterpart to ``makemodel-train-uk --target colour``.
Sibling to :mod:`infer` (make) and :mod:`bodytype_infer` (body type) -- same
crop + preprocessing machinery, a single ``colour`` head emitting
white/silver/grey/black/blue/red/green/... .

It exists because the live HSV vote (``common.color.vote_color``) reads a
*low-res sub-stream* crop with an 8-colour palette and scores ~40 % against
DVSA (light cars drift to black/blue; orange/brown/purple can't be emitted).
Reading the full-resolution 4K snap with a trained head is the fix -- the same
"resolution is the lever" move that lifted the make classifier. Predictions
carry ``colour_source="cnn"``, distinct from the HSV ``color`` field, which is
left untouched.

Reads the session's ``*_main_*.jpg`` snaps + per-snap BotSORT bboxes, crops
each to the tracked vehicle, classifies, and writes::

    <session>_colour.json            # per-image top-1 (colour, conf)
    <session>_colour_by_track.json   # per-track confidence-weighted best

CLI::

    streettracker colour <session_dir>
        [--model PATH]            # default: analysis/makemodel/models/colour_b0.pt
        [--conf-threshold 0.5] [--pad-frac 0.1] [--input-size N] [--limit N] [--cpu]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from streettracker.analysis.alpr.base import atomic_write_text
from streettracker.analysis.makemodel.cropper import VehicleCropper
from streettracker.analysis.makemodel.dataset import IMAGENET_MEAN, IMAGENET_STD
from streettracker.analysis.makemodel.model import load_checkpoint
from streettracker.analysis.snap_assets import (
    discover_vehicle_snaps,
    load_bbox_index,
    resolve_bbox_hint,
)

if TYPE_CHECKING:
    import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "colour_b0.pt"

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


class ColourClassifier:
    """Loaded colour checkpoint + cropper + preprocessing."""

    # Colour is read from the vehicle's paint, so crop tight (like body type,
    # unlike the make model's looser 0.25) to keep road/sky/adjacent cars out
    # of the vote. MUST match the training corpus crop (makemodel-build-uk
    # defaults to 0.1 for the colour/body-type corpora) -- a looser pad drags
    # background in and biases the prediction.
    DEFAULT_PAD_FRAC = 0.1

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        pad_frac: float = DEFAULT_PAD_FRAC,
        input_size: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model, meta = load_checkpoint(checkpoint, map_location=self.device)
        if "colour" not in self.model.head_sizes:
            raise ValueError(
                f"checkpoint {checkpoint} is not a colour model "
                f"(heads: {sorted(self.model.head_sizes)})"
            )
        self.colours: tuple[str, ...] = tuple(meta.get("colour_names") or ())
        if not self.colours:
            raise ValueError("colour checkpoint is missing 'colour_names' metadata")
        self.input_size: int = input_size or int(meta.get("input_size") or 224)
        self._cropper = VehicleCropper(pad_frac=pad_frac, output_size=self.input_size)
        self._mean = _MEAN.to(self.device)
        self._std = _STD.to(self.device)

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        import numpy as np

        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        tensor = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self._mean) / self._std

    @torch.no_grad()
    def classify(
        self, image_bgr: np.ndarray, bbox_hint: tuple[int, int, int, int] | None
    ) -> tuple[str, float] | None:
        """Top-1 ``(colour, conf)`` for the hinted vehicle, or ``None`` when
        there's no usable bbox (colour needs the car localised)."""
        crop = self._cropper.crop(image_bgr, bbox_hint)
        if crop is None:
            return None
        tensor = self._preprocess(crop).unsqueeze(0)
        probs = torch.softmax(self.model(tensor)["colour"], dim=1)[0]
        conf, idx = probs.max(dim=0)
        return self.colours[int(idx)], float(conf)


def aggregate_by_track(
    per_image_rows: list[dict[str, Any]], conf_threshold: float
) -> list[dict[str, Any]]:
    """Confidence-weighted vote of each snap's top-1 into a per-track best.

    Sums per-colour top-1 confidence across a track's snaps and takes the
    argmax. ``colour`` is emitted only when the winner's best single read
    clears ``conf_threshold`` (else ``None``); ``conf`` is kept regardless."""
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        if row["colour"] is not None:
            by_track[row["track_id"]].append(row)

    tracks: list[dict[str, Any]] = []
    for tid in sorted(by_track):
        rows = by_track[tid]
        weight: dict[str, float] = defaultdict(float)
        best_conf: dict[str, float] = defaultdict(float)
        n_high = 0
        for row in rows:
            col, c = row["colour"], row["conf"]
            weight[col] += c
            best_conf[col] = max(best_conf[col], c)
            if c >= conf_threshold:
                n_high += 1
        best = max(weight, key=lambda k: weight[k])
        conf = best_conf[best]
        confident = conf >= conf_threshold
        tracks.append(
            {
                "track_id": tid,
                "colour": best if confident else None,
                "colour_source": "cnn",
                "conf": round(conf, 4),
                "n_high_conf_reads": n_high,
                "n_reads": len(rows),
            }
        )
    return tracks


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``streettracker colour <session_dir>``."""
    import argparse

    parser = argparse.ArgumentParser(prog="streettracker colour")
    parser.add_argument("session_dir", type=Path, help="closed session directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="checkpoint .pt")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument(
        "--pad-frac",
        type=float,
        default=ColourClassifier.DEFAULT_PAD_FRAC,
        help="crop context pad; MUST match the training corpus (0.1) — a looser "
        "pad drags background into the vote (see class docstring)",
    )
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="first N snaps only (0=all)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    if not args.session_dir.is_dir():
        print(f"[colour] not a directory: {args.session_dir}")
        return 2
    if not args.model.exists():
        print(
            f"[colour] checkpoint not found: {args.model}\n"
            f"         train one (`makemodel-train-uk ... --target colour`) "
            f"and copy its best.pt here, or pass --model <path>."
        )
        return 1

    import cv2  # type: ignore[import-untyped]

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    clf = ColourClassifier(
        args.model, device=device, pad_frac=args.pad_frac, input_size=args.input_size
    )

    snaps = discover_vehicle_snaps(args.session_dir)
    if not snaps:
        print(f"[colour] no vehicle_*_main_*.jpg snaps in {args.session_dir}")
        return 1
    if args.limit > 0:
        snaps = snaps[: args.limit]
    bbox_index, sub_size = load_bbox_index(args.session_dir)
    print(
        f"[colour] {len(snaps)} snaps; {len(bbox_index)} per-snap bboxes; device={device}; "
        f"{len(clf.colours)} classes; input={clf.input_size}px"
    )

    per_image: list[dict[str, Any]] = []
    n_no_hint = 0
    for i, (path, tid, snap_index, _cls) in enumerate(snaps, 1):
        hint = resolve_bbox_hint(path, tid, snap_index, bbox_index, sub_size)
        if hint is None:
            n_no_hint += 1
        image = cv2.imread(str(path))
        pred = clf.classify(image, hint) if image is not None else None
        per_image.append(
            {
                "track_id": tid,
                "snap_index": snap_index,
                "image": path.name,
                "has_bbox": hint is not None,
                "colour": pred[0] if pred else None,
                "conf": round(pred[1], 4) if pred else None,
            }
        )
        if i % 200 == 0 or i == len(snaps):
            print(f"  {i}/{len(snaps)} classified")

    by_track = aggregate_by_track(per_image, args.conf_threshold)

    label = args.session_dir.name
    img_path = args.session_dir / f"{label}_colour.json"
    track_path = args.session_dir / f"{label}_colour_by_track.json"
    atomic_write_text(img_path, json.dumps(per_image, indent=2))
    atomic_write_text(track_path, json.dumps({"tracks": by_track}, indent=2))

    n_conf = sum(1 for t in by_track if t["colour"] is not None)
    dist: dict[str, int] = defaultdict(int)
    for t in by_track:
        if t["colour"]:
            dist[t["colour"]] += 1
    print(f"[colour] wrote {img_path}")
    print(f"[colour] wrote {track_path}")
    print(
        f"[colour] {len(by_track)} tracks, {n_conf} confident (>= {args.conf_threshold}); "
        + " ".join(f"{k}={v}" for k, v in sorted(dist.items(), key=lambda x: -x[1]))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
