"""Run the make/model classifier over a closed session's 4K snaps.

The off-device, batch counterpart to ``makemodel-train`` (design doc
§3). Mirrors ``alpr-run``: read the session's ``*_main_*.jpg`` snaps +
the per-snap BotSORT bboxes, crop each to the *tracked* vehicle (so we
classify the same car the plate detector targets, not whichever is
largest in frame), run the fine-tuned EfficientNet, and write::

    <session>_makemodel.json           # per-image top-k candidates
    <session>_makemodel_by_track.json  # per-track confidence-weighted best

Two checkpoint kinds load through the same path:

* the CompCars make+model net (``--backbone`` B0, 224 px) — metadata
  carries ``labels`` and the net has a fine ``model`` head; and
* the **UK make-only** net from ``streettracker makemodel-train-uk``
  (B0@512 / B4 / B5) — metadata carries ``make_names`` + ``input_size``
  and the net has only a ``make`` head. This is the production UK model;
  it emits make-only candidates (``model``/``year`` = ``None``).

The kind is auto-detected from the head set, and the crop edge defaults
to the checkpoint's recorded ``input_size`` so the promoted UK model
Just Works (no need to remember ``--input-size 512``).

CLI::

    streettracker makemodel <session_dir>
        [--model PATH]            # default: analysis/makemodel/models/makemodel_b0.pt
        [--top-k 5] [--conf-threshold 0.4] [--pad-frac 0.25]
        [--input-size N]          # default: checkpoint's, else 224
        [--limit N] [--cpu]

Predictions carry ``make_model_source = "cnn"`` once a separate apply
pass folds them onto ``data.json`` (mirrors ``dvsa-apply``); that
write-back is intentionally NOT done here.

Preprocessing note: the snap is cropped via
:class:`~streettracker.analysis.makemodel.cropper.VehicleCropper`
(context pad + square letterbox to ``input_size``) then ImageNet-
normalised. The training eval transform used Resize+CenterCrop (no
letterbox), so the black letterbox borders are a mild train/inference
framing skew -- a known refinement to evaluate during UK validation,
not yet tuned.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from streettracker.analysis.makemodel.dataset import SurveillanceLabels
from streettracker.analysis.makemodel.session_infer import Box, SessionInference, SnapClassifier

if TYPE_CHECKING:
    import argparse

    import numpy as np

    from streettracker.analysis.vehicle_locator import SnapVehicleLocator

# Conventional bundled-checkpoint location (gitignored via *.pt), mirroring
# alpr's DEFAULT_BESPOKE_MODEL. Copy a trained best.pt here, or pass --model.
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "makemodel_b0.pt"

# Inference pad for checkpoints that don't record their training pad (every
# model before 2026-09). Kept as-is so their predictions don't shift.
LEGACY_PAD_FRAC = 0.25


@dataclasses.dataclass(slots=True)
class Candidate:
    """One ranked (make, model, year, conf) prediction for a snap.

    ``model``/``year`` are ``None`` for the make-only UK model."""

    make: str
    model: str | None
    year: int | None
    conf: float


class MakeModelClassifier(SnapClassifier):
    """Loaded checkpoint + cropper + preprocessing, ready to classify
    a 4K snap given the tracked vehicle's bbox.

    Handles both checkpoint kinds (see module docstring). For the
    CompCars net, ``top_k`` candidates come from the **model** head and
    each candidate's make is its model's make (``make_of_model``), so
    make/model stay mutually consistent. For the UK make-only net,
    candidates come from the ``make`` head and ``model``/``year`` are
    ``None``.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        top_k: int = 5,
        pad_frac: float | None = None,
        input_size: int | None = None,
        crop_mode: str = "auto",
    ) -> None:
        # The crop edge must match the model's training resolution, and the
        # crop mode + pad follow the checkpoint's training corpus when it
        # recorded them; legacy checkpoints keep hint crops at pad 0.25.
        super().__init__(
            checkpoint,
            device=device,
            pad_frac=pad_frac,
            input_size=input_size,
            crop_mode=crop_mode,
            legacy_pad_frac=LEGACY_PAD_FRAC,
        )
        # Detect the checkpoint kind by its head set: the UK net has only a
        # "make" head, the CompCars net also has a fine "model" head.
        self.make_only: bool = "model" not in self.model.head_sizes
        self.labels: SurveillanceLabels | None
        if self.make_only:
            self.make_names: tuple[str, ...] = tuple(self.meta.get("make_names") or ())
            if not self.make_names:
                raise ValueError("make-only checkpoint is missing 'make_names' metadata")
            self.labels = None
        else:
            self.labels = SurveillanceLabels.from_metadata(self.meta["labels"])
            self.make_names = ()
        self.top_k = top_k

    @torch.no_grad()
    def classify(self, image_bgr: np.ndarray, bbox_hint: Box | None) -> list[Candidate]:
        """Top-k (make, [model, year], conf) for the hinted vehicle.

        Returns ``[]`` when there's no usable bbox (no hint, or a
        degenerate crop) -- make/model needs the car localised; the
        full 4K frame isn't a single-vehicle crop.
        """
        logits = self._logits(image_bgr, bbox_hint)
        if logits is None:
            return []

        if self.make_only:
            probs = torch.softmax(logits["make"], dim=1)[0]
            k = min(self.top_k, probs.shape[0])
            confs, idxs = probs.topk(k)
            return [
                Candidate(self.make_names[idx], None, None, float(conf))
                for conf, idx in zip(confs.tolist(), idxs.tolist(), strict=True)
            ]

        assert self.labels is not None  # noqa: S101 - guaranteed when not make_only
        probs = torch.softmax(logits["model"], dim=1)[0]
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
        weight: dict[tuple[str, str | None, int | None], float] = defaultdict(float)
        best_conf: dict[tuple[str, str | None, int | None], float] = defaultdict(float)
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


class _MakeInference(SessionInference[MakeModelClassifier]):
    tag = "makemodel"
    default_conf_threshold = 0.4
    train_command = "streettracker makemodel-train-uk ..."
    pad_help = "crop context pad (default: the checkpoint's training pad, else 0.25)"
    input_size_help = (
        "crop edge px fed to the model (default: the checkpoint's recorded "
        "input_size, else 224). The UK B0@512 model wants 512."
    )
    progress_every = 100

    def default_model(self) -> Path:
        return DEFAULT_MODEL

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--top-k", type=int, default=5)

    def load(self, args: argparse.Namespace, device: str) -> MakeModelClassifier:
        return MakeModelClassifier(
            args.model,
            device=device,
            top_k=args.top_k,
            pad_frac=args.pad_frac,
            input_size=args.input_size,
            crop_mode=args.crop_mode,
        )

    def describe(self, clf: MakeModelClassifier) -> str:
        if clf.labels is None:
            return f"{len(clf.make_names)} makes (make-only)"
        return f"{clf.labels.n_makes} makes / {clf.labels.n_models} models"

    def predict(
        self, clf: MakeModelClassifier, image: np.ndarray | None, box: Box | None
    ) -> dict[str, Any]:
        cands = clf.classify(image, box) if image is not None else []
        return {"top_k": [dataclasses.asdict(c) for c in cands]}

    def aggregate(self, rows: list[dict[str, Any]], conf_threshold: float) -> list[dict[str, Any]]:
        return aggregate_by_track(rows, conf_threshold)

    def summary(
        self,
        by_track: list[dict[str, Any]],
        conf_threshold: float,
        n_no_box: int,
        locator: SnapVehicleLocator,
    ) -> str:
        n_conf = sum(1 for t in by_track if t["make"] is not None)
        return (
            f"{len(by_track)} tracks classified, {n_conf} above conf "
            f"{conf_threshold} ({n_no_box} snaps had no vehicle box; crop sources "
            f"{dict(locator.source_counts)})"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``streettracker makemodel <session_dir>``."""
    return _MakeInference().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
