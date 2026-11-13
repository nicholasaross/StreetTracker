"""What the make, body-type and colour inference commands share.

``streettracker makemodel`` / ``bodytype`` / ``colour`` all do the same job
with a different head: read a closed session's vehicle 4K snaps, crop each to
the tracked car, run a fine-tuned EfficientNet over the crop, and write a
per-image file and a per-track file. Until 2026-11 each command carried its
own copy of that (:mod:`infer`, :mod:`bodytype_infer`, :mod:`colour_infer`,
about 85 % the same), so a fix had to be made three times.

This module holds the common parts:

* :class:`SnapClassifier`: a loaded checkpoint with the crop and the
  preprocessing its training corpus used;
* :class:`TopOneClassifier`: a single softmax head with named classes (body
  type, colour), giving the top class per crop;
* :func:`aggregate_top_one`: the per-track vote for such a head;
* :class:`SessionInference`: the command itself, from argument parsing to
  the files written, with the head-specific steps left to a subclass.

What each command prints is read by the control panel, and what it writes is
read by ``/stats`` and the showcase, so the three commands' output is kept
exactly as it was.
"""

from __future__ import annotations

import abc
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

import torch

from streettracker.analysis.alpr.fullframe import load_road_polygon
from streettracker.analysis.makemodel.cropper import VehicleCropper
from streettracker.analysis.makemodel.dataset import IMAGENET_MEAN, IMAGENET_STD
from streettracker.analysis.makemodel.model import load_checkpoint
from streettracker.analysis.snap_assets import discover_vehicle_snaps, read_image
from streettracker.analysis.vehicle_locator import SnapVehicleLocator, resolve_crop_settings
from streettracker.common.fileio import atomic_write_text

if TYPE_CHECKING:
    import numpy as np

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)

Box = tuple[int, int, int, int]


class SnapClassifier:
    """A loaded checkpoint, with the crop and preprocessing to feed it.

    The crop edge is the checkpoint's recorded ``input_size`` (224 for the
    checkpoints that don't record one), and the crop mode and pad follow the
    corpus the checkpoint was trained on when it recorded them. A checkpoint
    from before 2026-09 recorded neither: it keeps the legacy ``hint`` crop at
    ``legacy_pad_frac``, so its predictions don't shift.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        pad_frac: float | None = None,
        input_size: int | None = None,
        crop_mode: str = "auto",
        legacy_pad_frac: float,
    ) -> None:
        self.device = torch.device(device)
        self.model, self.meta = load_checkpoint(checkpoint, map_location=self.device)
        self.input_size: int = input_size or int(self.meta.get("input_size") or 224)
        self.crop_mode, self.pad_frac = resolve_crop_settings(
            self.meta, crop_mode, pad_frac, legacy_pad_frac
        )
        self._cropper = VehicleCropper(pad_frac=self.pad_frac, output_size=self.input_size)
        self._mean = _MEAN.to(self.device)
        self._std = _STD.to(self.device)

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        import numpy as np

        # The cropper outputs an NxN BGR uint8 crop -> RGB CHW float, ImageNet norm.
        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        tensor = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self._mean) / self._std

    def _logits(
        self, image_bgr: np.ndarray, bbox_hint: Box | None
    ) -> dict[str, torch.Tensor] | None:
        """Every head's logits for the hinted vehicle, or ``None`` when there
        is no usable box (no hint, or a degenerate crop). The car has to be
        localised: the full 4K frame is not a single-vehicle crop."""
        crop = self._cropper.crop(image_bgr, bbox_hint)
        if crop is None:
            return None
        logits: dict[str, torch.Tensor] = self.model(self._preprocess(crop).unsqueeze(0))
        return logits


class TopOneClassifier(SnapClassifier):
    """A checkpoint with one softmax head over named classes.

    ``head`` is the head's name in the checkpoint (its class names are the
    metadata's ``<head>_names``); ``label`` is how error messages name the
    model. ``DEFAULT_PAD_FRAC`` is the pad for a checkpoint that didn't record
    its own.
    """

    head: ClassVar[str]
    label: ClassVar[str]
    DEFAULT_PAD_FRAC: ClassVar[float] = 0.1

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        pad_frac: float | None = None,
        input_size: int | None = None,
        crop_mode: str = "auto",
    ) -> None:
        super().__init__(
            checkpoint,
            device=device,
            pad_frac=pad_frac,
            input_size=input_size,
            crop_mode=crop_mode,
            legacy_pad_frac=self.DEFAULT_PAD_FRAC,
        )
        if self.head not in self.model.head_sizes:
            raise ValueError(
                f"checkpoint {checkpoint} is not a {self.label} model "
                f"(heads: {sorted(self.model.head_sizes)})"
            )
        self.class_names: tuple[str, ...] = tuple(self.meta.get(f"{self.head}_names") or ())
        if not self.class_names:
            raise ValueError(f"{self.label} checkpoint is missing '{self.head}_names' metadata")

    @torch.no_grad()
    def classify(self, image_bgr: np.ndarray, bbox_hint: Box | None) -> tuple[str, float] | None:
        """Top-1 ``(class, conf)`` for the hinted vehicle, or ``None`` when
        there is no usable box."""
        logits = self._logits(image_bgr, bbox_hint)
        if logits is None:
            return None
        probs = torch.softmax(logits[self.head], dim=1)[0]
        conf, idx = probs.max(dim=0)
        return self.class_names[int(idx)], float(conf)


def aggregate_top_one(
    per_image_rows: list[dict[str, Any]],
    conf_threshold: float,
    *,
    field: str,
    source_field: str | None = None,
) -> list[dict[str, Any]]:
    """Confidence-weighted vote of each snap's top-1 into a per-track best.

    Sums each class's top-1 confidence across a track's snaps and takes the
    argmax. ``field`` is emitted only when the winner's best single read
    clears ``conf_threshold`` (else ``None``); ``conf`` is kept regardless,
    for debugging. With ``source_field``, each row also says the value is a
    prediction (``"cnn"``), to tell it from a register value.
    """
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        if row[field] is not None:
            by_track[row["track_id"]].append(row)

    tracks: list[dict[str, Any]] = []
    for tid in sorted(by_track):
        rows = by_track[tid]
        weight: dict[str, float] = defaultdict(float)
        best_conf: dict[str, float] = defaultdict(float)
        n_high = 0
        for row in rows:
            name, c = row[field], row["conf"]
            weight[name] += c
            best_conf[name] = max(best_conf[name], c)
            if c >= conf_threshold:
                n_high += 1
        best = max(weight, key=lambda k: weight[k])
        conf = best_conf[best]
        track: dict[str, Any] = {"track_id": tid, field: best if conf >= conf_threshold else None}
        if source_field is not None:
            track[source_field] = "cnn"
        track.update(conf=round(conf, 4), n_high_conf_reads=n_high, n_reads=len(rows))
        tracks.append(track)
    return tracks


C = TypeVar("C", bound=SnapClassifier)


class SessionInference(abc.ABC, Generic[C]):
    """One inference command: ``streettracker <tag> <session_dir>``.

    :meth:`main` is the whole command. A subclass says how to load its
    classifier, what a snap's row holds, how rows become tracks and how to
    sum the run up; everything else is here.

    The lines printed are an interface: the control panel shows the
    ``[tag] wrote ...`` lines and scans the rest for trouble. Don't reword
    them without changing ``control/progress.py`` and its tests.
    """

    #: The command's name: the print prefix and the output files' stem.
    tag: ClassVar[str]
    default_conf_threshold: ClassVar[float]
    #: The command that trains this head, quoted when the checkpoint is missing.
    train_command: ClassVar[str]
    pad_help: ClassVar[str]
    input_size_help: ClassVar[str] = (
        "crop edge px fed to the model (default: the checkpoint's recorded input_size, else 224)"
    )
    #: Print a progress line every this many snaps (and after the last).
    progress_every: ClassVar[int] = 200

    # ---- what a subclass supplies -------------------------------------

    @abc.abstractmethod
    def default_model(self) -> Path:
        """The production checkpoint ``--model`` defaults to. A method, so it
        is read when the command runs, not when the module is imported."""

    @abc.abstractmethod
    def load(self, args: argparse.Namespace, device: str) -> C:
        """Build the classifier from the parsed arguments."""

    @abc.abstractmethod
    def describe(self, clf: C) -> str:
        """The classifier's heads, for the first line printed."""

    @abc.abstractmethod
    def predict(self, clf: C, image: np.ndarray | None, box: Box | None) -> dict[str, Any]:
        """The head's fields of one snap's row. ``image`` is ``None`` for a
        snap that could not be read."""

    @abc.abstractmethod
    def aggregate(self, rows: list[dict[str, Any]], conf_threshold: float) -> list[dict[str, Any]]:
        """Per-track rows from the per-image ones."""

    @abc.abstractmethod
    def summary(
        self,
        by_track: list[dict[str, Any]],
        conf_threshold: float,
        n_no_box: int,
        locator: SnapVehicleLocator,
    ) -> str:
        """The last line printed, after the tag."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:  # noqa: B027
        """Arguments only this command takes. None by default."""

    # ---- the command --------------------------------------------------

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog=f"streettracker {self.tag}")
        parser.add_argument("session_dir", type=Path, help="closed session directory")
        parser.add_argument(
            "--model", type=Path, default=self.default_model(), help="checkpoint .pt"
        )
        self.add_arguments(parser)
        parser.add_argument("--conf-threshold", type=float, default=self.default_conf_threshold)
        parser.add_argument("--pad-frac", type=float, default=None, help=self.pad_help)
        parser.add_argument(
            "--crop-mode",
            choices=("auto", "hint", "fullframe"),
            default="auto",
            help="where to crop the car: auto (default) = what the checkpoint was trained "
            "on (fullframe for plate-anchored corpora, legacy hint otherwise); fullframe = "
            "plate-anchored / nearest on-road vehicle; hint = stale fire-time bbox",
        )
        parser.add_argument(
            "--road-polygon",
            type=Path,
            default=Path(".claude/triggers_proposal.json"),
            help="road outline (vertices_frac) for the fullframe on-road filter",
        )
        parser.add_argument("--input-size", type=int, default=None, help=self.input_size_help)
        parser.add_argument("--limit", type=int, default=0, help="first N snaps only (0=all)")
        parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA present")
        return parser

    def main(self, argv: list[str] | None = None) -> int:
        args = self.build_parser().parse_args(argv)
        tag = f"[{self.tag}]"

        if not args.session_dir.is_dir():
            print(f"{tag} not a directory: {args.session_dir}")
            return 2
        if not args.model.exists():
            print(
                f"{tag} checkpoint not found: {args.model}\n"
                f"{' ' * len(tag)} train one (`{self.train_command}`) and copy its "
                f"best.pt here, or pass --model <path>."
            )
            return 1

        device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
        clf = self.load(args, device)

        snaps = discover_vehicle_snaps(args.session_dir)
        if not snaps:
            print(f"{tag} no vehicle_*_main_*.jpg snaps in {args.session_dir}")
            return 1
        if args.limit > 0:
            snaps = snaps[: args.limit]
        locator = SnapVehicleLocator(
            args.session_dir,
            clf.crop_mode,
            road_polygon_frac=(
                load_road_polygon(args.road_polygon, log_prefix=tag)
                if clf.crop_mode == "fullframe"
                else None
            ),
        )
        print(
            f"{tag} {len(snaps)} snaps; {locator.n_bboxes} per-snap bboxes; device={device}; "
            f"{self.describe(clf)}; input={clf.input_size}px; "
            f"crop={clf.crop_mode} pad={clf.pad_frac}"
        )

        per_image: list[dict[str, Any]] = []
        n_no_box = 0
        for i, (path, tid, snap_index, _cls) in enumerate(snaps, 1):
            image = read_image(path)
            box, source = locator.locate(path, tid, snap_index, image)
            if box is None:
                n_no_box += 1
            per_image.append(
                {
                    "track_id": tid,
                    "snap_index": snap_index,
                    "image": path.name,
                    "has_bbox": box is not None,
                    "crop_source": source,
                    **self.predict(clf, image, box),
                }
            )
            if i % self.progress_every == 0 or i == len(snaps):
                print(f"  {i}/{len(snaps)} classified")

        locator.close()
        by_track = self.aggregate(per_image, args.conf_threshold)

        label = args.session_dir.name
        img_path = args.session_dir / f"{label}_{self.tag}.json"
        track_path = args.session_dir / f"{label}_{self.tag}_by_track.json"
        atomic_write_text(img_path, json.dumps(per_image, indent=2))
        atomic_write_text(track_path, json.dumps({"tracks": by_track}, indent=2))

        print(f"{tag} wrote {img_path}")
        print(f"{tag} wrote {track_path}")
        print(f"{tag} {self.summary(by_track, args.conf_threshold, n_no_box, locator)}")
        return 0


T = TypeVar("T", bound=TopOneClassifier)


class TopOneInference(SessionInference[T]):
    """The command for a :class:`TopOneClassifier` head: one class and its
    confidence per snap, one voted class per track."""

    classifier: ClassVar[type[TopOneClassifier]]
    #: Name a per-track row's value as a prediction under this key, if set.
    source_field: ClassVar[str | None] = None
    default_conf_threshold: ClassVar[float] = 0.5

    @property
    def field(self) -> str:
        return self.classifier.head

    def load(self, args: argparse.Namespace, device: str) -> T:
        clf = self.classifier(
            args.model,
            device=device,
            pad_frac=args.pad_frac,
            input_size=args.input_size,
            crop_mode=args.crop_mode,
        )
        return clf  # type: ignore[return-value]

    def describe(self, clf: T) -> str:
        return f"{len(clf.class_names)} classes"

    def predict(self, clf: T, image: np.ndarray | None, box: Box | None) -> dict[str, Any]:
        pred = clf.classify(image, box) if image is not None else None
        return {
            self.field: pred[0] if pred else None,
            "conf": round(pred[1], 4) if pred else None,
        }

    def aggregate(self, rows: list[dict[str, Any]], conf_threshold: float) -> list[dict[str, Any]]:
        return aggregate_top_one(
            rows, conf_threshold, field=self.field, source_field=self.source_field
        )

    def summary(
        self,
        by_track: list[dict[str, Any]],
        conf_threshold: float,
        n_no_box: int,
        locator: SnapVehicleLocator,
    ) -> str:
        n_conf = sum(1 for t in by_track if t[self.field] is not None)
        dist: dict[str, int] = defaultdict(int)
        for t in by_track:
            if t[self.field]:
                dist[t[self.field]] += 1
        return f"{len(by_track)} tracks, {n_conf} confident (>= {conf_threshold}); " + " ".join(
            f"{k}={v}" for k, v in sorted(dist.items(), key=lambda x: -x[1])
        )
