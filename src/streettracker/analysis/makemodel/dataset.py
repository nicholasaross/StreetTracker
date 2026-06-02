"""CompCars surveillance-subset data plumbing for the make/model CNN.

The surveillance-nature subset is the training corpus (``docs/
makemodel_design.md`` §2/§5): ~44.5k frontal images from real
surveillance cameras across 281 car *models* / 68 *makes* -- a much
closer domain match to StreetTracker's oblique-front street scene than
the web-collected subset.

On-disk layout (from ``CompCars/sv_data/README``)::

    sv_data/
      image/<surveillance_model_id>/<name>.jpg   # id is 1..281
      train_surveillance.txt                      # one "<id>/<name>.jpg" per line
      test_surveillance.txt                       # (no label column -- the
                                                  #  id IS the leading path part)
      sv_make_model_name.mat                      # id -> (make, model, web_id)
      color_list.mat                              # per-image colour (unused here)

Two deliberate choices vs the generic plan in §5.3:

* **Use the official train/test split**, not a custom car_id 80/20.
  The surveillance subset ships ``train_surveillance.txt`` /
  ``test_surveillance.txt`` (31148 / 13333) and has no car_id grouping
  to split on; these are also the splits the arXiv paper's accuracy
  numbers (our §6 targets) are measured against.
* **Make + model heads only, no year.** The surveillance labels carry
  no year (that lives in the web-nature path hierarchy). See
  :mod:`streettracker.analysis.makemodel.model`.

Label index conventions (contiguous, 0-based, for the model heads):

* ``model index = surveillance_model_id - 1`` (0..280) -- the ids are
  dense 1..281, so no remap table is needed.
* ``make index`` = position in the sorted-unique make-name list
  (0..67). Sorting makes the mapping deterministic across re-runs so a
  re-loaded checkpoint's class indices always line up.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from PIL import Image as PILImage

# EfficientNet-B0's ImageNet pretrain expects these normalisation stats;
# the surveillance fine-tune keeps them so the backbone weights stay valid.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = 224

_SPLIT_FILES = {
    "train": "train_surveillance.txt",
    "test": "test_surveillance.txt",
}


@dataclasses.dataclass(frozen=True, slots=True)
class SurveillanceLabels:
    """Index<->name maps for the make + model heads.

    ``make_names[i]`` is the i-th make; ``model_names[j]`` the j-th
    model ("Audi Q5"). ``make_of_model[j]`` is the make index of model
    j, so a model prediction also yields its make for free.
    """

    make_names: tuple[str, ...]
    model_names: tuple[str, ...]
    make_of_model: tuple[int, ...]

    @property
    def n_makes(self) -> int:
        return len(self.make_names)

    @property
    def n_models(self) -> int:
        return len(self.model_names)

    @property
    def head_sizes(self) -> dict[str, int]:
        """``{"make": 68, "model": 281}`` -- pass straight to
        :class:`~streettracker.analysis.makemodel.model.MakeModelNet`."""
        return {"make": self.n_makes, "model": self.n_models}

    def to_metadata(self) -> dict[str, list[Any]]:
        """JSON/checkpoint-friendly form (round-trips via
        :meth:`from_metadata`); rides in the model checkpoint so the
        inference path can turn a head argmax back into "Ford Focus"."""
        return {
            "make_names": list(self.make_names),
            "model_names": list(self.model_names),
            "make_of_model": list(self.make_of_model),
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> SurveillanceLabels:
        return cls(
            make_names=tuple(meta["make_names"]),
            model_names=tuple(meta["model_names"]),
            make_of_model=tuple(int(x) for x in meta["make_of_model"]),
        )


def read_make_model_rows(mat_path: str | Path) -> list[tuple[str, str, int]]:
    """Parse ``sv_make_model_name.mat`` into ``(make, model, web_id)``
    rows in surveillance_model_id order (row 0 == id 1).

    The .mat is a 281x3 MATLAB cell array; scipy loads each cell as a
    nested singleton ndarray, so unwrap down to the scalar.
    """
    import numpy as np
    from scipy.io import loadmat  # type: ignore[import-untyped]  # scipy.io ships no stubs

    def _unwrap(cell: Any) -> Any:
        v = cell
        while isinstance(v, np.ndarray) and v.size == 1:
            v = v.reshape(-1)[0]
        return v

    arr = loadmat(str(mat_path))["sv_make_model_name"]
    rows: list[tuple[str, str, int]] = []
    for i in range(arr.shape[0]):
        make = str(_unwrap(arr[i, 0])).strip()
        model = str(_unwrap(arr[i, 1])).strip()
        web_id = int(_unwrap(arr[i, 2]))
        rows.append((make, model, web_id))
    return rows


def build_labels(rows: Sequence[tuple[str, str, int]]) -> SurveillanceLabels:
    """Turn ``(make, model, web_id)`` rows (model-id order) into the
    contiguous index maps the heads consume."""
    make_names = tuple(sorted({make for make, _, _ in rows}))
    make_index = {name: i for i, name in enumerate(make_names)}
    model_names = tuple(model for _, model, _ in rows)
    make_of_model = tuple(make_index[make] for make, _, _ in rows)
    return SurveillanceLabels(make_names, model_names, make_of_model)


def load_surveillance_labels(sv_root: str | Path) -> SurveillanceLabels:
    """Read + build labels straight from a ``sv_data`` directory."""
    return build_labels(read_make_model_rows(Path(sv_root) / "sv_make_model_name.mat"))


@dataclasses.dataclass(slots=True)
class _Sample:
    rel_path: str  # relative to sv_data/image, e.g. "1/abc.jpg"
    model_index: int
    make_index: int


class CompCarsSurveillance(Dataset):
    """One official split of the CompCars surveillance subset.

    Yields ``(image, {"make": make_idx, "model": model_idx})``. With a
    tensor-producing ``transform`` the default DataLoader collate stacks
    the images and turns the label dict into ``{"make": LongTensor,
    "model": LongTensor}``, which the multi-head loss consumes directly.

    ``labels`` may be injected (the trainer loads them once and shares
    the same maps between the train + val datasets); when omitted they
    are read from ``sv_root``.
    """

    def __init__(
        self,
        sv_root: str | Path,
        split: str,
        *,
        labels: SurveillanceLabels | None = None,
        transform: Callable[[PILImage.Image], torch.Tensor] | None = None,
    ) -> None:
        if split not in _SPLIT_FILES:
            raise ValueError(f"split must be one of {sorted(_SPLIT_FILES)}, got {split!r}")
        root = Path(sv_root)
        self._image_root = root / "image"
        self.labels = labels if labels is not None else load_surveillance_labels(root)
        self._transform = transform

        n_models = self.labels.n_models
        samples: list[_Sample] = []
        for line in (root / _SPLIT_FILES[split]).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            model_id = int(line.split("/", 1)[0])
            model_index = model_id - 1
            if not 0 <= model_index < n_models:
                raise ValueError(
                    f"model id {model_id} in {split} split is outside the "
                    f"label table (1..{n_models}) -- labels/data mismatch"
                )
            samples.append(_Sample(line, model_index, self.labels.make_of_model[model_index]))
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, int]]:
        from PIL import Image

        sample = self._samples[index]
        # convert("RGB") guards against the odd grayscale / CMYK jpg.
        image = Image.open(self._image_root / sample.rel_path).convert("RGB")
        out: Any = self._transform(image) if self._transform is not None else image
        return out, {"make": sample.make_index, "model": sample.model_index}

    def class_counts(self) -> dict[str, list[int]]:
        """Per-head sample counts (no image I/O), for inverse-frequency
        class weighting. ``counts["make"][i]`` is the number of samples
        whose make index is ``i``."""
        make_counts = [0] * self.labels.n_makes
        model_counts = [0] * self.labels.n_models
        for sample in self._samples:
            make_counts[sample.make_index] += 1
            model_counts[sample.model_index] += 1
        return {"make": make_counts, "model": model_counts}


def build_train_transform(
    input_size: int = DEFAULT_INPUT_SIZE,
) -> Callable[[PILImage.Image], torch.Tensor]:
    """Augmentation pipeline from design doc §5.4.

    Rotation first (on the full PIL image), then a random-resized crop
    that trims most of the rotation's black corners, then colour jitter.
    **No horizontal flip** -- the surveillance labels are pose-specific
    (frontal), and mirroring a right-hand-drive car is a domain mismatch.
    """
    from torchvision import transforms  # type: ignore[import-untyped]  # no py.typed

    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=5),  # cars sit oblique on-scene
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(
    input_size: int = DEFAULT_INPUT_SIZE,
) -> Callable[[PILImage.Image], torch.Tensor]:
    """Deterministic val/inference transform: resize, centre-crop,
    normalise. No augmentation."""
    from torchvision import transforms  # type: ignore[import-untyped]  # no py.typed

    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_datasets(
    sv_root: str | Path,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> tuple[CompCarsSurveillance, CompCarsSurveillance, SurveillanceLabels]:
    """Convenience wiring for the trainer: load labels once, build the
    train (augmented) + test (deterministic) datasets sharing them.

    Returns ``(train_ds, val_ds, labels)``.
    """
    labels = load_surveillance_labels(sv_root)
    train_ds = CompCarsSurveillance(
        sv_root, "train", labels=labels, transform=build_train_transform(input_size)
    )
    val_ds = CompCarsSurveillance(
        sv_root, "test", labels=labels, transform=build_eval_transform(input_size)
    )
    return train_ds, val_ds, labels
