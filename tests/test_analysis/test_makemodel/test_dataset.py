"""CompCars surveillance loader: label maps, split parsing, collation.

Exercised against a synthetic ``sv_data`` tree (a handful of dummy
jpgs) plus one real ``scipy.io.savemat`` round-trip, so the .mat
parsing is checked against scipy's actual cell-array representation
rather than a hand-rolled stand-in. No dependency on the 12 GB dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat
from torch.utils.data import DataLoader, default_collate

from streettracker.analysis.makemodel.dataset import (
    CompCarsSurveillance,
    SurveillanceLabels,
    build_datasets,
    build_eval_transform,
    build_labels,
    build_train_transform,
    read_make_model_rows,
)

# (make, model, web_id) in surveillance_model_id order -> ids 1, 2, 3.
_ROWS = [
    ("Audi", "Audi Q5", 5),
    ("BMW", "BMW 3 Series", 10),
    ("Audi", "Audi A4", 20),
]


# ----------------------------------------------------------------------
# Label maps.


def test_build_labels_indices_are_sorted_and_linked() -> None:
    labels = build_labels(_ROWS)
    # Makes sorted-unique for determinism.
    assert labels.make_names == ("Audi", "BMW")
    # Model index == surveillance_model_id - 1, in file order.
    assert labels.model_names == ("Audi Q5", "BMW 3 Series", "Audi A4")
    # make_of_model[model_idx] -> make_idx. Models 0 & 2 are Audi (0), 1 is BMW (1).
    assert labels.make_of_model == (0, 1, 0)
    assert labels.head_sizes == {"make": 2, "model": 3}


def test_labels_metadata_roundtrip() -> None:
    labels = build_labels(_ROWS)
    restored = SurveillanceLabels.from_metadata(labels.to_metadata())
    assert restored == labels


def test_read_make_model_rows_parses_real_mat(tmp_path: Path) -> None:
    """savemat writes a genuine MATLAB cell array; read_make_model_rows
    must unwrap scipy's nested-singleton representation correctly."""
    cells = np.empty((3, 3), dtype=object)
    for i, (make, model, web_id) in enumerate(_ROWS):
        cells[i, 0] = np.array([make])
        cells[i, 1] = np.array([model])
        cells[i, 2] = np.array([[web_id]])
    mat_path = tmp_path / "sv_make_model_name.mat"
    savemat(mat_path, {"sv_make_model_name": cells})

    assert read_make_model_rows(mat_path) == _ROWS


# ----------------------------------------------------------------------
# Dataset over a synthetic sv_data tree.


@pytest.fixture
def sv_root(tmp_path: Path) -> Path:
    """A minimal sv_data tree: image/<id>/<name>.jpg for ids 1..3 plus
    a train + test split file."""
    root = tmp_path / "sv_data"
    img = root / "image"
    files = {
        "1/a.jpg": (200, 10, 10),
        "2/b.jpg": (10, 200, 10),
        "3/c.jpg": (10, 10, 200),
    }
    for rel, color in files.items():
        p = img / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 32), color).save(p, "JPEG")
    (root / "train_surveillance.txt").write_text("1/a.jpg\n3/c.jpg\n")
    (root / "test_surveillance.txt").write_text("2/b.jpg\n")
    return root


def test_dataset_yields_image_and_linked_labels(sv_root: Path) -> None:
    labels = build_labels(_ROWS)
    ds = CompCarsSurveillance(sv_root, "train", labels=labels, transform=build_eval_transform())
    assert len(ds) == 2

    img, lab = ds[0]  # "1/a.jpg" -> model 0 (Audi Q5), make 0 (Audi)
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert lab == {"make": 0, "model": 0}

    _, lab2 = ds[1]  # "3/c.jpg" -> model 2 (Audi A4), make 0 (Audi)
    assert lab2 == {"make": 0, "model": 2}


def test_dataset_default_collate_batches_label_dict(sv_root: Path) -> None:
    """The default collate must turn list[dict[str,int]] into
    dict[str, LongTensor] so the multi-head loss can index per head."""
    labels = build_labels(_ROWS)
    ds = CompCarsSurveillance(sv_root, "train", labels=labels, transform=build_eval_transform())
    images, batched = default_collate([ds[0], ds[1]])
    assert images.shape == (2, 3, 224, 224)
    assert torch.equal(batched["make"], torch.tensor([0, 0]))
    assert torch.equal(batched["model"], torch.tensor([0, 2]))


def test_dataloader_end_to_end(sv_root: Path) -> None:
    labels = build_labels(_ROWS)
    ds = CompCarsSurveillance(sv_root, "train", labels=labels, transform=build_train_transform())
    # num_workers=0 keeps it in-process (Windows-safe for the test).
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    images, batched = next(iter(loader))
    assert images.shape == (2, 3, 224, 224)
    assert set(batched) == {"make", "model"}
    assert batched["model"].shape == (2,)


def test_build_datasets_reads_labels_from_disk(sv_root: Path) -> None:
    """build_datasets is the trainer entry point; it must load the .mat
    itself and share one label map across train + val."""
    cells = np.empty((3, 3), dtype=object)
    for i, (make, model, web_id) in enumerate(_ROWS):
        cells[i, 0] = np.array([make])
        cells[i, 1] = np.array([model])
        cells[i, 2] = np.array([[web_id]])
    savemat(sv_root / "sv_make_model_name.mat", {"sv_make_model_name": cells})

    train_ds, val_ds, labels = build_datasets(sv_root)
    assert len(train_ds) == 2
    assert len(val_ds) == 1
    assert labels.head_sizes == {"make": 2, "model": 3}
    assert train_ds.labels is val_ds.labels  # shared, not reloaded


# ----------------------------------------------------------------------
# Guards.


def test_unknown_split_rejected(sv_root: Path) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        CompCarsSurveillance(sv_root, "validation", labels=build_labels(_ROWS))


def test_model_id_outside_label_table_rejected(sv_root: Path) -> None:
    """A split referencing id 3 with only a 2-model label table is a
    data/label mismatch -- fail loud, not with a silent bad index."""
    two_models = build_labels(_ROWS[:2])
    with pytest.raises(ValueError, match="outside the label table"):
        CompCarsSurveillance(sv_root, "train", labels=two_models)


# ----------------------------------------------------------------------
# Transforms.


def test_transforms_produce_normalised_chw_tensor() -> None:
    pil = Image.new("RGB", (300, 200), (128, 64, 32))
    for transform in (build_train_transform(), build_eval_transform()):
        out = transform(pil)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32
