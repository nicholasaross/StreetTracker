"""UK make dataset: make normalization, by-car split, extraction.

torch/torchvision/cv2 aren't on CI, so this module is skipped there.
The by-car split is the load-bearing correctness property -- a car's
near-duplicate frames must never straddle train and val.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("cv2")

from streettracker.analysis.makemodel.dataset import build_eval_transform  # noqa: E402
from streettracker.analysis.makemodel.uk_dataset import (  # noqa: E402
    UKMakeDataset,
    build_main,
    extract_crops,
    normalize_make,
    train_main,
)


def test_normalize_make() -> None:
    assert normalize_make("ford") == "FORD"
    assert normalize_make("  Mercedes-Benz ") == "MERCEDES-BENZ"
    assert normalize_make("land  rover") == "LAND ROVER"
    assert normalize_make(None) == ""


def _write_dataset(root: Path) -> None:
    """Manifest with 2 makes x 2 cars; FORD car P1 has 2 crops."""
    root.mkdir(parents=True, exist_ok=True)
    samples = [
        ("FORD", "P1", 2),
        ("FORD", "P2", 1),
        ("AUDI", "P3", 2),
        ("AUDI", "P4", 1),
    ]
    manifest_samples = []
    for make, car, n in samples:
        (root / make).mkdir(exist_ok=True)
        for i in range(n):
            rel = f"{make}/{car}_{i}.jpg"
            Image.new("RGB", (224, 224), (80, 80, 80)).save(root / rel)
            manifest_samples.append({"path": rel, "make": make, "car": car})
    (root / "manifest.json").write_text(
        json.dumps({"makes": ["AUDI", "FORD"], "samples": manifest_samples})
    )


def test_by_car_split_has_no_leakage(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    train = UKMakeDataset(tmp_path, "train", val_frac=0.5, seed=0)
    val = UKMakeDataset(tmp_path, "val", make_names=train.make_names, val_frac=0.5, seed=0)

    train_cars = {s.car for s in train._samples}
    val_cars = {s.car for s in val._samples}
    assert train_cars and val_cars
    assert train_cars.isdisjoint(val_cars)  # the core guarantee
    # All four crops + the P1 duplicate accounted for, none lost.
    assert len(train) + len(val) == 6
    assert train.make_names == val.make_names == ("AUDI", "FORD")


def test_dataset_yields_tensor_and_make_index(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    ds = UKMakeDataset(tmp_path, "train", transform=build_eval_transform(), val_frac=0.5)
    img, lab = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert lab["make"] in (0, 1)
    counts = ds.class_counts()
    assert set(counts) == {"make"}
    assert sum(counts["make"]) == len(ds)


def test_extract_crops_from_synthetic_session(tmp_path: Path) -> None:
    import cv2

    sess = tmp_path / "session_t"
    sess.mkdir()
    # Two snaps for track 1 (plate AB12CDE, a FORD).
    for n in (1, 2):
        cv2.imwrite(str(sess / f"vehicle_1_main_{n}.jpg"), np.full((300, 400, 3), 90, np.uint8))
    sess.joinpath("session_t_data.json").write_text(
        json.dumps(
            [
                {
                    "track_id": 1,
                    "main_snaps": [1, 2],
                    "main_snap_bboxes": [[60, 40, 200, 180], [60, 40, 200, 180]],
                }
            ]
        )
    )
    sess.joinpath("session_t_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    sess.joinpath("session_t_dvsa_labels.json").write_text(
        json.dumps({"labels": {"AB12CDE": {"make": "Ford", "model": "FOCUS", "track_ids": [1]}}})
    )

    out = tmp_path / "uk_crops"
    stats = extract_crops([sess], out, min_cars_per_make=1)
    assert stats["n_makes"] == 1
    assert stats["n_cars"] == 1
    assert stats["n_crops"] == 2  # both snaps cropped
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["makes"] == ["FORD"]
    assert len(manifest["samples"]) == 2
    assert all((out / s["path"]).exists() for s in manifest["samples"])


def test_extract_crops_drops_makes_below_threshold(tmp_path: Path) -> None:
    sess = tmp_path / "session_t"
    sess.mkdir()
    import cv2

    cv2.imwrite(str(sess / "vehicle_1_main_1.jpg"), np.full((300, 400, 3), 90, np.uint8))
    sess.joinpath("session_t_data.json").write_text(
        json.dumps([{"track_id": 1, "main_snaps": [1], "main_snap_bboxes": [[60, 40, 200, 180]]}])
    )
    sess.joinpath("session_t_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    sess.joinpath("session_t_dvsa_labels.json").write_text(
        json.dumps({"labels": {"AB12CDE": {"make": "Ford", "model": "X", "track_ids": [1]}}})
    )
    stats = extract_crops([sess], tmp_path / "out", min_cars_per_make=5)
    assert stats["n_makes"] == 0  # only 1 FORD car, below threshold 5


# ----------------------------------------------------------------------
# CLI entry points.


def test_build_main_cli(tmp_path: Path) -> None:
    import cv2

    sess = tmp_path / "session_t"
    sess.mkdir()
    cv2.imwrite(str(sess / "vehicle_1_main_1.jpg"), np.full((300, 400, 3), 90, np.uint8))
    sess.joinpath("session_t_data.json").write_text(
        json.dumps([{"track_id": 1, "main_snaps": [1], "main_snap_bboxes": [[60, 40, 200, 180]]}])
    )
    sess.joinpath("session_t_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    sess.joinpath("session_t_dvsa_labels.json").write_text(
        json.dumps({"labels": {"AB12CDE": {"make": "Ford", "model": "X", "track_ids": [1]}}})
    )
    out = tmp_path / "crops"
    rc = build_main([str(out), "--sessions", str(sess), "--min-cars-per-make", "1"])
    assert rc == 0
    assert (out / "manifest.json").exists()


def test_train_main_cli(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "crops")
    out = tmp_path / "run"
    rc = train_main(
        [
            str(tmp_path / "crops"),
            "--out",
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--no-pretrained",
            "--cpu",
        ]
    )
    assert rc == 0
    assert (out / "best.pt").exists()


def test_train_main_missing_manifest(tmp_path: Path) -> None:
    assert train_main([str(tmp_path / "nope")]) == 1
