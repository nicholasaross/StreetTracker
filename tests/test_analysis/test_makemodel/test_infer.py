"""Make/model inference: classifier, per-track voting, and the CLI.

Builds a tiny random-init checkpoint on the fly (no training, no hub
download) and runs the real classify + aggregate + CLI paths against a
synthetic session. torch / torchvision aren't on CI, so the module is
skipped there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from streettracker.analysis.makemodel.dataset import build_labels  # noqa: E402
from streettracker.analysis.makemodel.infer import (  # noqa: E402
    Candidate,
    MakeModelClassifier,
    aggregate_by_track,
    main,
)
from streettracker.analysis.makemodel.model import MakeModelNet, save_checkpoint  # noqa: E402

_ROWS = [("Audi", "Audi Q5", 5), ("BMW", "BMW 3 Series", 10), ("Audi", "Audi A4", 20)]
_MODEL_TO_MAKE = {"Audi Q5": "Audi", "BMW 3 Series": "BMW", "Audi A4": "Audi"}


def _checkpoint(tmp_path: Path) -> Path:
    labels = build_labels(_ROWS)
    net = MakeModelNet(labels.head_sizes, pretrained=False)
    ckpt = tmp_path / "mm.pt"
    save_checkpoint(net, ckpt, metadata={"labels": labels.to_metadata()})
    return ckpt


# ----------------------------------------------------------------------
# Classifier.


def test_classify_returns_consistent_topk(tmp_path: Path) -> None:
    clf = MakeModelClassifier(_checkpoint(tmp_path), device="cpu", top_k=2)
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cands = clf.classify(image, (50, 40, 200, 180))
    assert len(cands) == 2
    assert all(isinstance(c, Candidate) for c in cands)
    # Sorted high-to-low confidence.
    assert cands[0].conf >= cands[1].conf
    for c in cands:
        assert 0.0 <= c.conf <= 1.0
        assert c.year is None
        # make must be the predicted model's make (derived, not a separate head).
        assert c.make == _MODEL_TO_MAKE[c.model]


def test_classify_without_bbox_returns_empty(tmp_path: Path) -> None:
    clf = MakeModelClassifier(_checkpoint(tmp_path), device="cpu")
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    assert clf.classify(image, None) == []


# ----------------------------------------------------------------------
# Per-track aggregation (pure; no model).


def test_aggregate_confidence_weighted_vote() -> None:
    rows = [
        {"track_id": 7, "top_k": [{"make": "Audi", "model": "Audi Q5", "year": None, "conf": 0.9}]},
        {"track_id": 7, "top_k": [{"make": "Audi", "model": "Audi Q5", "year": None, "conf": 0.8}]},
        {
            "track_id": 7,
            "top_k": [{"make": "BMW", "model": "BMW 3 Series", "year": None, "conf": 0.7}],
        },
    ]
    out = aggregate_by_track(rows, conf_threshold=0.4)
    assert len(out) == 1
    t = out[0]
    # Audi Q5 weight 1.7 beats BMW 0.7.
    assert (t["make"], t["model"]) == ("Audi", "Audi Q5")
    assert t["conf"] == 0.9  # best single read of the winning class
    assert t["n_high_conf_reads"] == 3
    assert t["n_reads"] == 3


def test_aggregate_nulls_make_model_below_threshold() -> None:
    rows = [
        {"track_id": 1, "top_k": [{"make": "Audi", "model": "Audi Q5", "year": None, "conf": 0.2}]},
    ]
    out = aggregate_by_track(rows, conf_threshold=0.4)
    assert out[0]["make"] is None
    assert out[0]["model"] is None
    assert out[0]["conf"] == 0.2  # kept for debugging
    assert out[0]["n_high_conf_reads"] == 0


def test_aggregate_skips_tracks_with_no_candidates() -> None:
    rows = [{"track_id": 9, "top_k": []}]
    assert aggregate_by_track(rows, conf_threshold=0.4) == []


# ----------------------------------------------------------------------
# CLI.


def _session(tmp_path: Path) -> Path:
    sess = tmp_path / "session_x"
    sess.mkdir()
    Image.new("RGB", (400, 300), (120, 120, 120)).save(sess / "vehicle_1_main_1.jpg")
    sess.joinpath("session_x_data.json").write_text(
        json.dumps([{"track_id": 1, "main_snaps": [1], "main_snap_bboxes": [[64, 36, 200, 180]]}])
    )
    sess.joinpath("session_x_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    return sess


def test_cli_main_writes_outputs(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path)
    sess = _session(tmp_path)
    rc = main([str(sess), "--model", str(ckpt), "--cpu"])
    assert rc == 0

    per_image = json.loads((sess / "session_x_makemodel.json").read_text())
    assert per_image[0]["track_id"] == 1
    assert per_image[0]["has_bbox"] is True
    assert len(per_image[0]["top_k"]) >= 1

    by_track = json.loads((sess / "session_x_makemodel_by_track.json").read_text())
    assert by_track["tracks"][0]["track_id"] == 1


def test_cli_main_missing_checkpoint_returns_1(tmp_path: Path) -> None:
    sess = tmp_path / "s"
    sess.mkdir()
    assert main([str(sess), "--model", str(tmp_path / "nope.pt")]) == 1


def test_cli_main_not_a_dir_returns_2(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing"), "--model", str(_checkpoint(tmp_path))]) == 2
