"""makemodel-compare: shared held-out cars, per-track scoring, paired CI,
and the end-to-end head-to-head on tiny random-init checkpoints.

torch / torchvision / cv2 aren't on CI, so the end-to-end tests skip there;
the pure scoring helpers run everywhere torch imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("cv2")

from streettracker.analysis.makemodel.compare import (  # noqa: E402
    HeldOutCar,
    _colour_group,
    compare,
    held_out_cars,
    load_contender,
    main,
    paired_delta,
    score_rows,
)
from streettracker.analysis.makemodel.model import MakeModelNet, save_checkpoint  # noqa: E402
from streettracker.analysis.makemodel.uk_dataset import split_val_cars  # noqa: E402

SESS = "session_20260901_120000"
# 4 FORD + 4 AUDI cars, one track each (track id = index + 1).
_CARS = [(f"FD{i}", "FORD") for i in range(4)] + [(f"AU{i}", "AUDI") for i in range(4)]
_COLOUR = {"FD0": "white", "FD1": "silver", "FD2": "white", "FD3": "silver"} | {
    f"AU{i}": "black" for i in range(4)
}


def _corpus(root: Path) -> Path:
    corpus = root / "uk_crops_test"
    corpus.mkdir(parents=True)
    samples = [
        {
            "path": f"{mk}/{car}_{SESS}_{i + 1}_1.jpg",
            "make": mk,
            "car": car,
            "colour": _COLOUR[car],
            # AU3's model has no body-type mapping: dropped like the trainer drops it.
            "body_type": "" if car == "AU3" else ("hatchback" if mk == "FORD" else "suv"),
        }
        for i, (car, mk) in enumerate(_CARS)
    ]
    (corpus / "manifest.json").write_text(
        json.dumps({"makes": ["AUDI", "FORD"], "crop_mode": "plate", "samples": samples})
    )
    return corpus


def _session(output_root: Path) -> Path:
    sd = output_root / SESS
    sd.mkdir(parents=True)
    records = []
    for i in range(len(_CARS)):
        tid = i + 1
        for n in (1, 2):
            Image.new("RGB", (400, 300), (90, 90, 90)).save(sd / f"vehicle_{tid}_main_{n}.jpg")
        records.append(
            {
                "track_id": tid,
                "main_snaps": [1, 2],
                "main_snap_bboxes": [[64, 36, 200, 180], [64, 36, 200, 180]],
            }
        )
    (sd / f"{SESS}_data.json").write_text(json.dumps(records))
    (sd / f"{SESS}_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    return sd


def _ckpt(path: Path, makes: list[str], **meta: object) -> Path:
    net = MakeModelNet({"make": len(makes)}, pretrained=False)
    save_checkpoint(net, path, metadata={"make_names": makes, "input_size": 64, **meta})
    return path


def _stub_detector(image: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    return [(200.0, 100.0, 320.0, 200.0, 0.9)]


# ----------------------------------------------------------------------
# Held-out set + scoring (pure).


def test_held_out_cars_match_trainer_split_and_exclusions(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    samples = json.loads((corpus / "manifest.json").read_text())["samples"]
    val = split_val_cars(samples, val_frac=0.5)
    cars = held_out_cars(corpus, val_frac=0.5)
    assert {c.car for c in cars} == val  # exactly the trainer's val cars
    excluded = sorted(val)[0]
    cars2 = held_out_cars(corpus, val_frac=0.5, exclude_cars={excluded})
    assert {c.car for c in cars2} == val - {excluded}
    assert all(c.tracks == [(SESS, int(c.car[-1]) + (1 if c.label == "FORD" else 5))] for c in cars)


def test_score_rows_per_track_and_shared_makes() -> None:
    cars = [HeldOutCar("A", "FORD", [("s", 1), ("s", 2)]), HeldOutCar("B", "KIA", [("s", 3)])]
    preds = {
        "production": {("s", 1): [("FORD", 0.9)], ("s", 2): [("AUDI", 0.6)]},  # KIA unknown
        "candidate": {
            ("s", 1): [("FORD", 0.8)],
            ("s", 2): [("FORD", 0.7), ("AUDI", 0.3)],
            ("s", 3): [("KIA", 0.9)],
        },
    }
    known = {"production": {"FORD", "AUDI"}, "candidate": {"FORD", "AUDI", "KIA"}}
    rows = score_rows(cars, preds, known)
    assert rows["production"]["track_acc"] == round(1 / 3, 4)
    assert rows["production"]["coverage"] == round(2 / 3, 4)  # track 3 unpredicted = wrong
    assert rows["candidate"]["track_acc"] == 1.0
    # Shared-makes view drops KIA (production can't name it).
    assert rows["production"]["track_acc_shared_classes"] == 0.5
    assert rows["candidate"]["track_acc_shared_classes"] == 1.0


def test_paired_delta_bootstraps_by_car() -> None:
    a = {"c1": [1, 1], "c2": [1], "c3": [0]}
    b = {"c1": [0, 1], "c2": [0], "c3": [0]}
    point, lo, hi = paired_delta(a, b, iters=500)
    assert point == pytest.approx(2 / 4)
    assert lo <= point <= hi
    assert paired_delta(a, a, iters=200) == (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# End to end.


def test_compare_scores_each_model_with_its_own_crops(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    out = tmp_path / "output"
    _session(out)
    prod = _ckpt(tmp_path / "prod.pt", ["AUDI", "FORD"])  # legacy: hint crops
    (tmp_path / "run").mkdir()
    cand = _ckpt(
        tmp_path / "run" / "best.pt", ["AUDI", "FORD", "KIA"], crop_mode="plate", crop_pad_frac=0.1
    )

    report = compare(
        corpus,
        cand,
        prod,
        output_root=out,
        val_frac=0.5,
        vehicle_detector=_stub_detector,
    )
    rows = {r["name"]: r for r in report["rows"]}
    assert set(rows) == {"production", "production_clean_crops", "candidate"}
    assert rows["production"]["crop_mode"] == "hint" and rows["production"]["pad_frac"] == 0.25
    assert rows["production_clean_crops"]["crop_mode"] == "fullframe"
    assert rows["production_clean_crops"]["pad_frac"] == 0.1
    assert rows["candidate"]["crop_mode"] == "fullframe"
    assert report["n_cars"] == 4 and report["n_tracks"] == 4 and report["n_snaps"] == 8
    assert all(r["coverage"] == 1.0 for r in rows.values())
    lo, hi = report["delta"]["ci95"]
    assert lo <= report["delta"]["candidate_minus_production"] <= hi
    assert report["production"]["size"] == prod.stat().st_size
    # Production-on-clean-crops vs as-deployed gets its own paired CI.
    clean = report["production_clean_crop_delta"]
    assert clean["ci95"][0] <= clean["clean_minus_deployed"] <= clean["ci95"][1]


def test_main_requires_leakage_guard_then_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import streettracker.analysis.vehicle_locator as vl

    monkeypatch.setattr(vl, "yolo_vehicle_detector", lambda *a, **k: _stub_detector)
    corpus = _corpus(tmp_path)
    out = tmp_path / "output"
    _session(out)
    prod = _ckpt(tmp_path / "prod.pt", ["AUDI", "FORD"])
    (tmp_path / "run").mkdir()
    cand = _ckpt(tmp_path / "run" / "best.pt", ["AUDI", "FORD"], crop_mode="plate")
    base = [str(corpus), "--candidate", str(cand), "--production", str(prod), "--cpu"]
    base += ["--output-root", str(out), "--val-frac", "0.5"]
    base += ["--road-polygon", str(tmp_path / "none.json")]

    # No sidecar naming production's training corpus -> refuse (it could be
    # scored on cars it trained on).
    assert main(base + ["--runs-dir", str(tmp_path)]) == 1

    # Production's corpus = a disjoint set of cars: nothing excluded.
    other = tmp_path / "uk_crops_prod"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"samples": [{"car": "ZZ1"}]}))
    assert main(base + ["--exclude-corpus", str(other)]) == 0
    report = json.loads((tmp_path / "run" / "compare.json").read_text())
    assert report["n_cars"] == 4
    assert report["excluded_corpora"] == [str(other)]
    assert report["eval_sessions"] == []


def test_main_eval_session_scores_only_fresh_cars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import streettracker.analysis.vehicle_locator as vl

    monkeypatch.setattr(vl, "yolo_vehicle_detector", lambda *a, **k: _stub_detector)
    corpus = _corpus(tmp_path)
    out = tmp_path / "output"
    sd = _session(out)
    # FD0 is in the candidate's corpus (excluded); NEW1 / NEW2 are fresh cars.
    labels = {
        "FD0": {"make": "FORD", "track_ids": [1]},
        "NEW1": {"make": "FORD", "track_ids": [2]},
        "NEW2": {"make": "AUDI", "track_ids": [5]},
    }
    (sd / f"{SESS}_dvsa_labels.json").write_text(json.dumps({"labels": labels}))
    prod = _ckpt(tmp_path / "prod.pt", ["AUDI", "FORD"])
    (tmp_path / "run").mkdir()
    cand = _ckpt(tmp_path / "run" / "best.pt", ["AUDI", "FORD"], crop_mode="plate")
    (tmp_path / "run" / "compare.json").write_text('{"old": true}')
    other = tmp_path / "uk_crops_prod"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"samples": [{"car": "ZZ1"}]}))
    args = [str(corpus), "--candidate", str(cand), "--production", str(prod), "--cpu"]
    args += ["--output-root", str(out), "--road-polygon", str(tmp_path / "none.json")]
    args += ["--exclude-corpus", str(other)]

    assert main(args + ["--eval-session", "session_20990101_000000"]) == 1  # no labels there
    assert main(args + ["--eval-session", SESS]) == 0
    report = json.loads((tmp_path / "run" / "compare.json").read_text())
    assert report["eval_sessions"] == [SESS]
    assert report["n_cars"] == 2 and report["n_tracks"] == 2 and report["n_snaps"] == 4
    assert report["n_excluded_cars"] == 9  # 8 candidate-corpus cars + ZZ1
    # The report it replaced is kept beside it.
    assert json.loads((tmp_path / "run" / "compare.prev.json").read_text()) == {"old": True}

    # --include-trained-cars keeps FD0 (a corpus car) and needs no production
    # corpus to exclude; without --eval-session it is refused.
    base = [str(corpus), "--candidate", str(cand), "--production", str(prod), "--cpu"]
    base += ["--output-root", str(out), "--road-polygon", str(tmp_path / "none.json")]
    base += ["--runs-dir", str(tmp_path)]  # no production sidecar here
    assert main(base + ["--include-trained-cars"]) == 2
    assert main(base + ["--eval-session", SESS, "--include-trained-cars"]) == 0
    report = json.loads((tmp_path / "run" / "compare.json").read_text())
    assert report["include_trained_cars"] is True
    assert report["n_cars"] == 3 and report["n_excluded_cars"] == 0
    assert report["excluded_corpora"] == []


# ----------------------------------------------------------------------
# Colour + body-type heads (--target).

_COLOURS = ["white", "silver", "black"]


def _head_ckpt(path: Path, head: str, names: list[str], **meta: object) -> Path:
    net = MakeModelNet({head: len(names)}, pretrained=False)
    save_checkpoint(net, path, metadata={f"{head}_names": names, "input_size": 64, **meta})
    return path


def test_held_out_cars_by_colour_use_the_colour_split(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    samples = json.loads((corpus / "manifest.json").read_text())["samples"]
    val = split_val_cars(samples, label_field="colour", val_frac=0.5)
    cars = held_out_cars(corpus, val_frac=0.5, target="colour")
    assert {c.car for c in cars} == val  # the colour trainer's val cars
    assert {c.label for c in cars} <= set(_COLOURS)


def test_held_out_cars_drop_cars_without_the_target_label(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    cars = held_out_cars(corpus, val_frac=0.5, target="body_type")
    assert "AU3" not in {c.car for c in cars}  # no body-type mapping
    assert {c.label for c in cars} <= {"hatchback", "suv"}


def test_score_rows_grouped_colour() -> None:
    """Grouped colour counts a near-miss inside one family (silver for a
    white car) as right, but not a different family (blue for black)."""
    cars = [HeldOutCar("A", "white", [("s", 1)]), HeldOutCar("B", "black", [("s", 2)])]
    preds = {"production": {("s", 1): [("silver", 0.9)], ("s", 2): [("blue", 0.8)]}}
    rows = score_rows(cars, preds, {"production": set(_COLOURS)}, group=_colour_group)
    assert rows["production"]["track_acc"] == 0.0
    assert rows["production"]["track_acc_grouped"] == 0.5
    assert rows["production"]["_track_hits_grouped"] == {"A": [1], "B": [0]}


def test_compare_colour_end_to_end(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    out = tmp_path / "output"
    _session(out)
    prod = _head_ckpt(tmp_path / "colour_prod.pt", "colour", _COLOURS)  # legacy: hint crops
    (tmp_path / "run").mkdir()
    cand = _head_ckpt(
        tmp_path / "run" / "best.pt", "colour", _COLOURS, crop_mode="plate", crop_pad_frac=0.1
    )
    report = compare(
        corpus,
        cand,
        prod,
        output_root=out,
        val_frac=0.5,
        vehicle_detector=_stub_detector,
        target="colour",
    )
    assert report["target"] == "colour"
    rows = {r["name"]: r for r in report["rows"]}
    assert set(rows) == {"production", "production_clean_crops", "candidate"}
    assert rows["production"]["crop_mode"] == "hint" and rows["production"]["pad_frac"] == 0.1
    assert rows["candidate"]["crop_mode"] == "fullframe"
    assert all("track_acc_grouped" in r for r in rows.values())
    g = report["grouped_delta"]
    assert g["ci95"][0] <= g["candidate_minus_production"] <= g["ci95"][1]
    assert report["production"]["n_classes"] == 3 and "n_makes" not in report["production"]


def test_load_contender_body_type(tmp_path: Path) -> None:
    ckpt = _head_ckpt(tmp_path / "bt.pt", "body_type", ["hatchback", "suv"])
    c = load_contender("body_type", ckpt, device="cpu")
    assert c.classes == {"hatchback", "suv"}
    assert (c.crop_mode, c.pad_frac) == ("hint", 0.1)
    pred = c.predict(np.zeros((300, 400, 3), np.uint8), (50, 40, 200, 180))
    assert pred is not None and pred[0] in {"hatchback", "suv"}
    with pytest.raises(ValueError, match="target"):
        load_contender("wheels", ckpt, device="cpu")
