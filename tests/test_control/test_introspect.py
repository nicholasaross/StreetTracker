"""Tests for local introspection: session inventory, corpus + model stats,
training-run summaries, and the retrain / promote recommendations."""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.control import introspect
from streettracker.control.introspect import (
    CorpusInfo,
    ModelInfo,
    TrainingRunInfo,
    promote_recommendation,
    retrain_recommendation,
)

SESSION = "session_20260526_120000"
_JPEG = b"\xff\xd8\xff\xe0FAKE\xff\xd9"


def _make_session(root: Path, label: str = SESSION, *, n_snaps: int = 3) -> Path:
    d = root / label
    d.mkdir(parents=True)
    records = [
        {"track_id": 1, "class_name": "car", "time_start_unix": 1000.0, "time_end_unix": 1006.0},
        {"track_id": 2, "class_name": "car", "time_start_unix": 1010.0, "time_end_unix": 1020.0},
        {"track_id": 3, "class_name": "person", "time_start_unix": 1005.0, "time_end_unix": 1008.0},
    ]
    (d / f"{label}_data.json").write_text(json.dumps(records))
    (d / f"{label}_meta.json").write_text(
        json.dumps({"session_label": label, "session_start_unix": 990.0, "frame_size": [896, 512]})
    )
    for i in range(1, n_snaps + 1):
        (d / f"vehicle_{i}_main_1.jpg").write_bytes(_JPEG)
    # person crops must NOT be counted as snaps.
    (d / "person_9_main_1.jpg").write_bytes(_JPEG)
    return d


def test_session_info_counts_and_span(tmp_path: Path) -> None:
    d = _make_session(tmp_path, n_snaps=4)
    info = introspect.session_info(d)
    assert info.label == SESSION
    assert info.n_tracks == 3
    assert info.n_cars == 2
    assert info.n_persons == 1
    assert info.n_vehicle_snaps == 4  # vehicle main snaps
    assert info.n_person_snaps == 1  # the one person_9_main_1.jpg _make_session writes
    assert info.start_unix == 990.0  # meta wins over first track
    assert info.end_unix == 1020.0
    assert info.duration_s == 30.0
    # No enrichment files written yet.
    assert info.has_alpr is False
    assert info.n_dvsa_labels == 0
    assert info.has_vehicles is False
    assert info.has_makemodel is False
    assert info.has_people is False


def test_session_info_enrichment_badges(tmp_path: Path) -> None:
    d = _make_session(tmp_path)
    (d / f"{SESSION}_alpr_by_track.json").write_text("{}")
    (d / f"{SESSION}_dvsa_labels.json").write_text(
        json.dumps({"labels": {"AB12CDE": {}, "XY34ZZZ": {}}})
    )
    (d / f"{SESSION}_vehicles.json").write_text("[]")
    (d / f"{SESSION}_makemodel_by_track.json").write_text("{}")
    (d / f"{SESSION}_people.json").write_text("{}")
    info = introspect.session_info(d)
    assert info.has_alpr is True
    assert info.n_dvsa_labels == 2
    assert info.has_vehicles is True
    assert info.has_makemodel is True
    assert info.has_people is True


def test_session_info_events_fallback(tmp_path: Path) -> None:
    """With no data.json, counts come from events.jsonl (torn last line tolerated)."""
    label = "session_20260101_000000"
    d = tmp_path / label
    d.mkdir(parents=True)
    lines = [
        json.dumps(
            {"track_id": 1, "class_name": "car", "time_start_unix": 1.0, "time_end_unix": 2.0}
        ),
        json.dumps(
            {"track_id": 2, "class_name": "person", "time_start_unix": 3.0, "time_end_unix": 4.0}
        ),
        '{"track_id": 3, "class_name": "car"',  # torn tail
    ]
    (d / f"{label}_events.jsonl").write_text("\n".join(lines) + "\n")
    info = introspect.session_info(d)
    assert info.n_tracks == 2
    assert info.n_cars == 1


def test_discover_and_list_sessions_newest_first(tmp_path: Path) -> None:
    _make_session(tmp_path, "session_20260101_000000")
    _make_session(tmp_path, "session_20260201_000000")
    sessions = introspect.list_local_sessions(tmp_path)
    assert [s.label for s in sessions] == [
        "session_20260201_000000",
        "session_20260101_000000",
    ]


def test_latest_corpus_reads_manifest(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    d = runs / "uk_crops_0614_512"
    d.mkdir(parents=True)
    samples = [
        {"path": "FORD/a.jpg", "make": "FORD", "car": "AB12CDE"},
        {"path": "FORD/b.jpg", "make": "FORD", "car": "AB12CDE"},
        {"path": "FORD/c.jpg", "make": "FORD", "car": "CD34EFG"},
        {"path": "BMW/d.jpg", "make": "BMW", "car": "EF56GHI"},
    ]
    (d / "manifest.json").write_text(json.dumps({"makes": ["BMW", "FORD"], "samples": samples}))
    corpus = introspect.latest_corpus(runs)
    assert corpus is not None
    assert corpus.name == "uk_crops_0614_512"
    assert corpus.n_crops == 4
    assert corpus.n_makes == 2
    assert corpus.n_cars == 3
    assert corpus.per_make_cars == {"BMW": 1, "FORD": 2}


def test_latest_corpus_none_when_empty(tmp_path: Path) -> None:
    assert introspect.latest_corpus(tmp_path / "nope") is None
    (tmp_path / "runs").mkdir()
    assert introspect.latest_corpus(tmp_path / "runs") is None


def test_model_info_prefers_sidecar(tmp_path: Path) -> None:
    model = tmp_path / "makemodel_b0.pt"
    model.write_bytes(b"not-a-real-checkpoint")
    sidecar = tmp_path / "makemodel_b0.meta.json"
    sidecar.write_text(
        json.dumps(
            {
                "arch": "efficientnet_b5",
                "input_size": 456,
                "n_makes": 36,
                "val_make_top1": 0.402,
                "promoted_at": "2026-06-14T10:00:00+01:00",
                "trained_corpus": {"n_cars": 1200, "n_makes": 36, "n_crops": 32777},
            }
        )
    )
    # allow_torch True but the sidecar means the bogus checkpoint is never read.
    info = introspect.model_info(model, allow_torch=True)
    assert info is not None
    assert info.source == "sidecar"
    assert info.arch == "efficientnet_b5"
    assert info.val_make_top1 == 0.402
    assert info.trained_corpus["n_cars"] == 1200


def test_model_info_unavailable_without_torch(tmp_path: Path) -> None:
    model = tmp_path / "makemodel_b0.pt"
    model.write_bytes(b"x")
    info = introspect.model_info(model, allow_torch=False)
    assert info is not None
    assert info.source == "unavailable"


def test_model_info_none_when_missing(tmp_path: Path) -> None:
    assert introspect.model_info(tmp_path / "absent.pt") is None


def test_training_runs_sorted(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for name, acc in (("uk_make_0611_b5", 0.349), ("uk_make_0614_b5", 0.402)):
        d = runs / name
        d.mkdir(parents=True)
        (d / "history.json").write_text(
            json.dumps(
                {
                    "best_epoch": 16,
                    "best_val_make_top1": acc,
                    "makes": ["FORD", "BMW"],
                    "history": [{"epoch": 1}, {"epoch": 2}],
                }
            )
        )
    got = introspect.training_runs(runs)
    assert {r.name for r in got} == {"uk_make_0611_b5", "uk_make_0614_b5"}
    best = max(got, key=lambda r: r.best_val_make_top1 or 0)
    assert best.name == "uk_make_0614_b5"
    assert best.n_makes == 2
    assert best.n_epochs == 2


# ---- recommendations (logic-only, no fixtures) ----


def _model(acc: float | None, trained: dict | None) -> ModelInfo:
    return ModelInfo(
        path="m.pt", mtime=0.0, source="sidecar", val_make_top1=acc, trained_corpus=trained
    )


def _corpus(cars: int, makes: int, crops: int) -> CorpusInfo:
    return CorpusInfo(name="c", path="c", mtime=0.0, n_crops=crops, n_makes=makes, n_cars=cars)


def test_retrain_unknown_without_baseline() -> None:
    rec = retrain_recommendation(_corpus(1500, 36, 40000), _model(0.4, None))
    assert rec.verdict == "unknown"
    rec2 = retrain_recommendation(None, _model(0.4, {"n_cars": 1}))
    assert rec2.verdict == "unknown"


def test_retrain_recommend_on_growth() -> None:
    trained = {"n_cars": 1000, "n_makes": 33, "n_crops": 30000}
    rec = retrain_recommendation(_corpus(1500, 36, 40000), _model(0.4, trained))
    assert rec.verdict == "recommend"
    assert rec.evidence["deltas"]["cars"] == 500


def test_retrain_hold_below_threshold() -> None:
    trained = {"n_cars": 1000, "n_makes": 36, "n_crops": 30000}
    rec = retrain_recommendation(_corpus(1010, 36, 30100), _model(0.4, trained))
    assert rec.verdict == "hold"


def test_promote_recommend_when_candidate_wins() -> None:
    runs = [TrainingRunInfo("uk_make_0614_b5", "p", 1.0, 16, 0.402, 36, 30)]
    rec = promote_recommendation(_model(0.349, None), runs)
    assert rec.verdict == "recommend"
    assert rec.evidence["candidate"]["name"] == "uk_make_0614_b5"


def test_promote_hold_when_no_gain() -> None:
    runs = [TrainingRunInfo("uk_make_0614_b5", "p", 1.0, 16, 0.351, 36, 30)]
    rec = promote_recommendation(_model(0.349, None), runs)
    assert rec.verdict == "hold"


def test_promote_unknown_without_runs_or_prod_acc() -> None:
    assert promote_recommendation(_model(0.3, None), []).verdict == "unknown"
    runs = [TrainingRunInfo("uk_make_x", "p", 1.0, 1, 0.4, 36, 10)]
    assert promote_recommendation(_model(None, None), runs).verdict == "unknown"


def test_local_snapshot_composes(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _make_session(out)
    snap = introspect.local_snapshot(out, tmp_path / "runs", tmp_path / "absent.pt")
    assert snap["totals"]["n_sessions"] == 1
    assert snap["totals"]["n_cars"] == 2
    assert snap["corpus"] is None
    assert snap["model"] is None
    assert snap["recommendations"]["retrain"]["kind"] == "retrain"
    assert snap["recommendations"]["promote"]["kind"] == "promote"
