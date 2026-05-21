"""Lightweight tests for the ALPR CLI wrappers.

These avoid pulling in torch / EasyOCR by exercising:
- the top-level dispatcher's routing into each ALPR subcommand
- the snap-discovery and per-track rollup helpers in ``cli.alpr_run``
- the "no records yet" error paths in ``alpr-score`` and ``alpr-report``
"""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.cli import alpr_run, alpr_score


def test_top_level_help_lists_alpr_commands() -> None:
    from streettracker.cli import _HELP

    assert "alpr-run" in _HELP
    assert "alpr-score" in _HELP
    assert "alpr-label" in _HELP
    assert "alpr-report" in _HELP


def test_alpr_run_help_returns_zero(capsys) -> None:  # noqa: ANN001 — pytest fixture
    import pytest

    with pytest.raises(SystemExit) as exc:
        alpr_run.main(["--help"])
    assert exc.value.code == 0


# -------------------------- alpr_run helpers


def test_discover_snaps_filters_to_vehicle(tmp_path: Path) -> None:
    # Mixed vehicle + person + non-main files
    (tmp_path / "vehicle_1_main_1.jpg").write_bytes(b"")
    (tmp_path / "vehicle_1_main_2.jpg").write_bytes(b"")
    (tmp_path / "vehicle_2_main_1.jpg").write_bytes(b"")
    (tmp_path / "person_99_main_1.jpg").write_bytes(b"")
    (tmp_path / "vehicle_1_hq.jpg").write_bytes(b"")
    (tmp_path / "vehicle_1.jpg").write_bytes(b"")

    snaps = alpr_run._discover_snaps(tmp_path)
    names = [p.name for (p, *_rest) in snaps]
    # Vehicle main snaps only, sorted
    assert names == [
        "vehicle_1_main_1.jpg",
        "vehicle_1_main_2.jpg",
        "vehicle_2_main_1.jpg",
    ]
    # Parsed track_id / snap_index are passed through correctly
    by_name = {p.name: (tid, n) for (p, tid, n, _cls) in snaps}
    assert by_name["vehicle_1_main_2.jpg"] == (1, 2)


def test_rollup_by_track_picks_highest_confidence_per_pipeline() -> None:
    records = [
        {"pipeline": "bespoke", "track_id": 1, "snap_index": 1,
         "image": "a.jpg", "ocr_text": "ABC", "ocr_conf": 0.3, "det_conf": 0.9},
        {"pipeline": "bespoke", "track_id": 1, "snap_index": 2,
         "image": "b.jpg", "ocr_text": "ABC2", "ocr_conf": 0.7, "det_conf": 0.95},
        # Same track, different pipeline — independent best
        {"pipeline": "preferred", "track_id": 1, "snap_index": 1,
         "image": "a.jpg", "ocr_text": "ABC", "ocr_conf": 0.5, "det_conf": 0.9},
        # Records with no OCR are skipped entirely
        {"pipeline": "bespoke", "track_id": 2, "snap_index": 1,
         "image": "c.jpg", "ocr_text": None, "ocr_conf": None, "det_conf": None},
    ]
    rollup = alpr_run._rollup_by_track(records)
    by_tid = {t["track_id"]: t for t in rollup["tracks"]}
    assert 1 in by_tid
    assert by_tid[1]["best_bespoke"]["ocr_conf"] == 0.7
    assert by_tid[1]["best_bespoke"]["ocr_text"] == "ABC2"
    assert by_tid[1]["best_preferred"]["ocr_text"] == "ABC"
    # Track 2 had no OCR at all — should not appear
    assert 2 not in by_tid


# -------------------------- alpr_score paths


def test_alpr_score_returns_1_when_no_alpr_json(tmp_path: Path) -> None:
    rc = alpr_score.main([str(tmp_path)])
    assert rc == 1


def test_alpr_score_returns_2_when_session_dir_missing() -> None:
    rc = alpr_score.main([str(Path("/__does_not_exist__"))])
    assert rc == 2


def test_alpr_score_writes_scores_json_with_no_labels(
    tmp_path: Path, capsys  # noqa: ANN001
) -> None:
    """End-to-end happy path: ``records`` only, no labels file.

    Confirms ``alpr-score`` reads ``*_alpr.json``, computes the aggregate
    metrics, writes the sidecar JSON, and prints the human summary
    without crashing.
    """
    session = tmp_path / "session_demo"
    session.mkdir()
    records = [
        {
            "image": "vehicle_1_main_1.jpg", "image_path": "vehicle_1_main_1.jpg",
            "track_id": 1, "snap_index": 1, "class_name": "vehicle",
            "pipeline": "bespoke", "det_bbox": [1, 2, 3, 4], "det_conf": 0.9,
            "ocr_text": "ABC123", "ocr_raw": "ABC123", "ocr_conf": 0.8,
            "crop_path": None, "pipeline_ms": 1.0, "error": None,
        },
        {
            "image": "vehicle_1_main_1.jpg", "image_path": "vehicle_1_main_1.jpg",
            "track_id": 1, "snap_index": 1, "class_name": "vehicle",
            "pipeline": "preferred", "det_bbox": [1, 2, 3, 4], "det_conf": 0.85,
            "ocr_text": "ABC123", "ocr_raw": "ABC123", "ocr_conf": 0.9,
            "crop_path": None, "pipeline_ms": 1.0, "error": None,
        },
    ]
    (session / "session_demo_alpr.json").write_text(json.dumps(records))

    assert alpr_score.main([str(session)]) == 0

    scores = json.loads((session / "session_demo_alpr_scores.json").read_text())
    assert scores["n_records"] == 2
    assert scores["n_labels"] == 0
    assert scores["detection_rate"]["bespoke"] == 1.0
    assert scores["agreement"]["bespoke_vs_preferred_exact"] == 1.0
