"""``streettracker alpr-rescore``: recompute plate-read confidence from the
saved plate crops, without re-running detection.

The real recognizer needs fast-plate-ocr (not on CI), so a fake stands in:
it answers by the crop's width, which each test crop is written with.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from streettracker.analysis.alpr import preferred
from streettracker.analysis.alpr.base import PlateRead
from streettracker.cli import alpr_rescore

SESSION = "session_20260801_000000"

# crop width -> what the "OCR" reads on it
_READS: dict[int, PlateRead | None] = {
    40: PlateRead("AB12CDE", 0.94, "AB12CDE", [0.98, 0.98, 0.97, 0.96, 0.96, 0.96, 0.94]),
    50: PlateRead("AB12CDF", 0.30, "AB12CDF", [0.99, 0.99, 0.99, 0.99, 0.30, 0.99, 0.99]),
    60: None,
    70: PlateRead("XY12CDE", 0.91, "XY12CDE", [0.91, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99]),
}


def _fake_recognize(image: np.ndarray) -> PlateRead | None:
    return _READS[int(image.shape[1])]


class _FakeRecognizer:
    constructed = 0

    def __init__(self, model: str) -> None:
        type(self).constructed += 1
        self.model = model

    def recognize(self, image: np.ndarray) -> PlateRead | None:
        return _fake_recognize(image)


@pytest.fixture(autouse=True)
def _fake_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeRecognizer.constructed = 0
    monkeypatch.setattr(preferred, "FastPlateOcrRecognizer", _FakeRecognizer)


def _write_crop(path: Path, width: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((20, width, 3), 128, np.uint8))


def _rec(tid: int, pipeline: str, text: str | None, conf: float | None, crop: str | None) -> dict:
    image = f"vehicle_{tid}_main_1.jpg"
    return {
        "image": image,
        "image_path": f"/orig/{SESSION}/{image}",
        "track_id": tid,
        "snap_index": 1,
        "class_name": "vehicle",
        "pipeline": pipeline,
        "det_bbox": [1, 2, 3, 4] if crop is not None else None,
        "det_conf": 0.9 if crop is not None else None,
        "ocr_text": text,
        "ocr_raw": text,
        "ocr_conf": conf,
        "canonical_uk_shape": (text is not None and len(text) == 7) or None,
        "crop_path": crop,
        "pipeline_ms": 1.0,
        "error": None,
    }


def _session(tmp_path: Path, *, stamp: dict | None = None) -> Path:
    """Two preferred reads at the old ~1.0 confidence (a clean one and a
    shaky one), a bespoke read, and a snap with no detection."""
    d = tmp_path / SESSION
    crops = d / "alpr_crops"
    _write_crop(crops / "preferred" / "vehicle_1_main_1.jpg", 40)
    _write_crop(crops / "preferred" / "vehicle_2_main_1.jpg", 50)
    _write_crop(crops / "bespoke" / "vehicle_1_main_1.jpg", 40)
    records = [
        _rec(1, "preferred", "AB12CDE", 1.0, "C:/devbox/crops/vehicle_1_main_1.jpg"),
        _rec(2, "preferred", "AB12CDF", 0.998, "C:/devbox/crops/vehicle_2_main_1.jpg"),
        _rec(1, "bespoke", "ABC", 0.4, "C:/devbox/crops/b/vehicle_1_main_1.jpg"),
        _rec(3, "preferred", None, None, None),
    ]
    (d / f"{SESSION}_alpr.json").write_text(json.dumps(records))
    (d / f"{SESSION}_alpr_by_track.json").write_text(json.dumps({"tracks": []}))
    stamp = (
        {"crop_mode": "fullframe", "spots": [], "n_suspect_reads": 0} if stamp is None else stamp
    )
    (d / f"{SESSION}_static_plates.json").write_text(json.dumps(stamp))
    return d


def _records(d: Path) -> list[dict]:
    return json.loads((d / f"{SESSION}_alpr.json").read_text())


def _by_tid(d: Path, pipeline: str = "preferred") -> dict[int, dict]:
    rollup = json.loads((d / f"{SESSION}_alpr_by_track.json").read_text())
    return {
        t["track_id"]: t[f"best_{pipeline}"] for t in rollup["tracks"] if f"best_{pipeline}" in t
    }


def test_rescores_fast_plate_ocr_reads_and_rewrites_rollup(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    d = _session(tmp_path)
    assert alpr_rescore.main([str(d)]) == 0

    recs = {(r["pipeline"], r["track_id"]): r for r in _records(d)}
    clean, shaky = recs[("preferred", 1)], recs[("preferred", 2)]
    assert clean["ocr_conf"] == 0.94 and clean["ocr_char_probs"][6] == 0.94
    assert shaky["ocr_conf"] == 0.30 and shaky["canonical_uk_shape"] is True
    # EasyOCR's confidence is its own; a snap with no detection has no crop.
    assert recs[("bespoke", 1)]["ocr_conf"] == 0.4
    assert "ocr_char_probs" not in recs[("bespoke", 1)]
    assert recs[("preferred", 3)]["ocr_conf"] is None

    best = _by_tid(d)
    assert best[1]["ocr_conf"] == 0.94
    assert best[2]["ocr_conf"] == 0.30

    stamp = json.loads((d / f"{SESSION}_static_plates.json").read_text())
    assert stamp["ocr_conf"] == "min_char"
    assert stamp["crop_mode"] == "fullframe"  # other provenance kept
    assert "ocr_rescored_at" in stamp

    out = capsys.readouterr().out
    assert "[batch] 2/2 done" in out  # the control panel's progress format
    # Both tracks passed the 0.9 gate on the old confidence; one does now.
    assert "(what dvsa-label looks up): 2 -> 1" in out


def test_refuses_and_writes_nothing_when_crops_are_missing(tmp_path: Path) -> None:
    d = _session(tmp_path)
    (d / "alpr_crops" / "preferred" / "vehicle_2_main_1.jpg").unlink()
    before = (d / f"{SESSION}_alpr.json").read_text()
    assert alpr_rescore.main([str(d)]) == 2
    assert (d / f"{SESSION}_alpr.json").read_text() == before
    assert "ocr_conf" not in json.loads((d / f"{SESSION}_static_plates.json").read_text())
    assert _FakeRecognizer.constructed == 0  # checked before loading the model


def test_allow_missing_clears_the_unverifiable_confidence(tmp_path: Path) -> None:
    d = _session(tmp_path)
    (d / "alpr_crops" / "preferred" / "vehicle_2_main_1.jpg").unlink()
    assert alpr_rescore.main([str(d), "--allow-missing"]) == 0
    recs = {(r["pipeline"], r["track_id"]): r for r in _records(d)}
    # The old ~1.0 value is the bug, so it can't survive; the text stays.
    assert recs[("preferred", 2)]["ocr_conf"] is None
    assert recs[("preferred", 2)]["ocr_text"] == "AB12CDF"
    assert recs[("preferred", 1)]["ocr_conf"] == 0.94


def test_already_rescored_session_is_skipped(tmp_path: Path) -> None:
    d = _session(tmp_path, stamp={"crop_mode": "fullframe", "ocr_conf": "min_char"})
    before = (d / f"{SESSION}_alpr.json").read_text()
    assert alpr_rescore.main([str(d)]) == 0
    assert (d / f"{SESSION}_alpr.json").read_text() == before
    assert _FakeRecognizer.constructed == 0
    # --force re-scores anyway.
    assert alpr_rescore.main([str(d), "--force"]) == 0
    assert _records(d)[0]["ocr_conf"] == 0.94


def test_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    d = _session(tmp_path)
    before = (d / f"{SESSION}_alpr.json").read_text()
    assert alpr_rescore.main([str(d), "--dry-run"]) == 0
    assert (d / f"{SESSION}_alpr.json").read_text() == before
    assert "dry run" in capsys.readouterr().out


def test_missing_alpr_output_is_an_error(tmp_path: Path) -> None:
    d = tmp_path / SESSION
    d.mkdir()
    assert alpr_rescore.main([str(d)]) == 2


def test_help_returns_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        alpr_rescore.main(["--help"])
    assert exc.value.code == 0


class TestRescoreRecords:
    def test_counts_text_changes_and_reads_lost(self, tmp_path: Path) -> None:
        d = tmp_path / SESSION
        _write_crop(d / "alpr_crops" / "preferred" / "vehicle_1_main_1.jpg", 70)  # text changes
        _write_crop(d / "alpr_crops" / "preferred" / "vehicle_2_main_1.jpg", 60)  # reads nothing
        records = [
            _rec(1, "preferred", "AB12CDE", 1.0, "x"),
            _rec(2, "preferred", "AB12CDF", 1.0, "y"),
        ]
        stats = alpr_rescore.rescore_records(records, d, _fake_recognize)
        assert (stats.n_reads, stats.n_rescored, stats.n_missing) == (2, 2, 0)
        assert stats.n_text_changed == 2
        assert stats.n_now_unread == 1
        assert records[0]["ocr_text"] == "XY12CDE" and records[0]["ocr_conf"] == 0.91
        assert records[1]["ocr_text"] is None and records[1]["canonical_uk_shape"] is None
        assert stats.old_conf == [1.0, 1.0] and stats.new_conf == [0.91]

    def test_falls_back_to_the_recorded_crop_path(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "moved" / "vehicle_1_main_1.jpg"
        _write_crop(elsewhere, 40)
        records = [_rec(1, "preferred", "AB12CDE", 1.0, str(elsewhere))]
        stats = alpr_rescore.rescore_records(records, tmp_path / SESSION, _fake_recognize)
        assert stats.n_rescored == 1 and records[0]["ocr_conf"] == 0.94

    def test_gated_tracks_counts_the_dvsa_population(self) -> None:
        rollup = {
            "tracks": [
                {"track_id": 1, "best_preferred": {"ocr_conf": 0.95, "canonical_uk_shape": True}},
                {"track_id": 2, "best_preferred": {"ocr_conf": 0.95, "canonical_uk_shape": False}},
                {"track_id": 3, "best_preferred": {"ocr_conf": 0.5, "canonical_uk_shape": True}},
                {"track_id": 4, "best_bespoke": {"ocr_conf": 0.99, "canonical_uk_shape": True}},
            ]
        }
        assert alpr_rescore.gated_tracks(rollup) == 1


def test_summary_reports_against_the_shared_plate_gate(
    tmp_path: Path,
    capsys,  # noqa: ANN001
    _isolated_plate_conf_config: Path,
) -> None:
    # configs/alpr.json at 0.3: the shaky read (0.30) now clears the gate too.
    _isolated_plate_conf_config.parent.mkdir(parents=True)
    _isolated_plate_conf_config.write_text(json.dumps({"plate_conf_threshold": 0.3}))
    d = _session(tmp_path)
    assert alpr_rescore.main([str(d)]) == 0
    out = capsys.readouterr().out
    assert f"ocr_conf >= 0.3 ({_isolated_plate_conf_config})" in out
    assert "(what dvsa-label looks up): 2 -> 2" in out


def test_malformed_plate_gate_config_stops_before_writing(
    tmp_path: Path, _isolated_plate_conf_config: Path
) -> None:
    _isolated_plate_conf_config.parent.mkdir(parents=True)
    _isolated_plate_conf_config.write_text("{not json")
    d = _session(tmp_path)
    before = (d / f"{SESSION}_alpr.json").read_text()
    assert alpr_rescore.main([str(d)]) == 2
    assert (d / f"{SESSION}_alpr.json").read_text() == before
    assert _FakeRecognizer.constructed == 0
