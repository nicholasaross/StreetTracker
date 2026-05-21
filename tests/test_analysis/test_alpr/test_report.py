"""Smoke tests for the ALPR HTML report renderer."""

from __future__ import annotations

import re
from pathlib import Path

from streettracker.analysis.alpr.report import render_html


def _rec(image: str, pipeline: str, *, ocr: str | None,
         track_id: int = 1, snap_index: int = 1) -> dict:
    return {
        "image": image,
        "image_path": image,
        "track_id": track_id,
        "snap_index": snap_index,
        "class_name": "vehicle",
        "pipeline": pipeline,
        "det_bbox": [1, 2, 3, 4] if ocr else None,
        "det_conf": 0.8 if ocr else None,
        "ocr_text": ocr,
        "ocr_raw": ocr,
        "ocr_conf": 0.7 if ocr else None,
        "crop_path": None,
        "pipeline_ms": 1.0,
        "error": None,
    }


def test_render_html_contains_session_and_pipelines(tmp_path: Path) -> None:
    records = [
        _rec("vehicle_1_main_1.jpg", "bespoke", ocr="ABC123"),
        _rec("vehicle_1_main_1.jpg", "preferred", ocr="ABC123"),
    ]
    html = render_html(
        session_dir=tmp_path / "session_demo",
        records=records,
        labels={},
        generated_at="2026-05-19T12:00:00",
    )
    assert "session_demo" in html
    assert "bespoke" in html
    assert "preferred" in html
    assert "ABC123" in html
    # Status classes get rendered into the row markup
    assert 'class="row match"' in html


def test_render_html_disagreement_row_class(tmp_path: Path) -> None:
    records = [
        _rec("a.jpg", "bespoke", ocr="ABC"),
        _rec("a.jpg", "preferred", ocr="XYZ"),
    ]
    html = render_html(
        session_dir=tmp_path / "session_x",
        records=records,
        labels={},
        generated_at="2026-05-19T12:00:00",
    )
    assert 'class="row disagreement"' in html


def test_render_html_no_read_row_class(tmp_path: Path) -> None:
    records = [
        _rec("a.jpg", "bespoke", ocr=None),
        _rec("a.jpg", "preferred", ocr=None),
    ]
    html = render_html(
        session_dir=tmp_path / "session_x",
        records=records,
        labels={},
        generated_at="2026-05-19T12:00:00",
    )
    assert 'class="row no-read"' in html


def test_render_html_renders_labels_when_provided(tmp_path: Path) -> None:
    records = [_rec("vehicle_1_main_1.jpg", "bespoke", ocr="ABC123")]
    labels = {"vehicle_1_main_1.jpg": {"plate": "ABC123"}}
    html = render_html(
        session_dir=tmp_path / "s",
        records=records,
        labels=labels,
        generated_at="2026-05-19T12:00:00",
    )
    # The user-visible label appears in the table
    assert re.search(r'class="plate">\s*ABC123', html)
    # And the ✓ badge fires because the OCR matched the label
    assert "label-ok" in html


def test_render_html_filters_unlabeled_with_pending_badge(tmp_path: Path) -> None:
    records = [_rec("a.jpg", "bespoke", ocr="ABC")]
    html = render_html(
        session_dir=tmp_path / "s",
        records=records,
        labels={},
        generated_at="2026-05-19T12:00:00",
    )
    assert "label-pending" in html
    assert "unlabeled" in html
