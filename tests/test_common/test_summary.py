"""The session summary page: generate_html() and the incremental SummaryRows."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from streettracker.common import summary as summary_mod
from streettracker.common.schema import TrackRecord
from streettracker.common.summary import (
    SummaryRows,
    generate_html,
    render_summary,
    summary_row,
    write_summary,
)


def test_generate_html_writes_page_and_sidecar(tmp_path: Path, sample_track: TrackRecord) -> None:
    html_path = tmp_path / "session_summary.html"
    generate_html(
        records=[sample_track, sample_track],
        output_dir=tmp_path,
        html_path=html_path,
        session_label="session_test",
        meta={"frames_processed": 12000, "pipe_fps": 33.4},
        refresh_seconds=15,
    )
    assert html_path.exists()
    body = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in body
    assert "StreetTracker Summary" in body
    assert "session_test" in body
    # The embedded JSON should contain the track ID.
    assert '"track_id":42' in body

    sidecar = tmp_path / "vehicles.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["track_id"] == 42


def test_index_html_redirects_to_latest_summary(tmp_path: Path, sample_track: TrackRecord) -> None:
    html_path = tmp_path / "session_summary.html"
    generate_html(
        records=[sample_track],
        output_dir=tmp_path,
        html_path=html_path,
        session_label="session_test",
        meta={},
    )
    index = tmp_path / "index.html"
    assert index.exists()
    assert html_path.name in index.read_text()
    assert 'http-equiv="refresh"' in index.read_text()


def test_empty_records_renders_no_detections_yet(tmp_path: Path) -> None:
    html_path = tmp_path / "session_summary.html"
    generate_html(
        records=[],
        output_dir=tmp_path,
        html_path=html_path,
        session_label="empty_session",
        meta={},
    )
    body = html_path.read_text()
    assert "No detections yet" in body
    sidecar = json.loads((tmp_path / "vehicles.json").read_text())
    assert sidecar == []


def test_poll_disabled_when_refresh_zero(tmp_path: Path, sample_track: TrackRecord) -> None:
    html_path = tmp_path / "session_summary.html"
    generate_html(
        records=[sample_track],
        output_dir=tmp_path,
        html_path=html_path,
        session_label="t",
        meta={},
        refresh_seconds=0,
    )
    body = html_path.read_text()
    assert "POLL_MS=0;" in body


# ----------------------------------------------------------------------
# SummaryRows: the page's rows kept ready, one finalised track at a time.


def _some_records(sample_track: TrackRecord) -> list[TrackRecord | dict]:
    people = replace(
        sample_track, track_id=7, class_id=0, class_name="person", asset_prefix="person"
    )
    dog = replace(sample_track, track_id=8, class_id=16, class_name="dog", main_snaps=[])
    # A plain-dict record (what analysis code passes), without the optional keys.
    as_dict = {
        k: v
        for k, v in replace(sample_track, track_id=9).to_json_dict().items()
        if k not in ("asset_prefix", "main_snaps")
    }
    return [sample_track, people, dog, replace(sample_track, track_id=43), as_dict]


def test_summary_row_reads_a_record_the_same_as_its_dict(sample_track: TrackRecord) -> None:
    row = summary_row(sample_track)
    assert row == summary_row(sample_track.to_json_dict())
    # The page's JavaScript and analysis/recolor.py rely on these names; the
    # order is what vehicles.json has always carried.
    assert list(row) == [
        "track_id",
        "class_name",
        "color",
        "time_start",
        "time_start_unix",
        "duration_visible",
        "direction",
        "speed_px_s",
        "lane",
        "avg_confidence",
        "asset_prefix",
        "main_snaps",
    ]
    row["main_snaps"].append(99)
    assert sample_track.main_snaps == [1, 2]  # the row holds its own copy


def test_summary_row_defaults_for_dict_records_without_optional_keys() -> None:
    row = summary_row(
        {
            "track_id": 1,
            "class_name": "car",
            "color": "red",
            "time_start": "2026-05-17T14:32:05+01:00",
            "time_start_unix": 1.0,
            "duration_visible": 2.0,
            "direction": "left to right",
            "speed_px_s": 3.0,
            "lane": "top",
            "avg_confidence": 0.5,
        }
    )
    assert row["asset_prefix"] == "vehicle"
    assert row["main_snaps"] == []


def test_rows_serialise_exactly_as_the_one_shot_dump_did(sample_track: TrackRecord) -> None:
    records = _some_records(sample_track)
    rows = SummaryRows()
    for r in records:
        rows.add(r)
    assert len(rows) == 5
    assert rows.to_json() == json.dumps([summary_row(r) for r in records], separators=(",", ":"))
    assert SummaryRows().to_json() == "[]"


def test_rows_added_one_at_a_time_render_the_same_files(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    records = _some_records(sample_track)
    meta = {"session_label": "s", "frames_processed": 12, "pipe_fps": 9.1}
    rows = SummaryRows()
    for r in records:
        rows.add(r)
    snapshot = render_summary(rows, "s", meta, refresh_seconds=15)

    html_path = tmp_path / "s_summary.html"
    generate_html(records, tmp_path, html_path, "s", meta, refresh_seconds=15)
    assert html_path.read_bytes() == snapshot.page.encode("utf-8")
    assert (tmp_path / "vehicles.json").read_bytes() == snapshot.data_json.encode("utf-8")
    assert f'<script id="vehicles-data" type="application/json">{snapshot.data_json}</script>' in (
        snapshot.page
    )


def test_summary_text_counts_by_class(sample_track: TrackRecord) -> None:
    rows = SummaryRows()
    assert rows.summary_text() == "No detections yet"
    rows.add(sample_track)
    assert rows.summary_text() == "1 entry: 1 car"
    for r in _some_records(sample_track)[1:]:
        rows.add(r)
    assert rows.summary_text() == "5 entries: 3 cars, 1 dog, 1 person"
    assert rows.class_counts == {"car": 3, "person": 1, "dog": 1}


def test_rendering_again_does_no_per_record_work(
    sample_track: TrackRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The point of SummaryRows: a record is converted once, when it is added,
    # however many times the page is rendered afterwards.
    calls = {"n": 0}
    real = summary_mod.summary_row

    def counting(record: TrackRecord | dict) -> dict:
        calls["n"] += 1
        return real(record)

    monkeypatch.setattr(summary_mod, "summary_row", counting)
    rows = SummaryRows()
    for i in range(50):
        rows.add(replace(sample_track, track_id=i))
    first = render_summary(rows, "s", {}, 15)
    for _ in range(20):
        assert render_summary(rows, "s", {}, 15) == first
    assert calls["n"] == 50


def test_write_summary_replaces_files_and_leaves_no_temp(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    out = tmp_path / "session_x"  # created on demand
    html_path = out / "session_x_summary.html"
    rows = SummaryRows([sample_track])
    write_summary(render_summary(rows, "session_x", {}, 15), out, html_path)
    rows.add(replace(sample_track, track_id=43))
    write_summary(render_summary(rows, "session_x", {}, 15), out, html_path)

    assert sorted(p.name for p in out.iterdir()) == [
        "index.html",
        "session_x_summary.html",
        "vehicles.json",
    ]
    assert [r["track_id"] for r in json.loads((out / "vehicles.json").read_text())] == [42, 43]
    assert "session_x_summary.html" in (out / "index.html").read_text()
