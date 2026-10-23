"""Crashed-session recovery: rebuild data/meta/hourly JSON from the logs."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from streettracker.cli import recover_session as recover_cli
from streettracker.common.recover import (
    ir_periods_from_events,
    meta_from_summary_html,
    needs_recovery,
    recover_session,
)
from streettracker.common.schema import SessionMeta, TrackRecord
from streettracker.common.summary import generate_html

LABEL = "session_20261021_161948"


def _meta() -> SessionMeta:
    return SessionMeta(
        session_label=LABEL,
        session_start_unix=1792595988.6788986,
        frames_processed=1305971,
        pipe_fps=9.16,
        avg_infer_ms=108.64,
        snap_stats={
            "attempts": 6551,
            "latency": {"count": 6550, "p50_ms": 603.5},
            "fires_per_track": {"n": 14326, "zero": 12568},
        },
        frame_size=[896, 512],
    )


def _session(tmp_path: Path, records: list[TrackRecord], *, summary: bool = True) -> Path:
    """A session dir as a power cut leaves it: events log + idle summary,
    no end-of-session files."""
    d = tmp_path / LABEL
    d.mkdir()
    lines = [json.dumps(r.to_json_dict(), separators=(",", ":")) for r in records]
    (d / f"{LABEL}_events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / f"{LABEL}_ir_events.jsonl").write_text("", encoding="utf-8")
    if summary:
        generate_html(list(records), d, d / f"{LABEL}_summary.html", LABEL, _meta().to_json_dict())
    return d


def _track(tid: int, sample: TrackRecord) -> TrackRecord:
    return dataclasses.replace(
        sample,
        track_id=tid,
        time_start_unix=sample.time_start_unix + tid,
        time_end_unix=sample.time_end_unix + tid,
    )


def test_meta_round_trips_through_the_summary_html(tmp_path: Path) -> None:
    """The summary embeds the meta as str() reprs; parsing gives it back."""
    html_path = tmp_path / "s.html"
    generate_html([], tmp_path, html_path, LABEL, _meta().to_json_dict())
    parsed = meta_from_summary_html(html_path.read_text(encoding="utf-8"))
    assert parsed == _meta().to_json_dict()


def test_meta_from_summary_html_rejects_pages_without_it() -> None:
    assert meta_from_summary_html("<html><body>nothing</body></html>") is None
    # A meta div without the required keys (e.g. a test page) is not a meta.
    assert meta_from_summary_html('<div class="meta">pipe_fps: 9.1</div>') is None


def test_ir_periods_pair_events_and_close_the_open_one() -> None:
    t0 = 1792595988.0
    events = [
        {"event": "start", "unix": t0},
        {"event": "end", "unix": t0 + 600},
        {"event": "start", "unix": t0 + 1000},  # open when the power went
    ]
    periods = ir_periods_from_events(events, close_at_unix=t0 + 1300)
    assert [p["duration_s"] for p in periods] == [600.0, 300.0]
    assert ir_periods_from_events(events, close_at_unix=None)[-1]["duration_s"] == 600.0


def test_recover_session_rebuilds_the_end_of_session_files(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    records = [_track(i, sample_track) for i in (2, 5, 9)]
    d = _session(tmp_path, records)
    # The crash tore the last line mid-write.
    with (d / f"{LABEL}_events.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"track_id":12,"class_')
    assert needs_recovery(d)

    result = recover_session(d)

    data = json.loads((d / f"{LABEL}_data.json").read_text(encoding="utf-8"))
    assert data == [r.to_json_dict() for r in records]  # verbatim, finalise order
    meta = json.loads((d / f"{LABEL}_meta.json").read_text(encoding="utf-8"))
    assert meta["frame_size"] == [896, 512]
    assert meta["snap_stats"]["attempts"] == 6551
    assert SessionMeta.from_json_dict(meta).session_label == LABEL  # readers accept it
    assert meta["recovered"]["data_from"] == f"{LABEL}_events.jsonl"
    hourly = json.loads((d / f"{LABEL}_hourly.json").read_text(encoding="utf-8"))
    assert sum(h["count"] for h in hourly["hours"]) == 3
    assert result.n_records == 3
    assert result.meta_source == "summary"
    assert result.written[-1].name == f"{LABEL}_data.json"  # written last
    assert not needs_recovery(d)
    assert not list(d.glob("*.tmp"))


def test_recover_session_falls_back_to_a_label_meta(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    d = _session(tmp_path, [sample_track], summary=False)
    result = recover_session(d)
    meta = json.loads((d / f"{LABEL}_meta.json").read_text(encoding="utf-8"))
    assert result.meta_source == "label"
    assert meta["frame_size"] is None
    assert meta["session_start_unix"] > 0
    assert "no frame size" in result.describe()


def test_recover_session_takes_ir_periods_from_the_event_log(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    d = _session(tmp_path, [sample_track])
    start = sample_track.time_start_unix - 100
    (d / f"{LABEL}_ir_events.jsonl").write_text(
        json.dumps({"event": "start", "unix": start, "iso": "x"}) + "\n", encoding="utf-8"
    )
    recover_session(d)
    meta = json.loads((d / f"{LABEL}_meta.json").read_text(encoding="utf-8"))
    # Open at the crash: closed at the last track's end.
    assert meta["ir_periods"][0]["duration_s"] == pytest.approx(
        sample_track.time_end_unix - start, abs=0.1
    )


def test_recover_session_refuses_a_finalised_session(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    d = _session(tmp_path, [sample_track])
    (d / f"{LABEL}_data.json").write_text("[]", encoding="utf-8")
    assert not needs_recovery(d)
    with pytest.raises(FileExistsError):
        recover_session(d)


def test_cli_recovers_and_skips(
    tmp_path: Path, sample_track: TrackRecord, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _session(tmp_path, [sample_track])
    assert recover_cli.main([str(d)]) == 0
    assert "rebuilt" in capsys.readouterr().out
    assert recover_cli.main([str(d)]) == 0
    assert "nothing to do (already has _data.json)" in capsys.readouterr().out
