"""EventLog crash-safe append + JSON / JSONL read-back."""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.common.output import (
    EventLog,
    IREventLog,
    format_wall,
    read_data_json,
    read_events_jsonl,
    read_ir_events_jsonl,
    save_data_json,
    save_json,
    save_meta_json,
)
from streettracker.common.schema import SessionMeta, TrackRecord


def test_event_log_appends_one_line_per_record(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        log.append(sample_track)
        log.append(sample_track)
        log.append(sample_track)
    assert log.count == 3
    lines = [
        l for l in log_path.read_text().splitlines() if l.strip()  # noqa: E741
    ]
    assert len(lines) == 3
    parsed = json.loads(lines[0])
    assert parsed["track_id"] == sample_track.track_id


def test_event_log_appends_to_existing_file(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        log.append(sample_track)
    with EventLog(log_path) as log:
        log.append(sample_track)
    lines = [l for l in log_path.read_text().splitlines() if l.strip()]  # noqa: E741
    assert len(lines) == 2


def test_event_log_accepts_plain_dict(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        log.append({"track_id": 1, "class_name": "car"})
    parsed = json.loads(log_path.read_text().strip())
    assert parsed["track_id"] == 1


def test_save_data_json_round_trip(tmp_path: Path, sample_track: TrackRecord) -> None:
    p = tmp_path / "data.json"
    save_data_json([sample_track, sample_track], p)
    back = read_data_json(p)
    assert len(back) == 2
    assert back[0] == sample_track


def test_save_meta_json_round_trip(
    tmp_path: Path, sample_session_meta: SessionMeta
) -> None:
    p = tmp_path / "meta.json"
    save_meta_json(sample_session_meta, p)
    parsed = json.loads(p.read_text())
    assert parsed["session_label"] == sample_session_meta.session_label
    assert parsed["frames_processed"] == sample_session_meta.frames_processed
    assert len(parsed["ir_periods"]) == 1


def test_save_json_writes_both(
    tmp_path: Path, sample_track: TrackRecord, sample_session_meta: SessionMeta
) -> None:
    data = tmp_path / "data.json"
    meta = tmp_path / "meta.json"
    save_json([sample_track], sample_session_meta, data, meta)
    assert data.exists()
    assert meta.exists()


def test_read_events_jsonl_skips_blank_lines(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    p = tmp_path / "events.jsonl"
    with EventLog(p) as log:
        log.append(sample_track)
    # Simulate partial-write tail (a blank line)
    with open(p, "a") as f:
        f.write("\n\n")
    back = read_events_jsonl(p)
    assert len(back) == 1
    assert back[0] == sample_track


def test_format_wall_includes_tz_offset() -> None:
    s = format_wall(1747488725.0)
    # Either "+HH:MM" or "-HH:MM" or "Z" or "+HHMM"; Python's isoformat with
    # astimezone() returns "+HH:MM" form.
    assert "T" in s
    assert s[-6] in ("+", "-") or s.endswith("Z")


def test_read_events_jsonl_skips_torn_tail_without_aborting(
    tmp_path: Path, sample_track: TrackRecord, caplog
) -> None:
    """A SIGKILL between EventLog.write() and the OS journal flush can
    leave a partial last line. The reader must skip it (with a warning)
    instead of raising and losing every record before it."""
    import logging

    p = tmp_path / "events.jsonl"
    with EventLog(p) as log:
        log.append(sample_track)
        log.append(sample_track)
    # Append a torn last line: half a JSON record, no closing brace, no
    # newline. Exactly the failure mode SIGKILL produces on a 24/7 service.
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"track_id": 99, "class_name":')
    with caplog.at_level(logging.WARNING, logger="streettracker.common.output"):
        back = read_events_jsonl(p)
    # Two intact records preserved; the torn tail did not abort the read.
    assert len(back) == 2
    assert back[0] == sample_track
    assert back[1] == sample_track
    # Operator-visible signal: we log the skip so a silent data-loss
    # window doesn't open up.
    assert any("torn line" in r.message for r in caplog.records)


def test_ir_event_log_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "ir_events.jsonl"
    with IREventLog(p) as log:
        log.append_start(1747488725.0)
        log.append_end(1747490725.0)
    events = read_ir_events_jsonl(p)
    assert len(events) == 2
    assert events[0]["event"] == "start"
    assert events[0]["unix"] == 1747488725.0
    assert "T" in events[0]["iso"]
    assert events[1]["event"] == "end"
    assert events[1]["unix"] == 1747490725.0


def test_ir_event_log_records_unfinished_period_for_crash_recovery(
    tmp_path: Path,
) -> None:
    """A SIGKILL'd session that was mid-IR leaves a trailing unmatched
    "start" -- which is the whole point of the sidecar: meta.json
    never got written, but the start timestamp is still on disk."""
    p = tmp_path / "ir_events.jsonl"
    log = IREventLog(p)
    log.append_start(1747488725.0)
    # Simulate SIGKILL: no close() call. The IR -> day "end" never lands.
    del log
    events = read_ir_events_jsonl(p)
    assert len(events) == 1
    assert events[0]["event"] == "start"


def test_read_ir_events_jsonl_skips_torn_tail(tmp_path: Path, caplog) -> None:
    import logging

    p = tmp_path / "ir_events.jsonl"
    with IREventLog(p) as log:
        log.append_start(1747488725.0)
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"event": "end", "unix":')
    with caplog.at_level(logging.WARNING, logger="streettracker.common.output"):
        events = read_ir_events_jsonl(p)
    assert len(events) == 1
    assert events[0]["event"] == "start"
    assert any("torn IR-event line" in r.message for r in caplog.records)
