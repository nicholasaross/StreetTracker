"""Stationary parked-plate ("beacon") detection.

A parked car's static plate gets read across many DIFFERENT moving
tracks at (nearly) the same image position; the detector must convert
those reads into a suppression set + a ParkedEpisode, while leaving
genuinely moving reads (and the parked car's own slow track) alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from streettracker.analysis.parked import (
    best_unsuppressed_read,
    detect_parked,
    load_alpr_entries,
    merge_episodes,
)


def _entry(
    tid: int,
    snap: int,
    text: str,
    conf: float,
    cx: float,
    cy: float,
    w: float = 130.0,
    h: float = 34.0,
) -> dict:
    """One per-image ALPR read shaped like an ``_alpr.json`` entry."""
    x1, y1 = cx - w / 2, cy - h / 2
    return {
        "image": f"vehicle_{tid}_main_{snap}.jpg",
        "image_path": f"output/session_test/vehicle_{tid}_main_{snap}.jpg",
        "track_id": tid,
        "snap_index": snap,
        "class_name": "vehicle",
        "pipeline": "preferred",
        "det_bbox": [x1, y1, x1 + w, y1 + h],
        "det_conf": 0.8,
        "ocr_text": text,
        "ocr_raw": text,
        "ocr_conf": conf,
        "crop_path": None,
        "pipeline_ms": 10.0,
        "error": None,
    }


def _rec(tid: int, speed: float, t0: str, unix: float) -> dict:
    return {
        "track_id": tid,
        "class_name": "car",
        "speed_px_s": speed,
        "time_start": t0,
        "time_start_unix": unix,
    }


def _beacon_scene() -> tuple[list[dict], list[dict]]:
    """5 moving hosts + the parked car's own slow track, all reading the
    same plate at the same spot (with a little detector jitter)."""
    entries = [
        _entry(1, 1, "LX19PXR", 0.99, 2080, 455),
        _entry(2, 1, "LX19PXR", 0.99, 2090, 452),
        _entry(3, 2, "LX19PXR", 0.98, 2075, 458),
        _entry(4, 1, "LX19PXR", 0.99, 2085, 456),
        _entry(5, 3, "LX19PXR", 0.97, 2095, 450),
        _entry(9, 1, "LX19PXR", 0.99, 2082, 454),  # parked car's own track
    ]
    records = [
        _rec(1, 60.0, "2026-06-02T11:04:00+01:00", 1000.0),
        _rec(2, 55.0, "2026-06-02T11:10:00+01:00", 1360.0),
        _rec(3, 70.0, "2026-06-02T11:20:00+01:00", 1960.0),
        _rec(4, 48.0, "2026-06-02T11:35:00+01:00", 2860.0),
        _rec(5, 82.0, "2026-06-02T11:48:00+01:00", 3640.0),
        _rec(9, 1.2, "2026-06-02T11:02:00+01:00", 880.0),
    ]
    return entries, records


def test_beacon_detected_moving_reads_suppressed() -> None:
    entries, records = _beacon_scene()
    det = detect_parked(entries, records)

    assert len(det.episodes) == 1
    ep = det.episodes[0]
    assert ep.plate == "LX19PXR"
    assert ep.canonical_shape is True
    assert ep.n_reads == 6
    assert ep.n_tracks == 6  # 5 moving hosts + own slow track
    assert ep.track_ids == [1, 2, 3, 4, 5]  # suppressed = moving hosts only
    # Window spans the earliest (own slow track) to latest host.
    assert ep.first_seen == "2026-06-02T11:02:00+01:00"
    assert ep.last_seen == "2026-06-02T11:48:00+01:00"
    assert ep.duration_minutes == pytest.approx(46.0)
    assert ep.center[0] == pytest.approx(2084.5, abs=10)

    assert det.suppressed == {(1, 1), (2, 1), (3, 2), (4, 1), (5, 3)}
    assert (9, 1) not in det.suppressed  # the parked car keeps its self-read


def test_too_few_moving_hosts_is_not_a_beacon() -> None:
    entries, records = _beacon_scene()
    entries = [e for e in entries if e["track_id"] in (1, 2, 3, 9)]
    det = detect_parked(entries, records)
    assert det.episodes == []
    assert det.suppressed == set()


def test_slow_hosts_only_is_not_a_beacon() -> None:
    """BotSORT splitting one stationary car into several of its own slow
    tracks must not trip the detector."""
    entries, records = _beacon_scene()
    records = [{**r, "speed_px_s": 2.0} for r in records]
    det = detect_parked(entries, records)
    assert det.episodes == []
    assert det.suppressed == set()


def test_scattered_reads_are_not_a_beacon() -> None:
    """The same plate read at well-separated positions (a car genuinely
    driving through, read repeatedly) must not cluster."""
    entries = [_entry(t, 1, "FD61PVX", 0.99, 500 + 450 * t, 700 - 40 * t) for t in range(1, 6)]
    records = [_rec(t, 60.0, "2026-06-02T12:00:00+01:00", 5000.0) for t in range(1, 6)]
    det = detect_parked(entries, records)
    assert det.episodes == []
    assert det.suppressed == set()


def test_regular_commuter_consistent_spot_is_not_a_beacon() -> None:
    """A genuine regular whose plate happens to be read at the same road
    position on every pass clusters across its OWN tracks. Each of those
    hosts also reads the plate elsewhere along its transit, so the
    self-host exclusion must veto the beacon (real case: VK74TPO)."""
    entries: list[dict] = []
    records: list[dict] = []
    for i, tid in enumerate(range(1, 6)):
        # The consistent spot every pass gets read at...
        entries.append(_entry(tid, 1, "VK74TPO", 0.99, 1500, 600))
        # ...and the same plate read again further along the same transit.
        entries.append(_entry(tid, 2, "VK74TPO", 0.95, 2300 + 40 * i, 420))
        records.append(
            _rec(tid, 60.0, f"2026-06-0{1 + (i % 3)}T08:00:00+01:00", 1000.0 + i * 86400)
        )
    det = detect_parked(entries, records)
    assert det.episodes == []
    assert det.suppressed == set()


def test_merge_episodes_folds_ocr_variants() -> None:
    entries, records = _beacon_scene()
    # The same parked plate also misread as LX11PXR by 4 more hosts at the
    # same spot (own 9->1 OCR confusion).
    entries += [_entry(20 + i, 1, "LX11PXR", 0.99, 2083, 456) for i in range(4)]
    records += [_rec(20 + i, 50.0, "2026-06-02T11:30:00+01:00", 2500.0) for i in range(4)]
    det = detect_parked(entries, records)
    assert len(det.episodes) == 2

    merged = merge_episodes(det.episodes, {"LX11PXR": "LX19PXR", "LX19PXR": "LX19PXR"})
    assert len(merged) == 1
    ep = merged[0]
    assert ep.plate == "LX19PXR"
    assert ep.n_reads == 10
    assert set(ep.track_ids) == {1, 2, 3, 4, 5, 20, 21, 22, 23}


def test_best_unsuppressed_read_reanchors_to_own_plate() -> None:
    reads = [
        _entry(1, 1, "LX19PXR", 0.99, 2080, 455),  # beacon (suppressed)
        _entry(1, 2, "AB12CDE", 0.93, 900, 700),  # the car's own plate
        _entry(1, 3, "123456", 0.96, 905, 702),  # OCR garbage, non-canonical
        _entry(1, 4, "CD34EFG", 0.55, 910, 698),  # below threshold
    ]
    best = best_unsuppressed_read(reads, {(1, 1)}, conf_threshold=0.9, canonical_only=True)
    assert best is not None
    assert best["ocr_text"] == "AB12CDE"
    assert best["snap_index"] == 2

    none = best_unsuppressed_read(reads[:1], {(1, 1)}, conf_threshold=0.9, canonical_only=True)
    assert none is None


def test_best_unsuppressed_read_non_canonical_allowed_when_disabled() -> None:
    reads = [_entry(1, 3, "123456", 0.96, 905, 702)]
    best = best_unsuppressed_read(reads, set(), conf_threshold=0.9, canonical_only=False)
    assert best is not None
    assert best["ocr_text"] == "123456"


def test_load_alpr_entries_absent_file(tmp_path: Path) -> None:
    session = tmp_path / "session_test"
    session.mkdir()
    assert load_alpr_entries(session) == []
