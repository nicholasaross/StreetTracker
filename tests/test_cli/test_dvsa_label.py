"""DVSA label CLI tests.

The point is to lock down the per-session JSON shape + the
incremental-resume semantics (don't burn API budget re-querying a
plate we already have), without touching the live DVSA API. The
DvsaClient itself is exercised separately in
``tests/test_analysis/test_dvsa.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from streettracker.analysis.dvsa import MotLookupResult
from streettracker.cli import dvsa_label

# ----------------------------------------------------------------------
# Top-level dispatcher routes the subcommand.


def test_top_level_help_lists_dvsa_label() -> None:
    from streettracker.cli import _HELP

    assert "dvsa-label" in _HELP


def test_dvsa_label_help_returns_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        dvsa_label.main(["--help"])
    assert exc.value.code == 0


# ----------------------------------------------------------------------
# _collect_plate_requests groups by plate, filters by conf threshold.


def test_collect_plate_requests_groups_split_tracks() -> None:
    """BotSORT often splits a single physical car into multiple
    track_ids -- the FD61PVX failure mode. The accumulator must
    aggregate by plate so we bill DVSA once per car, not once per track."""
    by_track = {
        "tracks": [
            {
                "track_id": 1,
                "best_preferred": {
                    "track_id": 1, "snap_index": 1, "image": "x.jpg",
                    "ocr_text": "AE13SJX", "ocr_conf": 0.99, "det_conf": 0.9,
                },
            },
            {
                "track_id": 2,
                "best_preferred": {
                    "track_id": 2, "snap_index": 1, "image": "y.jpg",
                    "ocr_text": "AE13SJX", "ocr_conf": 0.95, "det_conf": 0.9,
                },
            },
            {
                "track_id": 3,
                "best_preferred": {
                    "track_id": 3, "snap_index": 1, "image": "z.jpg",
                    "ocr_text": "LD22BMG", "ocr_conf": 0.97, "det_conf": 0.9,
                },
            },
        ]
    }
    reqs = dvsa_label._collect_plate_requests(by_track, conf_threshold=0.9)
    by_plate = {r.plate: r for r in reqs}
    assert set(by_plate) == {"AE13SJX", "LD22BMG"}
    # Two tracks collapsed onto AE13SJX; highest conf preserved.
    assert sorted(by_plate["AE13SJX"].track_ids) == [1, 2]
    assert by_plate["AE13SJX"].plate_conf == pytest.approx(0.99)


def test_collect_plate_requests_drops_below_threshold() -> None:
    by_track = {
        "tracks": [
            {
                "track_id": 1,
                "best_preferred": {
                    "track_id": 1, "snap_index": 1, "image": "x.jpg",
                    "ocr_text": "AE13SJX", "ocr_conf": 0.85, "det_conf": 0.9,
                },
            },
            {
                "track_id": 2,
                "best_preferred": {
                    "track_id": 2, "snap_index": 1, "image": "y.jpg",
                    "ocr_text": "LD22BMG", "ocr_conf": 0.95, "det_conf": 0.9,
                },
            },
        ]
    }
    reqs = dvsa_label._collect_plate_requests(by_track, conf_threshold=0.9)
    assert [r.plate for r in reqs] == ["LD22BMG"]


def test_collect_plate_requests_normalises_plate_string() -> None:
    """OCR may emit lowercase or spaced plate variants; group them."""
    by_track = {
        "tracks": [
            {
                "track_id": 1,
                "best_preferred": {
                    "track_id": 1, "snap_index": 1, "image": "x.jpg",
                    "ocr_text": "ae13 sjx", "ocr_conf": 0.99, "det_conf": 0.9,
                },
            },
            {
                "track_id": 2,
                "best_preferred": {
                    "track_id": 2, "snap_index": 1, "image": "y.jpg",
                    "ocr_text": "AE13SJX", "ocr_conf": 0.95, "det_conf": 0.9,
                },
            },
        ]
    }
    reqs = dvsa_label._collect_plate_requests(by_track, conf_threshold=0.9)
    assert len(reqs) == 1
    assert reqs[0].plate == "AE13SJX"
    assert sorted(reqs[0].track_ids) == [1, 2]


# ----------------------------------------------------------------------
# Top-level main(): bail-early branches.


def test_main_returns_2_when_by_track_missing(tmp_path: Path) -> None:
    session = tmp_path / "session_demo"
    session.mkdir()
    rc = dvsa_label.main([str(session), "--config", str(tmp_path / "nope.json")])
    assert rc == 2


def test_main_returns_2_when_config_missing(tmp_path: Path) -> None:
    session = tmp_path / "session_demo"
    session.mkdir()
    (session / "session_demo_alpr_by_track.json").write_text(json.dumps({"tracks": []}))
    rc = dvsa_label.main([str(session), "--config", str(tmp_path / "nope.json")])
    assert rc == 2


# ----------------------------------------------------------------------
# Happy path: end-to-end via a mocked DvsaClient.


def _write_minimal_session(tmp_path: Path) -> Path:
    """Build a session_dir with an alpr_by_track.json + a dvsa.json
    credential file. Returns the session_dir path."""
    session = tmp_path / "session_demo"
    session.mkdir()
    (session / "session_demo_alpr_by_track.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "track_id": 1,
                        "best_preferred": {
                            "track_id": 1, "snap_index": 1, "image": "x.jpg",
                            "ocr_text": "AE13SJX", "ocr_conf": 0.99, "det_conf": 0.9,
                        },
                    },
                    {
                        "track_id": 2,
                        "best_preferred": {
                            "track_id": 2, "snap_index": 1, "image": "y.jpg",
                            "ocr_text": "LD22BMG", "ocr_conf": 0.97, "det_conf": 0.9,
                        },
                    },
                ]
            }
        )
    )
    cfg = tmp_path / "dvsa.json"
    cfg.write_text(
        json.dumps(
            {
                "x_api_key": "k",
                "oauth": {
                    "client_id": "cid",
                    "client_secret": "sec",
                    "token_url": "https://example.invalid/t",
                    "scope": "https://example.invalid/.default",
                },
            }
        )
    )
    return session


def _porsche_result(plate: str = "AE13SJX") -> MotLookupResult:
    return MotLookupResult(
        registration=plate,
        make="PORSCHE",
        model="911",
        year=2013,
        primary_colour="Black",
        fuel_type="Petrol",
        engine_size_cc=3800,
        looked_up_at="2026-05-29T16:00:00+00:00",
    )


def test_main_writes_labels_file_with_session_summary(tmp_path: Path) -> None:
    session = _write_minimal_session(tmp_path)
    cfg_path = tmp_path / "dvsa.json"

    with patch("streettracker.cli.dvsa_label.DvsaClient") as cls:
        client = cls.return_value
        client.lookup_plate.side_effect = lambda plate: _porsche_result(plate)
        rc = dvsa_label.main([str(session), "--config", str(cfg_path)])

    assert rc == 0
    out_path = session / "session_demo_dvsa_labels.json"
    payload = json.loads(out_path.read_text())
    assert payload["session"] == "session_demo"
    assert payload["n_high_conf_plates"] == 2
    assert payload["n_labelled"] == 2
    assert payload["n_unknown"] == 0
    assert set(payload["labels"]) == {"AE13SJX", "LD22BMG"}
    assert payload["labels"]["AE13SJX"]["make"] == "PORSCHE"
    assert payload["labels"]["AE13SJX"]["year"] == 2013


def test_main_records_unknown_plates_when_dvsa_returns_none(tmp_path: Path) -> None:
    session = _write_minimal_session(tmp_path)
    cfg_path = tmp_path / "dvsa.json"

    with patch("streettracker.cli.dvsa_label.DvsaClient") as cls:
        client = cls.return_value
        client.lookup_plate.side_effect = [None, _porsche_result("LD22BMG")]
        rc = dvsa_label.main([str(session), "--config", str(cfg_path)])

    assert rc == 0
    payload = json.loads((session / "session_demo_dvsa_labels.json").read_text())
    assert payload["n_labelled"] == 1
    assert payload["n_unknown"] == 1
    # The 404 plate (AE13SJX in this scenario) lands in unknown.
    assert payload["unknown"] == ["AE13SJX"]
    assert "AE13SJX" not in payload["labels"]


def test_main_is_incremental_by_default(tmp_path: Path) -> None:
    """Resume support: a second run only queries newly-present plates,
    so we don't burn API budget re-labelling known cars on every soak."""
    session = _write_minimal_session(tmp_path)
    cfg_path = tmp_path / "dvsa.json"

    # First run labels both plates.
    with patch("streettracker.cli.dvsa_label.DvsaClient") as cls:
        client = cls.return_value
        client.lookup_plate.side_effect = lambda plate: _porsche_result(plate)
        dvsa_label.main([str(session), "--config", str(cfg_path)])

    # Second run: no new plates. The client must not be called.
    with patch("streettracker.cli.dvsa_label.DvsaClient") as cls:
        client = cls.return_value
        client.lookup_plate.return_value = None  # would fail if called
        dvsa_label.main([str(session), "--config", str(cfg_path)])
        assert client.lookup_plate.call_count == 0


def test_main_re_label_flag_overwrites_existing(tmp_path: Path) -> None:
    """``--re-label`` forces re-query of every plate, including ones
    previously recorded as unknown (e.g. a car just turned 3 years old
    and is now MOT-eligible)."""
    session = _write_minimal_session(tmp_path)
    cfg_path = tmp_path / "dvsa.json"

    # First run: both plates land in "unknown".
    with patch("streettracker.cli.dvsa_label.DvsaClient") as cls:
        client = cls.return_value
        client.lookup_plate.return_value = None
        dvsa_label.main([str(session), "--config", str(cfg_path)])
    payload = json.loads((session / "session_demo_dvsa_labels.json").read_text())
    assert payload["n_unknown"] == 2

    # Re-run with --re-label: both plates now resolve.
    with patch("streettracker.cli.dvsa_label.DvsaClient") as cls:
        client = cls.return_value
        client.lookup_plate.side_effect = lambda plate: _porsche_result(plate)
        dvsa_label.main([str(session), "--config", str(cfg_path), "--re-label"])

    payload = json.loads((session / "session_demo_dvsa_labels.json").read_text())
    assert payload["n_labelled"] == 2
    assert payload["n_unknown"] == 0


def test_main_limit_caps_distinct_plates_processed(tmp_path: Path) -> None:
    session = _write_minimal_session(tmp_path)
    cfg_path = tmp_path / "dvsa.json"

    with patch("streettracker.cli.dvsa_label.DvsaClient") as cls:
        client = cls.return_value
        client.lookup_plate.side_effect = lambda plate: _porsche_result(plate)
        dvsa_label.main([str(session), "--config", str(cfg_path), "--limit", "1"])
        assert client.lookup_plate.call_count == 1

    payload = json.loads((session / "session_demo_dvsa_labels.json").read_text())
    assert payload["n_labelled"] == 1
