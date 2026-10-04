"""Tests for the shared plate gate (conf vs combined) and plate support."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from streettracker.analysis.alpr import base
from streettracker.analysis.alpr.gate import (
    PlateGate,
    load_plate_support,
    read_passes,
    resolve_plate_gate,
)


def _write(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg))


def test_conf_gate_is_a_plain_threshold() -> None:
    g = PlateGate()
    assert g.mode == "conf" and g.min_conf == 0.9 and not g.needs_support
    assert g.passes(0.95)
    assert not g.passes(0.85, n_agree=3, support=10)  # corroboration doesn't matter
    assert not g.passes(None)


def test_combined_gate_needs_confidence_and_corroboration() -> None:
    g = PlateGate(mode="combined", supported_conf=0.8, min_support=2)
    assert g.min_conf == 0.8 and g.needs_support
    assert g.passes(0.85, n_agree=1)  # another snap agrees
    assert g.passes(0.85, support=2)  # another track read it
    assert not g.passes(0.99, support=1)  # a lone confident read is not enough
    assert not g.passes(0.75, n_agree=4, support=9)  # below the confidence floor


def test_resolve_default_and_override(_isolated_plate_conf_config: Path) -> None:
    assert resolve_plate_gate() == (PlateGate(), "default")
    _write(_isolated_plate_conf_config, {"plate_gate": "combined"})
    gate, source = resolve_plate_gate(0.7)
    assert gate == PlateGate(conf_threshold=0.7) and source == "command line"


def test_resolve_combined_from_config(_isolated_plate_conf_config: Path) -> None:
    _write(
        _isolated_plate_conf_config,
        {
            "plate_gate": "combined",
            "supported_conf_threshold": 0.82,
            "min_support": 3,
            "plate_conf_threshold": 0.9,
            "_why": "comment keys are ignored",
        },
    )
    gate, source = resolve_plate_gate()
    assert gate == PlateGate("combined", 0.9, 0.82, 3)
    assert source == str(_isolated_plate_conf_config)
    # The plain threshold resolver still reads its own key from the same file.
    assert base.resolve_plate_conf_threshold() == (0.9, str(_isolated_plate_conf_config))


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        ({"plate_gate": "magic"}, "plate_gate must be one of"),
        ({"plate_gate": "combined", "min_support": 0}, "min_support"),
        ({"plate_gate": "combined", "min_support": True}, "min_support"),
        ({"supported_conf_threshold": 1.5}, r"\(0, 1\]"),
        ({"plate_gat": "combined"}, "unknown key"),
    ],
)
def test_resolve_rejects_bad_config(
    _isolated_plate_conf_config: Path, cfg: dict, match: str
) -> None:
    _write(_isolated_plate_conf_config, cfg)
    with pytest.raises(ValueError, match=match):
        resolve_plate_gate()


def _rollup(root: Path, session: str, reads: list[tuple[int, str]]) -> Path:
    d = root / session
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{session}_alpr_by_track.json"
    p.write_text(
        json.dumps(
            {
                "tracks": [
                    {"track_id": t, "best_preferred": {"ocr_text": plate, "ocr_conf": 0.9}}
                    for t, plate in reads
                ]
            }
        )
    )
    return p


def test_load_plate_support_counts_uk_shaped_best_reads(tmp_path: Path) -> None:
    _rollup(tmp_path, "session_20260101_000000", [(1, "AB12CDE"), (2, "AB12CDE"), (3, "X9")])
    _rollup(tmp_path, "session_20260102_000000", [(1, "AB12CDE"), (2, "LA68EWY")])
    support = load_plate_support(tmp_path)
    assert support == {"AB12CDE": 3, "LA68EWY": 1}  # "X9" isn't UK-shaped


def test_load_plate_support_refreshes_when_a_rollup_changes(tmp_path: Path) -> None:
    p = _rollup(tmp_path, "session_20260101_000000", [(1, "AB12CDE")])
    assert load_plate_support(tmp_path)["AB12CDE"] == 1
    _rollup(tmp_path, "session_20260101_000000", [(1, "AB12CDE"), (2, "AB12CDE")])
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # make the change visible
    assert load_plate_support(tmp_path)["AB12CDE"] == 2


def test_read_passes_uses_the_reads_own_agreement() -> None:
    g = PlateGate(mode="combined")
    read = {"ocr_text": "AB12 CDE", "ocr_conf": 0.85, "n_agree": 1}
    assert read_passes(g, read, {})
    assert not read_passes(g, {**read, "n_agree": 0}, {})
    assert read_passes(g, {**read, "n_agree": 0}, {"AB12CDE": 2})
    assert not read_passes(g, read, {}, n_agree=0)  # explicit override wins
