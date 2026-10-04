"""Tests for ``streettracker alpr-colour`` and the rollup's colour handling."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from streettracker.cli.alpr_colour import main
from streettracker.cli.alpr_run import _rollup_by_track

YELLOW = (40, 200, 235)
WHITE = (235, 215, 200)


def _plate(bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((46, 136, 3), np.uint8)
    img[4:-4, 4:-4] = bgr
    for x in range(20, 116, 16):
        img[12:-12, x : x + 6] = (25, 25, 25)
    return img


def _session(tmp_path: Path) -> Path:
    """Track 1 drives left to right (rear plate, yellow). Its most confident
    read is a white front plate -- an oncoming car's -- and two yellow reads
    agree on its own plate."""
    d = tmp_path / "session_20260101_000000"
    crops = d / "alpr_crops" / "preferred"
    crops.mkdir(parents=True)
    (d / f"{d.name}_data.json").write_text(
        json.dumps([{"track_id": 1, "direction": "left to right", "class_name": "car"}])
    )
    reads = [
        (1, "AB12CDE", 0.97, WHITE),
        (2, "CD34EFG", 0.90, YELLOW),
        (3, "CD34EFG", 0.85, YELLOW),
    ]
    records = []
    for snap, text, conf, colour in reads:
        name = f"vehicle_1_main_{snap}.jpg"
        cv2.imwrite(str(crops / name), _plate(colour))
        records.append(
            {
                "pipeline": "preferred",
                "track_id": 1,
                "snap_index": snap,
                "image": name,
                "ocr_text": text,
                "ocr_conf": conf,
                "canonical_uk_shape": True,
            }
        )
    (d / f"{d.name}_alpr.json").write_text(json.dumps(records))
    return d


def test_rollup_skips_colour_suspects_and_counts_agreement() -> None:
    recs = [
        {"pipeline": "preferred", "track_id": 1, "snap_index": 1, "image": "a",
         "ocr_text": "AB12CDE", "ocr_conf": 0.97, "colour_suspect": True},
        {"pipeline": "preferred", "track_id": 1, "snap_index": 2, "image": "b",
         "ocr_text": "CD34EFG", "ocr_conf": 0.90},
        {"pipeline": "preferred", "track_id": 1, "snap_index": 3, "image": "c",
         "ocr_text": "CD34EFG", "ocr_conf": 0.85},
    ]  # fmt: skip
    best = _rollup_by_track(recs)["tracks"][0]["best_preferred"]
    assert (best["ocr_text"], best["snap_index"], best["n_agree"]) == ("CD34EFG", 2, 1)


def test_alpr_colour_rewrites_the_rollup_and_stamps(tmp_path: Path) -> None:
    d = _session(tmp_path)
    assert main([str(d)]) == 0
    rollup = json.loads((d / f"{d.name}_alpr_by_track.json").read_text())
    best = rollup["tracks"][0]["best_preferred"]
    assert (best["ocr_text"], best["n_agree"]) == ("CD34EFG", 1)  # fell back off the white plate
    records = json.loads((d / f"{d.name}_alpr.json").read_text())
    assert [r.get("colour_suspect", False) for r in records] == [True, False, False]
    assert [r["plate_colour"] for r in records] == ["white", "yellow", "yellow"]
    stamp = json.loads((d / f"{d.name}_static_plates.json").read_text())
    assert stamp["plate_colour"] == "hsv_v1"


def test_alpr_colour_dry_run_writes_nothing(tmp_path: Path) -> None:
    d = _session(tmp_path)
    before = (d / f"{d.name}_alpr.json").read_text()
    assert main([str(d), "--dry-run"]) == 0
    assert (d / f"{d.name}_alpr.json").read_text() == before
    assert not (d / f"{d.name}_alpr_by_track.json").exists()
    assert not (d / f"{d.name}_static_plates.json").exists()


def test_alpr_colour_needs_alpr_output(tmp_path: Path) -> None:
    d = tmp_path / "session_20260101_000000"
    d.mkdir()
    assert main([str(d)]) == 2
