"""``streettracker alpr-run`` end to end, with a stub detector and recogniser.

The real pipeline needs the ``alpr`` extra and a GPU. Here ``main()`` runs
against a small synthetic session, with everything of ours real (snap
discovery, bbox hints, the ghost mask, ``PipelineRunner``, the static-plate
filter, the plate-colour check, the per-track rollup, the files written) and
only the two models replaced: a detector that returns the plate box painted
into each snap, and a recogniser that returns that snap's scripted text.

Also covered: which detector and recogniser ``_build_pipelines`` wires up for
each flag combination, and the small loaders around ``main()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from streettracker.analysis.alpr import bespoke as bespoke_mod
from streettracker.analysis.alpr import preferred as preferred_mod
from streettracker.analysis.alpr.base import PlateDetection, PlateRead
from streettracker.analysis.alpr.fullframe import TrajectoryCropDetector
from streettracker.analysis.alpr.precrop import PreCropDetector
from streettracker.analysis.alpr.runner import PipelineRunner
from streettracker.cli import alpr_run
from streettracker.control.progress import make_parser
from streettracker.control.prompts import summarize_issue

cv2 = pytest.importorskip("cv2")

# The synthetic "4K" snap and the sub-stream frame its bboxes are recorded in.
_W, _H = 640, 360
_SUB_W, _SUB_H = 320, 180

_BACKGROUND = (40, 30, 20)  # BGR, dark and not grey
_YELLOW = (0, 200, 255)  # a rear plate
_WHITE = (255, 245, 235)  # a front plate: white comes out slightly blue
# A bright block present in every snap, where the ghost mask is aimed.
_GHOST_BLOCK = (300, 10, 340, 30)
_GHOST_PROBE = (320, 20)  # (x, y) inside it

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class _Snap:
    """One vehicle snap of the scene: where its plate is painted, in which
    colour, and what the recogniser should read from it."""

    track: int
    index: int
    plate: Box | None = None
    colour: tuple[int, int, int] = _YELLOW
    text: str = ""
    conf: float = 0.0

    @property
    def name(self) -> str:
        return f"vehicle_{self.track}_main_{self.index}.jpg"


def _plate_at(cx: int, cy: int) -> Box:
    return (cx - 30, cy - 10, cx + 30, cy + 10)


_PARKED = _plate_at(60, 300)

# Sorted by file name, which is the order alpr-run processes them in.
_SCENE: tuple[_Snap, ...] = (
    # Track 1, left to right: the plate rides with the car. Two snaps agree.
    _Snap(1, 1, _plate_at(100, 180), _YELLOW, "AB12CDE", 0.95),
    _Snap(1, 2, _plate_at(200, 180), _YELLOW, "AB12CDE", 0.90),
    _Snap(1, 3, _plate_at(300, 180), _YELLOW, "AB12CDF", 0.60),
    # Track 2, right to left, so its own plate is white. Snap 1 picks up
    # another car's yellow rear plate, with the better confidence.
    _Snap(2, 1, _plate_at(580, 100), _YELLOW, "XY99ZZZ", 0.97),
    _Snap(2, 2, _plate_at(360, 280), _WHITE, "CD34EFG", 0.85),
    # Track 3: the car moves, the plate found does not. A parked car's plate.
    _Snap(3, 1, _PARKED, _YELLOW, "PK11PKD", 0.99),
    _Snap(3, 2, _PARKED, _YELLOW, "PK11PKD", 0.99),
    # Tracks 6 and 7: readable snaps with no plate in them.
    _Snap(6, 1),
    _Snap(7, 1),
)
_EMPTY_SNAP = "vehicle_4_main_1.jpg"  # 0 bytes: the device lost power mid-write
_CORRUPT_SNAP = "vehicle_5_main_1.jpg"  # full of junk: an interrupted pull
_N_SNAPS = len(_SCENE) + 2

# Sub-stream car boxes at fire time (a 4K box is twice these).
_CAR_BOXES: dict[int, list[list[int]]] = {
    1: [[20, 60, 80, 100], [70, 60, 130, 100], [120, 60, 180, 100]],
    2: [[200, 110, 260, 150], [150, 110, 210, 150]],
    3: [[100, 30, 160, 70], [160, 30, 220, 70]],
}
_DIRECTIONS = {1: "left to right", 2: "right to left", 3: "left to right"}


def _paint(snap: _Snap, snap_id: int) -> np.ndarray:
    image = np.full((_H, _W, 3), _BACKGROUND, dtype=np.uint8)
    image[0:16, 0:16] = snap_id * 8  # which snap this is, for the stub models
    x1, y1, x2, y2 = _GHOST_BLOCK
    image[y1:y2, x1:x2] = 255
    if snap.plate is not None:
        x1, y1, x2, y2 = snap.plate
        image[y1:y2, x1:x2] = snap.colour
    return image


class _SceneModels:
    """The stub detector and recogniser. The detector reads the snap's id
    from its corner and returns the plate box painted into it; the recogniser
    then returns that snap's scripted read."""

    name = "scene"

    def __init__(self) -> None:
        self.current: _Snap | None = None
        self.hints: dict[str, Box | None] = {}
        self.ghost_block_brightness: dict[str, int] = {}

    def detect(self, image: np.ndarray, *, bbox_hint: Box | None = None) -> PlateDetection | None:
        snap = _SCENE[round(float(image[4:12, 4:12].mean()) / 8) - 1]
        self.current = snap
        self.hints[snap.name] = bbox_hint
        x, y = _GHOST_PROBE
        self.ghost_block_brightness[snap.name] = int(image[y, x].max())
        if snap.plate is None:
            return None
        return PlateDetection(bbox=snap.plate, det_confidence=0.9)

    def recognize(self, crop_bgr: np.ndarray) -> PlateRead | None:
        snap = self.current
        assert snap is not None
        if not snap.text:
            return None
        return PlateRead(text=snap.text, ocr_confidence=snap.conf, raw_text=snap.text)


@pytest.fixture
def session(tmp_path: Path) -> Path:
    """A session directory holding the scene's snaps and its track records."""
    session_dir = tmp_path / "session_scene"
    session_dir.mkdir()
    for snap_id, snap in enumerate(_SCENE, start=1):
        quality = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
        assert cv2.imwrite(str(session_dir / snap.name), _paint(snap, snap_id), quality)
    (session_dir / _EMPTY_SNAP).write_bytes(b"")
    (session_dir / _CORRUPT_SNAP).write_bytes(b"\x00" * 4096)
    # Files alpr-run must leave alone: a person's snap, a tile, an HQ crop.
    cv2.imwrite(str(session_dir / "person_9_main_1.jpg"), _paint(_Snap(9, 1), 1))
    (session_dir / "vehicle_1.jpg").write_bytes(b"tile")
    (session_dir / "vehicle_1_hq.jpg").write_bytes(b"hq")

    records = []
    for track, boxes in _CAR_BOXES.items():
        record: dict[str, Any] = {
            "track_id": track,
            "direction": _DIRECTIONS[track],
            "main_snaps": list(range(1, len(boxes) + 1)),
            "main_snap_bboxes": boxes,
        }
        if track == 1:
            # Where the car was when snap 1 landed; none recorded for the rest.
            record["main_snap_bboxes_done"] = [[30, 60, 90, 100], None, None]
        records.append(record)
    (session_dir / "session_scene_data.json").write_text(json.dumps(records), encoding="utf-8")
    (session_dir / "session_scene_meta.json").write_text(
        json.dumps({"frame_size": [_SUB_W, _SUB_H]}), encoding="utf-8"
    )
    return session_dir


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> _SceneModels:
    """Make ``alpr-run`` build one pipeline, "preferred", from the stubs."""
    scene = _SceneModels()

    def build(args: Any) -> list[PipelineRunner]:
        return [PipelineRunner("preferred", scene, scene, plate_pad_frac=args.plate_pad_frac)]

    monkeypatch.setattr(alpr_run, "_build_pipelines", build)
    return scene


def _load(session_dir: Path, suffix: str) -> Any:
    return json.loads((session_dir / f"session_scene_{suffix}.json").read_text(encoding="utf-8"))


def _by_image(session_dir: Path) -> dict[str, dict[str, Any]]:
    return {record["image"]: record for record in _load(session_dir, "alpr")}


# ----------------------------------------------------------------------
# The whole run.


def test_run_writes_one_record_per_vehicle_snap(session: Path, models: _SceneModels) -> None:
    assert alpr_run.main([str(session)]) == 0

    records = _load(session, "alpr")
    expected_order = sorted([*(snap.name for snap in _SCENE), _EMPTY_SNAP, _CORRUPT_SNAP])
    assert [record["image"] for record in records] == expected_order
    assert {record["pipeline"] for record in records} == {"preferred"}

    first = records[0]
    assert first["track_id"] == 1 and first["snap_index"] == 1
    assert first["class_name"] == "vehicle"
    assert first["det_bbox"] == list(_plate_at(100, 180))
    assert first["det_conf"] == 0.9
    assert first["ocr_text"] == "AB12CDE" and first["ocr_conf"] == 0.95
    assert first["canonical_uk_shape"] is True
    assert first["error"] is None

    by_image = _by_image(session)
    no_plate = by_image["vehicle_6_main_1.jpg"]
    assert no_plate["det_bbox"] is None and no_plate["ocr_text"] is None
    assert no_plate["error"] is None
    assert by_image[_EMPTY_SNAP]["error"] == "imread_failed"
    assert by_image[_CORRUPT_SNAP]["error"] == "imread_failed"


def test_run_saves_each_plate_crop_under_the_pipeline_name(
    session: Path, models: _SceneModels
) -> None:
    assert alpr_run.main([str(session)]) == 0
    crops = sorted(p.name for p in (session / "alpr_crops" / "preferred").iterdir())
    assert crops == sorted(snap.name for snap in _SCENE if snap.plate is not None)

    crop = cv2.imread(str(session / "alpr_crops" / "preferred" / "vehicle_1_main_1.jpg"))
    # The 60x20 plate plus 10 % padding on each side.
    assert crop.shape[:2] == (24, 72)
    assert _by_image(session)["vehicle_1_main_1.jpg"]["crop_path"].endswith("vehicle_1_main_1.jpg")


def test_run_rollup_keeps_each_tracks_best_read_that_is_its_own(
    session: Path, models: _SceneModels
) -> None:
    assert alpr_run.main([str(session)]) == 0
    tracks = {track["track_id"]: track for track in _load(session, "alpr_by_track")["tracks"]}

    # Track 3 read only the parked car's plate, so it has no read at all.
    assert sorted(tracks) == [1, 2]

    best = tracks[1]["best_preferred"]
    assert (best["ocr_text"], best["ocr_conf"], best["snap_index"]) == ("AB12CDE", 0.95, 1)
    assert best["image"] == "vehicle_1_main_1.jpg"
    assert best["n_agree"] == 1  # snap 2 read the same string; snap 3 did not
    assert best["canonical_uk_shape"] is True
    assert tracks[1]["consensus_preferred"]["ocr_text"] == "AB12CDE"

    # Track 2's most confident read was another car's plate (yellow, on a
    # right-to-left track). Its best read is its own white plate.
    best = tracks[2]["best_preferred"]
    assert (best["ocr_text"], best["ocr_conf"], best["snap_index"]) == ("CD34EFG", 0.85, 2)
    assert best["n_agree"] == 0


def test_run_marks_the_parked_plate_static(session: Path, models: _SceneModels) -> None:
    assert alpr_run.main([str(session)]) == 0
    by_image = _by_image(session)
    marked = sorted(name for name, record in by_image.items() if record.get("static_suspect"))
    assert marked == ["vehicle_3_main_1.jpg", "vehicle_3_main_2.jpg"]

    static = _load(session, "static_plates")
    (spot,) = static["spots"]
    assert (spot["cx"], spot["cy"]) == (60.0, 300.0)
    assert spot["source"] == "within_track"
    assert spot["texts"] == ["PK11PKD"]
    assert static["n_suspect_reads"] == 2


def test_run_marks_a_plate_of_the_wrong_colour(session: Path, models: _SceneModels) -> None:
    assert alpr_run.main([str(session)]) == 0
    by_image = _by_image(session)
    assert by_image["vehicle_2_main_1.jpg"]["plate_colour"] == "yellow"
    assert by_image["vehicle_2_main_1.jpg"]["colour_suspect"] is True
    assert by_image["vehicle_2_main_2.jpg"]["plate_colour"] == "white"
    assert "colour_suspect" not in by_image["vehicle_2_main_2.jpg"]
    # A left-to-right car showing a yellow plate is as expected.
    assert by_image["vehicle_1_main_1.jpg"]["plate_colour"] == "yellow"
    assert "colour_suspect" not in by_image["vehicle_1_main_1.jpg"]
    # Snaps with no read get no colour.
    assert "plate_colour" not in by_image["vehicle_6_main_1.jpg"]


def test_run_stamps_which_fixes_the_session_has(session: Path, models: _SceneModels) -> None:
    assert alpr_run.main([str(session)]) == 0
    static = _load(session, "static_plates")
    assert static["crop_mode"] == "fullframe"
    assert static["ocr_conf"] == "min_char"
    assert static["plate_colour"] == "hsv_v1"
    assert static["generated_at"].endswith("+00:00")

    assert alpr_run.main([str(session), "--crop-mode", "hint"]) == 0
    assert _load(session, "static_plates")["crop_mode"] == "hint"


def test_run_gives_each_snap_a_hint_for_where_its_car_is(
    session: Path, models: _SceneModels
) -> None:
    assert alpr_run.main([str(session)]) == 0
    # A completion-time box is known: fire box and done box together, in 4K.
    assert models.hints["vehicle_1_main_1.jpg"] == (40, 120, 180, 200)
    # None known: this snap's fire box with the track's later ones.
    assert models.hints["vehicle_1_main_2.jpg"] == (140, 120, 360, 200)
    # The track's last snap: extended along the car's last step.
    assert models.hints["vehicle_1_main_3.jpg"] == (240, 120, 660, 200)
    # No record for the track: no hint.
    assert models.hints["vehicle_6_main_1.jpg"] is None


def test_run_with_no_lookahead_hints_the_fire_time_box_alone(
    session: Path, models: _SceneModels
) -> None:
    assert alpr_run.main([str(session), "--hint-lookahead", "0"]) == 0
    assert models.hints["vehicle_1_main_1.jpg"] == (40, 120, 160, 200)
    assert models.hints["vehicle_1_main_2.jpg"] == (140, 120, 260, 200)
    assert models.hints["vehicle_1_main_3.jpg"] == (240, 120, 360, 200)


def test_run_blanks_the_ghost_mask_before_detection(
    session: Path, models: _SceneModels, tmp_path: Path
) -> None:
    assert alpr_run.main([str(session)]) == 0
    assert all(value > 200 for value in models.ghost_block_brightness.values())

    # Rects are given at the size the mask was traced at and scaled to the
    # snap: 1280x720 here, twice the snap.
    x1, y1, x2, y2 = _GHOST_BLOCK
    mask = tmp_path / "ghost_mask.json"
    mask.write_text(
        json.dumps({"source_size": [1280, 720], "rects_4k": [[x1 * 2, y1 * 2, x2 * 2, y2 * 2]]}),
        encoding="utf-8",
    )
    models.ghost_block_brightness.clear()
    assert alpr_run.main([str(session), "--ghost-mask", str(mask)]) == 0
    assert len(models.ghost_block_brightness) == len(_SCENE)
    assert set(models.ghost_block_brightness.values()) == {0}


def test_run_passes_the_plate_padding_to_the_runner(session: Path, models: _SceneModels) -> None:
    assert alpr_run.main([str(session), "--plate-pad-frac", "0.5"]) == 0
    crop = cv2.imread(str(session / "alpr_crops" / "preferred" / "vehicle_1_main_1.jpg"))
    assert crop.shape[:2] == (40, 120)  # 60x20 with half its size added per side


def test_run_limit_processes_the_first_snaps_only(session: Path, models: _SceneModels) -> None:
    assert alpr_run.main([str(session), "--limit", "2"]) == 0
    assert [record["image"] for record in _load(session, "alpr")] == [
        "vehicle_1_main_1.jpg",
        "vehicle_1_main_2.jpg",
    ]


def test_run_without_the_static_filter_keeps_the_parked_read(
    session: Path, models: _SceneModels
) -> None:
    assert alpr_run.main([str(session), "--no-static-filter"]) == 0
    assert not any(record.get("static_suspect") for record in _load(session, "alpr"))
    tracks = {track["track_id"]: track for track in _load(session, "alpr_by_track")["tracks"]}
    assert tracks[3]["best_preferred"]["ocr_text"] == "PK11PKD"
    assert tracks[3]["best_preferred"]["n_agree"] == 1
    # No spot map, but the colour check still leaves its stamp.
    assert _load(session, "static_plates") == {"plate_colour": "hsv_v1"}


def test_run_without_the_colour_check_keeps_the_other_cars_plate(
    session: Path, models: _SceneModels
) -> None:
    assert alpr_run.main([str(session), "--no-colour-check"]) == 0
    records = _load(session, "alpr")
    assert not any("plate_colour" in record or "colour_suspect" in record for record in records)
    tracks = {track["track_id"]: track for track in _load(session, "alpr_by_track")["tracks"]}
    assert tracks[2]["best_preferred"]["ocr_text"] == "XY99ZZZ"
    assert "plate_colour" not in _load(session, "static_plates")


def test_run_checks_colour_on_the_preferred_pipeline_only(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _SceneModels()
    monkeypatch.setattr(
        alpr_run, "_build_pipelines", lambda args: [PipelineRunner("bespoke", scene, scene)]
    )
    assert alpr_run.main([str(session)]) == 0
    records = _load(session, "alpr")
    assert {record["pipeline"] for record in records} == {"bespoke"}
    assert not any("plate_colour" in record for record in records)
    assert (session / "alpr_crops" / "bespoke" / "vehicle_1_main_1.jpg").is_file()
    tracks = {track["track_id"]: track for track in _load(session, "alpr_by_track")["tracks"]}
    assert tracks[2]["best_bespoke"]["ocr_text"] == "XY99ZZZ"


def test_run_with_two_pipelines_records_each_snap_twice(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _SceneModels()
    monkeypatch.setattr(
        alpr_run,
        "_build_pipelines",
        lambda args: [
            PipelineRunner("bespoke", scene, scene),
            PipelineRunner("preferred", scene, scene),
        ],
    )
    assert alpr_run.main([str(session)]) == 0
    records = _load(session, "alpr")
    assert [record["pipeline"] for record in records] == ["bespoke"] * _N_SNAPS + [
        "preferred"
    ] * _N_SNAPS
    track = _load(session, "alpr_by_track")["tracks"][0]
    assert {"best_bespoke", "best_preferred", "consensus_bespoke", "consensus_preferred"} <= set(
        track
    )


# ----------------------------------------------------------------------
# What it prints. The control panel reads these lines.


def test_run_progress_lines_drive_the_panels_progress_bar(
    session: Path, models: _SceneModels, capsys: pytest.CaptureFixture[str]
) -> None:
    assert alpr_run.main([str(session)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert f"[alpr] {_N_SNAPS} vehicle snaps" in lines
    assert "[alpr] running pipeline: preferred" in lines
    # Every tenth snap, and the last.
    assert [line for line in lines if line.endswith(" done")] == [
        f"  [preferred] 10/{_N_SNAPS} done",
        f"  [preferred] {_N_SNAPS}/{_N_SNAPS} done",
    ]

    parser = make_parser("alpr-run")
    for line in lines:
        parser.feed(line)
    assert (parser.progress.current, parser.progress.total) == (_N_SNAPS, _N_SNAPS)
    assert parser.progress.frac == 1.0
    assert parser.progress.summary.startswith("[alpr] wrote ")
    assert parser.progress.summary.endswith("session_scene_alpr_by_track.json")


def test_run_reports_the_filters_and_the_files_it_wrote(
    session: Path, models: _SceneModels, capsys: pytest.CaptureFixture[str]
) -> None:
    assert alpr_run.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "[alpr] 1 completion-time bboxes available" in out
    assert "[alpr] loaded 7 per-snap bboxes (sub-stream (320, 180)); pre-crop hints active" in out
    assert "static-plate filter: 1 static spot(s), 2 detection(s) marked static_suspect" in out
    assert "plate-colour check: 1 read(s) marked colour_suspect" in out
    assert f"[alpr] wrote {session / 'session_scene_alpr.json'}" in out
    assert f"[alpr] wrote {session / 'session_scene_alpr_by_track.json'}" in out


def test_run_tells_an_empty_snap_from_a_corrupt_one(
    session: Path, models: _SceneModels, capsys: pytest.CaptureFixture[str]
) -> None:
    assert alpr_run.main([str(session)]) == 0
    out = capsys.readouterr().out
    # The empty file can't be fixed by a re-pull, so it is a note...
    assert f"  [preferred] {_EMPTY_SNAP}: skipped (empty file)" in out
    assert "[alpr] NOTE: 1 snap(s) are empty (0 bytes)" in out
    # ...and the corrupt one can, so it is a warning that says how.
    assert f"  [preferred] {_CORRUPT_SNAP}: ERROR imread_failed" in out
    assert f"[alpr] WARNING: 1/{_N_SNAPS} snaps unreadable (9.1%)" in out
    assert "streettracker pull --skip-existing" in out


def test_run_with_only_an_empty_snap_does_not_trip_the_panels_scan(
    session: Path, models: _SceneModels, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty snap is nothing the operator can fix, so a run that met one
    must not be reported as a problem. A corrupt snap must be."""
    (session / _CORRUPT_SNAP).unlink()
    assert alpr_run.main([str(session)]) == 0
    clean = capsys.readouterr().out.splitlines()
    assert any("skipped (empty file)" in line for line in clean)
    assert "WARNING" not in "\n".join(clean)
    assert summarize_issue(0, clean) is None

    (session / _CORRUPT_SNAP).write_bytes(b"\x00" * 4096)
    assert alpr_run.main([str(session)]) == 0
    assert summarize_issue(0, capsys.readouterr().out.splitlines()) is not None


def test_run_notes_when_a_session_has_no_recorded_boxes(
    session: Path, models: _SceneModels, capsys: pytest.CaptureFixture[str]
) -> None:
    (session / "session_scene_data.json").unlink()
    assert alpr_run.main([str(session), "--pre-crop"]) == 0
    out = capsys.readouterr().out
    assert "no per-snap bboxes in data.json" in out
    assert set(models.hints.values()) == {None}
    # With no directions either, no read can be called the wrong colour.
    assert not any(record.get("colour_suspect") for record in _load(session, "alpr"))


# ----------------------------------------------------------------------
# Refusals.


def test_run_refuses_a_path_that_is_not_a_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert alpr_run.main([str(tmp_path / "missing")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_run_refuses_a_session_with_no_vehicle_snaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def must_not_build(args: Any) -> list[PipelineRunner]:
        raise AssertionError("models were loaded for a session with nothing to read")

    monkeypatch.setattr(alpr_run, "_build_pipelines", must_not_build)
    (tmp_path / "person_1_main_1.jpg").write_bytes(b"x")
    assert alpr_run.main([str(tmp_path)]) == 1
    assert "no vehicle_*_main_*.jpg snaps" in capsys.readouterr().err


def test_run_refuses_when_no_pipeline_is_selected(
    session: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(alpr_run, "_build_pipelines", lambda args: [])
    assert alpr_run.main([str(session)]) == 2
    assert "no pipelines selected" in capsys.readouterr().err
    assert not (session / "session_scene_alpr.json").exists()


# ----------------------------------------------------------------------
# _build_pipelines: which detector and recogniser each flag combination gets.
# The four model classes load weights when built, so they are replaced; the
# crop wrappers load theirs on first use and are the real ones.


class _Model:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _BespokeDetector(_Model):
    name = "bespoke"


class _EasyOcr(_Model):
    name = "easyocr"


class _OimDetector(_Model):
    name = "oim"


class _FastOcr(_Model):
    name = "fast-plate-ocr"


@pytest.fixture
def stub_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bespoke_mod, "BespokeDetector", _BespokeDetector)
    monkeypatch.setattr(bespoke_mod, "EasyOcrRecognizer", _EasyOcr)
    monkeypatch.setattr(preferred_mod, "OpenImageModelsDetector", _OimDetector)
    monkeypatch.setattr(preferred_mod, "FastPlateOcrRecognizer", _FastOcr)


@pytest.fixture
def road_polygon(tmp_path: Path) -> Path:
    path = tmp_path / "road.json"
    path.write_text(
        json.dumps({"vertices_frac": [[0.1, 0.5], [0.9, 0.5], [0.9, 0.9], [0.1, 0.9]]}),
        encoding="utf-8",
    )
    return path


def _build(*flags: str, road_polygon: Path | None = None) -> list[PipelineRunner]:
    argv = ["some_session", *flags]
    if road_polygon is not None:
        argv += ["--road-polygon", str(road_polygon)]
    return alpr_run._build_pipelines(alpr_run._build_parser().parse_args(argv))


def _parts(runner: PipelineRunner) -> tuple[Any, Any]:
    return runner._detector, runner._recognizer


@pytest.mark.usefixtures("stub_models")
def test_preferred_pipeline_defaults_to_the_fullframe_crop(road_polygon: Path) -> None:
    (runner,) = _build("--pipeline", "preferred", road_polygon=road_polygon)
    detector, recognizer = _parts(runner)
    assert runner.name == "preferred"
    assert runner._plate_pad_frac == 0.10

    assert isinstance(detector, TrajectoryCropDetector)
    assert detector.name == "fullframe-preferred"
    assert detector._vehicle_model_spec == "yolov8m.pt"
    assert detector._poly_frac == [(0.1, 0.5), (0.9, 0.5), (0.9, 0.9), (0.1, 0.9)]
    assert isinstance(detector._plate_detector, _OimDetector)
    assert detector._plate_detector.args == ("yolo-v9-t-640-license-plate-end2end",)

    assert isinstance(recognizer, _FastOcr)
    assert recognizer.args == ("global-plates-mobile-vit-v2-model",)


@pytest.mark.usefixtures("stub_models")
def test_fullframe_runs_without_a_road_polygon_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (runner,) = _build("--pipeline", "preferred", road_polygon=tmp_path / "missing.json")
    assert runner._detector._poly_frac is None
    assert "runs without the on-road filter" in capsys.readouterr().err


@pytest.mark.usefixtures("stub_models")
def test_model_names_and_padding_come_from_the_flags(road_polygon: Path) -> None:
    (runner,) = _build(
        "--pipeline",
        "preferred",
        "--detector-model",
        "some-detector",
        "--ocr-model",
        "some-ocr",
        "--vehicle-model",
        "yolo11x.pt",
        "--plate-pad-frac",
        "0.3",
        road_polygon=road_polygon,
    )
    detector, recognizer = _parts(runner)
    assert detector._vehicle_model_spec == "yolo11x.pt"
    assert detector._plate_detector.args == ("some-detector",)
    assert recognizer.args == ("some-ocr",)
    assert runner._plate_pad_frac == 0.3


@pytest.mark.usefixtures("stub_models")
def test_hint_crop_mode_uses_the_plate_detector_directly() -> None:
    (runner,) = _build("--pipeline", "preferred", "--crop-mode", "hint")
    assert isinstance(runner._detector, _OimDetector)


@pytest.mark.usefixtures("stub_models")
def test_hint_crop_mode_with_pre_crop_wraps_the_detector() -> None:
    (runner,) = _build("--pipeline", "preferred", "--crop-mode", "hint", "--pre-crop")
    detector = runner._detector
    assert isinstance(detector, PreCropDetector)
    assert detector.name == "precrop-preferred"
    assert detector._vehicle_model_spec == "yolov8n.pt"
    assert isinstance(detector._plate_detector, _OimDetector)
    # The default hint is a motion window, so vehicles are found inside it.
    assert detector._vehicle_stage_in_hint is True

    (runner,) = _build(
        "--pipeline", "preferred", "--crop-mode", "hint", "--pre-crop", "--hint-lookahead", "0"
    )
    assert runner._detector._vehicle_stage_in_hint is False


@pytest.mark.usefixtures("stub_models")
def test_default_selection_builds_both_pipelines(road_polygon: Path) -> None:
    runners = _build(road_polygon=road_polygon)
    assert [runner.name for runner in runners] == ["bespoke", "preferred"]
    bespoke, preferred = runners
    assert bespoke._detector.name == "fullframe-bespoke"
    assert isinstance(bespoke._detector._plate_detector, _BespokeDetector)
    assert isinstance(bespoke._recognizer, _EasyOcr)
    assert bespoke._recognizer.kwargs == {"use_gpu": False}
    assert isinstance(preferred._recognizer, _FastOcr)


@pytest.mark.usefixtures("stub_models")
def test_bespoke_alone_loads_no_preferred_models(
    monkeypatch: pytest.MonkeyPatch, road_polygon: Path
) -> None:
    def must_not_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the preferred models were loaded for --pipeline bespoke")

    monkeypatch.setattr(preferred_mod, "OpenImageModelsDetector", must_not_load)
    monkeypatch.setattr(preferred_mod, "FastPlateOcrRecognizer", must_not_load)
    (runner,) = _build("--pipeline", "bespoke", "--gpu", road_polygon=road_polygon)
    assert runner.name == "bespoke"
    assert runner._recognizer.kwargs == {"use_gpu": True}


@pytest.mark.usefixtures("stub_models")
def test_preferred_alone_loads_no_bespoke_models(
    monkeypatch: pytest.MonkeyPatch, road_polygon: Path
) -> None:
    def must_not_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the bespoke models were loaded for --pipeline preferred")

    monkeypatch.setattr(bespoke_mod, "BespokeDetector", must_not_load)
    monkeypatch.setattr(bespoke_mod, "EasyOcrRecognizer", must_not_load)
    (runner,) = _build("--pipeline", "preferred", road_polygon=road_polygon)
    assert runner.name == "preferred"


@pytest.mark.usefixtures("stub_models")
def test_ablation_pairs_the_bespoke_detector_with_the_fast_ocr(road_polygon: Path) -> None:
    runners = _build("--ablation", road_polygon=road_polygon)
    assert [runner.name for runner in runners] == [
        "bespoke",
        "preferred",
        "ablation_bespokedet_fastocr",
    ]
    bespoke, _preferred, ablation = runners
    assert ablation._detector.name == "fullframe-ablation_bespokedet_fastocr"
    assert isinstance(ablation._recognizer, _FastOcr)
    # One set of bespoke weights, shared by both pipelines that use it.
    assert ablation._detector._plate_detector is bespoke._detector._plate_detector


@pytest.mark.usefixtures("stub_models")
def test_ablation_with_preferred_alone_still_loads_the_bespoke_detector(
    road_polygon: Path,
) -> None:
    runners = _build("--pipeline", "preferred", "--ablation", road_polygon=road_polygon)
    assert [runner.name for runner in runners] == ["preferred", "ablation_bespokedet_fastocr"]
    assert isinstance(runners[1]._detector._plate_detector, _BespokeDetector)


@pytest.mark.usefixtures("stub_models")
def test_a_relative_bespoke_model_path_is_taken_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, road_polygon: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (runner,) = _build(
        "--pipeline", "bespoke", "--bespoke-model", "weights/plates.pt", road_polygon=road_polygon
    )
    assert runner._detector._plate_detector.args == (tmp_path / "weights" / "plates.pt",)


# ----------------------------------------------------------------------
# The small loaders around main().


def test_ghost_mask_none_means_no_mask() -> None:
    assert alpr_run._load_ghost_mask(None) == ([], None)


def test_ghost_mask_missing_file_is_ignored_with_a_notice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert alpr_run._load_ghost_mask(tmp_path / "missing.json") == ([], None)
    assert "not found, ignoring" in capsys.readouterr().err


def test_ghost_mask_unparseable_file_is_ignored_with_a_notice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "mask.json"
    path.write_text("{not json", encoding="utf-8")
    assert alpr_run._load_ghost_mask(path) == ([], None)
    assert "parse error" in capsys.readouterr().err


def test_ghost_mask_keeps_the_usable_rects(tmp_path: Path) -> None:
    path = tmp_path / "mask.json"
    path.write_text(
        json.dumps(
            {
                "source_size": [4512, 2512],
                "rects_4k": [
                    [10, 20, 110, 220],
                    [5.9, 6.2, 50.5, 60.5],  # truncated to whole pixels
                    [100, 100, 100, 200],  # no width
                    [100, 200, 300, 100],  # upside down
                    [1, 2, 3],  # not a rect
                    ["a", "b", "c", "d"],
                    None,
                ],
            }
        ),
        encoding="utf-8",
    )
    rects, source_size = alpr_run._load_ghost_mask(path)
    assert rects == [(10, 20, 110, 220), (5, 6, 50, 60)]
    assert source_size == (4512, 2512)


@pytest.mark.parametrize("source_size", [None, [4512], [4512, "tall"], "4512x2512"])
def test_ghost_mask_without_a_usable_source_size(tmp_path: Path, source_size: object) -> None:
    path = tmp_path / "mask.json"
    path.write_text(
        json.dumps({"source_size": source_size, "rects_4k": [[1, 2, 3, 4]]}), encoding="utf-8"
    )
    assert alpr_run._load_ghost_mask(path) == ([(1, 2, 3, 4)], None)


def test_ghost_mask_with_no_rects(tmp_path: Path) -> None:
    path = tmp_path / "mask.json"
    path.write_text(json.dumps({"source_size": [10, 10]}), encoding="utf-8")
    assert alpr_run._load_ghost_mask(path) == ([], (10, 10))


def test_direction_by_track_reads_the_session_records(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_x"
    session_dir.mkdir()
    (session_dir / "session_x_data.json").write_text(
        json.dumps(
            [
                {"track_id": 1, "direction": "left to right"},
                {"track_id": "2", "direction": "right to left"},
                {"track_id": 3, "direction": None},
                {"track_id": 4},
                {"direction": "left to right"},  # no track id
                {"track_id": "x", "direction": "left to right"},
                # Entries that are not records at all are skipped too.
                "not a record",
                None,
                7,
                [1, "left to right"],
            ]
        ),
        encoding="utf-8",
    )
    assert alpr_run._direction_by_track(session_dir) == {
        1: "left to right",
        2: "right to left",
        3: "",
        4: "",
    }


@pytest.mark.parametrize("content", [None, "{not json", '{"track_id": 1}'])
def test_direction_by_track_is_empty_without_usable_records(
    tmp_path: Path, content: str | None
) -> None:
    session_dir = tmp_path / "session_x"
    session_dir.mkdir()
    if content is not None:
        (session_dir / "session_x_data.json").write_text(content, encoding="utf-8")
    assert alpr_run._direction_by_track(session_dir) == {}


def test_stamp_adds_to_the_existing_file(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_x"
    session_dir.mkdir()
    path = session_dir / "session_x_static_plates.json"

    alpr_run._stamp(session_dir, plate_colour="hsv_v1")
    assert json.loads(path.read_text(encoding="utf-8")) == {"plate_colour": "hsv_v1"}

    path.write_text(json.dumps({"crop_mode": "fullframe", "spots": []}), encoding="utf-8")
    alpr_run._stamp(session_dir, plate_colour="hsv_v1", ocr_conf="min_char")
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "crop_mode": "fullframe",
        "spots": [],
        "plate_colour": "hsv_v1",
        "ocr_conf": "min_char",
    }


@pytest.mark.parametrize("damaged", ["{not json", "[1, 2, 3]"])
def test_stamp_replaces_a_file_it_cannot_use(tmp_path: Path, damaged: str) -> None:
    session_dir = tmp_path / "session_x"
    session_dir.mkdir()
    path = session_dir / "session_x_static_plates.json"
    path.write_text(damaged, encoding="utf-8")
    alpr_run._stamp(session_dir, plate_colour="hsv_v1")
    assert json.loads(path.read_text(encoding="utf-8")) == {"plate_colour": "hsv_v1"}


def test_is_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    full = tmp_path / "full.jpg"
    full.write_bytes(b"x")
    assert alpr_run._is_empty(str(empty)) is True
    assert alpr_run._is_empty(str(full)) is False
    assert alpr_run._is_empty(str(tmp_path / "missing.jpg")) is False
    assert alpr_run._is_empty(None) is False


def test_first_snap_size_is_none_when_no_snap_is_readable(tmp_path: Path) -> None:
    empty = tmp_path / "vehicle_1_main_1.jpg"
    empty.write_bytes(b"")
    assert alpr_run._first_snap_size([(empty, 1, 1, "vehicle")]) is None
    assert alpr_run._first_snap_size([]) is None


# ----------------------------------------------------------------------
# The rollup's exclusions (its best-read choice is tested in test_alpr_dispatch).


def _record(track: int, snap: int, text: str | None, conf: float, **extra: Any) -> dict[str, Any]:
    return {
        "pipeline": "preferred",
        "track_id": track,
        "snap_index": snap,
        "image": f"vehicle_{track}_main_{snap}.jpg",
        "ocr_text": text,
        "ocr_conf": conf,
        "det_conf": 0.9,
        "canonical_uk_shape": True,
        **extra,
    }


def test_rollup_never_takes_a_suspect_read_as_best_or_as_agreement() -> None:
    rollup = alpr_run._rollup_by_track(
        [
            _record(1, 1, "PK11PKD", 0.99, static_suspect=True),
            _record(1, 2, "XY99ZZZ", 0.98, colour_suspect=True),
            _record(1, 3, "AB12CDE", 0.70),
            # Agrees with the best read, but is suspect itself: no support.
            _record(1, 4, "AB12CDE", 0.60, static_suspect=True),
            _record(1, 5, None, 0.0),
        ]
    )
    (track,) = rollup["tracks"]
    assert track["best_preferred"]["ocr_text"] == "AB12CDE"
    assert track["best_preferred"]["n_agree"] == 0
    assert track["consensus_preferred"]["n_reads_total"] == 1


def test_rollup_leaves_out_a_track_with_only_suspect_reads() -> None:
    rollup = alpr_run._rollup_by_track(
        [
            _record(1, 1, "PK11PKD", 0.99, static_suspect=True),
            _record(2, 1, "AB12CDE", 0.90),
        ]
    )
    assert [track["track_id"] for track in rollup["tracks"]] == [2]


def test_rollup_counts_agreeing_snaps_and_sorts_by_track() -> None:
    rollup = alpr_run._rollup_by_track(
        [
            _record(7, 1, "AB12CDE", 0.80),
            _record(7, 2, "AB12CDE", 0.95),
            _record(7, 3, "AB12CDE", 0.70),
            _record(7, 4, "AB12CDF", 0.60),
            _record(3, 1, "CD34EFG", 0.50),
        ]
    )
    assert [track["track_id"] for track in rollup["tracks"]] == [3, 7]
    best = rollup["tracks"][1]["best_preferred"]
    assert (best["snap_index"], best["n_agree"]) == (2, 2)


def test_rollup_of_nothing() -> None:
    assert alpr_run._rollup_by_track([]) == {"tracks": []}
