"""The inference command shared by ``makemodel``, ``bodytype`` and ``colour``.

``session_infer`` holds what the three commands have in common; these tests
run each command end to end on a small synthetic session with a tiny
random-init checkpoint (no training, no hub download), and pin what the
commands print and write, since the control panel and ``/stats`` read it.

torch isn't in CI's main job, so the module is skipped there and runs in the
``test-torch`` job.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from streettracker.analysis import vehicle_locator  # noqa: E402
from streettracker.analysis.makemodel import (  # noqa: E402
    bodytype_infer,
    colour_infer,
    infer,
)
from streettracker.analysis.makemodel.bodytype_infer import BodyTypeClassifier  # noqa: E402
from streettracker.analysis.makemodel.colour_infer import ColourClassifier  # noqa: E402
from streettracker.analysis.makemodel.model import MakeModelNet, save_checkpoint  # noqa: E402
from streettracker.analysis.makemodel.session_infer import (  # noqa: E402
    SnapClassifier,
    TopOneClassifier,
    aggregate_top_one,
)
from streettracker.control.progress import make_parser  # noqa: E402

_BODY_TYPES = ("hatchback", "suv", "van")
_COLOURS = ("white", "black", "red", "blue")
_MAKES = ("AUDI", "BMW", "FORD")

Main = Callable[[list[str]], int]


def _checkpoint(tmp_path: Path, head: str, names: tuple[str, ...], **extra: object) -> Path:
    torch.manual_seed(0)
    net = MakeModelNet({head: len(names)}, pretrained=False)
    path = tmp_path / f"{head}.pt"
    meta = {f"{head}_names": list(names), "input_size": 224, **extra}
    save_checkpoint(net, path, metadata=meta)
    return path


def _body_checkpoint(tmp_path: Path, **extra: object) -> Path:
    return _checkpoint(tmp_path, "body_type", _BODY_TYPES, **extra)


def _colour_checkpoint(tmp_path: Path, **extra: object) -> Path:
    return _checkpoint(tmp_path, "colour", _COLOURS, **extra)


def _make_checkpoint(tmp_path: Path) -> Path:
    return _checkpoint(tmp_path, "make", _MAKES, kind="uk_make")


def _session(tmp_path: Path) -> Path:
    """Five vehicle snaps: track 1 twice, track 2 once, track 3 with no
    recorded box, and track 4 with an empty (unreadable) file."""
    sess = tmp_path / "session_x"
    sess.mkdir()
    for name in ("vehicle_1_main_1", "vehicle_1_main_2", "vehicle_2_main_1", "vehicle_3_main_1"):
        Image.new("RGB", (400, 300), (120, 60, 30)).save(sess / f"{name}.jpg")
    (sess / "vehicle_4_main_1.jpg").write_bytes(b"")
    box = [64, 36, 200, 180]
    records = [
        {"track_id": 1, "main_snaps": [1, 2], "main_snap_bboxes": [box, box]},
        {"track_id": 2, "main_snaps": [1], "main_snap_bboxes": [box]},
        {"track_id": 4, "main_snaps": [1], "main_snap_bboxes": [box]},
    ]
    (sess / "session_x_data.json").write_text(json.dumps(records))
    (sess / "session_x_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    return sess


def _load(sess: Path, suffix: str) -> list | dict:
    return json.loads((sess / f"session_x_{suffix}.json").read_text(encoding="utf-8"))


_IMAGE = np.zeros((300, 400, 3), dtype=np.uint8)
_BOX = (50, 40, 200, 180)


# ----------------------------------------------------------------------
# Classifiers.


def test_classifier_classes_share_one_base() -> None:
    assert issubclass(BodyTypeClassifier, TopOneClassifier)
    assert issubclass(ColourClassifier, TopOneClassifier)
    assert issubclass(infer.MakeModelClassifier, SnapClassifier)


def test_body_type_classifier_gives_the_top_class(tmp_path: Path) -> None:
    clf = BodyTypeClassifier(_body_checkpoint(tmp_path), device="cpu")
    assert clf.body_types == _BODY_TYPES
    pred = clf.classify(_IMAGE, _BOX)
    assert pred is not None
    name, conf = pred
    assert name in _BODY_TYPES
    assert 1 / len(_BODY_TYPES) <= conf <= 1.0  # the top of a softmax over three


def test_colour_classifier_gives_the_top_class(tmp_path: Path) -> None:
    clf = ColourClassifier(_colour_checkpoint(tmp_path), device="cpu")
    assert clf.colours == _COLOURS
    pred = clf.classify(_IMAGE, _BOX)
    assert pred is not None and pred[0] in _COLOURS


def test_top_one_classifier_needs_a_box(tmp_path: Path) -> None:
    clf = ColourClassifier(_colour_checkpoint(tmp_path), device="cpu")
    assert clf.classify(_IMAGE, None) is None
    assert clf.classify(_IMAGE, (10, 10, 10, 10)) is None  # no area


def test_classifying_is_repeatable(tmp_path: Path) -> None:
    clf = BodyTypeClassifier(_body_checkpoint(tmp_path), device="cpu")
    assert clf.classify(_IMAGE, _BOX) == clf.classify(_IMAGE, _BOX)


def test_a_checkpoint_for_another_head_is_refused(tmp_path: Path) -> None:
    colour = _colour_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=r"is not a body-type model \(heads: \['colour'\]\)"):
        BodyTypeClassifier(colour, device="cpu")
    body = _body_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="is not a colour model"):
        ColourClassifier(body, device="cpu")


def test_a_checkpoint_without_class_names_is_refused(tmp_path: Path) -> None:
    net = MakeModelNet({"body_type": 3}, pretrained=False)
    path = tmp_path / "nameless.pt"
    save_checkpoint(net, path, metadata={"input_size": 224})
    with pytest.raises(ValueError, match="body-type checkpoint is missing 'body_type_names'"):
        BodyTypeClassifier(path, device="cpu")


@pytest.mark.parametrize(
    ("classifier", "head", "names"),
    [(BodyTypeClassifier, "body_type", _BODY_TYPES), (ColourClassifier, "colour", _COLOURS)],
)
def test_crop_settings_follow_the_checkpoint_or_the_arguments(
    tmp_path: Path, classifier: type[TopOneClassifier], head: str, names: tuple[str, ...]
) -> None:
    legacy = _checkpoint(tmp_path, head, names)
    clf = classifier(legacy, device="cpu")
    assert (clf.crop_mode, clf.pad_frac, clf.input_size) == ("hint", 0.1, 224)

    recorded = _checkpoint(
        tmp_path, head, names, crop_mode="plate", crop_pad_frac=0.2, input_size=384
    )
    clf = classifier(recorded, device="cpu")
    assert (clf.crop_mode, clf.pad_frac, clf.input_size) == ("fullframe", 0.2, 384)

    clf = classifier(recorded, device="cpu", crop_mode="hint", pad_frac=0.3, input_size=256)
    assert (clf.crop_mode, clf.pad_frac, clf.input_size) == ("hint", 0.3, 256)


# ----------------------------------------------------------------------
# The per-track vote.


def _row(track: int, value: str | None, conf: float | None) -> dict:
    return {"track_id": track, "kind": value, "conf": conf}


def test_vote_sums_confidence_and_reports_the_winners_best_read() -> None:
    rows = [_row(1, "suv", 0.6), _row(1, "van", 0.9), _row(1, "suv", 0.55)]
    (track,) = aggregate_top_one(rows, 0.5, field="kind")
    assert track == {
        "track_id": 1,
        "kind": "suv",  # 1.15 beats 0.9
        "conf": 0.6,
        "n_high_conf_reads": 3,
        "n_reads": 3,
    }


def test_vote_tie_goes_to_the_class_seen_first() -> None:
    rows = [_row(1, "van", 0.5), _row(1, "suv", 0.5)]
    assert aggregate_top_one(rows, 0.5, field="kind")[0]["kind"] == "van"


def test_vote_below_the_threshold_names_no_class_but_keeps_the_confidence() -> None:
    rows = [_row(7, "suv", 0.3), _row(7, "suv", 0.44444)]
    (track,) = aggregate_top_one(rows, 0.5, field="kind")
    assert track["kind"] is None
    assert track["conf"] == 0.4444
    assert track["n_high_conf_reads"] == 0


def test_vote_skips_snaps_with_no_prediction_and_sorts_by_track() -> None:
    rows = [_row(9, "suv", 0.8), _row(3, None, None), _row(2, "van", 0.7)]
    assert [t["track_id"] for t in aggregate_top_one(rows, 0.5, field="kind")] == [2, 9]


def test_vote_can_mark_its_value_as_a_prediction() -> None:
    (track,) = aggregate_top_one(
        [_row(1, "red", 0.9)], 0.5, field="kind", source_field="kind_source"
    )
    assert list(track) == [
        "track_id",
        "kind",
        "kind_source",
        "conf",
        "n_high_conf_reads",
        "n_reads",
    ]
    assert track["kind_source"] == "cnn"


def test_each_modules_vote_uses_its_own_field() -> None:
    body = bodytype_infer.aggregate_by_track(
        [{"track_id": 1, "body_type": "suv", "conf": 0.9}], 0.5
    )
    assert list(body[0]) == ["track_id", "body_type", "conf", "n_high_conf_reads", "n_reads"]
    colour = colour_infer.aggregate_by_track([{"track_id": 1, "colour": "red", "conf": 0.9}], 0.5)
    assert list(colour[0]) == [
        "track_id",
        "colour",
        "colour_source",
        "conf",
        "n_high_conf_reads",
        "n_reads",
    ]


# ----------------------------------------------------------------------
# The commands, end to end.


def test_bodytype_command_writes_both_files(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    rc = bodytype_infer.main(
        [str(sess), "--model", str(_body_checkpoint(tmp_path)), "--cpu", "--conf-threshold", "0"]
    )
    assert rc == 0

    per_image = _load(sess, "bodytype")
    assert [row["image"] for row in per_image] == [
        "vehicle_1_main_1.jpg",
        "vehicle_1_main_2.jpg",
        "vehicle_2_main_1.jpg",
        "vehicle_3_main_1.jpg",
        "vehicle_4_main_1.jpg",
    ]
    first = per_image[0]
    assert list(first) == [
        "track_id",
        "snap_index",
        "image",
        "has_bbox",
        "crop_source",
        "body_type",
        "conf",
    ]
    assert (first["track_id"], first["snap_index"]) == (1, 1)
    assert first["has_bbox"] is True and first["crop_source"] == "hint"
    assert first["body_type"] in _BODY_TYPES
    assert first["conf"] == round(first["conf"], 4)

    # No recorded box, and an unreadable file: a row each, with no prediction.
    for row in per_image[3:]:
        assert (row["body_type"], row["conf"]) == (None, None)
    assert per_image[3]["has_bbox"] is False

    tracks = _load(sess, "bodytype_by_track")["tracks"]
    assert [t["track_id"] for t in tracks] == [1, 2]
    assert list(tracks[0]) == ["track_id", "body_type", "conf", "n_high_conf_reads", "n_reads"]
    assert tracks[0]["n_reads"] == 2 and tracks[1]["n_reads"] == 1


def test_colour_command_marks_its_tracks_as_predictions(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    rc = colour_infer.main(
        [str(sess), "--model", str(_colour_checkpoint(tmp_path)), "--cpu", "--conf-threshold", "0"]
    )
    assert rc == 0
    assert list(_load(sess, "colour")[0])[-2:] == ["colour", "conf"]
    tracks = _load(sess, "colour_by_track")["tracks"]
    assert [t["track_id"] for t in tracks] == [1, 2]
    assert all(t["colour"] in _COLOURS and t["colour_source"] == "cnn" for t in tracks)


def test_make_command_writes_ranked_candidates(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    rc = infer.main(
        [str(sess), "--model", str(_make_checkpoint(tmp_path)), "--cpu", "--top-k", "2"]
    )
    assert rc == 0
    per_image = _load(sess, "makemodel")
    assert list(per_image[0]) == [
        "track_id",
        "snap_index",
        "image",
        "has_bbox",
        "crop_source",
        "top_k",
    ]
    assert [len(row["top_k"]) for row in per_image] == [2, 2, 2, 0, 0]
    assert list(per_image[0]["top_k"][0]) == ["make", "model", "year", "conf"]
    assert [t["track_id"] for t in _load(sess, "makemodel_by_track")["tracks"]] == [1, 2]


def test_a_high_threshold_leaves_every_track_unnamed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sess = _session(tmp_path)
    args = [str(sess), "--model", str(_body_checkpoint(tmp_path)), "--cpu"]
    assert bodytype_infer.main([*args, "--conf-threshold", "1.5"]) == 0
    tracks = _load(sess, "bodytype_by_track")["tracks"]
    assert [t["body_type"] for t in tracks] == [None, None]
    assert all(t["conf"] > 0 and t["n_high_conf_reads"] == 0 for t in tracks)
    assert capsys.readouterr().out.splitlines()[-1] == "[bodytype] 2 tracks, 0 confident (>= 1.5); "


def test_limit_takes_the_first_snaps(tmp_path: Path) -> None:
    sess = _session(tmp_path)
    args = [str(sess), "--model", str(_colour_checkpoint(tmp_path)), "--cpu", "--limit", "2"]
    assert colour_infer.main(args) == 0
    assert [row["image"] for row in _load(sess, "colour")] == [
        "vehicle_1_main_1.jpg",
        "vehicle_1_main_2.jpg",
    ]


def test_model_defaults_to_the_modules_production_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Read when the command runs, so pointing DEFAULT_MODEL elsewhere works.
    monkeypatch.setattr(bodytype_infer, "DEFAULT_MODEL", _body_checkpoint(tmp_path))
    sess = _session(tmp_path)
    assert bodytype_infer.main([str(sess), "--cpu"]) == 0
    assert (sess / "session_x_bodytype.json").is_file()


# ----------------------------------------------------------------------
# What the commands print. The control panel reads these lines.


def test_bodytype_command_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sess = _session(tmp_path)
    args = [str(sess), "--model", str(_body_checkpoint(tmp_path)), "--cpu"]
    assert bodytype_infer.main([*args, "--conf-threshold", "0"]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == (
        "[bodytype] 5 snaps; 4 per-snap bboxes; device=cpu; 3 classes; input=224px; "
        "crop=hint pad=0.1"
    )
    assert lines[1] == "  5/5 classified"
    assert lines[2] == f"[bodytype] wrote {sess / 'session_x_bodytype.json'}"
    assert lines[3] == f"[bodytype] wrote {sess / 'session_x_bodytype_by_track.json'}"
    assert re.fullmatch(
        r"\[bodytype\] 2 tracks, 2 confident \(>= 0\.0\); (\w+=\d+)( \w+=\d+)?", lines[4]
    )
    assert len(lines) == 5


def test_make_command_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sess = _session(tmp_path)
    args = [str(sess), "--model", str(_make_checkpoint(tmp_path)), "--cpu"]
    assert infer.main([*args, "--conf-threshold", "0"]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == (
        "[makemodel] 5 snaps; 4 per-snap bboxes; device=cpu; 3 makes (make-only); "
        "input=224px; crop=hint pad=0.25"
    )
    assert lines[1] == "  5/5 classified"
    assert lines[2] == f"[makemodel] wrote {sess / 'session_x_makemodel.json'}"
    assert lines[3] == f"[makemodel] wrote {sess / 'session_x_makemodel_by_track.json'}"
    assert lines[4].startswith(
        "[makemodel] 2 tracks classified, 2 above conf 0.0 (2 snaps had no vehicle box; "
        "crop sources {"
    )


@pytest.mark.parametrize(
    ("kind", "main", "checkpoint"),
    [
        ("makemodel", infer.main, _make_checkpoint),
        ("bodytype", bodytype_infer.main, _body_checkpoint),
        ("colour", colour_infer.main, _colour_checkpoint),
    ],
    ids=["makemodel", "bodytype", "colour"],
)
def test_the_panel_reads_progress_from_each_commands_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    main: Main,
    checkpoint: Callable[[Path], Path],
) -> None:
    """What the command prints, fed to the parser the control panel picks for
    it, must move the progress bar to the end. Until 2026-11 the panel matched
    none of these commands' progress lines, and no test joined the two."""
    command = {
        "makemodel": infer._MakeInference,
        "bodytype": bodytype_infer._BodyTypeInference,
        "colour": colour_infer._ColourInference,
    }[kind]
    monkeypatch.setattr(command, "progress_every", 2)  # several lines from five snaps
    sess = _session(tmp_path)
    assert main([str(sess), "--model", str(checkpoint(tmp_path)), "--cpu"]) == 0
    lines = capsys.readouterr().out.splitlines()

    parser = make_parser(kind)
    counts = []
    for line in lines:
        parser.feed(line)
        counts.append((parser.progress.current, parser.progress.total))
    assert (2, 5) in counts and (4, 5) in counts
    assert counts[-1] == (5, 5)
    assert parser.progress.frac == 1.0
    # The job's summary in the panel is the command's closing line.
    assert parser.progress.summary == lines[-1].strip()
    assert parser.progress.summary.startswith(f"[{kind}] 2 tracks")


@pytest.mark.parametrize(
    ("module", "private_class", "every"),
    [(infer, "_MakeInference", 100), (bodytype_infer, "_BodyTypeInference", 200)],
)
def test_progress_is_printed_at_a_fixed_interval_and_at_the_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module: object,
    private_class: str,
    every: int,
) -> None:
    command = getattr(module, private_class)
    assert command.progress_every == every

    monkeypatch.setattr(command, "progress_every", 2)
    sess = _session(tmp_path)
    make = command is infer._MakeInference
    ckpt = _make_checkpoint(tmp_path) if make else _body_checkpoint(tmp_path)
    assert module.main([str(sess), "--model", str(ckpt), "--cpu"]) == 0  # type: ignore[attr-defined]
    progress = [line for line in capsys.readouterr().out.splitlines() if "classified" in line]
    assert progress[:3] == ["  2/5 classified", "  4/5 classified", "  5/5 classified"]


_COMMANDS: list[tuple[str, Main, str]] = [
    ("makemodel", infer.main, "streettracker makemodel-train-uk ..."),
    ("bodytype", bodytype_infer.main, "makemodel-train-uk ... --target body_type"),
    ("colour", colour_infer.main, "makemodel-train-uk ... --target colour"),
]


@pytest.mark.parametrize(("tag", "main", "train_command"), _COMMANDS, ids=[c[0] for c in _COMMANDS])
def test_refusals(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tag: str,
    main: Main,
    train_command: str,
) -> None:
    missing_dir = tmp_path / "missing"
    assert main([str(missing_dir)]) == 2
    assert capsys.readouterr().out == f"[{tag}] not a directory: {missing_dir}\n"

    sess = tmp_path / "session_empty"
    sess.mkdir()
    no_model = tmp_path / "nope.pt"
    assert main([str(sess), "--model", str(no_model)]) == 1
    # The second line is aligned under the first line's text.
    assert capsys.readouterr().out == (
        f"[{tag}] checkpoint not found: {no_model}\n"
        f"{' ' * (len(tag) + 3)}train one (`{train_command}`) and copy its best.pt here, "
        f"or pass --model <path>.\n"
    )


# ----------------------------------------------------------------------
# A vehicle stopped in the road. Five tracks pass a van, a minute apart, and
# in each snap the van is the only vehicle the detector finds on the road.

_VAN = (300.0, 60.0, 380.0, 130.0, 0.9)


def _street(tmp_path: Path) -> Path:
    sess = tmp_path / "session_x"
    sess.mkdir()
    records = []
    for track in range(1, 6):
        Image.new("RGB", (400, 300), (120, 60, 30)).save(sess / f"vehicle_{track}_main_1.jpg")
        records.append(
            {
                "track_id": track,
                "main_snaps": [1],
                "main_snap_bboxes": [[320, 60, 448, 156]],  # beside the van, not on it
                "main_snap_fire_unix": [1000.0 + 60.0 * track],
            }
        )
    (sess / "session_x_data.json").write_text(json.dumps(records))
    (sess / "session_x_meta.json").write_text(json.dumps({"frame_size": [640, 360]}))
    return sess


def _stub_vehicle_detector(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Stand in for YOLO: every snap shows the van. Returns the call log."""
    calls: list[int] = []

    def factory(*_args: object, **_kwargs: object) -> Callable[[np.ndarray], list]:
        def detect(_image: np.ndarray) -> list:
            calls.append(1)
            return [_VAN]

        return detect

    monkeypatch.setattr(vehicle_locator, "yolo_vehicle_detector", factory)
    return calls


def _fullframe_args(tmp_path: Path, sess: Path) -> list[str]:
    ckpt = _colour_checkpoint(tmp_path, crop_mode="plate", crop_pad_frac=0.1)
    no_polygon = str(tmp_path / "no_road_polygon.json")
    return [str(sess), "--model", str(ckpt), "--cpu", "--road-polygon", no_polygon]


def test_a_stationary_vehicle_is_not_classified_as_the_cars_passing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_vehicle_detector(monkeypatch)
    sess = _street(tmp_path)
    assert colour_infer.main([*_fullframe_args(tmp_path, sess), "--conf-threshold", "0"]) == 0
    lines = capsys.readouterr().out.splitlines()

    # First and last tracks: nothing shows the van before the one or after
    # the other, so they keep it. The three between were only passing.
    rows = _load(sess, "colour")
    assert [(r["track_id"], r["has_bbox"], r["crop_source"]) for r in rows] == [
        (1, True, "trajectory"),
        (2, False, None),
        (3, False, None),
        (4, False, None),
        (5, True, "trajectory"),
    ]
    assert all(r["colour"] is None and r["conf"] is None for r in rows[1:4])
    assert [t["track_id"] for t in _load(sess, "colour_by_track")["tracks"]] == [1, 5]

    assert lines[1] == "  5/5 classified"
    assert lines[2] == "[colour] stationary vehicles: 3 crops refused (3 snaps left with no crop)"
    assert lines[3] == f"[colour] wrote {sess / 'session_x_colour.json'}"
    assert lines[-1].startswith("[colour] 2 tracks, 2 confident")
    assert len(calls) == 5  # the second look needs no detection

    vehicles = _load(sess, "stationary_vehicles")["vehicles"]
    assert [(v["box"], v["n_crops"], v["n_tracks"]) for v in vehicles] == [
        ([300, 60, 380, 130], 3, 3)
    ]

    # The panel still ends on the closing line as the job's summary.
    parser = make_parser("colour")
    for line in lines:
        parser.feed(line)
    assert (parser.progress.current, parser.progress.total) == (5, 5)
    assert parser.progress.summary == lines[-1].strip()


def test_a_later_command_knows_the_stationary_vehicles_from_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_vehicle_detector(monkeypatch)
    sess = _street(tmp_path)
    args = _fullframe_args(tmp_path, sess)
    assert colour_infer.main(args) == 0
    first = _load(sess, "colour")
    capsys.readouterr()

    # The boxes are cached now: the same rows, with no second look needed.
    assert colour_infer.main(args) == 0
    out = capsys.readouterr().out
    assert "[colour] stationary vehicles: 3 crops refused (3 snaps left with no crop)" in out
    assert _load(sess, "colour") == first
    assert len(calls) == 5


@pytest.mark.parametrize(
    ("tag", "main", "checkpoint"),
    [
        ("makemodel", infer.main, _make_checkpoint),
        ("bodytype", bodytype_infer.main, _body_checkpoint),
        ("colour", colour_infer.main, _colour_checkpoint),
    ],
    ids=["makemodel", "bodytype", "colour"],
)
def test_a_session_with_no_vehicle_snaps_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tag: str,
    main: Main,
    checkpoint: Callable[[Path], Path],
) -> None:
    sess = tmp_path / "session_empty"
    sess.mkdir()
    Image.new("RGB", (40, 30)).save(sess / "person_1_main_1.jpg")
    assert main([str(sess), "--model", str(checkpoint(tmp_path)), "--cpu"]) == 1
    assert capsys.readouterr().out == f"[{tag}] no vehicle_*_main_*.jpg snaps in {sess}\n"
    assert not list(sess.glob("*.json"))
