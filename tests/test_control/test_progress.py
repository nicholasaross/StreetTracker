"""Tests for the stdout progress parsers.

Each parser is pure (text in, structured state out), so these feed canned
transcripts — the exact lines the CLIs emit — and assert the resulting
:class:`Progress`.
"""

from __future__ import annotations

from streettracker.control import progress
from streettracker.control.progress import (
    BatchParser,
    BuildParser,
    ClassifyParser,
    DvsaLabelParser,
    GenericParser,
    PullParser,
    TrainParser,
    estimate_eta,
    make_parser,
)


def _feed(parser: progress.ProgressParser, *lines: str) -> progress.Progress:
    for line in lines:
        parser.feed(line)
    return parser.progress


# ----------------------------------------------------------------------
# estimate_eta + Progress.frac
# ----------------------------------------------------------------------


def test_estimate_eta_midway() -> None:
    # 100/1000 in 10s -> 10/s -> 900 left -> 90s.
    assert estimate_eta(100, 1000, 10.0) == 90.0


def test_estimate_eta_complete_is_zero() -> None:
    assert estimate_eta(1000, 1000, 5.0) == 0.0
    assert estimate_eta(1200, 1000, 5.0) == 0.0


def test_estimate_eta_insufficient_signal() -> None:
    assert estimate_eta(None, 1000, 10.0) is None
    assert estimate_eta(0, 1000, 10.0) is None
    assert estimate_eta(100, None, 10.0) is None
    assert estimate_eta(100, 1000, 0.0) is None
    assert estimate_eta(100, 0, 10.0) is None


def test_frac_derivation_and_json() -> None:
    p = progress.Progress(current=3, total=12, phase="x")
    assert p.frac == 0.25
    d = p.to_json_dict()
    assert d["frac"] == 0.25 and d["current"] == 3 and d["total"] == 12
    assert progress.Progress().frac is None  # no total


# ----------------------------------------------------------------------
# BatchParser (alpr-run, alpr-rescore, makemodel-compare)
# ----------------------------------------------------------------------


def test_batch_parser_counts_and_phase() -> None:
    p = _feed(
        BatchParser(),
        "[alpr] running pipeline: preferred",
        "  [preferred] 10/1200 done",
        "  [preferred] 240/1200 done",
    )
    assert p.current == 240
    assert p.total == 1200
    assert p.phase == "preferred 240/1200"
    assert abs((p.frac or 0) - 0.2) < 1e-9


def test_batch_parser_batch_tag_and_wrote_summary() -> None:
    p = _feed(
        BatchParser(),
        "  [batch] 50/100 done",
        "[alpr-rescore] wrote session_x_alpr.json + session_x_alpr_by_track.json",
    )
    assert p.current == 50 and p.total == 100
    assert "wrote" in p.summary


# ----------------------------------------------------------------------
# ClassifyParser (makemodel, bodytype, colour)
#
# The transcripts are what the commands print (SessionInference.main in
# analysis/makemodel/session_infer.py), taken from a real run with the paths
# shortened. tests/test_analysis/test_makemodel/test_session_infer.py feeds
# this parser each command's live output too, in the job that has torch.
# ----------------------------------------------------------------------

_MAKEMODEL_RUN = (
    "[makemodel] 115 snaps; 271 per-snap bboxes; device=cuda; 62 makes (make-only); "
    "input=528px; crop=fullframe pad=0.1",
    "  100/115 classified",
    "  115/115 classified",
    "[makemodel] wrote /out/session_x/session_x_makemodel.json",
    "[makemodel] wrote /out/session_x/session_x_makemodel_by_track.json",
    "[makemodel] 40 tracks classified, 37 above conf 0.4 (18 snaps had no vehicle box; "
    "crop sources {'trajectory': 97, 'none': 18})",
)
_BODYTYPE_RUN = (
    "[bodytype] 450 snaps; 900 per-snap bboxes; device=cuda; 7 classes; input=384px; "
    "crop=fullframe pad=0.1",
    "  200/450 classified",
    "  400/450 classified",
    "  450/450 classified",
    "[bodytype] wrote /out/session_x/session_x_bodytype.json",
    "[bodytype] wrote /out/session_x/session_x_bodytype_by_track.json",
    "[bodytype] 40 tracks, 39 confident (>= 0.5); suv=18 hatchback=15 saloon=5 mpv=1",
)


def test_classify_parser_follows_the_count() -> None:
    parser = ClassifyParser()
    seen = []
    for line in _BODYTYPE_RUN:
        parser.feed(line)
        seen.append((parser.progress.current, parser.progress.total, parser.progress.phase))
    assert seen[0] == (None, None, "")  # the header is not progress
    assert seen[1] == (200, 450, "classified 200/450")
    assert seen[2] == (400, 450, "classified 400/450")
    assert seen[3:] == [(450, 450, "classified 450/450")] * 4
    assert abs((_feed(ClassifyParser(), *_BODYTYPE_RUN[:2]).frac or 0) - 200 / 450) < 1e-9


def test_classify_parser_summary_is_the_closing_line() -> None:
    assert _feed(ClassifyParser(), *_BODYTYPE_RUN).summary == _BODYTYPE_RUN[-1]
    assert _feed(ClassifyParser(), *_MAKEMODEL_RUN).summary == _MAKEMODEL_RUN[-1]


def test_classify_parser_does_not_take_the_closing_line_for_progress() -> None:
    """makemodel's closing line reads "40 tracks classified, ...". It must
    not move the count."""
    p = _feed(ClassifyParser(), *_MAKEMODEL_RUN)
    assert (p.current, p.total, p.frac) == (115, 115, 1.0)
    assert p.phase == "classified 115/115"


def test_classify_parser_ignores_lookalike_lines() -> None:
    p = _feed(
        ClassifyParser(),
        "  12/40 classified so far",
        "classified 12/40",
        "[colour] 12/40 classified",
        "  x/40 classified",
    )
    assert (p.current, p.total) == (None, None)


def test_classifier_commands_drive_a_progress_bar() -> None:
    """The three commands as the panel runs them: each gets a parser that
    reads its count. Before 2026-11 none did, so these jobs showed no bar."""
    for kind, run in (
        ("makemodel", _MAKEMODEL_RUN),
        ("bodytype", _BODYTYPE_RUN),
        ("colour", tuple(line.replace("bodytype", "colour") for line in _BODYTYPE_RUN)),
    ):
        parser = make_parser(kind)
        assert isinstance(parser, ClassifyParser), kind
        for line in run[:2]:
            parser.feed(line)
        assert parser.progress.frac is not None and 0 < parser.progress.frac < 1, kind
        assert estimate_eta(parser.progress.current, parser.progress.total, 60.0) is not None
        for line in run[2:]:
            parser.feed(line)
        assert parser.progress.frac == 1.0, kind
        parser.finish(0)
        assert parser.progress.ok is True and parser.progress.summary == run[-1]


# ----------------------------------------------------------------------
# TrainParser (live loss curve)
# ----------------------------------------------------------------------


def test_train_parser_preamble_epochs_and_best() -> None:
    # Current trainer format (post --target): "target=make ... classes=N".
    p = _feed(
        TrainParser(),
        "[makemodel-uk] target=make device=cuda amp=True classes=36 "
        "train_crops=26500 val_crops=6277",
        "[makemodel-uk] epoch 1/30 loss=2.1456 make@1=0.325 make@5=0.612 (45s)",
        "[makemodel-uk] epoch 2/30 loss=1.8923 make@1=0.361 make@5=0.651 (44s)",
    )
    assert p.metrics["target"] == "make"
    assert p.metrics["device"] == "cuda"
    assert p.metrics["amp"] is True
    assert p.metrics["makes"] == 36
    assert p.metrics["train_crops"] == 26500
    assert p.metrics["val_crops"] == 6277
    assert len(p.metrics["epochs"]) == 2
    assert p.metrics["epochs"][0] == {
        "epoch": 1,
        "loss": 2.1456,
        "make1": 0.325,
        "make5": 0.612,
        "sec": 45,
    }
    assert p.current == 2 and p.total == 30
    assert p.phase == "epoch 2/30"
    assert p.metrics["best_make1"] == 0.361 and p.metrics["best_epoch"] == 2


def test_train_parser_accepts_legacy_preamble() -> None:
    # Pre --target format: no target=, "makes=" instead of "classes=".
    p = _feed(
        TrainParser(),
        "[makemodel-uk] device=cuda amp=true makes=36 train_crops=26500 val_crops=6277",
    )
    assert p.metrics["target"] == "make"
    assert p.metrics["makes"] == 36
    assert p.metrics["train_crops"] == 26500


def test_train_parser_body_type_target() -> None:
    # --target body_type runs print body_type@1/@5; the row keys stay
    # make1/make5 (template/history compatibility) and metrics["target"]
    # records which metric this actually is.
    p = _feed(
        TrainParser(),
        "[makemodel-uk] target=body_type device=cuda amp=True classes=7 "
        "train_crops=38000 val_crops=9500",
        "[makemodel-uk] epoch 1/20 loss=1.2345 body_type@1=0.581 body_type@5=0.998 (95s)",
        "[makemodel-uk] epoch 2/20 loss=0.9871 body_type@1=0.652 body_type@5=0.999 (93s)",
        "[makemodel-uk] done: best body_type@1=0.694 (epoch 12) -> /runs/uk_body_0708_b0/best.pt",
    )
    assert p.metrics["target"] == "body_type"
    assert p.metrics["makes"] == 7
    assert len(p.metrics["epochs"]) == 2
    assert p.metrics["epochs"][1] == {
        "epoch": 2,
        "loss": 0.9871,
        "make1": 0.652,
        "make5": 0.999,
        "sec": 93,
    }
    assert p.metrics["best_make1"] == 0.694 and p.metrics["best_epoch"] == 12
    assert p.metrics["checkpoint"] == "/runs/uk_body_0708_b0/best.pt"
    assert p.phase == "done"


def test_train_parser_best_does_not_regress() -> None:
    p = _feed(
        TrainParser(),
        "[makemodel-uk] epoch 1/30 loss=1.0 make@1=0.40 make@5=0.7 (10s)",
        "[makemodel-uk] epoch 2/30 loss=0.9 make@1=0.38 make@5=0.7 (10s)",
    )
    assert p.metrics["best_make1"] == 0.40 and p.metrics["best_epoch"] == 1


def test_train_parser_early_stop_and_done() -> None:
    p = _feed(
        TrainParser(),
        "[makemodel-uk] epoch 16/30 loss=0.41 make@1=0.402 make@5=0.80 (42s)",
        "[makemodel-uk] early stop after 21 epochs",
        "[makemodel-uk] done: best make@1=0.402 (epoch 16) -> /runs/uk_make_0614_b5/best.pt",
    )
    assert p.metrics["early_stopped_at"] == 21
    assert p.metrics["best_make1"] == 0.402
    assert p.metrics["best_epoch"] == 16
    assert p.metrics["checkpoint"] == "/runs/uk_make_0614_b5/best.pt"
    assert p.phase == "done"


# ----------------------------------------------------------------------
# PullParser
# ----------------------------------------------------------------------


def test_pull_parser_size_patterns_and_done() -> None:
    p = _feed(
        PullParser(),
        "[pull] size:    1.5 GB across 12345 files (1200 main snaps, 400 HQ crops, 5 jsonl)",
        "[pull] *_main_*.jpg: 1200 remote / 1000 local / 200 new",
        "[pull] *.json: 5 remote / 5 local / 0 new",
        "[pull] skip-existing: 200 new image(s) fetched",
        "[pull] done -> /output/session_x",
    )
    assert p.metrics["remote_size_h"] == "1.5 GB"
    assert p.metrics["remote_files"] == 12345
    assert p.metrics["patterns"]["*_main_*.jpg"]["new"] == 200
    assert p.metrics["new_total"] == 200
    assert p.metrics["new_fetched"] == 200
    assert p.phase == "done"


def test_pull_parser_exposes_watch_directive() -> None:
    # session + target + size_bytes together let the runner watch the local dir
    # fill toward the byte total.
    p = _feed(
        PullParser(),
        "[pull] session: session_x",
        "[pull] target:  D:\\Projects\\StreetTracker\\output",
        "[pull] size_bytes: 1610612736",
    )
    assert p.metrics["watch"] == {
        "path": "D:\\Projects\\StreetTracker\\output/session_x",
        "total_bytes": 1610612736,
    }


def test_pull_parser_no_watch_until_complete() -> None:
    p = _feed(PullParser(), "[pull] session: session_x")  # target + bytes missing
    assert "watch" not in p.metrics


# ----------------------------------------------------------------------
# DvsaLabelParser + BuildParser
# ----------------------------------------------------------------------


def test_dvsa_label_parser_total_and_tally() -> None:
    p = _feed(
        DvsaLabelParser(),
        "[dvsa-label] 42 distinct high-conf plates (>= 0.9) across 150 tracks",
        "[dvsa-label] wrote /out/session_x_dvsa_labels.json: 35 labelled, 7 unknown, 0 skipped",
    )
    assert p.total == 42
    assert p.metrics["labelled"] == 35
    assert p.metrics["unknown"] == 7
    assert p.current == 42
    assert p.phase == "done"


def test_build_parser_sessions_and_totals() -> None:
    p = _feed(
        BuildParser(),
        "[makemodel-build-uk] 14 session(s) -> /runs/uk_crops_0614_512",
        "[makemodel-build-uk] 36 makes, 2736 cars, 32777 crops",
    )
    assert p.metrics["sessions"] == 14
    assert p.metrics["makes"] == 36
    assert p.metrics["cars"] == 2736
    assert p.metrics["crops"] == 32777
    assert p.phase == "done"


def test_train_parser_reads_crop_mode() -> None:
    line = (
        "[makemodel-uk] target=make device=cuda amp=True classes=58 "
        "train_crops=70000 val_crops=17000 crop=plate"
    )
    assert _feed(TrainParser(), line).metrics["crop_mode"] == "plate"
    legacy = line.rsplit(" crop=", 1)[0]  # pre-2026-09-24 trainer output
    assert _feed(TrainParser(), legacy).metrics["crop_mode"] == "hint"


def test_compare_uses_batch_progress() -> None:
    p = make_parser("makemodel-compare")
    p.feed("[batch] 400/5000 done (61s)")
    assert (p.progress.current, p.progress.total) == (400, 5000)


def test_rescore_uses_batch_progress() -> None:
    p = make_parser("alpr-rescore")
    p.feed("  [batch] 500/12000 done")
    assert (p.progress.current, p.progress.total) == (500, 12000)
    p.feed("[alpr-rescore] wrote session_x_alpr.json + session_x_alpr_by_track.json")
    assert "wrote" in (p.progress.summary or "")


def test_build_parser_cropping_progress() -> None:
    p = _feed(
        BuildParser(),
        "[makemodel-build-uk] 44 session(s) -> runs/uk_crops_0924_576 (crop mode plate)",
        "[makemodel-build-uk] 1500/52000 cropped",
    )
    assert p.metrics["sessions"] == 44
    assert (p.current, p.total) == (1500, 52000)
    assert p.phase == "cropping"


# ----------------------------------------------------------------------
# Generic fallback + finish()
# ----------------------------------------------------------------------


def test_generic_parser_captures_summary() -> None:
    p = _feed(
        GenericParser(),
        "[vehicles] reading data...",
        "[vehicles] wrote /out/session_x_vehicles.json  (2847 vehicles)",
    )
    assert "2847 vehicles" in p.summary


def test_finish_sets_done_and_ok() -> None:
    parser = GenericParser()
    parser.finish(0)
    assert parser.progress.done is True and parser.progress.ok is True

    failed = BatchParser()
    failed.finish(1)
    assert failed.progress.done is True
    assert failed.progress.ok is False
    assert failed.progress.phase == "failed"


def test_make_parser_dispatch() -> None:
    assert isinstance(make_parser("pull"), PullParser)
    assert isinstance(make_parser("alpr-run"), BatchParser)
    assert isinstance(make_parser("makemodel"), ClassifyParser)
    assert isinstance(make_parser("bodytype"), ClassifyParser)
    assert isinstance(make_parser("colour"), ClassifyParser)
    assert isinstance(make_parser("makemodel-train-uk"), TrainParser)
    assert isinstance(make_parser("dvsa-label"), DvsaLabelParser)
    assert isinstance(make_parser("makemodel-build-uk"), BuildParser)
    # Unknown kind -> generic, never an error.
    assert isinstance(make_parser("dvsa-apply"), GenericParser)
    assert isinstance(make_parser("vehicles"), GenericParser)
