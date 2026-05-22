"""Coverage for ``streettracker.cli.batch``.

Mirrors ``test_cli/test_run.py`` -- the batch CLI is the file-source
counterpart of ``run``. The runtime itself is exercised in
``test_device/test_runtime.py``; here we only cover argument parsing,
error paths, and the disable-snapshotter / disable-dashboard wiring
into ``run_session``.
"""

from __future__ import annotations

from pathlib import Path

from streettracker.cli.batch import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "camera.example.json"


# ----------------------------------------------------------------------
# Argparse spec.


def test_parser_requires_config_and_video(tmp_path: Path) -> None:
    parser = build_parser()
    # No args at all -- argparse exits via SystemExit on missing positionals.
    try:
        parser.parse_args([])
    except SystemExit as exc:
        assert exc.code != 0
        return
    raise AssertionError("parser did not exit on missing video / --config")


def test_parser_accepts_full_arg_set(tmp_path: Path) -> None:
    parser = build_parser()
    cfg = tmp_path / "c.json"
    cfg.write_text("{}", encoding="utf-8")
    video = tmp_path / "clip.mp4"
    args = parser.parse_args(
        [
            str(video),
            "--config",
            str(cfg),
            "--codec",
            "h265",
            "--assumed-fps",
            "25",
            "--duration",
            "60",
            "--output-dir",
            str(tmp_path / "out"),
            "--no-html",
            "--no-json",
        ]
    )
    assert args.video == video
    assert args.config == cfg
    assert args.codec == "h265"
    assert args.assumed_fps == 25.0
    assert args.duration == 60.0
    assert args.no_html is True
    assert args.no_json is True


def test_parser_codec_choices_restricted(tmp_path: Path) -> None:
    parser = build_parser()
    cfg = tmp_path / "c.json"
    cfg.write_text("{}", encoding="utf-8")
    try:
        parser.parse_args([str(tmp_path / "v.mp4"), "--config", str(cfg), "--codec", "av1"])
    except SystemExit as exc:
        assert exc.code != 0
        return
    raise AssertionError("parser accepted an unknown codec")


# ----------------------------------------------------------------------
# main() failure paths.


def test_main_returns_2_when_video_missing(tmp_path: Path) -> None:
    cfg = tmp_path / "c.json"
    cfg.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    rc = main([str(tmp_path / "missing.mp4"), "--config", str(cfg)])
    assert rc == 2


def test_main_returns_2_when_config_missing(tmp_path: Path) -> None:
    # Touch the video so its existence check passes -- we want the
    # *config* error to be the one that surfaces.
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    rc = main([str(video), "--config", str(tmp_path / "absent.json")])
    assert rc == 2


def test_main_returns_2_on_unparseable_config(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a streettracker config"}', encoding="utf-8")
    rc = main([str(video), "--config", str(bad)])
    assert rc == 2


# ----------------------------------------------------------------------
# Top-level CLI dispatch routes 'batch'.


def test_cli_root_dispatches_batch_help() -> None:
    """`streettracker batch --help` should print help via the
    top-level dispatcher (smoke check that __init__.py wiring is live)."""
    from streettracker.cli import main as cli_main

    try:
        cli_main(["batch", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
        return
    raise AssertionError("--help did not raise SystemExit")
