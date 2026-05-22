"""Coverage for ``streettracker.cli.run``.

The CLI's job is narrow: load + validate the config, resolve the
chosen RTSP stream, compose the RTSP URL, and hand off to
``run_session``. The runtime itself is exercised in
``test_device/test_runtime.py``; here we only cover the wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streettracker.cli.run import (
    _build_rtsp_url,
    _redact_url,
    _resolve_stream_name,
    build_parser,
    main,
)
from streettracker.common.config import StreetTrackerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "camera.example.json"


def _example() -> StreetTrackerConfig:
    return StreetTrackerConfig.from_json_file(EXAMPLE_CONFIG)


# ----------------------------------------------------------------------
# Argparse spec.


def test_parser_requires_config() -> None:
    parser = build_parser()
    # argparse exits via SystemExit on missing required args.
    try:
        parser.parse_args([])
    except SystemExit as exc:
        assert exc.code != 0
        return
    raise AssertionError("parser did not exit on missing --config")


def test_parser_accepts_full_arg_set(tmp_path: Path) -> None:
    parser = build_parser()
    cfg = tmp_path / "c.json"
    cfg.write_text("{}", encoding="utf-8")
    args = parser.parse_args(
        [
            "--config",
            str(cfg),
            "--stream",
            "sub",
            "--duration",
            "60",
            "--output-dir",
            str(tmp_path / "out"),
            "--no-html",
            "--no-json",
            "--shutdown-grace-s",
            "5",
        ]
    )
    assert args.config == cfg
    assert args.stream == "sub"
    assert args.duration == 60.0
    assert args.no_html is True
    assert args.no_json is True
    assert args.shutdown_grace_s == 5.0


# ----------------------------------------------------------------------
# Stream resolution + URL composition.


def test_resolve_stream_uses_override_when_given() -> None:
    cfg = _example()
    assert _resolve_stream_name(cfg, "main") == "main"


def test_resolve_stream_falls_back_to_nano_preferred() -> None:
    cfg = _example()
    # The example has nano.preferred_stream == "sub".
    assert _resolve_stream_name(cfg, None) == "sub"


def test_build_rtsp_url_includes_credentials_and_path() -> None:
    cfg = _example()
    url = _build_rtsp_url(cfg, "sub")
    assert url.startswith("rtsp://admin:")
    assert "@192.168.1.XXX:554" in url
    assert "/h264Preview_01_sub" in url


def test_build_rtsp_url_url_encodes_password(tmp_path: Path) -> None:
    """Passwords with special characters must be URL-encoded so the
    full RTSP URL parses correctly."""
    cfg_dict = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    cfg_dict["camera"]["password"] = "p@ss/word#1"
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps(cfg_dict), encoding="utf-8")
    cfg = StreetTrackerConfig.from_json_file(cfg_path)
    url = _build_rtsp_url(cfg, "sub")
    assert "p%40ss%2Fword%231" in url
    assert "@" in url  # the credential separator (just one)


def test_redact_url_masks_password() -> None:
    masked = _redact_url("rtsp://admin:secret@192.168.1.1:554/path")
    assert "secret" not in masked
    assert masked == "rtsp://admin:***@192.168.1.1:554/path"


def test_redact_url_handles_passwordless_input() -> None:
    masked = _redact_url("rtsp://192.168.1.1:554/path")
    assert masked == "rtsp://192.168.1.1:554/path"


# ----------------------------------------------------------------------
# main() failure paths -- config errors should exit 2 (not crash).


def test_main_returns_nonzero_on_missing_config(tmp_path: Path) -> None:
    rc = main(["--config", str(tmp_path / "does-not-exist.json")])
    assert rc == 2


def test_main_returns_nonzero_on_unparseable_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a streettracker config"}', encoding="utf-8")
    rc = main(["--config", str(bad)])
    assert rc == 2


def _prepare_cwd_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Drop a copy of the example config under ``tmp_path`` + touch a
    fake engine file there so validate_paths passes. Returns the
    config path.

    Used by tests that need to reach past the engine-existence check
    in ``main()`` -- e.g. stream-resolution failure tests.
    """
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "yolov8m.engine").write_bytes(b"FAKE")
    monkeypatch.chdir(tmp_path)
    return cfg_path


def test_main_returns_nonzero_on_unknown_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _prepare_cwd_config(tmp_path, monkeypatch)
    rc = main(["--config", str(cfg_path), "--stream", "nope"])
    assert rc == 2


def test_main_returns_2_when_engine_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Engine validation must surface as a clear ConfigError before
    the runtime tries to load anything. Reproduces the second cutover
    failure (yolov8n_fp16.engine not on disk)."""
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    # Don't create yolov8m.engine -- validation should fail.
    rc = main(["--config", str(cfg_path)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "engine_path" in captured.err
    assert "yolov8m.engine" in captured.err


# ----------------------------------------------------------------------
# Top-level CLI dispatch still routes "run".


def test_cli_root_dispatches_run_help() -> None:
    """`streettracker run --help` should print help and exit 0 via the
    main entry point, confirming the dispatcher wires up the subcommand."""
    from streettracker.cli import main as cli_main

    try:
        cli_main(["run", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
        return
    raise AssertionError("--help did not raise SystemExit")
