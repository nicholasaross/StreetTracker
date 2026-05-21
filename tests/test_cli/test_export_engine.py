"""Tests for ``streettracker export-engine``.

We don't actually run TRT here (no GPU on CI, and engines aren't portable
anyway). Instead we cover argument parsing, the kwargs dict we forward
to ``YOLO.export``, and the input-validation error paths. The real export
is exercised on the Orin in phase-3 device tests.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from streettracker.cli import export_engine


def test_coerce_device_numeric_becomes_int() -> None:
    assert export_engine._coerce_device("0") == 0
    assert export_engine._coerce_device("1") == 1


def test_coerce_device_keeps_strings_as_is() -> None:
    assert export_engine._coerce_device("cpu") == "cpu"
    assert export_engine._coerce_device("mps") == "mps"


def test_export_kwargs_defaults_to_fp16() -> None:
    args = export_engine._build_parser().parse_args(["yolov8m.pt"])
    kwargs = export_engine._export_kwargs(args)
    assert kwargs["format"] == "engine"
    assert kwargs["half"] is True
    assert kwargs["int8"] is False
    assert kwargs["imgsz"] == 640
    assert kwargs["device"] == 0
    assert kwargs["workspace"] == 4
    assert kwargs["dynamic"] is False
    assert kwargs["simplify"] is True


def test_export_kwargs_fp32_disables_half() -> None:
    args = export_engine._build_parser().parse_args(["yolov8m.pt", "--fp32"])
    kwargs = export_engine._export_kwargs(args)
    assert kwargs["half"] is False
    assert kwargs["int8"] is False


def test_export_kwargs_int8_sets_flag_and_includes_data() -> None:
    args = export_engine._build_parser().parse_args(
        ["yolov8m.pt", "--int8", "--data", "calib.yaml"]
    )
    kwargs = export_engine._export_kwargs(args)
    assert kwargs["int8"] is True
    assert kwargs["half"] is False
    assert kwargs["data"] == "calib.yaml"


def test_export_kwargs_custom_imgsz_and_device() -> None:
    args = export_engine._build_parser().parse_args(
        ["best.pt", "--imgsz", "1280", "--device", "cpu", "--workspace", "8"]
    )
    kwargs = export_engine._export_kwargs(args)
    assert kwargs["imgsz"] == 1280
    assert kwargs["device"] == "cpu"
    assert kwargs["workspace"] == 8


def test_export_kwargs_no_simplify_flag() -> None:
    args = export_engine._build_parser().parse_args(
        ["best.pt", "--no-simplify"]
    )
    kwargs = export_engine._export_kwargs(args)
    assert kwargs["simplify"] is False


def test_main_returns_1_when_weights_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    assert export_engine.main([str(missing)]) == 1


def test_main_returns_2_when_int8_missing_data(tmp_path: Path) -> None:
    fake_pt = tmp_path / "best.pt"
    fake_pt.write_bytes(b"\x00")  # presence is all the CLI checks for
    rc = export_engine.main([str(fake_pt), "--int8"])
    assert rc == 2


def test_main_invokes_yolo_export_with_built_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exporter is called with the kwargs dict our builder produced.

    We inject a stub ``ultralytics`` module so the deferred import inside
    ``main()`` resolves to our mock — no torch / no real export.
    """
    fake_pt = tmp_path / "best.pt"
    fake_pt.write_bytes(b"\x00")
    fake_engine = tmp_path / "best.engine"
    fake_engine.write_bytes(b"\x00" * 1024)

    mock_model = MagicMock()
    mock_model.export.return_value = str(fake_engine)
    mock_yolo = MagicMock(return_value=mock_model)

    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = mock_yolo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    rc = export_engine.main([str(fake_pt), "--imgsz", "1024", "--fp32"])
    assert rc == 0
    mock_yolo.assert_called_once_with(str(fake_pt))
    kwargs = mock_model.export.call_args.kwargs
    assert kwargs["format"] == "engine"
    assert kwargs["imgsz"] == 1024
    assert kwargs["half"] is False


def test_main_returns_1_when_export_produces_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pt = tmp_path / "best.pt"
    fake_pt.write_bytes(b"\x00")

    mock_model = MagicMock()
    mock_model.export.return_value = str(tmp_path / "did_not_get_written.engine")
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = MagicMock(return_value=mock_model)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    assert export_engine.main([str(fake_pt)]) == 1
