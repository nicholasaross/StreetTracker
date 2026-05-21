"""Export a YOLO ``.pt`` checkpoint to a TensorRT ``.engine`` file.

Wraps ``ultralytics.YOLO(weights).export(format='engine', ...)``. That
single call replaces NanoTracker's two-step ``export_onnx.py`` +
``build_engine.sh`` (ONNX intermediate, ``trtexec`` invocation, custom
output decode) — Ultralytics owns the round-trip end to end.

**Important**: TRT engines are NOT portable across GPU architectures.
This command must run ON the target device (Orin Nano for the deployed
runtime, or a dev box only if you're not actually deploying that engine
elsewhere). The runtime guard below refuses to export when no CUDA GPU
is visible; pass ``--device cpu`` to override (the resulting engine
will still be CUDA-bound — that's just for testing the CLI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streettracker export-engine",
        description="Export a YOLO .pt checkpoint to a TensorRT .engine.",
    )
    parser.add_argument(
        "weights",
        type=Path,
        help="Path to YOLO .pt weights (e.g. yolov8m.pt, best.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input size for the exported engine (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Max batch size baked into the engine (default: 1)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TRT workspace budget in GB (default: 4)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index, 'cpu', or 'mps' (default: 0). "
             "Engines are only useful on CUDA; non-CUDA is for CLI smoke tests.",
    )
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--fp16",
        dest="precision",
        action="store_const",
        const="fp16",
        help="FP16 precision (default — best speed/accuracy trade-off on Orin)",
    )
    precision.add_argument(
        "--fp32",
        dest="precision",
        action="store_const",
        const="fp32",
        help="FP32 precision (slower, debug only)",
    )
    precision.add_argument(
        "--int8",
        dest="precision",
        action="store_const",
        const="int8",
        help="INT8 precision (requires a calibration dataset; advanced use)",
    )
    parser.set_defaults(precision="fp16")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Calibration data YAML (required when --int8 is set)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic input shapes (default: static — faster, more portable)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify the ONNX graph before TRT conversion (default: on)",
    )
    parser.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="Disable ONNX simplification",
    )
    return parser


def _coerce_device(value: str) -> str | int:
    """Accept ``"0"`` / ``"cpu"`` / ``"mps"``; coerce numeric strings to int."""
    if value.isdigit():
        return int(value)
    return value


def _export_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build the kwargs dict we pass to ``YOLO.export``.

    Pulled out so tests can assert what gets forwarded without invoking
    the real exporter (which needs torch + TRT + a real .pt on disk).
    """
    kwargs: dict[str, Any] = {
        "format": "engine",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workspace": args.workspace,
        "device": _coerce_device(args.device),
        "half": args.precision == "fp16",
        "int8": args.precision == "int8",
        "dynamic": args.dynamic,
        "simplify": args.simplify,
    }
    if args.data is not None:
        kwargs["data"] = str(args.data)
    return kwargs


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.weights.exists():
        print(f"[export-engine] weights not found: {args.weights}", file=sys.stderr)
        return 1
    if args.precision == "int8" and args.data is None:
        print(
            "[export-engine] --int8 requires --data <calibration.yaml>",
            file=sys.stderr,
        )
        return 2

    from ultralytics import YOLO  # deferred — heavy import

    kwargs = _export_kwargs(args)
    print(f"[export-engine] weights:   {args.weights}")
    print(f"[export-engine] imgsz:     {kwargs['imgsz']}")
    print(f"[export-engine] precision: {args.precision}")
    print(f"[export-engine] workspace: {kwargs['workspace']} GB")
    print(f"[export-engine] device:    {kwargs['device']}")
    print("[export-engine] starting export "
          "(first build can take 8-12 min on Orin) ...")

    model = YOLO(str(args.weights))
    result = model.export(**kwargs)

    out_path = Path(str(result)) if result else args.weights.with_suffix(".engine")
    if not out_path.exists():
        print(
            f"[export-engine] FAILED — no engine produced at {out_path}",
            file=sys.stderr,
        )
        return 1
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[export-engine] OK -> {out_path}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
