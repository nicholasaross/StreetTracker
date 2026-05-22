"""``streettracker batch`` -- process one local video file end-to-end.

A thin CLI wrapper around :func:`streettracker.device.runtime.run_session`
that swaps the live RTSP source for a :class:`FileSource` and disables
the two live-only subsystems (Reolink snapshotter, dashboard). Same
inference, tracking, IR detection, finalize, and session-output writers
as ``streettracker run`` -- so a batch session's
``data.json`` / ``meta.json`` / ``hourly.json`` / ``summary.html`` are
schema-identical to a live session's, and downstream analysers
(``recolor``, ``alpr-*``) treat them interchangeably.

Use cases:

* dev-box smoke runs against recorded clips before deploying engine
  changes;
* re-running inference over old footage when the tracker or weights
  improve;
* offline ALPR feeder -- batch produces ``vehicle_<id>.jpg`` +
  ``vehicle_<id>_hq.jpg`` thumbnails, which feed the ALPR pipelines.

Replaces VehicleTracker's ``main.py`` entry point. The fields written
to the session JSON match the live runtime exactly, by design.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from streettracker.common.config import ConfigError, StreetTrackerConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """``streettracker batch`` argparse spec.

    Pulled out so tests can drive ``parse_args`` directly without
    invoking the runtime."""
    p = argparse.ArgumentParser(
        prog="streettracker batch",
        description=(
            "Run the tracker over a local video file. Produces the same "
            "session output layout as `streettracker run` -- only the "
            "frame source changes."
        ),
    )
    p.add_argument(
        "video",
        type=Path,
        help="Path to an MP4 / MKV / MOV file to process.",
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "Path to a streettracker JSON config (see "
            "configs/camera.example.json). Snapshot + dashboard sections "
            "are accepted but ignored -- batch never fires Reolink snaps "
            "and never serves a live dashboard."
        ),
    )
    p.add_argument(
        "--codec",
        default="h264",
        choices=("h264", "h265"),
        help=(
            "Hint for the GStreamer NVDEC pipeline (Orin fast path). "
            "Ignored when OpenCV falls back to the FFmpeg software "
            "decoder (most dev boxes)."
        ),
    )
    p.add_argument(
        "--assumed-fps",
        type=float,
        default=30.0,
        help=(
            "Fallback FPS used to compute video time when the container "
            "doesn't advertise one. The probe overrides this when "
            "successful; this is just the safety net."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Hard cap on session length in seconds (default: run until "
            "EOF). Useful for quick smoke runs on long clips."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override OutputConfig.dir from the config.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip the summary HTML dashboard regen.",
    )
    p.add_argument(
        "--no-json",
        action="store_true",
        help="Skip the final data/meta/hourly JSON dumps.",
    )
    return p


async def _run(args: argparse.Namespace) -> int:
    """Async entry: load config + build file source + invoke run_session."""
    from streettracker.device.runtime import run_session
    from streettracker.sources.file import FileSource

    if not args.video.exists():
        print(f"[batch] video not found: {args.video}", file=sys.stderr)
        return 2

    try:
        config = StreetTrackerConfig.from_json_file(args.config)
        config.validate_paths()
    except ConfigError as exc:
        print(f"[batch] config error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, PermissionError) as exc:
        print(f"[batch] cannot read config: {exc}", file=sys.stderr)
        return 2

    source = FileSource(
        file_path=args.video,
        codec=args.codec,
        assumed_fps=args.assumed_fps,
    )

    output_root = args.output_dir or config.output.dir
    logger.info("[batch] processing %s -> %s", args.video, output_root)

    return await run_session(
        config,
        source,
        output_root=output_root,
        no_html=args.no_html,
        no_json=args.no_json,
        duration_s=args.duration,
        enable_snapshotter=False,
        enable_dashboard=False,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        # Mirror cli/run.py: the runtime's signal handler handles SIGINT
        # cleanly via shutdown_event, but the Ctrl-C path before the
        # handler installs still needs to exit non-zero.
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
