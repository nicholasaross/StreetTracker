"""``streettracker run`` -- start the live RTSP runtime.

Thin CLI wrapper around :func:`streettracker.device.runtime.run_session`.
Loads a strict-JSON config (:class:`StreetTrackerConfig`), resolves
which RTSP stream to consume, constructs the matching
:class:`RtspSource`, and hands control off to the asyncio runtime.

Intended to be invoked by the systemd unit on the Orin -- arguments
are kept minimal so the unit's ``ExecStart`` stays readable. The
config file is the single source of truth for tunables.
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
    """``streettracker run`` argparse spec.

    Pulled out so tests can drive ``parse_args`` directly without
    spinning up the runtime."""
    p = argparse.ArgumentParser(
        prog="streettracker run",
        description="Run the live RTSP tracker (Orin only).",
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the JSON config (see configs/camera.example.json).",
    )
    p.add_argument(
        "--stream",
        default=None,
        help=(
            "Stream name to consume (must match a streams[].name in the "
            "config). Defaults to the legacy NanoCompat.preferred_stream."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Hard cap on session length in seconds (default: run forever).",
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
    p.add_argument(
        "--shutdown-grace-s",
        type=float,
        default=30.0,
        help=(
            "Seconds to allow for in-flight track drain after SIGTERM. "
            "Should be < systemd unit's TimeoutStopSec (default 30s)."
        ),
    )
    return p


def _build_rtsp_url(config: StreetTrackerConfig, stream_name: str) -> str:
    """Compose the full ``rtsp://user:pw@ip:port/path`` URL for a stream.

    Lives here rather than on :class:`StreetTrackerConfig` so the
    config object stays focused on schema + validation; URL composition
    is a CLI concern.

    Uses :meth:`StreetTrackerConfig.stream_by_name_or_quality` so a
    config that names streams descriptively (``"RTSP sub (H.264)"``)
    while ``preferred_stream`` is the legacy short token (``"sub"``)
    still resolves. The strict-name variant is preserved for callers
    that want explicit failure on a typo.
    """
    stream = config.stream_by_name_or_quality(stream_name)
    import urllib.parse

    user = urllib.parse.quote(config.camera.username, safe="")
    pw = urllib.parse.quote(config.camera.password, safe="")
    return f"rtsp://{user}:{pw}@{config.camera.ip}:{config.ports.rtsp}{stream.path}"


def _resolve_stream_name(config: StreetTrackerConfig, override: str | None) -> str:
    if override is not None:
        return override
    return config.nano.preferred_stream


def _redact_url(url: str) -> str:
    """Mask the password segment of an RTSP URL for safe logging."""
    if "@" not in url:
        return url
    creds, host = url.rsplit("@", 1)
    scheme_user = creds.rsplit(":", 1)[0]
    return f"{scheme_user}:***@{host}"


async def _run(args: argparse.Namespace) -> int:
    """Async entry: load config + build source + invoke run_session."""
    from streettracker.device.runtime import run_session
    from streettracker.sources.rtsp import RtspSource

    try:
        config = StreetTrackerConfig.from_json_file(args.config)
        config.validate_paths()
    except ConfigError as exc:
        print(f"[run] config error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, PermissionError) as exc:
        print(f"[run] cannot read config: {exc}", file=sys.stderr)
        return 2

    try:
        stream_name = _resolve_stream_name(config, args.stream)
        stream = config.stream_by_name_or_quality(stream_name)
    except ConfigError as exc:
        print(f"[run] {exc}", file=sys.stderr)
        return 2

    rtsp_url = _build_rtsp_url(config, stream_name)
    logger.info("[run] using stream %r -> %s", stream_name, _redact_url(rtsp_url))

    source = RtspSource(
        rtsp_url=rtsp_url,
        codec=stream.codec,
        transport=config.nano.rtsp_transport,
    )

    output_root = args.output_dir or config.output.dir

    return await run_session(
        config,
        source,
        output_root=output_root,
        no_html=args.no_html,
        no_json=args.no_json,
        duration_s=args.duration,
        shutdown_grace_s=args.shutdown_grace_s,
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
        # Already handled by the runtime's signal handler; this is the
        # last-resort guard for the case where the signal arrives before
        # the handler installs.
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
