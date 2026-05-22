"""StreetTracker CLI entry point.

Subcommands:
- ``streettracker run`` — live RTSP (Orin only)
- ``streettracker batch <video>`` — file input
- ``streettracker pull`` — scp session from device
- ``streettracker recolor <session>`` — rerun color heuristic
- ``streettracker debug-color <crop.jpg>`` — inspect a single crop
- ``streettracker export-engine`` — ``.pt`` → ``.engine``
- ``streettracker alpr-run <session>`` — run ALPR pipelines on a session
- ``streettracker alpr-score <session>`` — score ALPR pipelines vs labels
- ``streettracker alpr-label <session>`` — interactive plate labeling
- ``streettracker alpr-report <session>`` — render comparison HTML

Each subcommand owns its own ``argparse.ArgumentParser`` in its module's
``main()`` — we dispatch on the first positional rather than using
``add_subparsers`` so subcommand parsers receive ``--help`` cleanly
instead of having the top-level parser intercept it.
"""

from __future__ import annotations

import sys

_HELP = """\
usage: streettracker [--version] <command> [<args>]

commands:
  run             live RTSP tracker (Orin only)
  batch           batch process a video file
  pull            scp a session from a remote device
  recolor         rerun color heuristic on a closed session
  debug-color     inspect HSV vote on one or more crop JPEGs
  export-engine   export .pt to .engine via Ultralytics
  alpr-run        run ALPR pipelines over a session's main snaps
  alpr-score      score ALPR pipelines against labels
  alpr-label      interactive plate labeling
  alpr-report     render the ALPR comparison HTML

Run ``streettracker <command> --help`` for per-command options.
"""


def _print_help() -> None:
    print(_HELP, end="")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_help()
        return 0

    head, rest = argv[0], argv[1:]

    if head in ("-h", "--help"):
        _print_help()
        return 0
    if head == "--version":
        from streettracker import __version__
        print(__version__)
        return 0

    if head == "recolor":
        from streettracker.analysis.recolor import main as recolor_main
        return recolor_main(rest)
    if head == "debug-color":
        from streettracker.analysis.debug_color import main as debug_color_main
        return debug_color_main(rest)
    if head == "pull":
        from streettracker.cli.pull import main as pull_main
        return pull_main(rest)
    if head == "export-engine":
        from streettracker.cli.export_engine import main as export_engine_main
        return export_engine_main(rest)
    if head == "alpr-run":
        from streettracker.cli.alpr_run import main as alpr_run_main
        return alpr_run_main(rest)
    if head == "alpr-score":
        from streettracker.cli.alpr_score import main as alpr_score_main
        return alpr_score_main(rest)
    if head == "alpr-label":
        from streettracker.cli.alpr_label import main as alpr_label_main
        return alpr_label_main(rest)
    if head == "alpr-report":
        from streettracker.cli.alpr_report import main as alpr_report_main
        return alpr_report_main(rest)

    if head == "run":
        from streettracker.cli.run import main as run_main
        return run_main(rest)
    if head == "batch":
        from streettracker.cli.batch import main as batch_main
        return batch_main(rest)

    print(f"[streettracker] unknown subcommand: {head}", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
