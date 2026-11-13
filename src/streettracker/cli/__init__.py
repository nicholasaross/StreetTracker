"""StreetTracker CLI entry point.

``streettracker <command> [<args>]``. The commands are listed in one place,
:data:`streettracker.cli.commands.COMMANDS`; the help text and the dispatch
below are both built from it, so a new subcommand is one line there.

Each subcommand owns its own ``argparse.ArgumentParser`` in its module's
``main()`` — we dispatch on the first positional rather than using
``add_subparsers`` so subcommand parsers receive ``--help`` cleanly
instead of having the top-level parser intercept it.
"""

from __future__ import annotations

import os
import sys

from streettracker.cli.commands import find_command, render_help

_HELP = render_help()


def _print_help() -> None:
    print(_HELP, end="")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_help()
        return 0

    head, rest = argv[0], argv[1:]

    # Stop Ultralytics installing packages into the venv mid-job (its
    # check_requirements "AutoUpdate"). Every dependency is pinned in
    # pyproject.toml, and an unplanned install can break the venv (a CPU
    # onnxruntime install wrecks onnxruntime-gpu). On 2026-10-23 a 0-byte
    # snap made it try `uv pip install pi-heif` during an alpr-run. Ultralytics
    # reads this when first imported, which every subcommand does lazily,
    # after this line. Set YOLO_AUTOINSTALL=true to allow it.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    if head in ("-h", "--help"):
        _print_help()
        return 0
    if head == "--version":
        from streettracker import __version__

        print(__version__)
        return 0

    command = find_command(head)
    if command is None:
        print(f"[streettracker] unknown subcommand: {head}", file=sys.stderr)
        _print_help()
        return 2
    return command.load()(rest)


if __name__ == "__main__":
    raise SystemExit(main())
