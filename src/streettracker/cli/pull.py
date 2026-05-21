"""Pull a StreetTracker session directory from the device to the local box.

By default fetches the most recent ``session_*`` directory under
``/home/streettracker/streettracker/output/`` on the device into
``./output/`` on the local box, preserving the session-dir layout so
paths embedded in the session HTML / JSON keep resolving locally.

Ported from NanoTracker's ``scripts/pull_session.py``. Differences:

- Modernized for Python 3.12: ``pathlib``, PEP-604 unions, ``@dataclass``
  for the inventory result, ``shlex.join`` instead of manual quoting.
- Lives as a proper CLI subcommand (``streettracker pull``) instead of a
  standalone script — keeps the SSH/scp orchestration the same.
- Defaults bumped to a ``streettracker`` remote user/path; the original
  ``claude@nano`` defaults are no longer wired in.

Designed to run on the Windows dev box where the built-in OpenSSH client
provides ``ssh`` and ``scp``; works equally on Linux / macOS. Idempotent:
re-running over an existing local copy overwrites with the latest remote
state (scp merges into the existing session dir).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "orin"
DEFAULT_USER = "streettracker"
DEFAULT_KEY = "~/.ssh/streettracker"
DEFAULT_REMOTE_PARENT = "/home/streettracker/streettracker/output"
DEFAULT_LOCAL_PARENT = "./output"


@dataclass(slots=True)
class RemoteInventory:
    """Coarse stats about a remote session directory."""

    bytes: int = 0
    files: int = 0
    main_snaps: int = 0
    hq_crops: int = 0
    jsonl: int = 0


def _ssh_run(
    host: str, user: str, key: str, cmd: str, *, check: bool = True
) -> str:
    """Run ``cmd`` on the device via SSH and return stripped stdout.

    On failure with ``check=True``, prints the stderr to our stderr and
    exits with the SSH exit code (or 1).
    """
    args = [
        "ssh",
        "-i",
        key,
        "-o",
        "BatchMode=yes",
        f"{user}@{host}",
        cmd,
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    if check and proc.returncode != 0:
        sys.stderr.write(
            f"[pull] ssh failed (exit {proc.returncode}):\n{proc.stderr.strip()}\n"
        )
        sys.exit(proc.returncode or 1)
    return proc.stdout.strip()


def find_latest_session(host: str, user: str, key: str, remote_parent: str) -> str:
    """Return the lexicographically-last ``session_*`` directory name on
    the remote box. ``session_*`` directories are timestamped, so the
    lexical max is the newest.

    Exits non-zero if the parent has no such directory.
    """
    cmd = (
        f"ls -1 {shlex.quote(remote_parent)} 2>/dev/null "
        "| grep '^session_' | sort | tail -1"
    )
    name = _ssh_run(host, user, key, cmd)
    if not name:
        sys.exit(
            f"[pull] No session_* directories found in {host}:{remote_parent}"
        )
    return name


def remote_inventory(
    host: str, user: str, key: str, remote_path: str
) -> RemoteInventory:
    """Total size, file count, and counts split by filename suffix.

    A single SSH round-trip keeps this snappy. ``-maxdepth 1`` keeps us
    from descending into nested dirs that newer pipelines might write.
    """
    cmd = (
        f"cd {shlex.quote(remote_path)} 2>/dev/null && "
        "du -sb . 2>/dev/null | awk '{print \"BYTES \" $1}' ; "
        "find . -maxdepth 1 -type f | wc -l | awk '{print \"FILES \" $1}' ; "
        "find . -maxdepth 1 -name '*_main_*.jpg' | wc -l | awk '{print \"MAIN \" $1}' ; "
        "find . -maxdepth 1 -name '*_hq.jpg'     | wc -l | awk '{print \"HQ \" $1}' ; "
        "find . -maxdepth 1 -name '*.jsonl'      | wc -l | awk '{print \"JSONL \" $1}'"
    )
    out = _ssh_run(host, user, key, cmd, check=False)
    inv = RemoteInventory()
    fields = {"BYTES": "bytes", "FILES": "files", "MAIN": "main_snaps",
              "HQ": "hq_crops", "JSONL": "jsonl"}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in fields:
            with contextlib.suppress(ValueError):
                setattr(inv, fields[parts[0]], int(parts[1]))
    return inv


def human_bytes(n: int) -> str:
    """Format ``n`` bytes as a single-decimal human string (1.5 MB)."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"  # unreachable but mypy-friendly


def _scp_commands(
    host: str,
    user: str,
    key: str,
    remote_path: str,
    local_parent: Path,
    only_main: bool,
) -> list[list[str]]:
    """Build the list of scp invocations to execute.

    With ``only_main``, we issue per-pattern scps for just the main snaps
    + JSON metadata (skipping thumbs / HQ / HTML). Per-pattern scp keeps
    the include list simple without needing rsync (Windows OpenSSH does
    not bundle it).
    """
    if not only_main:
        remote_target = f"{user}@{host}:{remote_path}"
        return [["scp", "-i", key, "-r", "-p", remote_target, str(local_parent)]]

    session_name = remote_path.rstrip("/").rsplit("/", 1)[-1]
    local_session = local_parent / session_name
    local_session.mkdir(parents=True, exist_ok=True)
    patterns = (
        "*_main_*.jpg",
        "*.json",
        "*.jsonl",
        "*_summary.html",
        "index.html",
    )
    return [
        ["scp", "-i", key, "-p", "-q",
         f"{user}@{host}:{remote_path}/{pat}", str(local_session)]
        for pat in patterns
    ]


def scp_pull(
    host: str,
    user: str,
    key: str,
    remote_path: str,
    local_parent: Path,
    *,
    only_main: bool,
    dry_run: bool,
) -> None:
    """Fetch the remote session dir into ``local_parent``.

    Exits non-zero on scp failure unless ``only_main`` is set, in which
    case a missing-pattern failure is acceptable (a session with no main
    snaps is still valid output).
    """
    local_parent.mkdir(parents=True, exist_ok=True)
    for args in _scp_commands(host, user, key, remote_path, local_parent, only_main):
        if dry_run:
            print("[dry-run]", shlex.join(args))
            continue
        proc = subprocess.run(args)
        if proc.returncode != 0 and not only_main:
            sys.exit(f"[pull] scp failed (exit {proc.returncode})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streettracker pull",
        description="Pull a StreetTracker session from the device via ssh/scp.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"SSH host or alias (default: {DEFAULT_HOST})")
    parser.add_argument("--user", default=DEFAULT_USER,
                        help=f"SSH user (default: {DEFAULT_USER})")
    parser.add_argument("--key", default=DEFAULT_KEY,
                        help=f"SSH private key (default: {DEFAULT_KEY})")
    parser.add_argument("--remote-parent", default=DEFAULT_REMOTE_PARENT,
                        help="Parent output directory on the device")
    parser.add_argument("--target", default=DEFAULT_LOCAL_PARENT,
                        help="Local parent directory to receive the session "
                             "(session subdir is created inside)")
    parser.add_argument("--session", default=None,
                        help="Specific session label (default: latest)")
    parser.add_argument("--only-main", action="store_true",
                        help="Pull only main-stream snaps + JSON metadata "
                             "(skip thumbnails / HQ crops / summary HTML)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the scp invocation and remote inventory; "
                             "do not transfer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    key = os.path.expanduser(args.key)
    if not Path(key).exists():
        print(f"[pull] SSH key not found: {key}", file=sys.stderr)
        return 1

    session = args.session or find_latest_session(
        args.host, args.user, key, args.remote_parent
    )
    remote_path = f"{args.remote_parent.rstrip('/')}/{session}"
    local_parent = Path(args.target).expanduser().resolve()

    inv = remote_inventory(args.host, args.user, key, remote_path)
    print(f"[pull] session: {session}")
    print(f"[pull] remote:  {args.user}@{args.host}:{remote_path}")
    print(f"[pull] target:  {local_parent}")
    print(
        f"[pull] size:    {human_bytes(inv.bytes)} across {inv.files} files "
        f"({inv.main_snaps} main snaps, {inv.hq_crops} HQ crops, {inv.jsonl} jsonl)"
    )
    if args.only_main:
        print("[pull] mode:    --only-main (skipping thumbs + HQ + HTML)")

    scp_pull(
        args.host, args.user, key, remote_path, local_parent,
        only_main=args.only_main, dry_run=args.dry_run,
    )

    if not args.dry_run:
        landed = local_parent / session
        print(f"[pull] done -> {landed}")
        html = landed / f"{session}_summary.html"
        if html.exists():
            url = str(html).replace("\\", "/")
            print(f"[pull] open:   file:///{url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
