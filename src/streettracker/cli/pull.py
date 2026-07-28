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
provides ``ssh`` / ``scp`` / ``sftp``; works equally on Linux / macOS.
Idempotent: re-running over an existing local copy overwrites with the
latest remote state (scp merges into the existing session dir). With
``--skip-existing`` only image files missing locally are fetched (snaps
are write-once), so re-pulling a grown or still-live session is cheap.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "orin"
DEFAULT_USER = "streettracker"
DEFAULT_KEY = "~/.ssh/streettracker"
DEFAULT_REMOTE_PARENT = "/home/streettracker/streettracker/output"
DEFAULT_LOCAL_PARENT = "./output"
# Parallel sftp streams for --skip-existing. Measured against the live Orin
# (200-file arms, disjoint slices): 1 -> x1.00, 2 -> x1.41, 4 -> x1.61,
# 8 -> x1.70. Four is the knee -- doubling again buys ~5 % for twice the
# sshd sessions on a device that is also running the live tracker.
DEFAULT_JOBS = 4


@dataclass(slots=True)
class RemoteInventory:
    """Coarse stats about a remote session directory."""

    bytes: int = 0
    files: int = 0
    main_snaps: int = 0  # all *_main_*.jpg (vehicle + person)
    vehicle_main_snaps: int = 0  # vehicle_*_main_*.jpg only (person = main - vehicle)
    main_bytes: int = 0  # summed size of *_main_*.jpg (the --only-main payload)
    hq_crops: int = 0
    jsonl: int = 0


def _ssh_run(host: str, user: str, key: str, cmd: str, *, check: bool = True) -> str:
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
        sys.stderr.write(f"[pull] ssh failed (exit {proc.returncode}):\n{proc.stderr.strip()}\n")
        sys.exit(proc.returncode or 1)
    return proc.stdout.strip()


def find_latest_session(host: str, user: str, key: str, remote_parent: str) -> str:
    """Return the lexicographically-last ``session_*`` directory name on
    the remote box. ``session_*`` directories are timestamped, so the
    lexical max is the newest.

    Exits non-zero if the parent has no such directory.
    """
    cmd = f"ls -1 {shlex.quote(remote_parent)} 2>/dev/null | grep '^session_' | sort | tail -1"
    name = _ssh_run(host, user, key, cmd)
    if not name:
        sys.exit(f"[pull] No session_* directories found in {host}:{remote_parent}")
    return name


def remote_inventory(host: str, user: str, key: str, remote_path: str) -> RemoteInventory:
    """Total size, file count, and counts split by filename suffix.

    A single SSH round-trip keeps this snappy. ``-maxdepth 1`` keeps us
    from descending into nested dirs that newer pipelines might write.
    """
    cmd = (
        f"cd {shlex.quote(remote_path)} 2>/dev/null && "
        "du -sb . 2>/dev/null | awk '{print \"BYTES \" $1}' ; "
        "find . -maxdepth 1 -type f | wc -l | awk '{print \"FILES \" $1}' ; "
        "find . -maxdepth 1 -name '*_main_*.jpg' | wc -l | awk '{print \"MAIN \" $1}' ; "
        "find . -maxdepth 1 -name 'vehicle_*_main_*.jpg' | wc -l | awk '{print \"VEHMAIN \" $1}' ; "
        "find . -maxdepth 1 -name '*_main_*.jpg' -printf '%s\\n' 2>/dev/null"
        " | awk '{s+=$1} END{print \"MAINBYTES \" s+0}' ; "
        "find . -maxdepth 1 -name '*_hq.jpg'     | wc -l | awk '{print \"HQ \" $1}' ; "
        "find . -maxdepth 1 -name '*.jsonl'      | wc -l | awk '{print \"JSONL \" $1}'"
    )
    out = _ssh_run(host, user, key, cmd, check=False)
    inv = RemoteInventory()
    fields = {
        "BYTES": "bytes",
        "FILES": "files",
        "MAIN": "main_snaps",
        "VEHMAIN": "vehicle_main_snaps",
        "MAINBYTES": "main_bytes",
        "HQ": "hq_crops",
        "JSONL": "jsonl",
    }
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
        ["scp", "-i", key, "-p", "-q", f"{user}@{host}:{remote_path}/{pat}", str(local_session)]
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


def _immutable_image_patterns(only_main: bool) -> tuple[str, ...]:
    """Globs whose files are write-once -- 4K snaps are never rewritten
    once saved, so a remote-minus-local *name* diff is exact and needs no
    checksum. ``--only-main`` fetches just the main snaps; a full pull
    treats every ``*.jpg`` (tile / HQ / main) as immutable."""
    return ("*_main_*.jpg",) if only_main else ("*.jpg",)


def _remote_entries(
    host: str, user: str, key: str, remote_path: str, pattern: str
) -> dict[str, float]:
    """Map ``{basename: mtime}`` for remote files matching ``pattern`` in
    ``remote_path`` (one SSH round-trip; empty dict when nothing matches).

    Uses ``find ... -printf`` rather than ``ls -1 <glob>``: the shell
    expands ``ls -1 *.jpg`` client-of-the-remote-shell into one argv per
    match, which overflows ``ARG_MAX`` on a busy session (a 4.6-day soak
    holds 40k+ main snaps), so the ``ls`` errors, ``2>/dev/null`` eats it,
    and the empty result silently reports "0 remote" -- making
    ``--skip-existing`` fetch nothing for exactly the large sessions the
    mining workflow targets. ``find -maxdepth 1 -name`` matches one
    argument server-side with no expansion (the same approach
    :func:`remote_inventory` already relies on).

    ``-printf '%T@ %f\\n'`` emits ``<epoch-seconds> <basename>``. The mtime
    is what lets :func:`sftp_get_missing` fetch oldest-first -- the device
    prunes snaps by age, so age order is the only order that makes an
    interrupted pull lose the *newest* (still-safe) tail rather than files
    scattered through the session. Snap basenames never contain spaces, but
    we split on the first one only so an odd name can't corrupt the stamp."""
    cmd = (
        f"cd {shlex.quote(remote_path)} 2>/dev/null && "
        f"find . -maxdepth 1 -name {shlex.quote(pattern)} -printf '%T@ %f\\n' 2>/dev/null"
    )
    out = _ssh_run(host, user, key, cmd, check=False)
    entries: dict[str, float] = {}
    for raw in out.splitlines():
        stamp, sep, name = raw.strip().partition(" ")
        if not sep or not name:
            continue
        with contextlib.suppress(ValueError):
            entries[name] = float(stamp)
    return entries


def _is_intact_jpeg(path: Path) -> bool:
    """Cheap validity gate for a locally-present snap.

    A complete JPEG starts with SOI (``FF D8``) and ends with EOI
    (``FF D9``). An interrupted ``sftp get`` writes directly to the
    final name, so a killed transfer (the dev box idle-sleeps and the
    app helper can recycle mid-pull -- both reap the sftp process)
    leaves a full-size zero-filled or tail-truncated file at the right
    basename. Without this check the name-only diff below would treat
    that corpse as "already have it" and never re-fetch it. Surfaced
    2026-07-28: one session carried 634 zero-filled snaps (195 cars
    lost every snap) that had survived every subsequent
    ``--skip-existing`` pull.
    """
    try:
        with path.open("rb") as f:
            if f.read(2) != b"\xff\xd8":
                return False
            f.seek(-2, 2)
            return f.read(2) == b"\xff\xd9"
    except OSError:
        return False


def _stripe(names: list[str], n: int) -> list[list[str]]:
    """Deal ``names`` round-robin into ``n`` lists, preserving order within
    each.

    Striping rather than slicing into contiguous blocks is what keeps the
    parallel fetch age-ordered *globally*: every worker walks its own share
    from oldest to newest at roughly the same rate, so the combined
    "fetched so far" frontier advances by age. Contiguous blocks would have
    the last worker start at the newest files immediately, which is exactly
    the ordering the prune race needs to avoid."""
    return [names[i::n] for i in range(n)]


def _run_sftp_batch(
    host: str,
    user: str,
    key: str,
    remote_path: str,
    local_session: Path,
    names: list[str],
) -> int:
    """Fetch ``names`` over one ``sftp -b -`` connection; returns its exit code.

    Each ``get`` is prefixed with ``-`` so a per-file error does not abort
    the batch. That matters because this pull deliberately races the
    device's hourly prune: a file that vanishes between the ``find`` and
    its ``get`` would otherwise kill the whole stream and strand every
    later file in it. Connection-level failures still surface as a
    non-zero exit code.
    """
    # Forward slashes for the local lcd target: sftp treats backslash as an
    # escape, and it accepts forward-slash paths on Windows.
    local_lcd = str(local_session).replace("\\", "/")
    lines = [f'cd "{remote_path}"', f'lcd "{local_lcd}"']
    lines += [f'-get -p "{name}"' for name in names]
    args = ["sftp", "-i", key, "-o", "BatchMode=yes", "-q", "-b", "-", f"{user}@{host}"]
    proc = subprocess.run(args, input="\n".join(lines) + "\n", text=True)
    return proc.returncode


def sftp_get_missing(
    host: str,
    user: str,
    key: str,
    remote_path: str,
    local_session: Path,
    pattern: str,
    *,
    dry_run: bool,
    jobs: int = DEFAULT_JOBS,
) -> int:
    """Fetch the ``pattern`` files not already in ``local_session``, oldest
    first, across ``jobs`` parallel ``sftp`` batches. Returns the count that
    landed intact.

    Exact because snaps are immutable: a basename present locally is the
    same bytes as the remote one, so a name diff suffices. Uses ``sftp``
    rather than ``scp`` because Windows OpenSSH's SFTP-backed ``scp`` will
    not shell-expand an explicit ``{a,b}`` file list, whereas ``sftp -b -``
    runs one ``get`` per line over a single connection (no per-file
    handshake, no command-line-length limit).

    A single stream is latency- and cipher-bound well below the gigabit
    link, so the file list is split across ``jobs`` connections -- a
    measured x1.61 at the default 4 (see :data:`DEFAULT_JOBS`). Absolute
    rates swing with how cold the snaps are on the device: a week-old
    session read from disk at ~16 MB/s single-stream where a same-day one
    served from page cache hit ~57 MB/s. The multiplier is what holds.
    """
    remote = _remote_entries(host, user, key, remote_path, pattern)
    # Immutability lets us diff by basename -- but only for files that
    # actually arrived intact. A corrupt local snap (interrupted prior
    # transfer) whose name is still on the Orin is re-fetched, which
    # overwrites the corpse; one the Orin has since pruned can't be
    # recovered and is left as-is.
    local_intact: set[str] = set()
    n_corrupt = 0
    for p in local_session.glob(pattern):
        if _is_intact_jpeg(p):
            local_intact.add(p.name)
        else:
            n_corrupt += 1
    # Oldest-first (name as a stable tie-break) so an interrupted pull loses
    # the newest tail -- the files the device will prune last, and so the
    # ones a re-run can still recover.
    missing = sorted(remote.keys() - local_intact, key=lambda n: (remote[n], n))
    corrupt_note = f" ({n_corrupt} corrupt, re-fetching)" if n_corrupt else ""
    # Keep the "N remote / N local / N new" wording: the control panel's
    # PullParser regex keys off it for the per-pattern progress tally.
    print(
        f"[pull] {pattern}: {len(remote)} remote / {len(local_intact)} local / "
        f"{len(missing)} new{corrupt_note}"
    )
    if not missing:
        return 0
    if dry_run:
        print(f"[dry-run] sftp -b - would fetch {len(missing)} file(s) into {local_session}")
        return 0

    batches = [b for b in _stripe(missing, max(1, min(jobs, len(missing)))) if b]
    if len(batches) > 1:
        print(f"[pull] {len(batches)} parallel sftp streams, oldest snap first", flush=True)
    if len(batches) == 1:
        failed = (
            1 if _run_sftp_batch(host, user, key, remote_path, local_session, batches[0]) else 0
        )
    else:
        with ThreadPoolExecutor(max_workers=len(batches)) as pool:
            futures = [
                pool.submit(_run_sftp_batch, host, user, key, remote_path, local_session, b)
                for b in batches
            ]
            failed = sum(1 for f in futures if f.result() != 0)

    # Per-file errors are swallowed by the `-get` prefix, so count what
    # actually arrived rather than assuming the whole list landed.
    landed = sum(1 for name in missing if _is_intact_jpeg(local_session / name))
    if landed < len(missing):
        print(
            f"[pull] WARNING: {len(missing) - landed} of {len(missing)} did not land "
            "(pruned on the device mid-pull, or a transfer error) -- re-run to retry"
        )
    if failed:
        sys.exit(f"[pull] sftp failed ({failed} of {len(batches)} stream(s))")
    return landed


def skip_existing_pull(
    host: str,
    user: str,
    key: str,
    remote_path: str,
    local_parent: Path,
    *,
    only_main: bool,
    dry_run: bool,
    jobs: int = DEFAULT_JOBS,
) -> None:
    """Incremental pull: name-diff the immutable image globs (sftp only
    the new snaps) and re-fetch the mutable metadata in full via scp
    (``events.jsonl`` grows; ``*.json`` is rewritten at session end). Cheap
    re-pull of a grown or still-live session."""
    session_name = remote_path.rstrip("/").rsplit("/", 1)[-1]
    local_session = local_parent / session_name
    local_session.mkdir(parents=True, exist_ok=True)

    total_new = 0
    for pat in _immutable_image_patterns(only_main):
        total_new += sftp_get_missing(
            host, user, key, remote_path, local_session, pat, dry_run=dry_run, jobs=jobs
        )

    mutable = (
        ("*.json", "*.jsonl", "*_summary.html", "index.html")
        if only_main
        else ("*.json", "*.jsonl", "*.html")
    )
    for pat in mutable:
        args = [
            "scp",
            "-i",
            key,
            "-p",
            "-q",
            f"{user}@{host}:{remote_path}/{pat}",
            str(local_session),
        ]
        if dry_run:
            print("[dry-run]", shlex.join(args))
            continue
        subprocess.run(args)  # missing-pattern failures are acceptable
    if not dry_run:
        print(f"[pull] skip-existing: {total_new} new image(s) fetched")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streettracker pull",
        description="Pull a StreetTracker session from the device via ssh/scp.",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"SSH host or alias (default: {DEFAULT_HOST})"
    )
    parser.add_argument("--user", default=DEFAULT_USER, help=f"SSH user (default: {DEFAULT_USER})")
    parser.add_argument(
        "--key", default=DEFAULT_KEY, help=f"SSH private key (default: {DEFAULT_KEY})"
    )
    parser.add_argument(
        "--remote-parent",
        default=DEFAULT_REMOTE_PARENT,
        help="Parent output directory on the device",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_LOCAL_PARENT,
        help="Local parent directory to receive the session (session subdir is created inside)",
    )
    parser.add_argument("--session", default=None, help="Specific session label (default: latest)")
    parser.add_argument(
        "--only-main",
        action="store_true",
        help="Pull only main-stream snaps + JSON metadata "
        "(skip thumbnails / HQ crops / summary HTML)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Incremental: fetch only image files missing "
        "locally (snaps are write-once) via sftp, and "
        "re-fetch the mutable JSON/jsonl in full. Cheap "
        "re-pull of a grown or still-live session.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        metavar="N",
        help=f"Parallel sftp streams for --skip-existing (default: {DEFAULT_JOBS}; "
        "1 restores single-stream). Files are fetched oldest-first regardless, "
        "so an interrupted pull loses the newest tail rather than a random spread",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scp invocation and remote inventory; do not transfer",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    key = os.path.expanduser(args.key)
    if not Path(key).exists():
        print(f"[pull] SSH key not found: {key}", file=sys.stderr)
        return 1

    session = args.session or find_latest_session(args.host, args.user, key, args.remote_parent)
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
    # Machine-readable transfer total for the control panel's progress watcher:
    # the bytes that *this* pull will copy (just the main snaps under
    # --only-main, the whole dir otherwise), so the ETA tracks reality.
    pull_total = inv.main_bytes if (args.only_main and inv.main_bytes) else inv.bytes
    print(f"[pull] size_bytes: {pull_total}", flush=True)
    if args.only_main:
        print("[pull] mode:    --only-main (skipping thumbs + HQ + HTML)")

    if args.skip_existing:
        if args.jobs < 1:
            print("[pull] --jobs must be >= 1", file=sys.stderr)
            return 1
        skip_existing_pull(
            args.host,
            args.user,
            key,
            remote_path,
            local_parent,
            only_main=args.only_main,
            dry_run=args.dry_run,
            jobs=args.jobs,
        )
    else:
        scp_pull(
            args.host,
            args.user,
            key,
            remote_path,
            local_parent,
            only_main=args.only_main,
            dry_run=args.dry_run,
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
