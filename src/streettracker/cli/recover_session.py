"""``streettracker recover-session <session>`` -- rebuild a crashed session's
``_data.json`` / ``_meta.json`` / ``_hourly.json`` from its crash-safe logs.

``pull`` runs this automatically when it fetches a session the device never
finalised; use the command directly for a session pulled before that, or
one whose recovery the pull could not confirm (device unreachable). Do not
run it on a partial copy of the device's LIVE session: the device writes the
real files when that session ends.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from streettracker.common.recover import needs_recovery, recover_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="streettracker recover-session",
        description="Rebuild a crashed session's data/meta/hourly JSON from its events log.",
    )
    parser.add_argument("session_dirs", nargs="+", type=Path, help="session_* directories")
    args = parser.parse_args(argv)

    rc = 0
    for session_dir in args.session_dirs:
        label = session_dir.name
        if not needs_recovery(session_dir):
            reason = (
                "already has _data.json"
                if (session_dir / f"{label}_data.json").exists()
                else "has no _events.jsonl"
            )
            print(f"[recover] {label}: nothing to do ({reason})")
            continue
        try:
            result = recover_session(session_dir)
        except OSError as exc:
            print(f"[recover] {label}: FAILED: {exc}")
            rc = 1
            continue
        print(f"[recover] {label}: {result.describe()}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
