"""Phase 0 read-only checks from docs/data_integrity_review.md §4.

Reads ``output/`` (and, for E0.6, optionally the Orin over SSH); writes
nothing except the optional ``--json`` report. One check per function; run a
subset with ``--checks``.

    uv run python .claude/phase0_checks.py --checks e06 --orin streettracker@orin
    uv run python .claude/phase0_checks.py --checks e06 --json .claude/phase0_e06.json

Implemented so far:

- **E0.6 (R15) time base.** Orin clock sync + timezone (``timedatectl``), and
  per session: negative durations, ``time_end < time_start``, ISO-vs-unix
  disagreement, missing or unexpected UTC offsets (a fixed-offset or UTC
  zone would show as all-``+00:00`` or never-changing offsets across a clock
  change), session label vs ``session_start_unix``, first track before the
  session start, and backward steps in ``events.jsonl`` finalize order
  (tracks are appended as they finish, so ``time_end`` should only creep
  backwards by the tracker's lost-track buffer; a step of minutes means the
  wall clock jumped).

Still to add (revised plan step 3): E0.3, E0.4, E0.5, E0.7, E0.9.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Finalize order vs time_end: BotSORT keeps a lost track for a few seconds
# before it finalizes, and a long-lived track finishing late can land after
# shorter ones, so small backward steps are normal. Flag anything larger.
_BACKSTEP_FLAG_S = 120.0
# format_wall() writes whole seconds; time_*_unix is rounded to 0.01 s.
_ISO_UNIX_TOL_S = 1.5
# The session label is the start time to the second; the first track should
# start after it, within the time it takes the first car to come by.
_LABEL_TOL_S = 2.0


def _sessions(root: Path) -> list[Path]:
    return sorted(
        d for d in root.glob("session_*") if d.is_dir() and (d / f"{d.name}_data.json").exists()
    )


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _offset(iso: str) -> str | None:
    try:
        off = datetime.fromisoformat(iso).utcoffset()
    except ValueError:
        return "unparseable"
    if off is None:
        return None
    sign = "-" if off < timedelta(0) else "+"
    mins = abs(int(off.total_seconds())) // 60
    return f"{sign}{mins // 60:02d}:{mins % 60:02d}"


def orin_time(host: str) -> dict[str, Any]:
    """``timedatectl show`` on the Orin: timezone + NTP sync state."""
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "timedatectl show"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    if out.returncode != 0:
        return {"error": out.stderr.strip() or f"exit {out.returncode}"}
    kv = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    return {
        k: kv.get(k)
        for k in ("Timezone", "NTP", "NTPSynchronized", "LocalRTC", "TimeUSec", "RTCTimeUSec")
    }


def e06_session(d: Path) -> dict[str, Any]:
    name = d.name
    data = _load_json(d / f"{name}_data.json") or []
    meta = _load_json(d / f"{name}_meta.json") or {}
    res: dict[str, Any] = {"session": name, "tracks": len(data)}

    offsets: Counter[str] = Counter()
    neg_dur = end_before_start = iso_unix_mismatch = 0
    first_start: float | None = None
    first_offset: str | None = None
    for r in data:
        ts, us = r.get("time_start"), r.get("time_start_unix")
        if (r.get("duration_visible") or 0) < 0:
            neg_dur += 1
        if us is not None and r.get("time_end_unix") is not None and r["time_end_unix"] < us:
            end_before_start += 1
        if not ts:
            offsets["missing"] += 1
            continue
        off = _offset(ts)
        offsets[off or "naive"] += 1
        if off and off != "unparseable" and us is not None:
            if abs(datetime.fromisoformat(ts).timestamp() - float(us)) > _ISO_UNIX_TOL_S:
                iso_unix_mismatch += 1
            if first_start is None or float(us) < first_start:
                first_start, first_offset = float(us), off
    res.update(
        offsets=dict(offsets),
        negative_duration=neg_dur,
        end_before_start=end_before_start,
        iso_unix_mismatch=iso_unix_mismatch,
    )

    # Session label (local wall time at start) vs session_start_unix, read in
    # the zone the session's own timestamps carry.
    start = meta.get("session_start_unix")
    res["session_start_unix"] = start
    if start is not None and first_offset and first_offset not in ("unparseable",):
        try:
            label_dt = datetime.strptime(name.removeprefix("session_"), "%Y%m%d_%H%M%S")
            tz = datetime.fromisoformat(f"2000-01-01T00:00:00{first_offset}").tzinfo
            start_local = datetime.fromtimestamp(float(start), tz=tz).replace(tzinfo=None)
            res["label_minus_start_s"] = round((label_dt - start_local).total_seconds(), 1)
        except ValueError:
            res["label_minus_start_s"] = None
    if start is not None and first_start is not None:
        res["first_track_after_start_s"] = round(first_start - float(start), 1)

    # Finalize order: events.jsonl is appended as tracks finish.
    worst_back = 0.0
    n_back = 0
    prev_end: float | None = None
    ev = d / f"{name}_events.jsonl"
    if ev.exists():
        with ev.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    end = float(json.loads(line).get("time_end_unix"))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if prev_end is not None and end < prev_end:
                    back = prev_end - end
                    worst_back = max(worst_back, back)
                    n_back += back > _BACKSTEP_FLAG_S
                prev_end = end if prev_end is None else max(prev_end, end)
    res["events_worst_backstep_s"] = round(worst_back, 1)
    res["events_backsteps_flagged"] = n_back

    flags = []
    if neg_dur:
        flags.append(f"{neg_dur} negative durations")
    if end_before_start:
        flags.append(f"{end_before_start} end<start")
    if iso_unix_mismatch:
        flags.append(f"{iso_unix_mismatch} ISO/unix mismatches")
    if offsets.get("naive") or offsets.get("missing") or offsets.get("unparseable"):
        flags.append("naive/missing/unparseable time_start")
    lms = res.get("label_minus_start_s")
    if lms is not None and abs(lms) > _LABEL_TOL_S:
        flags.append(f"label off start by {lms} s")
    fts = res.get("first_track_after_start_s")
    if fts is not None and fts < 0:
        flags.append(f"first track {-fts} s before session start")
    if n_back:
        flags.append(f"{n_back} finalize backsteps > {_BACKSTEP_FLAG_S:.0f} s")
    res["flags"] = flags
    return res


def check_e06(root: Path, orin: str | None) -> dict[str, Any]:
    print("== E0.6 time base ==")
    report: dict[str, Any] = {}
    if orin:
        report["orin"] = orin_time(orin)
        print(f"Orin {orin}: {report['orin']}")
    rows = [e06_session(d) for d in _sessions(root)]
    report["sessions"] = rows
    all_offsets: Counter[str] = Counter()
    for r in rows:
        all_offsets.update(r["offsets"])
    print(f"{len(rows)} sessions, {sum(r['tracks'] for r in rows)} tracks")
    print(f"time_start offsets: {dict(all_offsets)}")
    worst = max(rows, key=lambda r: r["events_worst_backstep_s"], default=None)
    if worst:
        print(
            f"largest finalize backstep: {worst['events_worst_backstep_s']} s "
            f"({worst['session']}; flag threshold {_BACKSTEP_FLAG_S:.0f} s)"
        )
    lms = [r["label_minus_start_s"] for r in rows if r.get("label_minus_start_s") is not None]
    if lms:
        print(f"label - start: min {min(lms)} s, max {max(lms)} s")
    fts = [r["first_track_after_start_s"] for r in rows if "first_track_after_start_s" in r]
    if fts:
        print(f"first track after start: min {min(fts)} s, max {max(fts)} s")
    flagged = [r for r in rows if r["flags"]]
    print(f"sessions flagged: {len(flagged)}")
    for r in flagged:
        print(f"  {r['session']}: {'; '.join(r['flags'])}")
    report["n_flagged"] = len(flagged)
    return report


# Each check takes (output_root, orin_ssh_target_or_None) and returns its report.
CHECKS = {"e06": check_e06}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--checks", default=",".join(CHECKS), help="comma list: " + ",".join(CHECKS))
    ap.add_argument(
        "--orin", default=None, help="SSH target for E0.6's timedatectl, e.g. streettracker@orin"
    )
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = ap.parse_args()

    report: dict[str, Any] = {"as_of": datetime.now().isoformat(timespec="seconds")}
    for key in args.checks.split(","):
        key = key.strip().lower()
        if key not in CHECKS:
            raise SystemExit(f"unknown check {key!r}; have {', '.join(CHECKS)}")
        report[key] = CHECKS[key](args.output_root, args.orin)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
