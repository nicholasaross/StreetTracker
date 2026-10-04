"""Phase 0 read-only checks from docs/data_integrity_review.md §4.

Reads ``output/`` and ``runs/`` (and, for E0.6, optionally the Orin over
SSH); writes nothing except the optional ``--json`` report. One check per
function; run a subset with ``--checks``.

    uv run python .claude/phase0_checks.py                       # all checks
    uv run python .claude/phase0_checks.py --checks e06 --orin streettracker@orin
    uv run python .claude/phase0_checks.py --checks e03,e05 --json .claude/phase0.json

Local times use the dev box's zone (the Orin and the dev box are both
Europe/London; this venv has no tzdata for ``zoneinfo``).

- **E0.3 (R8) observed time.** Session spans minus IR periods, minus daytime
  (07-19 h) gaps of > 120 s between track starts (an outage proxy; reported
  separately because a quiet minute can be real). Coverage per date and per
  weekday x hour cell, over every minute from the first session's date to the
  last's. (The review's ``frames_processed / pipe_fps`` comparison is
  circular: ``pipe_fps`` is frames over elapsed time.)
- **E0.4 (R3, R7) corpus identity.** In the production corpus, plates that
  fuzzy-match (``vehicles`` rule: ratio >= 85, same length) across the
  train/val split, per-plate read support from every ``_alpr.json``, and the
  "fresh" cars of the post-corpus sessions that fuzzy-match a corpus plate.
- **E0.5 (R2, R9) plate collisions.** The same plate (fuzzy) as best read on
  two tracks whose time windows are within 10 s: opposite direction (at least
  one misattributed) vs same direction (a BotSORT split).
- **E0.6 (R15) time base.** Orin clock sync + timezone (``timedatectl``), and
  per session: negative durations, ``time_end < time_start``, ISO-vs-unix
  disagreement, missing or unexpected UTC offsets (a fixed-offset or UTC
  zone would show as all-``+00:00`` or never-changing offsets across a clock
  change), session label vs ``session_start_unix``, first track before the
  session start, and backward steps in ``events.jsonl`` finalize order
  (tracks are appended as they finish, so ``time_end`` should only creep
  backwards by the tracker's lost-track buffer; a step of minutes means the
  wall clock jumped).
- **E0.7 (R10) jogger vs pavement.** Person speeds split by pavement (median
  y of entry/exit points), with each pavement's speed histogram and the
  perspective ratio from person bbox heights.
- **E0.9 (R3) labels describing another car.** On sessions no head trained
  on, DVSA colour group vs the colour CNN's per-track read, split by read
  support, snap agreement and confidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

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


def check_e06(args: argparse.Namespace) -> dict[str, Any]:
    root, orin = args.output_root, args.orin
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


# ----------------------------------------------------------------------
# Shared helpers for E0.3-E0.9

_SESSION_RE = re.compile(r"session_\d{8}_\d{6}")
_SAMPLE_TRACK_RE = re.compile(r"_(session_\d{8}_\d{6})_(\d+)_\d+\.[a-z]+$")
_FUZZY_CUTOFF = 85  # analysis.vehicles' default merge ratio (same length only)


def _local(ts: float) -> datetime:
    return datetime.fromtimestamp(ts).astimezone()


def _iso_unix(iso: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(iso)).timestamp()
    except (TypeError, ValueError):
        return None


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = _wilson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def _fmt(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "n=0"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:5.1f}% ({r['k']}/{r['n']}, CI {100 * lo:.1f}-{100 * hi:.1f})"


def _plate(text: Any) -> str:
    return str(text or "").replace(" ", "").upper()


def _fuzzy_pairs(plates: list[str]) -> list[tuple[str, str]]:
    """Pairs of distinct plates the ``vehicles`` fuzzy rule would merge."""
    import numpy as np
    from rapidfuzz import fuzz, process

    by_len: dict[int, list[str]] = defaultdict(list)
    for p in sorted(set(plates)):
        by_len[len(p)].append(p)
    pairs: list[tuple[str, str]] = []
    for group in by_len.values():
        if len(group) < 2:
            continue
        m = process.cdist(
            group, group, scorer=fuzz.ratio, score_cutoff=_FUZZY_CUTOFF, dtype=np.uint8, workers=-1
        )
        ii, jj = np.nonzero(np.triu(m, 1))
        pairs.extend((group[i], group[j]) for i, j in zip(ii.tolist(), jj.tolist(), strict=True))
    return pairs


def _fuzzy_neighbour(plate: str, pool_by_len: dict[int, list[str]]) -> str | None:
    """The closest *different* plate in the pool the fuzzy rule would merge."""
    from rapidfuzz import fuzz, process

    cands = [p for p in pool_by_len.get(len(plate), []) if p != plate]
    hit = process.extractOne(plate, cands, scorer=fuzz.ratio, score_cutoff=_FUZZY_CUTOFF)
    return hit[0] if hit else None


def _clusters(plates: list[str], pairs: list[tuple[str, str]]) -> list[set[str]]:
    parent = {p: p for p in plates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, set[str]] = defaultdict(set)
    for p in plates:
        groups[find(p)].add(p)
    return list(groups.values())


def _production_corpus(args: argparse.Namespace) -> Path:
    """``--corpus``, else the corpus the production make model trained on."""
    if args.corpus:
        return Path(args.corpus)
    meta = _load_json(
        _HERE.parent / "src/streettracker/analysis/makemodel/models/makemodel_b0.meta.json"
    )
    tc = (meta or {}).get("trained_corpus")
    name = tc.get("name") if isinstance(tc, dict) else tc
    if not name:
        raise SystemExit("no --corpus and no trained_corpus in the make model's sidecar")
    return args.runs / str(name)


def _manifest_samples(corpus: Path) -> list[dict[str, Any]]:
    man = _load_json(corpus / "manifest.json")
    if not isinstance(man, dict):
        raise SystemExit(f"{corpus}: no readable manifest.json")
    return list(man.get("samples") or [])


def _corpus_sessions(samples: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for s in samples:
        m = _SESSION_RE.search(str(s.get("path", "")))
        if m:
            out.add(m.group(0))
    return out


def _all_corpus_plates(runs: Path) -> set[str]:
    """Every car in any training corpus under ``runs/`` (uk_crops_*)."""
    plates: set[str] = set()
    for man in sorted(runs.glob("uk_crops_*/manifest.json")):
        doc = _load_json(man)
        if isinstance(doc, dict):
            plates |= {s["car"] for s in doc.get("samples", []) if s.get("car")}
    return plates


def _post_corpus_sessions(root: Path, corpus_sessions: set[str]) -> list[Path]:
    """Sessions recorded after the corpus's newest session (no head trained on them)."""
    newest = max(corpus_sessions, default="")
    return [d for d in _sessions(root) if d.name not in corpus_sessions and d.name > newest]


# ----------------------------------------------------------------------
# E0.3 observed time

_GAP_PROXY_S = 120.0
_DAY_HOURS = (7, 19)
_COVERED = 0.95


def check_e03(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root
    print("== E0.3 observed time ==")
    spans: list[tuple[float, float, str]] = []
    irs: list[tuple[float, float]] = []
    gaps: list[tuple[float, float]] = []
    for d in _sessions(root):
        n = d.name
        meta = _load_json(d / f"{n}_meta.json") or {}
        data = _load_json(d / f"{n}_data.json") or []
        start = meta.get("session_start_unix")
        if start is None:
            continue
        ir: list[tuple[float, float]] = []
        for p in meta.get("ir_periods") or []:
            a, b = _iso_unix(p.get("start")), _iso_unix(p.get("end"))
            if a is not None and b is not None and b > a:
                ir.append((a, b))
        ends = [float(r["time_end_unix"]) for r in data if r.get("time_end_unix")]
        end = max(ends + [b for _, b in ir], default=None)
        if end is None or end <= float(start):
            continue
        spans.append((float(start), end, n))
        irs.extend(ir)
        # Outage proxy: a daytime stretch with no track starting at all.
        prev = float(start)
        starts = sorted(float(r["time_start_unix"]) for r in data if r.get("time_start_unix"))
        for s in [*starts, end]:
            mid = (s + prev) / 2
            daytime = _DAY_HOURS[0] <= _local(mid).hour < _DAY_HOURS[1]
            in_ir = any(a <= mid < b for a, b in ir)  # IR is already unobserved
            if s - prev > _GAP_PROXY_S and daytime and not in_ir:
                gaps.append((prev, s))
            prev = max(prev, s)
    if not spans:
        print("no sessions")
        return {}

    first = _local(min(s for s, _, _ in spans)).replace(hour=0, minute=0, second=0, microsecond=0)
    last = _local(max(e for _, e, _ in spans)).replace(hour=0, minute=0, second=0, microsecond=0)
    m0 = int(first.timestamp() // 60)
    m1 = int((last + timedelta(days=1)).timestamp() // 60)
    size = m1 - m0

    def mark(arr: bytearray, a: float, b: float, v: int) -> None:
        # Minute k is covered when its midpoint lies in [a, b).
        i = max(0, math.ceil((a - 30) / 60) - m0)
        j = min(size, math.ceil((b - 30) / 60) - m0)
        if j > i:
            arr[i:j] = bytes([v]) * (j - i)

    sess = bytearray(size)
    for a, b, _ in spans:
        mark(sess, a, b, 1)
    no_ir = bytearray(sess)
    for a, b in irs:
        mark(no_ir, a, b, 0)
    no_gap = bytearray(no_ir)
    for a, b in gaps:
        mark(no_gap, a, b, 0)

    # Per local (date, hour): [minutes, in a session, minus IR, minus gap proxy, in IR].
    cells: dict[tuple[date, int], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for k in range(size):
        dt = _local((m0 + k) * 60)
        c = cells[(dt.date(), dt.hour)]
        c[0] += 1
        c[1] += sess[k]
        c[2] += no_ir[k]
        c[3] += no_gap[k]
        c[4] += sess[k] and not no_ir[k]

    tot = [sum(c[i] for c in cells.values()) for i in range(4)]
    h = [t / 60 for t in tot]
    print(
        f"{first.date()} .. {last.date()}: {h[0]:.0f} h; in a session {h[1]:.0f} h "
        f"({100 * tot[1] / tot[0]:.1f} %), minus IR {h[2]:.0f} h ({100 * tot[2] / tot[0]:.1f} %), "
        f"minus daytime gaps > {_GAP_PROXY_S:.0f} s {h[3]:.0f} h ({100 * tot[3] / tot[0]:.1f} %)"
    )
    print(
        f"IR periods: {len(irs)} ({sum(b - a for a, b in irs) / 3600:.1f} h); daytime gap-proxy "
        f"stretches: {len(gaps)} ({sum(b - a for a, b in gaps) / 3600:.1f} h)"
    )

    # IR share of recorded time, by month: the camera's night mode skips inference.
    by_month: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (dd, _hr), c in cells.items():
        by_month[dd.strftime("%Y-%m")][0] += c[1]
        by_month[dd.strftime("%Y-%m")][1] += c[4]
    ir_by_month = {m: round(v[1] / v[0], 4) if v[0] else None for m, v in sorted(by_month.items())}
    print(
        "IR (no inference) share of session time by month: "
        + ", ".join(f"{m} {100 * (v or 0):.1f} %" for m, v in ir_by_month.items())
    )

    by_date: dict[date, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for (dd, _hr), c in cells.items():
        for i in range(5):
            by_date[dd][i] += c[i]
    date_cov = {dd.isoformat(): round(c[2] / c[0], 4) for dd, c in sorted(by_date.items())}
    low_dates = [(dd, v) for dd, v in date_cov.items() if v < _COVERED]
    print(
        f"dates: {len(date_cov)}, under {100 * _COVERED:.0f} % observed (minus IR): "
        f"{len(low_dates)}"
    )
    for dd, v in low_dates:
        print(f"  {dd} {datetime.fromisoformat(dd).strftime('%a')} {100 * v:5.1f} %")

    grid = [[[0, 0, 0, 0] for _ in range(24)] for _ in range(7)]
    for (dd, hr), c in cells.items():
        g = grid[dd.weekday()][hr]
        g[0] += c[0]
        g[1] += c[2]
        g[2] += c[1]
        g[3] += c[4]
    grid_cov = [[round(g[1] / g[0], 4) if g[0] else None for g in row] for row in grid]
    grid_sess = [[round(g[2] / g[0], 4) if g[0] else None for g in row] for row in grid]
    grid_ir = [[round(g[3] / g[0], 4) if g[0] else None for g in row] for row in grid]
    n_low = sum(1 for row in grid_cov for v in row if v is not None and v < _COVERED)
    print(f"weekday x hour cells under {100 * _COVERED:.0f} % (minus IR): {n_low} of 168")
    print("      " + " ".join(f"{hr:02d}" for hr in range(24)))
    for wd, row in enumerate(grid_cov):
        cells_txt = " ".join(
            "--" if v is None else "##" if v >= 0.995 else f"{min(99, int(100 * v)):2d}"
            for v in row
        )
        print(f"  {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][wd]} {cells_txt}")
    print("  (## = 100 %, numbers = % of that weekday-hour's minutes observed)")
    for title, g in (("in a session", grid_sess), ("in IR (no inference)", grid_ir)):
        print(f"weekday x hour, % of minutes {title}:")
        for wd, row in enumerate(g):
            cells_txt = " ".join(
                "--" if v is None else "##" if v >= 0.995 else f"{min(99, int(100 * v)):2d}"
                for v in row
            )
            print(f"  {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][wd]} {cells_txt}")

    # Longest unobserved stretches between sessions (unpulled sessions or downtime).
    holes: list[tuple[int, int]] = []
    k = 0
    while k < size:
        if sess[k]:
            k += 1
            continue
        j = k
        while j < size and not sess[j]:
            j += 1
        holes.append((k, j))
        k = j
    holes.sort(key=lambda x: x[1] - x[0], reverse=True)
    print("longest stretches with no session:")
    hole_rows = []
    for a, b in holes[:8]:
        t = _local((m0 + a) * 60)
        hole_rows.append(
            {"start": t.isoformat(timespec="minutes"), "hours": round((b - a) / 60, 1)}
        )
        print(f"  {t:%Y-%m-%d %a %H:%M}  {(b - a) / 60:6.1f} h")

    worst = min(date_cov.values())
    print(
        "decision: "
        + (
            "coverage gaps exist -> build E3.4 (observed-hours denominators) before quoting "
            "daily means or heatmaps"
            if low_dates or n_low
            else "every date and weekday-hour cell >= 95 % observed"
        )
    )
    return {
        "hours": {"range": h[0], "session": h[1], "minus_ir": h[2], "minus_gap_proxy": h[3]},
        "ir_periods": len(irs),
        "gap_proxy_stretches": len(gaps),
        "date_coverage_minus_ir": date_cov,
        "n_dates_under": len(low_dates),
        "weekday_hour_coverage_minus_ir": grid_cov,
        "weekday_hour_in_session": grid_sess,
        "weekday_hour_in_ir": grid_ir,
        "ir_share_by_month": ir_by_month,
        "n_cells_under": n_low,
        "worst_date_coverage": worst,
        "longest_holes": hole_rows,
    }


# ----------------------------------------------------------------------
# E0.4 corpus identity


def check_e04(args: argparse.Namespace) -> dict[str, Any]:
    from streettracker.analysis.dvsa import is_canonical_uk_plate
    from streettracker.analysis.makemodel.colour import colour_class_for
    from streettracker.analysis.makemodel.uk_dataset import split_val_cars
    from streettracker.analysis.vehicles import _colour_group

    root = args.output_root
    corpus = _production_corpus(args)
    print(f"== E0.4 corpus identity ({corpus.name}) ==")
    samples = _manifest_samples(corpus)
    cars = sorted({s["car"] for s in samples if s.get("car")})
    make_of: dict[str, str | None] = {}
    for s in samples:
        make_of.setdefault(s["car"], s.get("make"))
    val = split_val_cars(samples)  # the make head's held-out cars

    def track_key(s: dict[str, Any]) -> tuple[str, int] | None:
        m = _SAMPLE_TRACK_RE.search(str(s.get("path", "")))
        return (m.group(1), int(m.group(2))) if m else None

    pairs = _fuzzy_pairs(cars)
    cross = [(a, b) for a, b in pairs if (a in val) != (b in val)]
    clusters = _clusters(cars, pairs)
    spanning = [c for c in clusters if (c & val) and (c - val)]
    span_cars = set().union(*spanning) if spanning else set()
    val_samples = [s for s in samples if s["car"] in val]
    val_tracks = {track_key(s) for s in val_samples}
    span_val_tracks = {track_key(s) for s in val_samples if s["car"] in span_cars}
    leak = len(span_val_tracks) / len(val_tracks) if val_tracks else 0.0
    same_pairs = sum(make_of[a] == make_of[b] for a, b in pairs)
    same_cross = sum(make_of[a] == make_of[b] for a, b in cross)
    print(
        f"{len(cars)} cars, {len(val)} val; fuzzy pairs {len(pairs)} ({same_pairs} same make), "
        f"across train/val {len(cross)} ({same_cross} same make)"
    )
    print(
        f"clusters spanning train and val: {len(spanning)}; their val tracks "
        f"{len(span_val_tracks)} of {len(val_tracks)} ({100 * leak:.1f} %)"
    )

    # Read support: how many snaps / tracks read each corpus plate exactly.
    snaps: Counter[str] = Counter()
    tracks: dict[str, set[tuple[str, int]]] = defaultdict(set)
    car_set = set(cars)
    for d in _sessions(root):
        entries = _load_json(d / f"{d.name}_alpr.json")
        if not isinstance(entries, list):
            continue
        for r in entries:
            if r.get("pipeline") != "preferred" or r.get("static_suspect"):
                continue
            p = _plate(r.get("ocr_text"))
            if p in car_set:
                snaps[p] += 1
                tracks[p].add((d.name, int(r.get("track_id", -1))))
    one_snap = sum(1 for c in cars if snaps.get(c, 0) <= 1)
    one_track = sum(1 for c in cars if len(tracks.get(c, ())) <= 1)
    print(
        f"read support: {one_snap} cars ({100 * one_snap / len(cars):.1f} %) read on <= 1 snap, "
        f"{one_track} ({100 * one_track / len(cars):.1f} %) on <= 1 track"
    )

    # Fresh cars of the post-corpus sessions that fuzzy-match a corpus car.
    all_corpus = _all_corpus_plates(args.runs)
    pool: dict[int, list[str]] = defaultdict(list)
    for p in all_corpus:
        pool[len(p)].append(p)
    post = _post_corpus_sessions(root, _corpus_sessions(samples))
    fresh: dict[str, dict[str, Any]] = {}
    mism: dict[bool, list[int]] = {True: [0, 0], False: [0, 0]}
    for d in post:
        doc = _load_json(d / f"{d.name}_dvsa_labels.json") or {}
        col = {
            int(t["track_id"]): t.get("colour")
            for t in (_load_json(d / f"{d.name}_colour_by_track.json") or {}).get("tracks", [])
            if t.get("track_id") is not None
        }
        for plate, row in (doc.get("labels") or {}).items():
            if not row.get("track_ids") or plate in all_corpus or not is_canonical_uk_plate(plate):
                continue
            f = fresh.setdefault(plate, {"neighbour": _fuzzy_neighbour(plate, pool)})
            g_dvsa = _colour_group(colour_class_for(row.get("primary_colour")))
            for tid in row["track_ids"]:
                g_cnn = _colour_group(col.get(int(tid)))
                if g_dvsa and g_cnn:
                    m = mism[f["neighbour"] is not None]
                    m[0] += g_dvsa != g_cnn
                    m[1] += 1
    with_n = [p for p, f in fresh.items() if f["neighbour"]]
    share_n = 100 * len(with_n) / max(1, len(fresh))
    print(
        f"post-corpus sessions {[d.name for d in post]}: {len(fresh)} labelled cars in no "
        f"corpus, {len(with_n)} ({share_n:.1f} %) one fuzzy step from a corpus car"
    )
    r_with, r_without = _rate(*mism[True]), _rate(*mism[False])
    print(f"  DVSA-vs-CNN colour mismatch per track: with a neighbour {_fmt(r_with)}")
    print(f"                                         without          {_fmt(r_without)}")
    print(
        "decision: "
        + (
            "leakage > 2 % of val tracks -> re-run makemodel-compare with cluster-aware exclusion; "
            if leak > 0.02
            else "leakage <= 2 % of val tracks; "
        )
        + (
            "single-read > 10 % of corpus cars -> prioritise E1.4"
            if one_snap / len(cars) > 0.10
            else "single-read <= 10 % of corpus cars"
        )
    )
    return {
        "corpus": corpus.name,
        "n_cars": len(cars),
        "n_val_cars": len(val),
        "fuzzy_pairs": len(pairs),
        "cross_split_pairs": len(cross),
        "cross_split_same_make": same_cross,
        "spanning_clusters": len(spanning),
        "val_tracks": len(val_tracks),
        "val_tracks_in_spanning": len(span_val_tracks),
        "cross_split_examples": [[a, make_of[a], b, make_of[b]] for a, b in cross[:20]],
        "cars_one_snap": one_snap,
        "cars_one_track": one_track,
        "post_corpus_sessions": [d.name for d in post],
        "fresh_cars": len(fresh),
        "fresh_with_corpus_neighbour": len(with_n),
        "fresh_neighbour_examples": [[p, fresh[p]["neighbour"]] for p in with_n[:20]],
        "colour_mismatch_with_neighbour": r_with,
        "colour_mismatch_without_neighbour": r_without,
    }


# ----------------------------------------------------------------------
# E0.5 plate collisions

_COLLIDE_WINDOW_S = 10.0
_CONCURRENT_S = 1.0


def check_e05(args: argparse.Namespace) -> dict[str, Any]:
    from rapidfuzz import fuzz

    from streettracker.analysis.alpr.base import plate_conf_threshold
    from streettracker.analysis.dvsa import is_canonical_uk_plate
    from streettracker.analysis.parked import detect_parked

    gate = plate_conf_threshold()
    print(f"== E0.5 plate collisions (gate {gate}) ==")
    totals = {k: Counter() for k in ("all", "gated")}
    examples: list[dict[str, Any]] = []
    for d in _sessions(args.output_root):
        n = d.name
        by_track = _load_json(d / f"{n}_alpr_by_track.json")
        data = _load_json(d / f"{n}_data.json")
        if not isinstance(by_track, dict) or not isinstance(data, list):
            continue
        entries = _load_json(d / f"{n}_alpr.json")
        suppressed = detect_parked(entries, data).suppressed if isinstance(entries, list) else set()
        rec = {int(r["track_id"]): r for r in data if r.get("track_id") is not None}
        reads = []
        for t in by_track.get("tracks", []):
            b = t.get("best_preferred")
            if not isinstance(b, dict):
                continue
            tid, plate = int(t["track_id"]), _plate(b.get("ocr_text"))
            if not plate or not is_canonical_uk_plate(plate):
                continue
            if (tid, int(b.get("snap_index", -1))) in suppressed:
                continue
            r = rec.get(tid)
            if not r or r.get("time_start_unix") is None or r.get("time_end_unix") is None:
                continue
            reads.append(
                (
                    float(r["time_start_unix"]),
                    float(r["time_end_unix"]),
                    tid,
                    plate,
                    float(b.get("ocr_conf") or 0.0),
                    r.get("direction"),
                )
            )
        reads.sort()
        for label, rows in (("all", reads), ("gated", [x for x in reads if x[4] >= gate])):
            c = totals[label]
            c["read_tracks"] += len(rows)
            involved: dict[str, set[int]] = defaultdict(set)
            for i, (s1, e1, t1, p1, c1, d1) in enumerate(rows):
                for j in range(i + 1, len(rows)):
                    s2, e2, t2, p2, c2, d2 = rows[j]
                    if s2 > e1 + _COLLIDE_WINDOW_S:
                        break
                    if len(p1) != len(p2) or fuzz.ratio(p1, p2) < _FUZZY_CUTOFF:
                        continue
                    c["exact_pairs"] += p1 == p2
                    # Concurrent (both tracks live at once for > 1 s) = two objects
                    # carrying one plate; sequential = one object whose track split.
                    overlap = min(e1, e2) - max(s1, s2)
                    kind = ("opposite" if d1 != d2 else "same_dir") + (
                        "_concurrent" if overlap > _CONCURRENT_S else "_sequential"
                    )
                    c[f"{kind}_pairs"] += 1
                    involved[kind] |= {t1, t2}
                    if label == "gated" and kind == "opposite_concurrent" and len(examples) < 15:
                        examples.append(
                            {
                                "session": n,
                                "overlap_s": round(overlap, 1),
                                "a": [
                                    t1,
                                    p1,
                                    round(c1, 3),
                                    d1,
                                    f"{_local(s1):%H:%M:%S}",
                                    round(e1 - s1, 1),
                                ],
                                "b": [
                                    t2,
                                    p2,
                                    round(c2, 3),
                                    d2,
                                    f"{_local(s2):%H:%M:%S}",
                                    round(e2 - s2, 1),
                                ],
                            }
                        )
            for kind, tids in involved.items():
                c[f"{kind}_tracks"] += len(tids)
    out: dict[str, Any] = {"gate": gate}
    kinds = (
        "opposite_concurrent",
        "opposite_sequential",
        "same_dir_concurrent",
        "same_dir_sequential",
    )
    for label, c in totals.items():
        nr = c["read_tracks"]
        rates = {k: _rate(c[f"{k}_tracks"], nr) for k in kinds}
        out[label] = {**dict(c), "rates": rates}
        print(f"{label}: {nr} read tracks, {c['exact_pairs']} exact-string pairs")
        for k in kinds:
            print(f"  {k:20s} {c[f'{k}_pairs']:5d} pairs, tracks {_fmt(rates[k])}")
    out["opposite_concurrent_examples"] = examples
    print("concurrent opposite-direction examples (gated; [id, plate, conf, dir, start, dur s]):")
    for e in examples[:8]:
        print(f"  {e['session']} overlap {e['overlap_s']} s: {e['a']}  <->  {e['b']}")
    worst = out["gated"]["rates"]["opposite_concurrent"]["rate"] or 0.0
    print(
        "decision: "
        + (
            "concurrent opposite-direction collisions > 1 % of read tracks -> R2 is real, "
            "prioritise E1.2/E1.3"
            if worst > 0.01
            else "concurrent opposite-direction collisions <= 1 % of gated read tracks; "
            "sequential pairs are track splits (one fragment's direction is wrong)"
        )
        + "; the same-direction rates are the first car split-rate estimate"
    )
    return out


# ----------------------------------------------------------------------
# E0.7 jogger vs pavement

_JOG_M_S = 2.5
_SPEED_BIN = 0.25


def _otsu(values: list[float], bins: int = 50) -> float:
    lo, hi = min(values), max(values)
    width = (hi - lo) / bins or 1.0
    hist = [0] * bins
    for v in values:
        hist[min(bins - 1, int((v - lo) / width))] += 1
    total = len(values)
    sum_all = sum(i * h for i, h in enumerate(hist))
    best, best_t, w0, sum0 = -1.0, 0, 0, 0.0
    for t in range(bins):
        w0 += hist[t]
        if w0 == 0 or w0 == total:
            continue
        sum0 += t * hist[t]
        m0, m1 = sum0 / w0, (sum_all - sum0) / (total - w0)
        between = w0 * (total - w0) * (m0 - m1) ** 2
        if between > best:
            best, best_t = between, t
    return lo + (best_t + 1) * width


def _peaks(hist: list[int]) -> list[int]:
    """Local maxima of a 3-bin smoothed histogram holding >= 5 % of the peak."""
    sm = [statistics.mean(hist[max(0, i - 1) : i + 2]) for i in range(len(hist))]
    top = max(sm) if sm else 0
    return [
        i
        for i in range(1, len(sm) - 1)
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] >= 0.05 * top
    ]


def check_e07(args: argparse.Namespace) -> dict[str, Any]:
    print("== E0.7 jogger vs pavement ==")
    cfg = _load_json(_HERE.parent / "configs" / "showcase.json") or {}
    m_per_px = cfg.get("m_per_px") or (
        float(cfg["road_length_m"]) / 801.0 if cfg.get("road_length_m") else None
    )
    unit = "m/s" if m_per_px else "px/s"
    rows: list[tuple[float, float, float | None]] = []
    for d in _sessions(args.output_root):
        for r in _load_json(d / f"{d.name}_data.json") or []:
            if r.get("class_name") != "person" or r.get("class_suspect"):
                continue
            if (r.get("num_detections") or 0) < 6:
                continue
            en, ex = r.get("entry_point_frac"), r.get("exit_point_frac")
            if not en or not ex:
                continue
            sp = float(r.get("speed_px_s") or 0.0) * (m_per_px or 1.0)
            boxes = r.get("main_snap_bboxes_done") or r.get("main_snap_bboxes") or []
            hs = [b[3] - b[1] for b in boxes if b]
            rows.append(((en[1] + ex[1]) / 2, sp, statistics.median(hs) if hs else None))
    if len(rows) < 50:
        print(f"only {len(rows)} person tracks with entry/exit points")
        return {"n": len(rows)}
    ys = [y for y, _, _ in rows]
    thr = _otsu(ys)
    print(f"{len(rows)} person tracks (>= 6 detections, entry/exit points); speed in {unit}")
    yb = [0] * 20
    for y in ys:
        yb[min(19, int(y * 20))] += 1
    print("track y (mean of entry/exit, 0 = top of frame), 5 % bins:")
    for i, c in enumerate(yb):
        if c:
            bar = "#" * max(1, round(60 * c / max(yb)))
            print(f"  {i * 0.05:.2f}-{(i + 1) * 0.05:.2f} {c:6d} {bar}")
    print(f"pavement split (Otsu) at y = {thr:.3f}")

    out: dict[str, Any] = {"n": len(rows), "y_split": round(thr, 4), "unit": unit, "pavements": {}}
    n_bins = int(5.0 / _SPEED_BIN) if m_per_px else 40
    bin_w = _SPEED_BIN if m_per_px else (max(s for _, s, _ in rows) / 40 or 1.0)
    groups_by_y = (
        ("far (top)", [r for r in rows if r[0] < thr]),
        ("near (bottom)", [r for r in rows if r[0] >= thr]),
    )
    for name, sel in groups_by_y:
        sp = [s for _, s, _ in sel]
        hs = [h for _, _, h in sel if h]
        hist = [0] * n_bins
        for s in sp:
            hist[min(n_bins - 1, int(s / bin_w))] += 1
        pk = _peaks(hist)
        jog = sum(1 for s in sp if s >= _JOG_M_S) if m_per_px else None
        info = {
            "n": len(sp),
            "median_speed": round(statistics.median(sp), 3),
            "peaks": [round((i + 0.5) * bin_w, 2) for i in pk],
            "jogger_share": round(jog / len(sp), 4) if jog is not None and sp else None,
            "median_bbox_h_px": statistics.median(hs) if hs else None,
            "hist": hist,
        }
        out["pavements"][name] = info
        print(
            f"{name}: n {len(sp)}, median {info['median_speed']} {unit}, peaks at {info['peaks']}, "
            f">= {_JOG_M_S} m/s {100 * (info['jogger_share'] or 0):.1f} %, median bbox h "
            f"{info['median_bbox_h_px']} px"
        )
        top = max(hist) or 1
        for i, c in enumerate(hist):
            if c:
                print(f"    {i * bin_w:4.2f} {c:6d} {'#' * max(1, round(50 * c / top))}")
    far, near = out["pavements"]["far (top)"], out["pavements"]["near (bottom)"]
    if far["median_bbox_h_px"] and near["median_bbox_h_px"]:
        persp = near["median_bbox_h_px"] / far["median_bbox_h_px"]
        speed_ratio = near["median_speed"] / far["median_speed"] if far["median_speed"] else None
        out["perspective_ratio"] = round(persp, 3)
        out["median_speed_ratio"] = round(speed_ratio, 3) if speed_ratio else None
        print(
            f"perspective ratio (near/far person bbox height) {persp:.2f}; "
            f"median speed ratio near/far {speed_ratio:.2f}"
        )
    print(
        "decision: if each pavement is unimodal and the speed ratio matches the perspective ratio, "
        "the jogger class is a near-pavement artifact -> suspend jogger stats until E3.2"
    )
    return out


# ----------------------------------------------------------------------
# E0.9 labels describing another car


def check_e09(args: argparse.Namespace) -> dict[str, Any]:
    import ocr_conf_calibration as occ

    from streettracker.analysis.dvsa import is_canonical_uk_plate

    root = args.output_root
    corpus = _production_corpus(args)
    samples = _manifest_samples(corpus)
    post = _post_corpus_sessions(root, _corpus_sessions(samples))
    print(f"== E0.9 labels describing another car (sessions after {corpus.name}) ==")
    if not post:
        print("no post-corpus sessions")
        return {}
    every = _sessions(root)
    dvsa = occ.dvsa_outcomes(every)
    all_corpus = _all_corpus_plates(args.runs)
    # Cross-track support: tracks anywhere whose UK-shaped best read is this plate.
    support: Counter[str] = Counter()
    for d in every:
        for t in (_load_json(d / f"{d.name}_alpr_by_track.json") or {}).get("tracks", []):
            b = t.get("best_preferred")
            if isinstance(b, dict):
                p = _plate(b.get("ocr_text"))
                if p and is_canonical_uk_plate(p):
                    support[p] += 1
    as_of = date.today()
    old_before = date.fromordinal(as_of.toordinal() - int(round(4 * 365.25)))
    rows = []
    for d in post:
        got = occ.session_reads(d, dvsa, old_before=old_before, exclude_colour=set())
        if got is None:
            print(f"  {d.name}: not re-scored, skipped")
            continue
        rows.extend(got)
    print(f"sessions {[d.name for d in post]}: {len(rows)} UK-shaped best reads")

    def conf_group(r: Any) -> str:
        c = r.conf["min"]
        return ">=0.95" if c >= 0.95 else "0.90-0.95" if c >= 0.90 else "<0.90"

    def sup_group(r: Any) -> str:
        s = support.get(r.plate, 0)
        return "1 track" if s <= 1 else "2-4 tracks" if s <= 4 else ">=5 tracks"

    def agree_group(r: Any) -> str:
        return {None: "single snap", False: "a snap agrees", True: "no snap agrees"}[r.no_agreement]

    def fresh_group(r: Any) -> str:
        return "known car" if r.plate in all_corpus else "fresh car"

    splits = {
        "read support": sup_group,
        "snap agreement": agree_group,
        "min-char confidence": conf_group,
        "in a corpus": fresh_group,
    }
    out: dict[str, Any] = {"sessions": [d.name for d in post], "n_reads": len(rows), "splits": {}}
    for metric in ("colour_mismatch", "not_on_register"):
        vals = [r for r in rows if getattr(r, metric) is not None]
        overall = _rate(sum(getattr(r, metric) for r in vals), len(vals))
        print(f"{metric.replace('_', ' ')}: overall {_fmt(overall)}")
        out["splits"][metric] = {"overall": overall}
        for sname, fn in splits.items():
            groups: dict[str, list[Any]] = defaultdict(list)
            for r in vals:
                groups[fn(r)].append(r)
            res = {
                g: _rate(sum(getattr(r, metric) for r in rs), len(rs)) for g, rs in groups.items()
            }
            out["splits"][metric][sname] = res
            print(f"  by {sname}:")
            for g in sorted(res):
                print(f"    {g:15s} {_fmt(res[g])}")
    # The safest labels set the colour classifier's own error floor.
    safe = [
        r
        for r in rows
        if r.colour_mismatch is not None
        and support.get(r.plate, 0) >= 5
        and r.no_agreement is False
        and r.conf["min"] >= 0.95
    ]
    risky = [r for r in rows if r.colour_mismatch is not None and support.get(r.plate, 0) <= 1]
    floor = _rate(sum(r.colour_mismatch for r in safe), len(safe))
    single = _rate(sum(r.colour_mismatch for r in risky), len(risky))
    out["floor_safest"] = floor
    out["single_track"] = single
    excess = (single["rate"] or 0) - (floor["rate"] or 0)
    print(f"floor (>= 5 tracks, a snap agrees, conf >= 0.95): {_fmt(floor)}")
    print(f"single-track labels: {_fmt(single)}  -> excess {100 * excess:+.1f} pp")
    print(
        "decision: single-read labels mismatching well above the floor -> the excess estimates "
        "the misread-to-real-car rate; adopt the E1.4 filter before the next retrain"
    )
    return out


# Each check takes the parsed arguments and returns its report.
CHECKS = {
    "e03": check_e03,
    "e04": check_e04,
    "e05": check_e05,
    "e06": check_e06,
    "e07": check_e07,
    "e09": check_e09,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument(
        "--corpus", default=None, help="corpus dir for E0.4/E0.9 (default: production's)"
    )
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
        report[key] = CHECKS[key](args)
        print()
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
