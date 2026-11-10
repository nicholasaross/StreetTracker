"""Does regenerating the summary page stall the tracker? (2026-11-10)

``device/runtime.py`` calls ``write_html`` inline in ``process_frame`` once
``html_idle_seconds`` (10 s) have passed since the last KEPT track was
finalised, and a track is finalised ``max_misses + 1`` processed frames after
its last detection. So the page is rebuilt at a predictable instant:

    T = time_end of the last kept track + (max_misses + 1) / fps + idle

``generate_html`` converts every record held, on the event loop. While it runs
no frame is processed, so no track logs a point. Sessions recorded since
2026-10-04 carry ``_trajectories.jsonl`` (every 3rd box of every kept track,
with wall-clock times), which makes the stall visible after the fact:

1. track ENDS pile up at T (the object is not re-associated after the stall);
2. no track logs a point for the length of the stall, then new tracks start;
3. the stall grows with the number of records the session holds.

Read-only. Usage:

    uv run --no-sync python .claude/loop_stall_check.py [output_root] [session ...]

With no session names every session that has a trajectory log is pooled.
Name one or more sessions to look at those alone, e.g. the first session
recorded after a fix.

After the runtime is fixed, sessions recorded with the fix should show no pile
of ends at T and a stall equal to the control (one logging interval, ~0.34 s).
"""

from __future__ import annotations

import bisect
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from streettracker.common.fileio import iter_jsonl
from streettracker.common.session_files import SessionFiles

IDLE_S = 10.0  # output.html_idle_seconds (the default; the live config doesn't override it)
MAX_MISSES = 15  # tracking.max_misses
AT_WINDOW = (-0.15, 0.35)  # a track "ends at the instant" if its last point is in T + this
PICKUP_S = 4.0  # a same-class track starting this soon after a cut counts as the same pass


def quantile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def largest_gap(points: list[float], lo: float, hi: float) -> float | None:
    """Largest gap between consecutive logged points whose first lies in [lo, hi]."""
    i, j = bisect.bisect_left(points, lo), bisect.bisect_right(points, hi)
    if j <= i or j >= len(points):
        return None
    return max(points[x + 1] - points[x] for x in range(i, j))


def bars(counts: list[int], lo: float, step: float) -> None:
    peak = max(counts) or 1
    for i, c in enumerate(counts):
        print(f"    {lo + i * step:+6.2f} s {c:6d} {'#' * int(50 * c / peak)}")


def main() -> int:  # noqa: C901 - one linear measurement script
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    only = set(sys.argv[2:])
    sessions = sorted(
        d
        for d in root.glob("session_*")
        if SessionFiles(d).trajectories.is_file() and (not only or d.name in only)
    )
    # The end/start pictures pool the late part of long sessions, where the
    # stall was longest. For named sessions show everything they have.
    hist_min_records = 0 if only else 4000
    if not sessions:
        print(f"no sessions with a trajectory log under {root}")
        return 1

    tot: Counter[str] = Counter()
    ends_rel: list[float] = []  # track ends relative to T (see hist_min_records)
    starts_rel: list[float] = []
    stall: list[tuple[int, float]] = []  # (records held, seconds)
    control: list[tuple[int, float]] = []
    snap_at: list[float] = []
    snap_other: list[float] = []
    cut_by_class: Counter[str] = Counter()
    regen_k: list[tuple[str, int]] = []
    span_s: dict[str, float] = {}

    print("session                     records  regenerations  fps")
    for sd in sessions:
        files = SessionFiles(sd)
        events = list(iter_jsonl(files.events))
        trajs = list(iter_jsonl(files.trajectories))
        if len(events) != len(trajs):
            print(f"  {sd.name}: events/trajectories line counts differ; skipped")
            continue
        fps = json.loads(files.meta.read_text(encoding="utf-8"))["pipe_fps"]
        lag = (MAX_MISSES + 1) / fps
        points = sorted(tr["t0_unix"] + p[0] for tr in trajs for p in tr["points"])
        recs = sorted(
            ((tr["t0_unix"], ev["time_end_unix"], ev["class_name"], ev["track_id"]) for ev, tr in zip(events, trajs)),
            key=lambda r: r[1],
        )
        ends = [r[1] for r in recs]
        by_start = sorted(recs, key=lambda r: r[0])
        starts = [r[0] for r in by_start]
        span_s[sd.name] = ends[-1] - ends[0]
        tot["tracks"] += len(recs)

        # Predicted regeneration instants: a kept finalise followed by >= idle of quiet.
        instants: list[tuple[int, float]] = []
        for k, e in enumerate(ends):
            nxt = ends[k + 1] if k + 1 < len(ends) else float("inf")
            if nxt - e > IDLE_S + 0.3:
                instants.append((k + 1, e + lag + IDLE_S))
        tot["regens"] += len(instants)
        regen_k.extend((sd.name, k) for k, _t in instants)
        print(f"  {sd.name}  {len(recs):6d}  {len(instants):6d}        {fps}")

        def same_class_pickup(end: float, cls: str) -> bool:
            a = bisect.bisect_right(starts, end + 0.05)
            b = bisect.bisect_right(starts, end + PICKUP_S)
            return any(by_start[x][2] == cls for x in range(a, b))

        cut_ids: set[int] = set()
        for k, t in instants:
            a = bisect.bisect_left(ends, t + AT_WINDOW[0])
            b = bisect.bisect_right(ends, t + AT_WINDOW[1])
            for j in range(a, b):
                _s0, e0, cls, tid = recs[j]
                cut_ids.add(tid)
                tot["cut"] += 1
                cut_by_class[cls] += 1
                tot["cut_picked_up"] += same_class_pickup(e0, cls)
            if b > a:
                tot["regens_with_cut"] += 1
                g = largest_gap(points, t + AT_WINDOW[0], t + AT_WINDOW[1])
                if g is not None and g < 30:
                    stall.append((k, g))
            g = largest_gap(points, t - 3.0 + AT_WINDOW[0], t - 3.0 + AT_WINDOW[1])
            if g is not None and g < 30:
                control.append((k, g))
            if k >= hist_min_records:
                for arr, out in ((ends, ends_rel), (starts, starts_rel)):
                    lo, hi = bisect.bisect_left(arr, t - 3.0), bisect.bisect_right(arr, t + 6.0)
                    out.extend(x - t for x in arr[lo:hi])
        # Chance level for "ends at an instant": the session's mean end rate.
        tot["cut_by_chance_x1000"] += int(1000 * len(instants) * (AT_WINDOW[1] - AT_WINDOW[0]) * len(ends) / span_s[sd.name])
        for _s0, e0, cls, tid in recs:
            if tid not in cut_ids:
                tot["other"] += 1
                tot["other_picked_up"] += same_class_pickup(e0, cls)

        # 4K snaps in flight when the page is rebuilt finish late.
        ts = [t for _k, t in instants]
        for ev in events:
            for fired, done in zip(ev.get("main_snap_fire_unix") or [], ev.get("main_snap_done_unix") or []):
                if fired is None or done is None:
                    continue
                i = bisect.bisect_left(ts, fired)
                (snap_at if i < len(ts) and ts[i] - fired < 0.7 else snap_other).append(done - fired)

    print(f"\ntracks {tot['tracks']}, predicted regenerations {tot['regens']}")

    print(f"\n[1] track ENDS relative to the predicted instant (records held >= {hist_min_records}, 0.25 s bins)")
    print("    (no ends before -1.5 s by construction: an instant needs a quiet spell first)")
    b = [0] * 24
    for d in ends_rel:
        if -3.0 <= d < 3.0:
            b[int((d + 3.0) / 0.25)] += 1
    bars(b, -3.0, 0.25)
    print("\n[2] track STARTS relative to the predicted instant (same sessions)")
    b = [0] * 36
    for d in starts_rel:
        if -3.0 <= d < 6.0:
            b[int((d + 3.0) / 0.25)] += 1
    bars(b, -3.0, 0.25)

    print("\n[3] stall length (s): largest gap between logged points at the instant, where a track was cut")
    print("    records held |     n   p25   med   p75 | control (3 s earlier): med   p75")
    edges = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 12000, 20000]
    fit: list[tuple[float, float, int]] = []
    for lo, hi in zip(edges, edges[1:]):
        v = [g for k, g in stall if lo <= k < hi]
        c = [g for k, g in control if lo <= k < hi]
        if len(v) >= 15 and c:
            fit.append(((lo + hi) / 2, statistics.median(v), len(v)))
            print(
                f"    {lo:>6d}-{hi:<5d} | {len(v):5d} {quantile(v, 0.25):5.2f} {statistics.median(v):5.2f} "
                f"{quantile(v, 0.75):5.2f} | {statistics.median(c):27.2f} {quantile(c, 0.75):5.2f}"
            )
    slope = intercept = 0.0
    if len(fit) >= 3:
        w = sum(f[2] for f in fit)
        mx = sum(f[0] * f[2] for f in fit) / w
        my = sum(f[1] * f[2] for f in fit) / w
        slope = sum(f[2] * (f[0] - mx) * (f[1] - my) for f in fit) / sum(f[2] * (f[0] - mx) ** 2 for f in fit)
        intercept = my - slope * mx
        print(f"    fit to the medians: {intercept:.2f} s + {slope * 1000:.3f} ms per record held")

    chance = tot["cut_by_chance_x1000"] / 1000
    excess = tot["cut"] - chance
    p_cut = tot["cut_picked_up"] / max(1, tot["cut"])
    p_other = tot["other_picked_up"] / max(1, tot["other"])
    pairs = tot["cut"] * (p_cut - p_other)
    print("\n[4] cost")
    print(f"    tracks ending at an instant: {tot['cut']} (chance {chance:.0f}); excess {excess:.0f} = {excess / tot['tracks']:.1%} of kept tracks")
    print(f"    by class: {dict(cut_by_class.most_common())}")
    print(f"    regenerations that cut a track: {tot['regens_with_cut']} of {tot['regens']} ({tot['regens_with_cut'] / tot['regens']:.1%})")
    print(f"    same-class track starts within {PICKUP_S:.0f} s: {p_cut:.1%} after a cut, {p_other:.1%} after any other end")
    print(f"    => about {pairs:.0f} passes recorded as two tracks = {pairs / tot['tracks']:.1%} of kept tracks")
    for label, v in (("fired in the 0.7 s before an instant", snap_at), ("all other snaps", snap_other)):
        if v:
            print(
                f"    4K snap fire->saved, {label:37s} n={len(v):6d} median {statistics.median(v):.2f} s  "
                f"p90 {quantile(v, 0.9):.2f}  >=1.5 s: {sum(x >= 1.5 for x in v) / len(v):.1%}"
            )

    if slope:
        print("\n[5] time spent rebuilding the page, from the fit")
        for name, span in span_s.items():
            ks = [k for n, k in regen_k if n == name]
            blind = sum(intercept + slope * k for k in ks)
            print(f"    {name}: {len(ks)} rebuilds, {blind / 60:6.1f} min blind over {span / 86400:.1f} days ({blind / span:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
