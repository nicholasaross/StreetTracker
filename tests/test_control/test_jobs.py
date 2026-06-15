"""Tests for the subprocess job runner.

Drives a trivial ``python -c`` child (via ``base_argv``) instead of the real
``uv run streettracker``, so the run loop, output capture, exit classification,
issue detection, cancel, and the copy-prompt-on-failure are all exercised
against a real subprocess deterministically.
"""

from __future__ import annotations

import asyncio
import sys

from streettracker.control.jobs import JobRunner, JobSpec, _lane_for

PY = sys.executable


def _runner() -> JobRunner:
    # kind carries the python source; args stay empty -> argv = [py, -u, -c, <code>].
    return JobRunner(base_argv=[PY, "-u", "-c"])


async def test_job_success_captures_output_and_no_prompt() -> None:
    runner = _runner()
    job = runner.submit(JobSpec(kind="print('[x] hello'); print('[x] done')"))
    await runner.wait(job.id)
    assert job.status == "succeeded"
    assert job.returncode == 0
    assert job.issue is None
    assert any("hello" in line for line in job.log)
    snap = job.snapshot()
    assert "prompt" not in snap  # nothing to ask about


async def test_job_failure_produces_copyable_prompt() -> None:
    runner = _runner()
    job = runner.submit(JobSpec(kind="import sys; print('boom'); sys.exit(3)"))
    await runner.wait(job.id)
    assert job.status == "failed"
    assert job.returncode == 3
    assert job.issue == "process exited with code 3"
    snap = job.snapshot()
    assert "prompt" in snap
    # The prompt is self-contained: command + exit code + captured output.
    assert snap["command"] in snap["prompt"]
    assert "exit code 3" in snap["prompt"]
    assert "boom" in snap["prompt"]


async def test_job_clean_exit_with_traceback_still_flags_issue() -> None:
    runner = _runner()
    job = runner.submit(
        JobSpec(kind="print('Traceback (most recent call last):'); print('ValueError: nope')")
    )
    await runner.wait(job.id)
    assert job.status == "succeeded"  # exit 0
    assert job.issue == "Python traceback in output"
    assert "prompt" in job.snapshot()  # surfaced despite the clean exit


async def test_job_cancel_marks_cancelled() -> None:
    runner = _runner()
    job = runner.submit(JobSpec(kind="import time; print('start'); time.sleep(30)"))
    # Wait until the child is actually running before cancelling.
    for _ in range(200):
        if job.status == "running" and job.proc is not None:
            break
        await asyncio.sleep(0.02)
    assert await runner.cancel(job.id) is True
    await runner.wait(job.id)
    assert job.status == "cancelled"
    # A cancel isn't an error to ask Claude about.
    assert "prompt" not in job.snapshot()


async def test_runner_serialises_jobs() -> None:
    runner = _runner()
    a = runner.submit(JobSpec(kind="import time; time.sleep(0.3); print('a')"))
    b = runner.submit(JobSpec(kind="print('b')"))
    # b must not finish before a (single lane).
    await runner.wait(a.id)
    assert b.status in ("queued", "running", "succeeded")
    await runner.wait(b.id)
    assert a.status == "succeeded" and b.status == "succeeded"
    assert a.started_at is not None and b.started_at is not None
    assert b.started_at >= a.started_at


async def test_submit_returns_immediately_with_id() -> None:
    runner = _runner()
    job = runner.submit(JobSpec(kind="print('x')", label="demo"))
    assert job.id and runner.get(job.id) is job
    assert runner.list()[0] is job
    snap = job.snapshot()
    assert snap["label"] == "demo"
    await runner.wait(job.id)


async def test_history_persists_and_survives_restart(tmp_path: object) -> None:
    hp = tmp_path / "jobs_history.json"  # type: ignore[operator]
    runner = JobRunner(base_argv=[PY, "-u", "-c"], history_path=hp)
    job = runner.submit(JobSpec(kind="print('x')"))
    await runner.wait(job.id)
    assert hp.is_file()
    # A fresh runner (simulating a panel restart) sees the prior job + merges live.
    reborn = JobRunner(base_argv=[PY, "-u", "-c"], history_path=hp)
    assert any(s["status"] == "succeeded" for s in reborn.snapshots())
    j2 = reborn.submit(JobSpec(kind="print('y')"))
    await reborn.wait(j2.id)
    assert len(reborn.snapshots()) >= 2


def test_lane_for_mapping() -> None:
    assert _lane_for("makemodel-train-uk") == "gpu"
    assert _lane_for("alpr-run") == "gpu"
    assert _lane_for("export-engine") == "gpu"
    assert _lane_for("pull") == "net"
    assert _lane_for("vehicles") == "net"
    assert _lane_for("anything-else") == "net"


def _overlap(a: object, b: object) -> bool:
    # True iff the two run intervals [started, ended] overlapped in time.
    return a.started_at < b.ended_at and b.started_at < a.ended_at  # type: ignore[attr-defined]


async def test_cross_lane_jobs_run_concurrently() -> None:
    runner = _runner()
    gpu = runner.submit(JobSpec(kind="import time; time.sleep(0.5)", lane="gpu"))
    net = runner.submit(JobSpec(kind="import time; time.sleep(0.5)", lane="net"))
    await runner.wait(gpu.id)
    await runner.wait(net.id)
    assert gpu.lane == "gpu" and net.lane == "net"
    assert _overlap(gpu, net), "different lanes should overlap"


async def test_same_lane_jobs_serialise() -> None:
    runner = _runner()
    first = runner.submit(JobSpec(kind="import time; time.sleep(0.4)", lane="gpu"))
    second = runner.submit(JobSpec(kind="print('second')", lane="gpu"))
    await runner.wait(first.id)
    await runner.wait(second.id)
    assert first.ended_at is not None and second.started_at is not None
    # Same lane -> the second can't start until the first finishes.
    assert second.started_at >= first.ended_at - 0.05


async def test_snapshot_exposes_lane() -> None:
    runner = _runner()
    job = runner.submit(JobSpec(kind="print('x')", lane="gpu"))
    assert job.snapshot()["lane"] == "gpu"
    await runner.wait(job.id)


async def test_pull_byte_watcher_tracks_local_dir(tmp_path: object) -> None:
    """A pull job's progress/ETA come from the local dir growing toward the
    remote byte total the command prints. Uses a stand-in 'pull' script that
    prints the real lines + writes bytes into the local session dir."""
    target = str(tmp_path / "out")  # type: ignore[operator]
    session = "session_w"
    script = tmp_path / "fakepull.py"  # type: ignore[operator]
    script.write_text(
        "import os, sys, time\n"
        "target, session = sys.argv[2], sys.argv[3]\n"
        "d = os.path.join(target, session)\n"
        "os.makedirs(d, exist_ok=True)\n"
        "print('[pull] session: ' + session, flush=True)\n"
        "print('[pull] target:  ' + target, flush=True)\n"
        "print('[pull] size_bytes: 1000', flush=True)\n"
        "open(os.path.join(d, 'vehicle_1_main_1.jpg'), 'wb').write(b'x' * 600)\n"
        "time.sleep(0.2)\n"
        "open(os.path.join(d, 'vehicle_2_main_1.jpg'), 'wb').write(b'x' * 400)\n"
        "time.sleep(0.1)\n"
    )
    # kind="pull" selects the PullParser; argv = [PY, -u, script, "pull", target, session].
    runner = JobRunner(base_argv=[PY, "-u", str(script)])
    job = runner.submit(JobSpec(kind="pull", args=[target, session]))
    await runner.wait(job.id)
    assert job.status == "succeeded"
    assert job.bytes_total == 1000
    assert job.bytes_done == 1000  # final read after both files landed
    prog = job.snapshot()["progress"]
    assert prog["frac"] == 1.0
    assert prog["bytes_human"]  # "1000.0 B / 1000.0 B"
