"""Subprocess job runner for the control panel.

Runs a StreetTracker command (``uv run streettracker <kind> …``) as a child
process, streams its stdout line-by-line into the matching
:mod:`~streettracker.control.progress` parser + a bounded log buffer, and on
exit classifies the outcome. When a job **fails** — or finishes but trips
:func:`~streettracker.control.prompts.summarize_issue` — its snapshot carries a
ready-to-paste Claude prompt (``snapshot()["prompt"]``) so the operator can ask
for help in one copy, without hand-writing the context.

Scope of this slice:

* **Per-lane execution.** Two lanes serialise independently: a **GPU lane**
  (alpr / makemodel / build / train / export-engine — they share one device) and
  a **net lane** (pull / dvsa / vehicles / people). A pull can run while a train is in
  flight, but two GPU jobs never overlap. ``JobSpec.lane`` overrides the
  kind-derived lane.
* **Wake-lock while busy** — the dev box idle-sleeps and has killed overnight
  jobs; an in-process ``SetThreadExecutionState`` keeps the system awake for as
  long as any job runs (Windows; a no-op elsewhere).
* **Process-tree cancel** — ``uv run`` spawns a child python, so cancelling
  uses ``taskkill /T`` on Windows to reap the whole tree.

In-memory only for now (jobs reset on panel restart); on-disk history is a
follow-up.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from streettracker.cli.pull import human_bytes
from streettracker.control import history
from streettracker.control.progress import estimate_eta, make_parser
from streettracker.control.prompts import JobReport, build_prompt, summarize_issue

logger = logging.getLogger(__name__)

# How the runner invokes the project CLI. Overridable so tests can drive a
# trivial python child instead of the real `uv run streettracker`.
#
# ``--no-sync`` is load-bearing, not a nicety. Plain ``uv run`` re-syncs the
# environment first, and when the project itself needs reinstalling that
# means replacing ``.venv/Scripts/streettracker.exe`` -- a file the running
# panel is itself holding open, because the panel *is* a `uv run
# streettracker control` process. So the panel would break every one of its
# own jobs with
#
#     error: failed to remove file `...\Scripts\streettracker.exe`:
#     The process cannot access the file because it is being used by
#     another process. (os error 32)
#
# Hit for real on 2026-07-20: every panel job failed with exit code 2 and
# ~0s runtime after a deploy changed the source. The dependency install is
# an operator step (`uv sync --extra alpr`), deliberately not something a
# job does implicitly -- the same reason the Orin's systemd unit passes
# ``--no-sync``.
DEFAULT_BASE_ARGV = ["uv", "run", "--no-sync", "streettracker"]

_LOG_MAXLEN = 500  # ring-buffer of recent output lines kept per job
_TAIL = 40  # lines embedded in the help prompt / surfaced in the snapshot
_WATCH_INTERVAL = 1.5  # seconds between local-dir size samples (pull byte-ETA)

# Jobs that contend for the one GPU share a lane and run one-at-a-time;
# everything else runs in the "net" lane, which may overlap the GPU lane.
_GPU_KINDS = frozenset(
    {
        "alpr-run",
        "makemodel",
        "bodytype",
        "makemodel-build-uk",
        "makemodel-train-uk",
        "export-engine",
    }
)


def _lane_for(kind: str) -> str:
    return "gpu" if kind in _GPU_KINDS else "net"


@dataclass
class JobSpec:
    """What to run: a ``streettracker`` subcommand (``kind``) plus its args."""

    kind: str
    args: list[str] = field(default_factory=list)
    label: str = ""  # optional human label; defaults to the command string
    lane: str | None = None  # override the kind-derived lane ("gpu" | "net")


class Job:
    """A single command execution and its live state."""

    def __init__(self, spec: JobSpec, base_argv: list[str], cwd: Path) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.spec = spec
        self.lane = spec.lane or _lane_for(spec.kind)
        self.cwd = str(cwd)
        self.argv = [*base_argv, spec.kind, *spec.args]
        self.status = "queued"  # queued | running | succeeded | failed | cancelled
        self.created_at = time.time()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.returncode: int | None = None
        self.parser = make_parser(spec.kind)
        self.log: deque[str] = deque(maxlen=_LOG_MAXLEN)
        self.issue: str | None = None
        self.cancelled = False
        self.task: asyncio.Task[None] | None = None
        self.proc: asyncio.subprocess.Process | None = None
        # Byte-progress for transfer jobs (pull): filled by the dir-watcher when
        # the parser exposes a "watch" directive. None -> no byte progress.
        self.bytes_done: int | None = None
        self.bytes_total: int | None = None

    @property
    def command(self) -> str:
        return " ".join(self.argv)

    @property
    def elapsed_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else time.time()
        return end - self.started_at

    def report(self) -> JobReport:
        return JobReport(
            kind=self.spec.kind,
            command=self.command,
            cwd=self.cwd,
            status=self.status,
            returncode=self.returncode,
            elapsed_s=self.elapsed_s,
            log_tail=list(self.log)[-_TAIL:],
            issue=self.issue,
            phase=self.parser.progress.phase,
            summary=self.parser.progress.summary,
        )

    def snapshot(self) -> dict[str, Any]:
        prog = self.parser.progress.to_json_dict()
        if self.bytes_total:
            # A transfer job: progress + ETA are byte-based (overrides the
            # parser's count-based frac).
            done = self.bytes_done or 0
            prog["bytes_done"] = done
            prog["bytes_total"] = self.bytes_total
            prog["bytes_human"] = f"{human_bytes(done)} / {human_bytes(self.bytes_total)}"
            prog["frac"] = max(0.0, min(1.0, done / self.bytes_total))
            prog["eta_s"] = estimate_eta(done, self.bytes_total, self.elapsed_s or 0.0)
        else:
            prog["eta_s"] = estimate_eta(
                prog.get("current"), prog.get("total"), self.elapsed_s or 0.0
            )
        snap: dict[str, Any] = {
            "id": self.id,
            "kind": self.spec.kind,
            "lane": self.lane,
            "label": self.spec.label or self.command,
            "command": self.command,
            "status": self.status,
            "returncode": self.returncode,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_s": self.elapsed_s,
            "progress": prog,
            "issue": self.issue,
            "log_tail": list(self.log)[-_TAIL:],
        }
        # The headline feature: a fully-formed Claude prompt, present only when
        # there's something worth asking about.
        if self.issue:
            snap["prompt"] = build_prompt(self.report())
        return snap


class JobRunner:
    """Submits + tracks jobs, serialising execution behind one lock."""

    def __init__(
        self,
        base_argv: list[str] | None = None,
        cwd: Path | None = None,
        history_path: Path | None = None,
    ) -> None:
        self.base_argv = base_argv if base_argv is not None else list(DEFAULT_BASE_ARGV)
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.history_path = history_path
        self._history = history.load(history_path)  # terminal snapshots from prior runs
        self.jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._locks: dict[str, asyncio.Lock] = {}  # one per lane
        self._wake_refs = 0

    def snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        """Persisted (prior-run) snapshots + live jobs, most recent ``limit``.

        Prior-run history older than ``history.MAX_AGE_S`` (60h) is dropped so the
        panel list stays bounded even across a long-running panel (the on-disk file
        is pruned on append)."""
        combined = [*history.prune_old(self._history), *(j.snapshot() for j in self.list())]
        return combined[-limit:]

    def _lane_lock(self, lane: str) -> asyncio.Lock:
        lock = self._locks.get(lane)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[lane] = lock
        return lock

    def submit(self, spec: JobSpec) -> Job:
        job = Job(spec, self.base_argv, self.cwd)
        self.jobs[job.id] = job
        self._order.append(job.id)
        job.task = asyncio.create_task(self._run(job))
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return [self.jobs[i] for i in self._order]

    async def wait(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is not None and job.task is not None:
            await job.task

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        job.cancelled = True
        if job.proc is not None and job.proc.returncode is None:
            _kill_tree(job.proc)
        return True

    def _persist(self, job: Job) -> None:
        if self.history_path is not None:
            with contextlib.suppress(Exception):
                history.append(self.history_path, job.snapshot())

    async def _run(self, job: Job) -> None:
        async with self._lane_lock(job.lane):
            if job.cancelled:
                # Cancelled before it ever started (still queued behind the lane
                # lock): nothing ran and there's nothing to clean up, so — unlike
                # a cancel mid-run — no help prompt is surfaced.
                job.status = "cancelled"
                job.ended_at = time.time()
                self._persist(job)
                return
            job.status = "running"
            job.started_at = time.time()
            self._acquire_wake()
            watcher: asyncio.Task[None] | None = None
            try:
                job.proc = await asyncio.create_subprocess_exec(
                    *job.argv,
                    cwd=str(self.cwd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=_child_env(),
                )
                # A cancel that arrived while the process was being spawned
                # (status was already "running") couldn't kill a not-yet-existent
                # proc — reap it now so cancellation is never silently dropped.
                if job.cancelled:
                    _kill_tree(job.proc)
                watcher = asyncio.create_task(self._watch(job))
                assert job.proc.stdout is not None
                async for raw in job.proc.stdout:
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    job.log.append(line)
                    job.parser.feed(line)
                rc = await job.proc.wait()
                job.returncode = rc
                job.ended_at = time.time()
                job.status = (
                    "cancelled" if job.cancelled else ("succeeded" if rc == 0 else "failed")
                )
                job.parser.finish(rc)
                # A cancel and a genuine crash both yield a non-zero rc on
                # Windows (taskkill), so rc alone can't tell them apart. A job
                # that ran but didn't finish — failed OR cancelled — still gets a
                # copy-paste help prompt (build_prompt has a "cancelled" variant:
                # "check whether anything needs cleaning up before I re-run it").
                # For a cancel we don't trust rc; we surface a real trouble marker
                # if the output already showed one, else a plain cancellation note.
                if job.cancelled:
                    job.issue = summarize_issue(None, list(job.log)) or (
                        "job was cancelled before it finished"
                    )
                else:
                    job.issue = summarize_issue(rc, list(job.log))
                _read_watch_bytes(job)  # final, exact byte measurement
            except Exception as exc:  # spawn failure, decode error, etc.
                job.ended_at = time.time()
                job.status = "failed"
                job.returncode = job.returncode if job.returncode is not None else -1
                job.log.append(f"[control] runner error: {exc}")
                job.issue = f"runner could not execute the command: {exc}"
                logger.exception("[control] job %s crashed", job.id)
            finally:
                if watcher is not None:
                    watcher.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await watcher
                self._release_wake()
        self._persist(job)

    async def _watch(self, job: Job) -> None:
        """While the process runs, sample the local dir the parser points at so
        a transfer's progress bar + ETA reflect bytes actually copied."""
        try:
            while job.proc is not None and job.proc.returncode is None:
                await asyncio.to_thread(_read_watch_bytes, job)
                await asyncio.sleep(_WATCH_INTERVAL)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("[control] watcher error for job %s", job.id, exc_info=True)

    # -- wake-lock (reference-counted across concurrent submissions) --

    def _acquire_wake(self) -> None:
        self._wake_refs += 1
        if self._wake_refs == 1:
            _set_wakelock(True)

    def _release_wake(self) -> None:
        self._wake_refs = max(0, self._wake_refs - 1)
        if self._wake_refs == 0:
            _set_wakelock(False)


def _child_env() -> dict[str, str]:
    """Child env with unbuffered stdout, so the panel sees output lines live
    regardless of whether a given command flushes its prints."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _read_watch_bytes(job: Job) -> None:
    """Update ``job.bytes_done/total`` from the parser's ``watch`` directive
    (set once a pull has printed its target + byte total). No-op otherwise."""
    watch = job.parser.progress.metrics.get("watch")
    if not isinstance(watch, dict):
        return
    total, path = watch.get("total_bytes"), watch.get("path")
    if not total or not path:
        return
    job.bytes_total = int(total)
    job.bytes_done = _dir_bytes(str(path))


def _dir_bytes(path: str) -> int:
    """Summed size of the files directly under ``path`` (session dirs are flat).
    Returns 0 if the directory isn't there yet."""
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _set_wakelock(on: bool) -> None:
    """Keep the system awake while a job runs (Windows; no-op elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        es_continuous = 0x80000000
        es_system_required = 0x00000001
        flags = es_continuous | (es_system_required if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]
    except Exception:
        logger.debug("[control] wake-lock toggle failed", exc_info=True)


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate ``proc`` and its children (``uv run`` spawns a child python)."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            proc.kill()
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()
