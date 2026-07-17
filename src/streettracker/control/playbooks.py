"""Multi-step playbooks — the one-click operational workflows.

A **playbook** runs an ordered list of **steps** to completion, stopping on the
first failure (remaining steps are marked *skipped*). Each step is either:

* a **job** — a command run through the shared :class:`~streettracker.control.jobs.JobRunner`
  (so it inherits the lane model, the live progress/ETA, and the copy-paste
  Claude prompt on failure); or
* an **action** — an in-process coroutine returning a :class:`StepResult` (for
  steps that aren't a subprocess, e.g. an SSH service restart or a model file
  swap — used by the device/promote playbooks).

Playbooks run sequentially within themselves (each step is awaited before the
next), so the lane model only matters for *other* concurrent work.

Shipped playbooks are pure job-chains:

* **enrich** — ``alpr-run`` → ``dvsa-label`` → ``dvsa-apply`` → ``vehicles`` →
  ``makemodel`` → ``bodytype`` → ``people`` on one pulled session (the
  documented enrichment order).
* **build-train** — ``makemodel-build-uk`` → ``makemodel-train-uk``.

The action-based playbooks (roll-session+pull, promote, re-infer+refresh) build
on the same engine and land next, behind confirmation gates.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from streettracker.control import history, introspect, orin, prompts
from streettracker.control.jobs import JobRunner, JobSpec

logger = logging.getLogger(__name__)

# Canonical enrichment args (see CLAUDE.md "ANPR tuning loop" / make-model).
GHOST_MASK = ".claude/ghost_mask.json"


@dataclass
class StepResult:
    """Outcome of an in-process action step."""

    ok: bool
    message: str = ""


ActionFn = Callable[[], Awaitable[StepResult]]


@dataclass
class Step:
    """One playbook step: exactly one of ``job`` / ``action`` is set."""

    label: str
    job: JobSpec | None = None
    action: ActionFn | None = None


class Playbook:
    """A running (or finished) sequence of steps + its per-step state."""

    def __init__(self, name: str, steps: list[Step], label: str = "") -> None:
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.label = label or name
        self.steps = steps
        self.status = "queued"  # queued | running | succeeded | failed | cancelled
        self.created_at = time.time()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.current = -1
        self.cancelled = False
        self.task: asyncio.Task[None] | None = None
        self.states: list[dict[str, Any]] = [
            {"label": s.label, "status": "pending", "job_id": None, "message": ""} for s in steps
        ]

    @property
    def elapsed_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else time.time()
        return end - self.started_at


def _action_prompt(pb: Playbook, step_label: str, message: str, *, cancelled: bool = False) -> str:
    """A copy-paste Claude prompt for a failed/cancelled in-process *action* step.

    Job steps get their prompt from the job runner; action steps (SSH restart,
    model swap, showcase refresh) have no subprocess, so build the same kind of
    self-contained prompt from the playbook name, the step, and its message."""
    lines = [message] if message else []
    report = prompts.JobReport(
        kind=pb.name,
        command=f"playbook '{pb.name}' → action step '{step_label}'",
        cwd=str(Path.cwd()),
        status="cancelled" if cancelled else "failed",
        log_tail=lines,
        issue=prompts.summarize_issue(None, lines),
        summary=message,
    )
    return prompts.build_prompt(report)


class PlaybookRunner:
    """Submits + tracks playbooks, running each step's job on the shared
    :class:`JobRunner` so the UI shows live per-step progress and prompts."""

    def __init__(self, jobs: JobRunner, history_path: Path | None = None) -> None:
        self.jobs = jobs
        self.history_path = history_path
        self._history = history.load(history_path)  # terminal snapshots from prior runs
        self.playbooks: dict[str, Playbook] = {}
        self._order: list[str] = []

    def snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        """Persisted (prior-run) snapshots + live playbooks, most recent ``limit``.

        Prior-run history older than ``history.MAX_AGE_S`` (60h) is dropped so the
        panel list stays bounded even across a long-running panel (the on-disk file
        is pruned on append)."""
        combined = [*history.prune_old(self._history), *(self.snapshot(pb) for pb in self.list())]
        return combined[-limit:]

    def submit(self, name: str, steps: list[Step], label: str = "") -> Playbook:
        pb = Playbook(name, steps, label)
        self.playbooks[pb.id] = pb
        self._order.append(pb.id)
        pb.task = asyncio.create_task(self._run(pb))
        return pb

    def get(self, pid: str) -> Playbook | None:
        return self.playbooks.get(pid)

    def list(self) -> list[Playbook]:
        return [self.playbooks[i] for i in self._order]

    async def wait(self, pid: str) -> None:
        pb = self.playbooks.get(pid)
        if pb is not None and pb.task is not None:
            await pb.task

    async def cancel(self, pid: str) -> bool:
        pb = self.playbooks.get(pid)
        if pb is None:
            return False
        pb.cancelled = True
        # Propagate to the step currently in flight, if it's a job.
        if 0 <= pb.current < len(pb.states):
            jid = pb.states[pb.current]["job_id"]
            if jid:
                await self.jobs.cancel(jid)
        return True

    def snapshot(self, pb: Playbook) -> dict[str, Any]:
        steps = []
        for st in pb.states:
            entry: dict[str, Any] = {
                "label": st["label"],
                "status": st["status"],
                "message": st["message"],
            }
            if st.get("prompt"):
                entry["prompt"] = st["prompt"]
            if st["job_id"]:
                job = self.jobs.get(st["job_id"])
                entry["job"] = job.snapshot() if job is not None else None
            steps.append(entry)
        return {
            "id": pb.id,
            "name": pb.name,
            "label": pb.label,
            "status": pb.status,
            "created_at": pb.created_at,
            "started_at": pb.started_at,
            "ended_at": pb.ended_at,
            "elapsed_s": pb.elapsed_s,
            "current": pb.current,
            "steps": steps,
        }

    async def _run(self, pb: Playbook) -> None:
        pb.status = "running"
        pb.started_at = time.time()
        try:
            for i, step in enumerate(pb.steps):
                if pb.cancelled:
                    pb.status = "cancelled"
                    break
                pb.current = i
                pb.states[i]["status"] = "running"
                ok, status = await self._run_step(pb, i, step)
                pb.states[i]["status"] = status
                if not ok:
                    pb.status = "cancelled" if status == "cancelled" else "failed"
                    break
            else:
                pb.status = "succeeded"
            # Steps that never ran become "skipped".
            for st in pb.states:
                if st["status"] == "pending":
                    st["status"] = "skipped"
        except Exception:
            pb.status = "failed"
            logger.exception("[control] playbook %s crashed", pb.id)
        finally:
            pb.ended_at = time.time()
            if pb.status == "running":  # safety net
                pb.status = "failed"
            if self.history_path is not None:
                with contextlib.suppress(Exception):
                    history.append(self.history_path, self.snapshot(pb))

    async def _run_step(self, pb: Playbook, i: int, step: Step) -> tuple[bool, str]:
        if step.job is not None:
            job = self.jobs.submit(step.job)
            pb.states[i]["job_id"] = job.id
            await self.jobs.wait(job.id)
            return (job.status == "succeeded", job.status)
        if step.action is not None:
            try:
                res = await step.action()
            except Exception as exc:
                msg = f"action error: {exc}"
                pb.states[i]["message"] = msg
                pb.states[i]["prompt"] = _action_prompt(pb, step.label, msg)
                logger.exception("[control] playbook %s step %d action crashed", pb.id, i)
                return (False, "failed")
            pb.states[i]["message"] = res.message
            if not res.ok:
                pb.states[i]["prompt"] = _action_prompt(pb, step.label, res.message)
            return (res.ok, "succeeded" if res.ok else "failed")
        msg = "step has neither a job nor an action"
        pb.states[i]["message"] = msg
        pb.states[i]["prompt"] = _action_prompt(pb, step.label, msg)
        return (False, "failed")


# ----------------------------------------------------------------------
# Playbook factories + the name registry the server dispatches on
# ----------------------------------------------------------------------


def enrich_steps(session_dir: str) -> list[Step]:
    """The documented per-session enrichment chain."""
    return [
        Step(
            "ALPR plate OCR",
            job=JobSpec(
                "alpr-run",
                [session_dir, "--pipeline", "preferred", "--pre-crop", "--ghost-mask", GHOST_MASK],
            ),
        ),
        Step("DVSA make/model lookup", job=JobSpec("dvsa-label", [session_dir])),
        Step("Apply DVSA labels", job=JobSpec("dvsa-apply", [session_dir])),
        Step("Per-vehicle aggregation", job=JobSpec("vehicles", [session_dir])),
        Step("Make/model classifier", job=JobSpec("makemodel", [session_dir])),
        Step("Body-type classifier", job=JobSpec("bodytype", [session_dir])),
        Step("People enrichment", job=JobSpec("people", [session_dir])),
    ]


def build_train_steps(
    corpus_dir: str,
    out_dir: str,
    *,
    # Defaults reproduce the PRODUCTION recipe: B6@528 on 576px crops (the
    # promoted uk_make_0707_b6 run, make@1 0.451). Bumped from B5@456/512
    # on 2026-07-15 -- the playbook had gone stale against the B6 promotion.
    output_size: str = "576",
    # Must be a canonical arch name -- the makemodel-train-uk CLI's --backbone
    # validates against SUPPORTED_ARCHS (efficientnet_b0/b4/b5/b6/b7), not the
    # "b6" shorthand used in docs.
    backbone: str = "efficientnet_b6",
    input_size: str = "528",
    # 20, not 30: CosineAnnealingLR's T_max follows --epochs, and the late-LR
    # lift is where the last ~1pp comes from (0707 run: 0.442->0.451 over
    # epochs 15-20). At --epochs 30 the anneal is too slow for --patience 5 --
    # the 0715 run early-stopped at epoch 18 (0.433) before its tail arrived.
    epochs: str = "20",
    # The CLI default batch (64) OOMs big backbones on the dev-box 3080
    # (10 GB). 8 fits B6@528 (proven by the promoted 0707 run on this box)
    # and leaves headroom alongside the showcase/control GPU usage. Lower
    # for a smaller card.
    batch_size: str = "8",
    # The control panel (and thus build-train) runs on the Windows dev box,
    # where a multi-worker DataLoader stalls -- workers and GPU sit idle, no
    # epoch completes. num_workers=0 (synchronous, single-process loading) is
    # reliable; the per-epoch cost is acceptable since crops cache in RAM.
    num_workers: str = "0",
) -> list[Step]:
    """Rebuild the UK crop corpus, then train the make classifier on it."""
    return [
        Step(
            "Build UK crop corpus",
            job=JobSpec("makemodel-build-uk", [corpus_dir, "--output-size", output_size]),
        ),
        Step(
            "Train UK make classifier",
            job=JobSpec(
                "makemodel-train-uk",
                [
                    corpus_dir,
                    "--out",
                    out_dir,
                    "--backbone",
                    backbone,
                    "--input-size",
                    input_size,
                    "--epochs",
                    epochs,
                    "--batch-size",
                    batch_size,
                    "--num-workers",
                    num_workers,
                ],
            ),
        ),
    ]


@dataclass
class PlaybookContext:
    """Config the action playbooks need: where data lives + how to reach the
    device. Built by the server from its run config + state."""

    output_root: Path
    runs_dir: Path
    model_path: Path
    orin_host: str = orin.DEFAULT_HOST
    orin_user: str = orin.DEFAULT_USER
    orin_key: str = orin.DEFAULT_KEY
    remote_parent: str = orin.DEFAULT_REMOTE_PARENT
    target: str = "output"
    showcase_url: str = "http://127.0.0.1:8090"


def _do_roll(old_session: str, ctx: PlaybookContext) -> StepResult:
    """Restart the device service (finalising ``old_session`` + starting a new
    one) and report the handover. Blocking — call via ``asyncio.to_thread``."""
    ok, detail = orin.restart_service(ctx.orin_host, ctx.orin_user, ctx.orin_key)
    if not ok:
        return StepResult(False, f"service restart failed: {detail}")
    time.sleep(4.0)  # let the unit come back up and create the next session
    snap = orin.orin_live_snapshot(ctx.orin_host, ctx.orin_user, ctx.orin_key, ctx.remote_parent)
    if not snap.reachable:
        return StepResult(False, "device unreachable after restart")
    if snap.service_active != "active":
        return StepResult(False, f"service not active after restart ({snap.service_active})")
    if snap.live_session == old_session:
        return StepResult(
            False, f"a new session was not created (still {old_session}); not pulling"
        )
    n = orin.count_finalized_tracks(
        ctx.orin_host, ctx.orin_user, ctx.orin_key, ctx.remote_parent, old_session
    )
    finalised = f"{n} tracks finalised" if n is not None else "track count unavailable"
    return StepResult(
        True, f"finalised {old_session} ({finalised}); now recording {snap.live_session}"
    )


def roll_and_pull_steps(old_session: str, ctx: PlaybookContext) -> list[Step]:
    """Restart the device (close ``old_session``, open a new one), then pull
    ``old_session`` to the dev box."""

    async def roll() -> StepResult:
        return await asyncio.to_thread(_do_roll, old_session, ctx)

    pull_args = [
        "--session",
        old_session,
        "--host",
        ctx.orin_host,
        "--user",
        ctx.orin_user,
        "--key",
        ctx.orin_key,
        "--remote-parent",
        ctx.remote_parent,
        "--target",
        ctx.target,
        "--only-main",
        "--skip-existing",
    ]
    return [
        Step(f"Roll device — finalise {old_session}, start new session", action=roll),
        Step(f"Pull {old_session}", job=JobSpec("pull", pull_args)),
    ]


def promote_model(
    ctx: PlaybookContext,
    run_name: str | None = None,
    *,
    reader: Callable[[Path], introspect.ModelInfo | None] = introspect.model_info,
    now: datetime | None = None,
) -> StepResult:
    """Promote a training run's ``best.pt`` to the production model: back the
    current model up, copy the candidate in, and write the metadata sidecar +
    corpus fingerprint. Blocking (file IO) — call via ``asyncio.to_thread``."""
    runs = introspect.training_runs(ctx.runs_dir)
    if run_name:
        run = next((r for r in runs if r.name == run_name), None)
    else:
        scored = [r for r in runs if r.best_val_make_top1 is not None]
        run = max(scored, key=lambda r: r.best_val_make_top1 or 0.0, default=None)
    if run is None:
        return StepResult(False, "no scored training run available to promote")
    cand = Path(run.path) / "best.pt"
    if not cand.is_file():
        return StepResult(False, f"{run.name} has no best.pt")
    meta = reader(cand)
    if meta is None:
        return StepResult(False, f"could not read candidate checkpoint {cand}")

    when = now or datetime.now()
    model_path = ctx.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    backup_name = None
    if model_path.is_file():
        backup = model_path.parent / f"{model_path.stem}.{when.strftime('%Y%m%dT%H%M%S')}.pt"
        shutil.copy2(model_path, backup)
        backup_name = backup.name
    shutil.copy2(cand, model_path)

    corpus = introspect.latest_corpus(ctx.runs_dir)
    sidecar = {
        "arch": meta.arch,
        "input_size": meta.input_size,
        "n_makes": meta.n_makes,
        "val_make_top1": meta.val_make_top1,
        "promoted_at": when.strftime("%Y-%m-%d"),
        "source_run": run.name,
        "trained_corpus": (
            {
                "name": corpus.name,
                "n_cars": corpus.n_cars,
                "n_makes": corpus.n_makes,
                "n_crops": corpus.n_crops,
            }
            if corpus is not None
            else None
        ),
    }
    introspect.model_sidecar_path(model_path).write_text(json.dumps(sidecar, indent=2))
    acc = f"{meta.val_make_top1:.3f}" if meta.val_make_top1 is not None else "?"
    msg = f"Promoted {run.name} (make@1 {acc}) -> {model_path.name}"
    if backup_name:
        msg += f"; backed up prior model to {backup_name}"
    return StepResult(True, msg)


def promote_steps(ctx: PlaybookContext, run_name: str | None = None) -> list[Step]:
    async def act() -> StepResult:
        return await asyncio.to_thread(promote_model, ctx, run_name)

    return [Step("Promote model (backup + swap + sidecar)", action=act)]


# /api/refresh rebuilds every session's vehicle aggregation, so it can take well
# over a minute on a large output root (~45s observed at ~4.5k cars). The old 30s
# cap timed out — and aiohttp's timeout raises an empty-str TimeoutError, which
# surfaced as the unhelpful "not reachable (  )".
SHOWCASE_REFRESH_TIMEOUT_S = 300


async def refresh_showcase(
    url: str, *, timeout_s: float = SHOWCASE_REFRESH_TIMEOUT_S
) -> StepResult:
    """POST ``/api/refresh`` so the showcase site re-aggregates new predictions."""
    target = url.rstrip("/") + "/api/refresh"
    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as session,
            session.post(target) as resp,
        ):
            if resp.status != 200:
                return StepResult(False, f"showcase refresh returned HTTP {resp.status}")
            data = await resp.json()
            return StepResult(True, f"showcase refreshed ({data.get('cars', '?')} cars)")
    except asyncio.TimeoutError:
        return StepResult(
            False,
            f"showcase refresh timed out after {timeout_s:.0f}s at {url} — the re-aggregation "
            "may still be running; check the showcase, then re-run reinfer.",
        )
    except aiohttp.ClientError as exc:
        return StepResult(
            False,
            f"showcase not reachable at {url} ({exc!r}) — is `streettracker showcase` running "
            "on that port?",
        )
    except Exception as exc:  # never surface an empty "( )"
        return StepResult(False, f"showcase refresh failed at {url} ({type(exc).__name__}: {exc})")


def reinfer_steps(ctx: PlaybookContext) -> list[Step]:
    """Re-run the make/model classifier on every local session, then refresh the
    showcase so it reflects the new predictions. One GPU job per session
    (sequential), so this is a long, background-style playbook."""
    sessions = introspect.discover_session_dirs(ctx.output_root)
    steps: list[Step] = [
        Step(f"Make/model: {d.name}", job=JobSpec("makemodel", [str(d)])) for d in sessions
    ]

    async def refresh() -> StepResult:
        return await refresh_showcase(ctx.showcase_url)

    steps.append(Step("Refresh showcase", action=refresh))
    return steps


# name -> metadata for the launcher + the server's validation/gating.
PLAYBOOKS: dict[str, dict[str, Any]] = {
    "enrich": {"label": "Enrich a session", "needs_session": True, "destructive": False},
    "build-train": {
        "label": "Build corpus + train CNN",
        "needs_session": False,
        "destructive": False,
    },
    "roll": {
        "label": "Roll session + pull (restarts the camera)",
        "needs_session": False,
        "destructive": True,
    },
    "promote": {"label": "Promote best model", "needs_session": False, "destructive": True},
    "reinfer": {
        "label": "Re-infer all + refresh showcase",
        "needs_session": False,
        "destructive": False,
    },
}


def build_playbook(
    name: str,
    ctx: PlaybookContext,
    *,
    session: str | None = None,
    old_session: str | None = None,
    run_name: str | None = None,
) -> tuple[str, list[Step]]:
    """Resolve a playbook ``name`` (+ params) to a ``(label, steps)`` pair.

    The server supplies ``ctx`` (paths + device config) and resolves any live
    state (e.g. the current session for ``roll``) before calling. Raises
    :class:`ValueError` for an unknown name or a missing required param.
    """
    if name == "enrich":
        if not session:
            raise ValueError("enrich requires a session")
        return f"Enrich {session}", enrich_steps(str(ctx.output_root / session))
    if name == "build-train":
        mmdd = datetime.now().strftime("%m%d")
        corpus_dir = str(ctx.runs_dir / f"uk_crops_{mmdd}_576")
        out_dir = str(ctx.runs_dir / f"uk_make_{mmdd}_b6")
        return f"Build + train -> {out_dir}", build_train_steps(corpus_dir, out_dir)
    if name == "roll":
        if not old_session:
            raise ValueError("roll requires the current live session")
        return f"Roll + pull {old_session}", roll_and_pull_steps(old_session, ctx)
    if name == "promote":
        label = f"Promote {run_name}" if run_name else "Promote best model"
        return label, promote_steps(ctx, run_name)
    if name == "reinfer":
        return "Re-infer all sessions + refresh showcase", reinfer_steps(ctx)
    raise ValueError(f"unknown playbook: {name!r}")
