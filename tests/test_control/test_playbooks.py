"""Tests for the playbook engine + the shipped job-chain playbooks.

Drives job steps through a ``python -c`` JobRunner (no real ``uv run``), so
sequencing, stop-on-failure, action steps, cancellation, and the inlined
per-step job snapshots are all exercised deterministically.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from streettracker.analysis.makemodel.model import SUPPORTED_ARCHS
from streettracker.control.introspect import ModelInfo
from streettracker.control.jobs import JobRunner, JobSpec
from streettracker.control.playbooks import (
    PlaybookContext,
    PlaybookRunner,
    Step,
    StepResult,
    build_playbook,
    enrich_steps,
    promote_model,
    refresh_showcase,
    reinfer_steps,
)

PY = sys.executable


def _runner() -> PlaybookRunner:
    return PlaybookRunner(JobRunner(base_argv=[PY, "-u", "-c"]))


async def test_all_steps_succeed() -> None:
    pr = _runner()
    pb = pr.submit(
        "t",
        [Step("a", job=JobSpec(kind="print('a')")), Step("b", job=JobSpec(kind="print('b')"))],
    )
    await pr.wait(pb.id)
    assert pb.status == "succeeded"
    snap = pr.snapshot(pb)
    assert [s["status"] for s in snap["steps"]] == ["succeeded", "succeeded"]
    assert snap["steps"][0]["job"]["status"] == "succeeded"


async def test_stop_on_failure_skips_rest_and_carries_prompt() -> None:
    pr = _runner()
    pb = pr.submit(
        "t",
        [
            Step("boom", job=JobSpec(kind="import sys; print('x'); sys.exit(1)")),
            Step("never", job=JobSpec(kind="print('never')")),
        ],
    )
    await pr.wait(pb.id)
    assert pb.status == "failed"
    snap = pr.snapshot(pb)
    assert snap["steps"][0]["status"] == "failed"
    assert snap["steps"][1]["status"] == "skipped"
    # The failing step's job still surfaces the copy-paste Claude prompt.
    assert "prompt" in snap["steps"][0]["job"]


async def test_action_steps_success_and_failure() -> None:
    pr = _runner()

    async def ok() -> StepResult:
        return StepResult(True, "did it")

    async def bad() -> StepResult:
        return StepResult(False, "nope")

    pb = pr.submit(
        "t",
        [
            Step("ok", action=ok),
            Step("bad", action=bad),
            Step("after", job=JobSpec(kind="print('x')")),
        ],
    )
    await pr.wait(pb.id)
    assert pb.status == "failed"
    snap = pr.snapshot(pb)
    assert snap["steps"][0]["status"] == "succeeded" and snap["steps"][0]["message"] == "did it"
    assert snap["steps"][1]["status"] == "failed" and snap["steps"][1]["message"] == "nope"
    assert snap["steps"][2]["status"] == "skipped"


async def test_action_exception_is_caught() -> None:
    pr = _runner()

    async def boom() -> StepResult:
        raise RuntimeError("kaboom")

    pb = pr.submit("t", [Step("x", action=boom)])
    await pr.wait(pb.id)
    assert pb.status == "failed"
    assert "kaboom" in pr.snapshot(pb)["steps"][0]["message"]


async def test_cancel_playbook_stops_and_skips_rest() -> None:
    pr = _runner()
    pb = pr.submit(
        "t",
        [
            Step("slow", job=JobSpec(kind="import time; print('s'); time.sleep(30)")),
            Step("after", job=JobSpec(kind="print('x')")),
        ],
    )
    # Wait until the first step's job is actually running before cancelling.
    for _ in range(300):
        jid = pb.states[0]["job_id"]
        job = pr.jobs.get(jid) if jid else None
        if job is not None and job.status == "running":
            break
        await asyncio.sleep(0.02)
    assert await pr.cancel(pb.id) is True
    await pr.wait(pb.id)
    assert pb.status == "cancelled"
    assert pr.snapshot(pb)["steps"][1]["status"] == "skipped"


# ---- the registry / factories ----


def _ctx(tmp_path: Path) -> PlaybookContext:
    return PlaybookContext(
        output_root=tmp_path / "output",
        runs_dir=tmp_path / "runs",
        model_path=tmp_path / "models" / "makemodel_b0.pt",
    )


def test_enrich_steps_chain() -> None:
    steps = enrich_steps("output/session_x")
    assert [s.job.kind for s in steps] == [  # type: ignore[union-attr]
        "alpr-run",
        "dvsa-label",
        "dvsa-apply",
        "vehicles",
        "makemodel",
    ]
    assert "--ghost-mask" in steps[0].job.args  # type: ignore[union-attr]


def test_build_playbook_enrich(tmp_path: Path) -> None:
    label, steps = build_playbook("enrich", _ctx(tmp_path), session="session_x")
    assert "session_x" in label
    assert steps[0].job.kind == "alpr-run"  # type: ignore[union-attr]
    assert "session_x" in steps[0].job.args[0]  # type: ignore[union-attr]


def test_build_playbook_build_train_generates_dated_dirs(tmp_path: Path) -> None:
    _label, steps = build_playbook("build-train", _ctx(tmp_path))
    assert [s.job.kind for s in steps] == [  # type: ignore[union-attr]
        "makemodel-build-uk",
        "makemodel-train-uk",
    ]
    assert "uk_crops_" in steps[0].job.args[0]  # type: ignore[union-attr]
    # The --backbone value must be a real arch the CLI accepts (choices=
    # SUPPORTED_ARCHS), not the "b5" shorthand from the docs -- argparse
    # rejects an unknown choice with exit code 2 before training starts.
    train_args = steps[1].job.args  # type: ignore[union-attr]
    backbone = train_args[train_args.index("--backbone") + 1]
    assert backbone in SUPPORTED_ARCHS


def test_build_playbook_roll(tmp_path: Path) -> None:
    label, steps = build_playbook("roll", _ctx(tmp_path), old_session="session_a")
    assert "session_a" in label
    assert len(steps) == 2
    assert steps[0].action is not None  # restart action
    assert steps[1].job.kind == "pull"  # type: ignore[union-attr]
    assert "session_a" in steps[1].job.args  # type: ignore[union-attr]
    assert "--only-main" in steps[1].job.args  # type: ignore[union-attr]


def test_build_playbook_promote(tmp_path: Path) -> None:
    label, steps = build_playbook("promote", _ctx(tmp_path))
    assert "Promote" in label
    assert len(steps) == 1 and steps[0].action is not None


def test_build_playbook_errors(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError):
        build_playbook("nope", ctx)
    with pytest.raises(ValueError):
        build_playbook("enrich", ctx)  # missing session
    with pytest.raises(ValueError):
        build_playbook("roll", ctx)  # missing old_session


def test_promote_model_swaps_backs_up_and_writes_sidecar(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    rundir = runs / "uk_make_0615_b5"
    rundir.mkdir(parents=True)
    (rundir / "history.json").write_text(
        json.dumps(
            {"best_epoch": 16, "best_val_make_top1": 0.41, "makes": ["FORD"], "history": [{}]}
        )
    )
    (rundir / "best.pt").write_bytes(b"CANDIDATE")
    model_path = tmp_path / "models" / "makemodel_b0.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"OLD-MODEL")
    ctx = PlaybookContext(output_root=tmp_path / "output", runs_dir=runs, model_path=model_path)

    def reader(_p: Path) -> ModelInfo:  # avoid a real torch read
        return ModelInfo(
            path="x",
            mtime=0.0,
            source="checkpoint",
            arch="efficientnet_b5",
            input_size=456,
            n_makes=36,
            val_make_top1=0.41,
        )

    res = promote_model(ctx, reader=reader)
    assert res.ok
    assert model_path.read_bytes() == b"CANDIDATE"  # candidate swapped in
    sidecar = json.loads(model_path.with_suffix(".meta.json").read_text())
    assert sidecar["arch"] == "efficientnet_b5"
    assert sidecar["val_make_top1"] == 0.41
    assert sidecar["source_run"] == "uk_make_0615_b5"
    # the prior model was preserved under a timestamped backup
    backups = list(model_path.parent.glob("makemodel_b0.*.pt"))
    assert any(b.read_bytes() == b"OLD-MODEL" for b in backups)


def test_promote_model_no_runs_is_graceful(tmp_path: Path) -> None:
    ctx = PlaybookContext(
        output_root=tmp_path / "o", runs_dir=tmp_path / "runs", model_path=tmp_path / "m.pt"
    )
    res = promote_model(ctx, reader=lambda _p: None)
    assert res.ok is False


def test_reinfer_steps_one_job_per_session_plus_refresh(tmp_path: Path) -> None:
    out = tmp_path / "output"
    (out / "session_20260101_000000").mkdir(parents=True)
    (out / "session_20260102_000000").mkdir(parents=True)
    ctx = PlaybookContext(output_root=out, runs_dir=tmp_path / "runs", model_path=tmp_path / "m.pt")
    steps = reinfer_steps(ctx)
    assert sum(1 for s in steps if s.job and s.job.kind == "makemodel") == 2
    assert steps[-1].action is not None  # the showcase refresh


async def test_refresh_showcase_unreachable_is_graceful() -> None:
    res = await refresh_showcase("http://127.0.0.1:1")  # nothing listening there
    assert res.ok is False


def test_build_playbook_reinfer(tmp_path: Path) -> None:
    label, steps = build_playbook("reinfer", _ctx(tmp_path))
    assert "Re-infer" in label
    assert steps[-1].action is not None


async def test_playbook_history_persists_and_survives_restart(tmp_path: Path) -> None:
    hp = tmp_path / "pb_history.json"
    runner = PlaybookRunner(JobRunner(base_argv=[PY, "-u", "-c"]), history_path=hp)
    pb = runner.submit("t", [Step("a", job=JobSpec(kind="print('a')"))])
    await runner.wait(pb.id)
    assert hp.is_file()
    reborn = PlaybookRunner(JobRunner(base_argv=[PY, "-u", "-c"]), history_path=hp)
    assert any(s["status"] == "succeeded" for s in reborn.snapshots())
