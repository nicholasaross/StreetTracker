"""Tests for the playbook engine + the shipped job-chain playbooks.

Drives job steps through a ``python -c`` JobRunner (no real ``uv run``), so
sequencing, stop-on-failure, action steps, cancellation, and the inlined
per-step job snapshots are all exercised deterministically.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from streettracker.analysis.makemodel.archs import SUPPORTED_ARCHS
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
    relabel_steps,
    rescore_steps,
    sessions_needing_rescore,
    sessions_rescored,
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
    # A failed *action* step (no subprocess) still carries a copy-paste Claude
    # prompt, just like a failed job step does.
    assert "prompt" not in snap["steps"][0]  # the succeeded action has none
    assert "nope" in snap["steps"][1]["prompt"]
    assert "bad" in snap["steps"][1]["prompt"]  # the step label is named


async def test_action_exception_is_caught() -> None:
    pr = _runner()

    async def boom() -> StepResult:
        raise RuntimeError("kaboom")

    pb = pr.submit("t", [Step("x", action=boom)])
    await pr.wait(pb.id)
    assert pb.status == "failed"
    snap = pr.snapshot(pb)
    assert "kaboom" in snap["steps"][0]["message"]
    assert "kaboom" in snap["steps"][0]["prompt"]  # crash also yields a help prompt


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


async def test_playbook_holds_wakelock_across_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A playbook keeps the box awake from its first step to its last,
    including the gaps between jobs and action steps -- the reinfer playbook
    slept the PC between two jobs on 2026-09-25. Grace is 0 here so only the
    playbook's own hold can bridge the gaps."""
    import streettracker.control.jobs as jobs_mod

    calls: list[bool] = []
    monkeypatch.setattr(jobs_mod, "_set_wakelock", calls.append)
    pr = _runner()
    pr.jobs.wake_release_grace_s = 0

    async def pause() -> StepResult:
        await asyncio.sleep(0.05)
        return StepResult(True, "ok")

    pb = pr.submit(
        "t",
        [
            Step("a", job=JobSpec(kind="print('a')")),
            Step("wait", action=pause),
            Step("b", job=JobSpec(kind="print('b')")),
        ],
    )
    await pr.wait(pb.id)
    await asyncio.sleep(0.05)
    assert pb.status == "succeeded"
    assert calls == [True, False]  # held once for the whole playbook


async def test_refresh_showcase_timeout_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow re-aggregation (TimeoutError has an empty str) must still produce a
    clear, non-empty message — this was the '(  )' bug."""

    class _FakeSession:
        def __init__(self, *a: object, **k: object) -> None: ...
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *a: object) -> bool:
            return False

        def post(self, *a: object, **k: object) -> object:
            raise asyncio.TimeoutError()

    monkeypatch.setattr("streettracker.control.playbooks.aiohttp.ClientSession", _FakeSession)
    res = await refresh_showcase("http://127.0.0.1:8090", timeout_s=5)
    assert res.ok is False
    assert "timed out" in res.message
    assert res.message.strip().endswith((".",))  # not an empty "( )"


async def test_refresh_showcase_unreachable_message() -> None:
    """A genuinely-down showcase yields a non-empty, actionable message."""
    # Port 1 is never a showcase -> a fast ClientConnectorError.
    res = await refresh_showcase("http://127.0.0.1:1", timeout_s=5)
    assert res.ok is False
    assert "not reachable" in res.message
    assert "()" not in res.message  # the exception repr is included, never empty


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
        "bodytype",
        "colour",
        "people",
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
        "makemodel-compare",
    ]
    assert "uk_crops_" in steps[0].job.args[0]  # type: ignore[union-attr]
    # The head-to-head scores the run just trained, on the corpus just built.
    cmp_args = steps[2].job.args  # type: ignore[union-attr]
    assert cmp_args[0] == steps[0].job.args[0]  # type: ignore[union-attr]
    out_dir = steps[1].job.args[steps[1].job.args.index("--out") + 1]  # type: ignore[union-attr]
    assert cmp_args[cmp_args.index("--candidate") + 1] == str(Path(out_dir) / "best.pt")
    assert cmp_args[cmp_args.index("--output-root") + 1] == str(tmp_path / "output")
    # The --backbone value must be a real arch the CLI accepts (choices=
    # SUPPORTED_ARCHS), not the "b6" shorthand from the docs -- argparse
    # rejects an unknown choice with exit code 2 before training starts.
    train_args = steps[1].job.args  # type: ignore[union-attr]
    backbone = train_args[train_args.index("--backbone") + 1]
    assert backbone in SUPPORTED_ARCHS
    # Pin the PRODUCTION recipe (B6@528 on 576px crops -- the promoted
    # uk_make_0707_b6 run) so the playbook can't silently drift from what's
    # actually shipped again (it sat on B5@456/512 for two promotions).
    build_args = steps[0].job.args  # type: ignore[union-attr]
    assert build_args[build_args.index("--output-size") + 1] == "576"
    assert "--max-per-car" not in build_args  # uncapped unless asked
    assert backbone == "efficientnet_b6"
    assert train_args[train_args.index("--input-size") + 1] == "528"
    # --epochs 20, matching the production run's anneal schedule: T_max=30 was
    # evidenced (0715 run) to early-stop under patience 5 before the LR tail.
    assert train_args[train_args.index("--epochs") + 1] == "20"
    # Batch 6: B6@528 at batch 8 spills out of the 3080's 10 GB into system
    # RAM beside the Windows desktop and trains 5.4x slower (measured
    # 2026-09-29).
    assert train_args[train_args.index("--batch-size") + 1] == "6"


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


def test_promote_records_the_runs_own_corpus_not_the_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The recorded corpus path is relative; resolve it under tmp_path, not a
    # dev box's real runs/ (which holds a real uk_crops_0929_576).
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    for name, n_cars in (("uk_crops_0929_576", 3), ("uk_crops_1001_576_cap30", 2)):
        d = runs / name
        d.mkdir(parents=True)
        samples = [{"make": "FORD", "car": f"C{i}"} for i in range(n_cars)]
        (d / "manifest.json").write_text(json.dumps({"makes": ["FORD"], "samples": samples}))
    os.utime(runs / "uk_crops_0929_576", (1, 1))  # the capped corpus is the newest
    rundir = runs / "uk_make_0929_b6"
    rundir.mkdir()
    summary = {"best_epoch": 15, "best_val_make_top1": 0.93, "makes": ["FORD"], "history": [{}]}
    summary["corpus"] = r"runs\uk_crops_0929_576"  # as the trainer records it on Windows
    (rundir / "history.json").write_text(json.dumps(summary))
    (rundir / "best.pt").write_bytes(b"CANDIDATE")
    model_path = tmp_path / "models" / "makemodel_b0.pt"
    model_path.parent.mkdir(parents=True)
    ctx = PlaybookContext(output_root=tmp_path / "output", runs_dir=runs, model_path=model_path)

    def reader(_p: Path) -> ModelInfo:
        return ModelInfo(path="x", mtime=0.0, source="checkpoint", n_makes=1, val_make_top1=0.93)

    assert promote_model(ctx, "uk_make_0929_b6", reader=reader).ok
    trained = json.loads(model_path.with_suffix(".meta.json").read_text())["trained_corpus"]
    assert trained["name"] == "uk_crops_0929_576" and trained["n_cars"] == 3


def _plate_run_dir(runs: Path, *, compare: dict | None = None) -> Path:
    d = runs / "uk_make_0924_b6"
    d.mkdir(parents=True)
    (d / "history.json").write_text(
        json.dumps(
            {
                "best_epoch": 18,
                "best_val_make_top1": 0.62,
                "makes": ["FORD"],
                "history": [{}],
                "crop_mode": "plate",
                "crop_pad_frac": 0.1,
            }
        )
    )
    (d / "best.pt").write_bytes(b"PLATE-CANDIDATE")
    if compare is not None:
        (d / "compare.json").write_text(json.dumps(compare))
    return d


def test_one_click_promote_refuses_incomparable_run(tmp_path: Path) -> None:
    """ "Promote best model" must not swap in a plate-trained run on the
    strength of a val make@1 that isn't comparable with production's."""
    runs = tmp_path / "runs"
    _plate_run_dir(runs)
    model_path = tmp_path / "models" / "makemodel_b0.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"OLD-MODEL")
    ctx = PlaybookContext(output_root=tmp_path / "output", runs_dir=runs, model_path=model_path)

    def reader(p: Path) -> ModelInfo:
        return ModelInfo(path=str(p), mtime=0.0, source="sidecar", val_make_top1=0.451, size=9)

    res = promote_model(ctx, reader=reader)
    assert res.ok is False
    assert "makemodel-compare" in res.message
    assert model_path.read_bytes() == b"OLD-MODEL"  # untouched


def test_promote_records_crop_mode_and_head_to_head(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    model_path = tmp_path / "models" / "makemodel_b0.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"OLD-MODEL")
    st = model_path.stat()
    report = {
        "production": {"size": st.st_size, "mtime": st.st_mtime},
        "n_cars": 800,
        "n_tracks": 1900,
        "delta": {"candidate_minus_production": 0.08, "ci95": [0.05, 0.11]},
    }
    _plate_run_dir(runs, compare=report)
    ctx = PlaybookContext(output_root=tmp_path / "output", runs_dir=runs, model_path=model_path)

    def reader(p: Path) -> ModelInfo:
        s = p.stat()
        return ModelInfo(
            path=str(p), mtime=s.st_mtime, source="checkpoint", val_make_top1=0.6, size=s.st_size
        )

    res = promote_model(ctx, reader=reader)
    assert res.ok, res.message
    assert model_path.read_bytes() == b"PLATE-CANDIDATE"
    sidecar = json.loads(model_path.with_suffix(".meta.json").read_text())
    assert sidecar["crop_mode"] == "plate"
    assert sidecar["crop_pad_frac"] == 0.1
    assert sidecar["head_to_head"]["delta"]["candidate_minus_production"] == 0.08


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


def _alpr_session(out: Path, label: str, stamp: dict | None) -> Path:
    d = out / label
    d.mkdir(parents=True)
    (d / f"{label}_alpr.json").write_text("[]")
    if stamp is not None:
        (d / f"{label}_static_plates.json").write_text(json.dumps(stamp))
    return d


def test_rescore_targets_only_sessions_with_the_old_confidence(tmp_path: Path) -> None:
    out = tmp_path / "output"
    old = _alpr_session(out, "session_20260101_000000", {"crop_mode": "fullframe"})
    unstamped = _alpr_session(out, "session_20260102_000000", None)
    _alpr_session(
        out, "session_20260103_000000", {"crop_mode": "fullframe", "ocr_conf": "min_char"}
    )
    (out / "session_20260104_000000").mkdir()  # never ALPR'd
    assert sorted(sessions_needing_rescore(out)) == sorted([old, unstamped])

    ctx = PlaybookContext(output_root=out, runs_dir=tmp_path / "runs", model_path=tmp_path / "m.pt")
    steps = rescore_steps(ctx)
    kinds = [s.job.kind for s in steps if s.job]
    # Per session: re-score, then carry it through to the DVSA labels.
    assert kinds == ["alpr-rescore", "dvsa-label", "dvsa-apply", "vehicles"] * 2
    assert steps[0].job.args == [str(unstamped)]  # type: ignore[union-attr]  # newest first
    assert steps[-1].action is not None  # the showcase refresh


def test_build_playbook_rescore(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError, match="nothing to re-score"):
        build_playbook("rescore", ctx)
    _alpr_session(ctx.output_root, "session_20260101_000000", {"crop_mode": "fullframe"})
    label, steps = build_playbook("rescore", ctx)
    assert "Re-score" in label
    assert steps[0].job.kind == "alpr-rescore"  # type: ignore[union-attr]


def test_relabel_targets_only_rescored_sessions(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _alpr_session(out, "session_20260101_000000", {"crop_mode": "fullframe"})  # not re-scored
    done = _alpr_session(
        out, "session_20260102_000000", {"crop_mode": "fullframe", "ocr_conf": "min_char"}
    )
    (out / "session_20260103_000000").mkdir()  # never ALPR'd
    assert sessions_rescored(out) == [done]

    ctx = PlaybookContext(output_root=out, runs_dir=tmp_path / "runs", model_path=tmp_path / "m.pt")
    steps = relabel_steps(ctx)
    # No re-score: only the DVSA harvest, apply and aggregation re-run.
    assert [s.job.kind for s in steps if s.job] == ["dvsa-label", "dvsa-apply", "vehicles"]
    assert steps[0].job.args == [str(done)]  # type: ignore[union-attr]
    assert steps[-1].action is not None  # the showcase refresh


def test_build_playbook_relabel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streettracker.analysis.alpr import base

    ctx = _ctx(tmp_path)
    cfg = tmp_path / "alpr.json"
    monkeypatch.setattr(base, "PLATE_CONF_CONFIG", cfg)
    _alpr_session(ctx.output_root, "session_20260101_000000", {"crop_mode": "fullframe"})
    with pytest.raises(ValueError, match="run the rescore playbook first"):
        build_playbook("relabel", ctx)

    stamp = {"crop_mode": "fullframe", "ocr_conf": "min_char"}
    _alpr_session(ctx.output_root, "session_20260102_000000", stamp)
    cfg.write_text('{"plate_conf_threshold": 0.75}')
    label, steps = build_playbook("relabel", ctx)
    assert "0.75" in label
    assert steps[0].job.kind == "dvsa-label"  # type: ignore[union-attr]

    # A malformed gate file is refused up front, not after the first step.
    cfg.write_text('{"plate_conf_threshold": 7}')
    with pytest.raises(ValueError, match="threshold"):
        build_playbook("relabel", ctx)


def test_build_train_cap_reaches_the_build_and_gets_its_own_dirs(tmp_path: Path) -> None:
    label, steps = build_playbook("build-train", _ctx(tmp_path), max_per_car=30)
    build_args = steps[0].job.args  # type: ignore[union-attr]
    assert build_args[build_args.index("--max-per-car") + 1] == "30"
    corpus = build_args[0]
    assert corpus.endswith("_576_cap30")
    train_args = steps[1].job.args  # type: ignore[union-attr]
    out_dir = train_args[train_args.index("--out") + 1]
    assert out_dir.endswith("_b6_cap30") and "cap30" in label
    # Train + head-to-head both use the capped corpus, not the uncapped one.
    assert train_args[0] == corpus
    assert steps[2].job.args[0] == corpus  # type: ignore[union-attr]
    # Never the same dirs as an uncapped build on the same day.
    _l, plain = build_playbook("build-train", _ctx(tmp_path))
    assert plain[0].job.args[0] != corpus  # type: ignore[union-attr]


@pytest.mark.parametrize("bad", [0, -5, True, "30", 2.5])
def test_build_train_rejects_a_bad_cap(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ValueError, match="max_per_car"):
        build_playbook("build-train", _ctx(tmp_path), max_per_car=bad)  # type: ignore[arg-type]
