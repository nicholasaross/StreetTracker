"""HTTP surface of the control panel, driven in-process with aiohttp's
``test_utils`` (no real socket, matching asyncio_mode=auto).

Background pollers are disabled (``background=False``) so tests never shell out
to ``ssh``; the eager local scan populates state from tmp fixtures.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from streettracker.cli.pull import RemoteInventory
from streettracker.control import server
from streettracker.control.orin import OrinStatus

SESSION = "session_20260526_120000"
_JPEG = b"\xff\xd8\xff\xe0FAKE\xff\xd9"


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    out = tmp_path / "output"
    d = out / SESSION
    d.mkdir(parents=True)
    (d / f"{SESSION}_data.json").write_text(
        json.dumps(
            [{"track_id": 1, "class_name": "car", "time_start_unix": 1.0, "time_end_unix": 7.0}]
        )
    )
    (d / f"{SESSION}_meta.json").write_text(
        json.dumps({"session_label": SESSION, "session_start_unix": 0.0})
    )
    (d / "vehicle_1_main_1.jpg").write_bytes(_JPEG)

    runs = tmp_path / "runs"
    cd = runs / "uk_crops_0614_512"
    cd.mkdir(parents=True)
    (cd / "manifest.json").write_text(
        json.dumps(
            {
                "makes": ["FORD"],
                "samples": [{"path": "FORD/a.jpg", "make": "FORD", "car": "AB12CDE"}],
            }
        )
    )

    model = tmp_path / "makemodel_b0.pt"
    model.write_bytes(b"x")
    (tmp_path / "makemodel_b0.meta.json").write_text(
        json.dumps(
            {"arch": "efficientnet_b5", "input_size": 456, "n_makes": 36, "val_make_top1": 0.402}
        )
    )
    return {"out": out, "runs": runs, "model": model}


@pytest.fixture
async def client(env: dict[str, Path]) -> AsyncIterator[TestClient]:
    app = server.build_app(
        env["out"], runs_dir=env["runs"], model_path=env["model"], background=False
    )
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_status_shape(client: TestClient) -> None:
    r = await client.get("/api/status")
    assert r.status == 200
    body = await r.json()
    assert body["orin"] is None  # no poll ran
    assert body["is_local"] is True  # test client is loopback
    assert len(body["local"]["sessions"]) == 1
    assert body["local"]["totals"]["n_cars"] == 1
    assert body["local"]["model"]["val_make_top1"] == 0.402
    assert body["local"]["corpus"]["n_crops"] == 1


async def test_dashboard_renders(client: TestClient) -> None:
    r = await client.get("/")
    assert r.status == 200
    html = await r.text()
    assert "StreetTracker Control" in html
    assert SESSION in html
    assert 'href="/training"' in html  # nav link to the training page


async def test_training_page_renders(client: TestClient) -> None:
    r = await client.get("/training")
    assert r.status == 200
    html = await r.text()
    assert "StreetTracker Control" in html
    assert "renderTraining" in html  # the training page's renderer
    assert 'href="/training"' in html


async def test_sessions_corpus_model_endpoints(client: TestClient) -> None:
    sessions = await (await client.get("/api/sessions")).json()
    assert sessions[0]["label"] == SESSION
    corpus = await (await client.get("/api/corpus")).json()
    assert corpus["name"] == "uk_crops_0614_512"
    model = await (await client.get("/api/model")).json()
    assert model["arch"] == "efficientnet_b5"


async def test_image_served_and_guarded(client: TestClient) -> None:
    ok = await client.get(f"/images/{SESSION}/vehicle_1_main_1.jpg")
    assert ok.status == 200
    assert (await ok.read()).startswith(b"\xff\xd8\xff")
    # unknown session (valid shape) -> 404
    assert (await client.get("/images/session_99999999_999999/vehicle_1_main_1.jpg")).status == 404
    # bad-shape session -> 404
    assert (await client.get("/images/etc/vehicle_1_main_1.jpg")).status == 404
    # non-jpg in the served dir -> 404 (the data.json must not leak)
    assert (await client.get(f"/images/{SESSION}/{SESSION}_data.json")).status == 404


async def test_pull_preview(client: TestClient) -> None:
    inv = RemoteInventory(bytes=50 * 1024 * 1024, files=20, main_snaps=18)
    with patch("streettracker.control.orin.pull_preview", return_value=inv):
        r = await client.get(f"/api/pull-preview/{SESSION}")
    assert r.status == 200
    body = await r.json()
    assert body["bytes"] == 50 * 1024 * 1024
    assert body["main_snaps"] == 18
    assert body["est_seconds"] > 0


async def test_pull_preview_bad_session_400(client: TestClient) -> None:
    assert (await client.get("/api/pull-preview/not-a-session")).status == 400


async def test_refresh_localhost_only(client: TestClient) -> None:
    # The test client is loopback -> allowed. Patch the snapshot calls so no ssh.
    with (
        patch("streettracker.control.server._State.refresh_orin"),
        patch("streettracker.control.server._State.refresh_local"),
    ):
        r = await client.post("/api/refresh")
    assert r.status == 200


async def test_refresh_rejected_from_non_local(client: TestClient) -> None:
    with patch("streettracker.control.server._is_local", return_value=False):
        r = await client.post("/api/refresh")
    assert r.status == 403


def test_is_local_classification() -> None:
    assert server._is_local(SimpleNamespace(remote="127.0.0.1")) is True
    assert server._is_local(SimpleNamespace(remote="::1")) is True
    assert server._is_local(SimpleNamespace(remote="10.0.0.5")) is False
    assert server._is_local(SimpleNamespace(remote=None)) is False


# ----------------------------------------------------------------------
# Jobs (Phase 2): the routes are tested without spawning real subprocesses.


async def test_status_and_list_include_jobs(client: TestClient) -> None:
    body = await (await client.get("/api/status")).json()
    assert body["jobs"] == []
    assert await (await client.get("/api/jobs")).json() == []
    assert (await client.get("/api/jobs/nope")).status == 404


async def test_submit_job_rejects_disallowed_kind(client: TestClient) -> None:
    r = await client.post("/api/jobs", json={"kind": "rm-rf", "args": []})
    assert r.status == 400


async def test_submit_job_rejects_non_list_args(client: TestClient) -> None:
    r = await client.post("/api/jobs", json={"kind": "pull", "args": "notalist"})
    assert r.status == 400


async def test_submit_job_dispatches_allowed_kind(client: TestClient) -> None:
    # Patch submit so the route is exercised without launching `uv run`.
    fake = SimpleNamespace(snapshot=lambda: {"id": "abc", "kind": "pull", "status": "queued"})
    with patch("streettracker.control.server.jobs.JobRunner.submit", return_value=fake) as m:
        r = await client.post(
            "/api/jobs", json={"kind": "pull", "args": ["--session", "session_x"]}
        )
    assert r.status == 200
    assert (await r.json())["id"] == "abc"
    spec = m.call_args.args[0]
    assert spec.kind == "pull" and spec.args == ["--session", "session_x"]


async def test_submit_and_cancel_are_localhost_only(client: TestClient) -> None:
    with patch("streettracker.control.server._is_local", return_value=False):
        assert (await client.post("/api/jobs", json={"kind": "pull"})).status == 403
        assert (await client.post("/api/jobs/anything/cancel")).status == 403


async def test_cancel_unknown_job_404(client: TestClient) -> None:
    assert (await client.post("/api/jobs/nope/cancel")).status == 404


# ----------------------------------------------------------------------
# Playbooks: routes tested with build_playbook patched so nothing real launches.


async def test_status_and_list_include_playbooks(client: TestClient) -> None:
    body = await (await client.get("/api/status")).json()
    assert body["playbooks"] == []
    assert await (await client.get("/api/playbooks")).json() == []
    assert (await client.get("/api/playbooks/nope")).status == 404


async def test_submit_playbook_rejects_unknown_name(client: TestClient) -> None:
    r = await client.post("/api/playbooks", json={"name": "wat"})
    assert r.status == 400


async def test_submit_enrich_requires_valid_session(client: TestClient) -> None:
    assert (await client.post("/api/playbooks", json={"name": "enrich"})).status == 400
    bad = await client.post("/api/playbooks", json={"name": "enrich", "session": "nope"})
    assert bad.status == 400


async def test_submit_enrich_dispatches_with_session(client: TestClient) -> None:
    with patch(
        "streettracker.control.server.playbooks.build_playbook", return_value=("Enrich", [])
    ) as m:
        r = await client.post("/api/playbooks", json={"name": "enrich", "session": SESSION})
    assert r.status == 200
    assert m.call_args.args[0] == "enrich"
    assert m.call_args.kwargs["session"] == SESSION


async def test_submit_build_train_dispatches(client: TestClient) -> None:
    with patch(
        "streettracker.control.server.playbooks.build_playbook", return_value=("Build", [])
    ) as m:
        r = await client.post("/api/playbooks", json={"name": "build-train"})
    assert r.status == 200
    assert m.call_args.args[0] == "build-train"


async def test_destructive_playbooks_need_confirm(client: TestClient) -> None:
    assert (await client.post("/api/playbooks", json={"name": "roll"})).status == 400
    assert (await client.post("/api/playbooks", json={"name": "promote"})).status == 400


async def test_roll_requires_reachable_recording_device(client: TestClient) -> None:
    down = OrinStatus(reachable=False, checked_at=0.0)
    with patch("streettracker.control.server.orin.orin_live_snapshot", return_value=down):
        r = await client.post("/api/playbooks", json={"name": "roll", "confirm": True})
    assert r.status == 400


async def test_roll_resolves_live_session_and_dispatches(client: TestClient) -> None:
    live = OrinStatus(
        reachable=True, checked_at=0.0, service_active="active", live_session="session_live"
    )
    with (
        patch("streettracker.control.server.orin.orin_live_snapshot", return_value=live),
        patch(
            "streettracker.control.server.playbooks.build_playbook", return_value=("Roll", [])
        ) as m,
    ):
        r = await client.post("/api/playbooks", json={"name": "roll", "confirm": True})
    assert r.status == 200
    assert m.call_args.kwargs["old_session"] == "session_live"


async def test_promote_dispatches_with_confirm(client: TestClient) -> None:
    with patch(
        "streettracker.control.server.playbooks.build_playbook", return_value=("Promote", [])
    ) as m:
        r = await client.post("/api/playbooks", json={"name": "promote", "confirm": True})
    assert r.status == 200
    assert m.call_args.args[0] == "promote"


async def test_submit_reinfer_dispatches(client: TestClient) -> None:
    # reinfer is non-destructive -> no confirm needed.
    with patch(
        "streettracker.control.server.playbooks.build_playbook", return_value=("Reinfer", [])
    ) as m:
        r = await client.post("/api/playbooks", json={"name": "reinfer"})
    assert r.status == 200
    assert m.call_args.args[0] == "reinfer"


async def test_playbook_submit_and_cancel_localhost_only(client: TestClient) -> None:
    with patch("streettracker.control.server._is_local", return_value=False):
        assert (await client.post("/api/playbooks", json={"name": "enrich"})).status == 403
        assert (await client.post("/api/playbooks/x/cancel")).status == 403


async def test_cancel_unknown_playbook_404(client: TestClient) -> None:
    assert (await client.post("/api/playbooks/nope/cancel")).status == 404
