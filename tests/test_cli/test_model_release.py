"""``streettracker model-release``: back the production make model up as a
GitHub Release, safely (public repo) and idempotently. ``gh`` is faked."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from streettracker.cli import model_release as mr


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "makemodel_b0.pt"
    model.write_bytes(b"weights" * 1000)
    (tmp_path / "makemodel_b0.meta.json").write_text(
        json.dumps(
            {
                "arch": "efficientnet_b6",
                "input_size": 528,
                "n_makes": 62,
                "val_make_top1": 0.9323,
                "promoted_at": "2026-10-01",
                "source_run": "uk_make_0929_b6",
                "head_to_head": {
                    "n_cars": 67,
                    "delta": {"candidate_minus_production": 0.013, "ci95": [-0.026, 0.0625]},
                },
                "trained_corpus": {
                    "name": "uk_crops_0929_576",
                    "n_cars": 4347,
                    "n_makes": 62,
                    "n_crops": 76033,
                },
            }
        ),
        encoding="utf-8",
    )
    return model


_CLEAN_CKPT = {
    "arch": "efficientnet_b6",
    "metadata": {"make_names": ["AUDI", "BMW", "FORD"], "input_size": 528},
}


class _FakeGh:
    """Just enough of ``gh release view/create/upload/edit`` for main()."""

    def __init__(self, release: dict[str, Any] | None = None, *, view_error: str = "") -> None:
        self.release = release  # {"isDraft": bool, "assets": [...]}
        self.view_error = view_error
        self.calls: list[tuple[str, ...]] = []

    @staticmethod
    def _asset(path: str) -> dict[str, Any]:
        data = Path(path).read_bytes()
        return {
            "name": Path(path).name,
            "size": len(data),
            "state": "uploaded",
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        }

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        verb = args[1]
        if verb == "view":
            if self.view_error:
                return subprocess.CompletedProcess(args, 1, "", self.view_error)
            if self.release is None:
                return subprocess.CompletedProcess(args, 1, "", "release not found")
            return subprocess.CompletedProcess(args, 0, json.dumps(self.release), "")
        if verb == "create":
            files = [a for a in args[3:] if a.endswith((".pt", ".json"))]
            self.release = {"isDraft": False, "assets": [self._asset(f) for f in files]}
            return subprocess.CompletedProcess(args, 0, "https://example/releases/x\n", "")
        if verb == "upload":
            assert self.release is not None
            files = [a for a in args[3:] if a.endswith((".pt", ".json"))]
            names = {Path(f).name for f in files}
            kept = [a for a in self.release["assets"] if a["name"] not in names]
            self.release["assets"] = kept + [self._asset(f) for f in files]
            return subprocess.CompletedProcess(args, 0, "", "")
        if verb == "edit":
            assert self.release is not None
            self.release["isDraft"] = False
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"unexpected gh call {args}")


def _run(tmp_path: Path, gh: _FakeGh, ckpt: Any = _CLEAN_CKPT, *extra: str) -> int:
    model = _model(tmp_path)
    with (
        patch.object(mr, "_load_checkpoint", return_value=ckpt),
        patch.object(mr, "_gh", side_effect=gh),
    ):
        return mr.main(["--model", str(model), *extra])


def _verbs(gh: _FakeGh) -> list[str]:
    return [c[1] for c in gh.calls]


def test_creates_the_release_and_verifies_the_upload(tmp_path: Path) -> None:
    gh = _FakeGh()
    assert _run(tmp_path, gh) == 0
    assert _verbs(gh) == ["view", "create", "view"]  # verified after upload
    create = gh.calls[1]
    assert create[2] == "makemodel-uk_make_0929_b6"  # tag from the sidecar's run
    assert create[create.index("--target") + 1] == "main"
    assert {Path(a).name for a in create if a.endswith((".pt", ".json"))} == {
        "makemodel_b0.pt",
        "makemodel_b0.meta.json",
    }


def test_rerun_on_a_published_release_does_nothing(tmp_path: Path) -> None:
    gh = _FakeGh()
    assert _run(tmp_path, gh) == 0
    gh.calls.clear()
    assert _run(tmp_path, gh) == 0
    assert _verbs(gh) == ["view"]  # no upload


def test_finishes_a_half_done_draft(tmp_path: Path) -> None:
    """gh leaves a draft when an upload fails; a re-run completes it."""
    gh = _FakeGh({"isDraft": True, "assets": []})
    assert _run(tmp_path, gh) == 0
    assert _verbs(gh) == ["view", "upload", "edit", "view"]
    assert "--clobber" in gh.calls[1]
    assert gh.release is not None and gh.release["isDraft"] is False


def test_refuses_a_tag_holding_different_bytes(tmp_path: Path) -> None:
    other = {"name": "makemodel_b0.pt", "size": 7000, "state": "uploaded", "digest": "sha256:0"}
    gh = _FakeGh({"isDraft": False, "assets": [other]})
    assert _run(tmp_path, gh) == 1
    assert _verbs(gh) == ["view"]  # never overwritten


@pytest.mark.parametrize(
    "ckpt",
    [
        {"metadata": {"val_cars": ["AB12 CDE"]}},  # a plate in the metadata
        {"metadata": {"corpus": "D:\\Projects\\StreetTracker\\runs\\uk_crops"}},  # a dev-box path
        {"AB12CDE": 1},  # a plate as a dict key
    ],
)
def test_refuses_a_checkpoint_carrying_plates_or_paths(tmp_path: Path, ckpt: Any) -> None:
    gh = _FakeGh()
    assert _run(tmp_path, gh, ckpt) == 1
    assert gh.calls == []  # nothing reaches GitHub


def test_dry_run_checks_but_uploads_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gh = _FakeGh()
    assert _run(tmp_path, gh, _CLEAN_CKPT, "--dry-run") == 0
    assert gh.calls == []
    out = capsys.readouterr().out
    assert "uk_crops_0929_576" in out and "+1.3 pp per track" in out
    assert hashlib.sha256((tmp_path / "makemodel_b0.pt").read_bytes()).hexdigest() in out


def test_gh_failure_other_than_not_found_stops(tmp_path: Path) -> None:
    gh = _FakeGh(view_error="HTTP 401: Bad credentials")
    assert _run(tmp_path, gh) == 1
    assert _verbs(gh) == ["view"]


def test_unsafe_strings_walks_keys_values_and_lists() -> None:
    assert mr.unsafe_strings(_CLEAN_CKPT, check_paths=True) == []
    assert mr.unsafe_strings({"a": [{"b": "ab12cde"}]}, check_paths=False) == ["ab12cde"]
    # Paths only count when asked: the sidecar legitimately records one.
    sidecar_like = {"production": {"path": "D:\\Projects\\x.pt"}}
    assert mr.unsafe_strings(sidecar_like, check_paths=False) == []
    assert mr.unsafe_strings(sidecar_like, check_paths=True) == ["D:\\Projects\\x.pt"]
