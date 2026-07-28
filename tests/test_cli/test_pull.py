"""Tests for ``streettracker pull``.

The pull CLI is mostly an orchestrator around ``ssh`` and ``scp``.
These tests cover the pure helpers (inventory parsing, scp command
construction, byte formatting) and patch ``subprocess.run`` for the
network-touching paths.
"""

from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from streettracker.cli import pull

# A minimal well-formed JPEG: SOI ... EOI. The integrity gate in
# sftp_get_missing only checks the first/last two bytes.
_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01payload\xff\xd9"

_GET_RE = re.compile(r'get -p "([^"]+)"')


def _sftp_stub(
    local_session: Path,
    captured: dict[str, Any],
    *,
    land: set[str] | None = None,
    returncode: int = 0,
) -> Any:
    """Fake ``sftp -b -``: record each batch body and write an intact JPEG
    for every file it asked for, so the caller's landed-count check sees
    them arrive. ``land`` restricts which names actually materialise --
    used to simulate a snap pruned on the device mid-pull."""
    lock = threading.Lock()

    def _run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        body = kwargs.get("input", "")
        names = _GET_RE.findall(body)
        with lock:
            captured["args"] = args
            captured.setdefault("bodies", []).append(body)
            captured["input"] = captured.get("input", "") + body
            captured.setdefault("names", []).extend(names)
        for name in names:
            if land is None or name in land:
                (local_session / name).write_bytes(_JPEG)
        return _fake_completed(returncode=returncode)

    return _run


def test_human_bytes_scales_through_units() -> None:
    assert pull.human_bytes(0) == "0.0 B"
    assert pull.human_bytes(512) == "512.0 B"
    assert pull.human_bytes(2048) == "2.0 KB"
    assert pull.human_bytes(5 * 1024 * 1024) == "5.0 MB"
    assert pull.human_bytes(3 * 1024**3) == "3.0 GB"


def test_scp_commands_full_pull_uses_recursive_scp(tmp_path: Path) -> None:
    cmds = pull._scp_commands(
        host="orin",
        user="streettracker",
        key="/k/id",
        remote_path="/srv/output/session_x",
        local_parent=tmp_path,
        only_main=False,
    )
    assert len(cmds) == 1
    args = cmds[0]
    assert args[0] == "scp"
    assert "-r" in args
    assert "streettracker@orin:/srv/output/session_x" in args
    assert str(tmp_path) in args


def test_scp_commands_only_main_uses_pattern_list(tmp_path: Path) -> None:
    cmds = pull._scp_commands(
        host="orin",
        user="streettracker",
        key="/k/id",
        remote_path="/srv/output/session_x",
        local_parent=tmp_path,
        only_main=True,
    )
    # one command per pattern; thumbs (<id>.jpg) and HQ (<id>_hq.jpg) excluded
    patterns_seen = [c[-2].split(":")[-1].rsplit("/", 1)[-1] for c in cmds]
    assert "*_main_*.jpg" in patterns_seen
    assert "*.json" in patterns_seen
    assert "*.jsonl" in patterns_seen
    assert "*_summary.html" in patterns_seen
    assert "index.html" in patterns_seen
    assert "*_hq.jpg" not in patterns_seen
    # session subdir was created
    assert (tmp_path / "session_x").is_dir()


def _fake_completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_find_latest_session_returns_remote_basename() -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed("session_20260518_141443\n")
        result = pull.find_latest_session("orin", "u", "/k", "/srv/output")
    assert result == "session_20260518_141443"
    # ssh got the right host/user
    args = mock_run.call_args.args[0]
    assert args[0] == "ssh"
    assert "u@orin" in args


def test_find_latest_session_exits_on_empty_listing() -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed("")
        with pytest.raises(SystemExit):
            pull.find_latest_session("orin", "u", "/k", "/srv/output")


def test_remote_inventory_parses_du_and_find_output() -> None:
    fake_out = "BYTES 1572864\nFILES 42\nMAIN 7\nHQ 14\nJSONL 1\n"
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(fake_out)
        inv = pull.remote_inventory("orin", "u", "/k", "/srv/output/session_x")
    assert inv.bytes == 1572864
    assert inv.files == 42
    assert inv.main_snaps == 7
    assert inv.hq_crops == 14
    assert inv.jsonl == 1


def test_remote_inventory_tolerates_garbage_lines() -> None:
    """Out-of-order or unknown labels are ignored, not fatal."""
    fake_out = "BYTES 100\nUNKNOWN 9999\nMAIN garbage\nFILES 3\n"
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(fake_out)
        inv = pull.remote_inventory("orin", "u", "/k", "/x")
    assert inv.bytes == 100
    assert inv.files == 3
    assert inv.main_snaps == 0


def test_remote_inventory_parses_vehicle_main_split() -> None:
    """VEHMAIN gives the vehicle subset of MAIN (person = main - vehicle)."""
    fake_out = "MAIN 10\nVEHMAIN 4\n"
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(fake_out)
        inv = pull.remote_inventory("orin", "u", "/k", "/x")
    assert inv.main_snaps == 10
    assert inv.vehicle_main_snaps == 4


def test_remote_inventory_parses_main_bytes() -> None:
    """MAINBYTES gives the --only-main payload size for the pull ETA."""
    fake_out = "BYTES 1000\nMAINBYTES 600\nMAIN 4\n"
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(fake_out)
        inv = pull.remote_inventory("orin", "u", "/k", "/x")
    assert inv.bytes == 1000
    assert inv.main_bytes == 600
    assert inv.main_snaps == 4


def test_scp_pull_dry_run_does_not_invoke_subprocess(tmp_path: Path) -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        pull.scp_pull(
            host="orin",
            user="u",
            key="/k",
            remote_path="/srv/output/session_x",
            local_parent=tmp_path,
            only_main=False,
            dry_run=True,
        )
    mock_run.assert_not_called()


def test_scp_pull_runs_scp_when_not_dry_run(tmp_path: Path) -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(returncode=0)
        pull.scp_pull(
            host="orin",
            user="u",
            key="/k",
            remote_path="/srv/output/session_x",
            local_parent=tmp_path,
            only_main=False,
            dry_run=False,
        )
    assert mock_run.call_count == 1
    assert mock_run.call_args.args[0][0] == "scp"


def test_scp_pull_only_main_tolerates_missing_patterns(tmp_path: Path) -> None:
    """With --only-main, a non-zero scp exit (no files matched a pattern)
    is acceptable — sessions sometimes have no main snaps."""
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(returncode=1)
        pull.scp_pull(
            host="orin",
            user="u",
            key="/k",
            remote_path="/srv/output/session_x",
            local_parent=tmp_path,
            only_main=True,
            dry_run=False,
        )


def test_scp_pull_full_pull_exits_on_scp_failure(tmp_path: Path) -> None:
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(returncode=1)
        with pytest.raises(SystemExit):
            pull.scp_pull(
                host="orin",
                user="u",
                key="/k",
                remote_path="/srv/output/session_x",
                local_parent=tmp_path,
                only_main=False,
                dry_run=False,
            )


def test_main_returns_1_when_key_missing(tmp_path: Path) -> None:
    fake_key = tmp_path / "does_not_exist"
    rc = pull.main(["--key", str(fake_key), "--session", "session_x"])
    assert rc == 1


def test_main_dry_run_end_to_end(tmp_path: Path) -> None:
    """Full happy-path dry-run: key exists, ssh stubbed, no scp invoked."""
    fake_key = tmp_path / "id"
    fake_key.write_text("not-a-real-key")

    def _ssh_stub(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # remote_inventory; find_latest_session is skipped because we pass --session
        return _fake_completed("BYTES 0\nFILES 0\nMAIN 0\nHQ 0\nJSONL 0\n")

    with patch("streettracker.cli.pull.subprocess.run", side_effect=_ssh_stub):
        rc = pull.main(
            [
                "--key",
                str(fake_key),
                "--session",
                "session_test",
                "--target",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
    assert rc == 0


# ----------------------------------------------------------------------
# --skip-existing: incremental pull (name-diff the immutable snaps).


def test_immutable_image_patterns_by_mode() -> None:
    assert pull._immutable_image_patterns(only_main=True) == ("*_main_*.jpg",)
    assert pull._immutable_image_patterns(only_main=False) == ("*.jpg",)


def test_remote_entries_uses_find_not_glob_and_parses_mtimes() -> None:
    """Regression: a busy session holds 40k+ snaps, so the enumeration
    must NOT shell-expand ``ls -1 *.jpg`` (overflows ARG_MAX -> silent
    empty -> skip-existing fetches nothing). It must use a server-side
    ``find ... -printf`` that matches the pattern as a single argument --
    and emit the mtime alongside the name so the fetch can be age-ordered."""
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(
            "1784631599.5287271520 vehicle_1_main_1.jpg\n1785179818.7 vehicle_2_main_1.jpg\n"
        )
        entries = pull._remote_entries("orin", "u", "/k", "/srv/output/session_x", "*.jpg")
    assert entries == {
        "vehicle_1_main_1.jpg": pytest.approx(1784631599.528727),
        "vehicle_2_main_1.jpg": pytest.approx(1785179818.7),
    }
    remote_cmd = mock_run.call_args.args[0][-1]  # ssh's trailing command string
    assert "find . -maxdepth 1 -name" in remote_cmd
    assert "-printf" in remote_cmd
    assert "%T@" in remote_cmd  # mtime is what makes oldest-first possible
    assert "ls -1 *.jpg" not in remote_cmd  # the overflow-prone form is gone


def test_remote_entries_skips_malformed_lines() -> None:
    """A line without a stamp/name split, or a non-numeric stamp, is dropped
    rather than crashing the whole enumeration."""
    with patch("streettracker.cli.pull.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(
            "\nnostamp\nnotanumber vehicle_9_main_1.jpg\n123.5 vehicle_1_main_1.jpg\n"
        )
        entries = pull._remote_entries("orin", "u", "/k", "/p", "*.jpg")
    assert entries == {"vehicle_1_main_1.jpg": pytest.approx(123.5)}


def test_stripe_deals_round_robin_preserving_order() -> None:
    """Striping (not contiguous slicing) is what keeps a parallel fetch
    age-ordered overall: each worker walks oldest->newest in step."""
    assert pull._stripe(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]]
    assert pull._stripe(["a", "b"], 3) == [["a"], ["b"], []]
    assert pull._stripe(["a", "b", "c"], 1) == [["a", "b", "c"]]


def test_sftp_get_missing_fetches_only_new(tmp_path: Path) -> None:
    """The name diff fetches just the remote files absent locally; an
    already-present snap is skipped (immutable -> no re-transfer)."""
    local_session = tmp_path / "session_x"
    local_session.mkdir()
    (local_session / "vehicle_1_main_1.jpg").write_bytes(_JPEG)
    remote = {
        "vehicle_1_main_1.jpg": 100.0,
        "vehicle_2_main_1.jpg": 200.0,
        "vehicle_3_main_1.jpg": 300.0,
    }
    captured: dict[str, Any] = {}

    with (
        patch("streettracker.cli.pull._remote_entries", return_value=remote),
        patch(
            "streettracker.cli.pull.subprocess.run",
            side_effect=_sftp_stub(local_session, captured),
        ),
    ):
        n = pull.sftp_get_missing(
            "orin",
            "u",
            "/k",
            "/srv/output/session_x",
            local_session,
            "*_main_*.jpg",
            dry_run=False,
            jobs=1,
        )
    assert n == 2
    assert captured["args"][0] == "sftp"
    assert "-b" in captured["args"]
    body = captured["input"]
    assert 'get -p "vehicle_2_main_1.jpg"' in body
    assert 'get -p "vehicle_3_main_1.jpg"' in body
    assert "vehicle_1_main_1.jpg" not in body  # already local -> not re-fetched


def test_sftp_get_missing_fetches_oldest_first(tmp_path: Path) -> None:
    """The device prunes snaps by mtime, so the fetch order must be age
    ascending -- an interrupted pull then loses only the newest tail, which
    a re-run can still recover. Remote name order must not leak through."""
    local_session = tmp_path / "session_x"
    local_session.mkdir()
    # Newest snap sorts FIRST alphabetically, so a name sort would invert age.
    remote = {
        "vehicle_1_main_1.jpg": 900.0,  # newest
        "vehicle_2_main_1.jpg": 100.0,  # oldest
        "vehicle_3_main_1.jpg": 500.0,
    }
    captured: dict[str, Any] = {}

    with (
        patch("streettracker.cli.pull._remote_entries", return_value=remote),
        patch(
            "streettracker.cli.pull.subprocess.run",
            side_effect=_sftp_stub(local_session, captured),
        ),
    ):
        pull.sftp_get_missing(
            "orin", "u", "/k", "/p", local_session, "*_main_*.jpg", dry_run=False, jobs=1
        )
    assert captured["names"] == [
        "vehicle_2_main_1.jpg",
        "vehicle_3_main_1.jpg",
        "vehicle_1_main_1.jpg",
    ]


def test_sftp_get_missing_ignores_per_file_errors(tmp_path: Path) -> None:
    """Each get is ``-`` prefixed so a snap pruned between the enumeration
    and its transfer cannot abort the batch and strand every later file."""
    local_session = tmp_path / "session_x"
    local_session.mkdir()
    captured: dict[str, Any] = {}
    with (
        patch("streettracker.cli.pull._remote_entries", return_value={"vehicle_9_main_1.jpg": 1.0}),
        patch(
            "streettracker.cli.pull.subprocess.run",
            side_effect=_sftp_stub(local_session, captured),
        ),
    ):
        pull.sftp_get_missing(
            "orin", "u", "/k", "/p", local_session, "*_main_*.jpg", dry_run=False, jobs=1
        )
    assert '-get -p "vehicle_9_main_1.jpg"' in captured["input"]


def test_sftp_get_missing_splits_across_parallel_streams(tmp_path: Path) -> None:
    """--jobs N fans the age-ordered list across N sftp connections, striped
    so every stream advances through the age range together."""
    local_session = tmp_path / "session_x"
    local_session.mkdir()
    remote = {f"vehicle_{i}_main_1.jpg": float(i) for i in range(1, 7)}
    captured: dict[str, Any] = {}

    with (
        patch("streettracker.cli.pull._remote_entries", return_value=remote),
        patch(
            "streettracker.cli.pull.subprocess.run",
            side_effect=_sftp_stub(local_session, captured),
        ),
    ):
        n = pull.sftp_get_missing(
            "orin", "u", "/k", "/p", local_session, "*_main_*.jpg", dry_run=False, jobs=3
        )
    assert n == 6
    assert len(captured["bodies"]) == 3  # three separate sftp batches
    batches = [_GET_RE.findall(b) for b in captured["bodies"]]
    assert sorted(sum(batches, [])) == sorted(remote)  # every file dealt exactly once
    # Striped, not sliced: the oldest three land in three different streams.
    assert sorted(b[0] for b in batches) == [
        "vehicle_1_main_1.jpg",
        "vehicle_2_main_1.jpg",
        "vehicle_3_main_1.jpg",
    ]


def test_sftp_get_missing_reports_only_what_landed(tmp_path: Path) -> None:
    """A file pruned on the device mid-pull does not arrive; the return
    count reflects reality rather than the size of the request."""
    local_session = tmp_path / "session_x"
    local_session.mkdir()
    remote = {"vehicle_1_main_1.jpg": 1.0, "vehicle_2_main_1.jpg": 2.0}
    captured: dict[str, Any] = {}

    with (
        patch("streettracker.cli.pull._remote_entries", return_value=remote),
        patch(
            "streettracker.cli.pull.subprocess.run",
            side_effect=_sftp_stub(local_session, captured, land={"vehicle_1_main_1.jpg"}),
        ),
    ):
        n = pull.sftp_get_missing(
            "orin", "u", "/k", "/p", local_session, "*_main_*.jpg", dry_run=False, jobs=1
        )
    assert n == 1  # requested 2, one was pruned away before its get


def test_is_intact_jpeg_rejects_corruption(tmp_path: Path) -> None:
    ok = tmp_path / "ok.jpg"
    ok.write_bytes(_JPEG)
    assert pull._is_intact_jpeg(ok) is True
    # Full-size zero-filled corpse from an interrupted transfer.
    zeroed = tmp_path / "zeroed.jpg"
    zeroed.write_bytes(b"\x00" * 4096)
    assert pull._is_intact_jpeg(zeroed) is False
    # Header present, EOI missing (tail-truncated mid-write).
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(b"\xff\xd8\xff\xe0" + b"\x11" * 4096)
    assert pull._is_intact_jpeg(truncated) is False
    # Absent file.
    assert pull._is_intact_jpeg(tmp_path / "nope.jpg") is False


def test_sftp_get_missing_refetches_corrupt_local(tmp_path: Path) -> None:
    """A locally-present but corrupt snap (zero-filled / truncated) is
    re-fetched, overwriting the corpse -- name presence alone is not
    trusted. Regression guard for the 2026-07-28 zero-filled-snap bug."""
    local_session = tmp_path / "session_x"
    local_session.mkdir()
    (local_session / "vehicle_1_main_1.jpg").write_bytes(_JPEG)  # intact -> skip
    (local_session / "vehicle_2_main_1.jpg").write_bytes(b"\x00" * 2048)  # zeroed -> refetch
    (local_session / "vehicle_3_main_1.jpg").write_bytes(b"\xff\xd8bad")  # no EOI -> refetch
    remote = {
        "vehicle_1_main_1.jpg": 100.0,
        "vehicle_2_main_1.jpg": 200.0,
        "vehicle_3_main_1.jpg": 300.0,
    }
    captured: dict[str, Any] = {}

    with (
        patch("streettracker.cli.pull._remote_entries", return_value=remote),
        patch(
            "streettracker.cli.pull.subprocess.run",
            side_effect=_sftp_stub(local_session, captured),
        ),
    ):
        n = pull.sftp_get_missing(
            "orin",
            "u",
            "/k",
            "/srv/output/session_x",
            local_session,
            "*_main_*.jpg",
            dry_run=False,
            jobs=1,
        )
    assert n == 2
    body = captured["input"]
    assert 'get -p "vehicle_2_main_1.jpg"' in body  # zeroed re-fetched
    assert 'get -p "vehicle_3_main_1.jpg"' in body  # truncated re-fetched
    assert "vehicle_1_main_1.jpg" not in body  # intact -> not re-fetched


def test_sftp_get_missing_nothing_new_skips_sftp(tmp_path: Path) -> None:
    local_session = tmp_path / "session_x"
    local_session.mkdir()
    (local_session / "vehicle_1_main_1.jpg").write_bytes(_JPEG)
    with (
        patch("streettracker.cli.pull._remote_entries", return_value={"vehicle_1_main_1.jpg": 1.0}),
        patch("streettracker.cli.pull.subprocess.run") as mock_run,
    ):
        n = pull.sftp_get_missing(
            "orin",
            "u",
            "/k",
            "/p",
            local_session,
            "*_main_*.jpg",
            dry_run=False,
        )
    assert n == 0
    mock_run.assert_not_called()


def test_sftp_get_missing_dry_run_skips_transfer(tmp_path: Path) -> None:
    local_session = tmp_path / "session_x"
    local_session.mkdir()
    with (
        patch("streettracker.cli.pull._remote_entries", return_value={"vehicle_9_main_1.jpg": 1.0}),
        patch("streettracker.cli.pull.subprocess.run") as mock_run,
    ):
        n = pull.sftp_get_missing(
            "orin",
            "u",
            "/k",
            "/p",
            local_session,
            "*_main_*.jpg",
            dry_run=True,
        )
    assert n == 0
    mock_run.assert_not_called()


def test_skip_existing_pull_routes_images_to_sftp_metadata_to_scp(tmp_path: Path) -> None:
    """Immutable snaps go through the sftp name-diff; mutable JSON/jsonl/
    HTML are re-fetched in full via scp."""
    scp_targets: list[str] = []

    def _capture_scp(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        scp_targets.append(args[-2])  # "user@host:remote/<pattern>"
        return _fake_completed(returncode=0)

    with (
        patch("streettracker.cli.pull.sftp_get_missing", return_value=3) as mock_sftp,
        patch("streettracker.cli.pull.subprocess.run", side_effect=_capture_scp),
    ):
        pull.skip_existing_pull(
            "orin",
            "u",
            "/k",
            "/srv/output/session_x",
            tmp_path,
            only_main=True,
            dry_run=False,
        )
    mock_sftp.assert_called_once()
    assert mock_sftp.call_args.args[-1] == "*_main_*.jpg"  # the immutable glob
    scp_patterns = [t.rsplit("/", 1)[-1] for t in scp_targets]
    assert scp_patterns == ["*.json", "*.jsonl", "*_summary.html", "index.html"]


def test_main_skip_existing_dry_run(tmp_path: Path) -> None:
    """--skip-existing dry-run happy path returns 0 and transfers nothing."""
    fake_key = tmp_path / "id"
    fake_key.write_text("not-a-real-key")

    def _ssh_stub(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # remote_inventory + the find inside _remote_entries both land here.
        return _fake_completed("BYTES 0\nFILES 0\nMAIN 0\nHQ 0\nJSONL 0\n")

    with patch("streettracker.cli.pull.subprocess.run", side_effect=_ssh_stub):
        rc = pull.main(
            [
                "--key",
                str(fake_key),
                "--session",
                "session_test",
                "--target",
                str(tmp_path / "out"),
                "--only-main",
                "--skip-existing",
                "--dry-run",
            ]
        )
    assert rc == 0
