"""Tests for the copy-pasteable Claude prompt builder."""

from __future__ import annotations

from streettracker.control.prompts import JobReport, build_prompt, summarize_issue


def test_summarize_issue_nonzero_exit() -> None:
    assert summarize_issue(1, ["all fine"]) == "process exited with code 1"
    assert summarize_issue(255, []) == "process exited with code 255"


def test_summarize_issue_clean() -> None:
    assert summarize_issue(0, ["[pull] done -> /output/x"]) is None
    assert summarize_issue(None, []) is None


def test_summarize_issue_scans_output_on_clean_exit() -> None:
    assert summarize_issue(0, ["...", "Traceback (most recent call last):"]) == (
        "Python traceback in output"
    )
    assert (
        summarize_issue(0, ["RuntimeError: CUDA out of memory. Tried..."]) == "CUDA out of memory"
    )


def test_summarize_issue_priority_order() -> None:
    # Both an OOM and a generic "error" present -> the stronger, specific one wins.
    lines = ["error: something", "CUDA out of memory"]
    assert summarize_issue(0, lines) == "CUDA out of memory"


def _report(**kw: object) -> JobReport:
    base: dict[str, object] = {
        "kind": "pull",
        "command": "uv run streettracker pull --session session_x --only-main",
        "cwd": "D:\\Projects\\StreetTracker",
        "status": "failed",
        "returncode": 1,
        "elapsed_s": 73.0,
        "log_tail": ["[pull] session: session_x", "ssh: connect to host orin port 22: timed out"],
        "issue": "could not reach the device over SSH",
    }
    base.update(kw)
    return JobReport(**base)  # type: ignore[arg-type]


def test_build_prompt_failed_is_self_contained() -> None:
    text = build_prompt(_report())
    # Names the command + cwd.
    assert "uv run streettracker pull --session session_x --only-main" in text
    assert "D:\\Projects\\StreetTracker" in text
    # Outcome + issue.
    assert "exit code 1" in text
    assert "could not reach the device over SSH" in text
    # Embeds the captured output in a fenced block.
    assert "```" in text
    assert "ssh: connect to host orin port 22: timed out" in text
    # Kind-specific pointer + the project reference + the ask.
    assert "ssh" in text.lower()
    assert "CLAUDE.md" in text
    assert "failed" in text.lower()
    assert "re-run to confirm" in text


def test_build_prompt_succeeded_with_issue_framing() -> None:
    text = build_prompt(_report(status="succeeded", returncode=0, issue="error reported in output"))
    assert "room for improvement" in text
    assert "Assess whether action is warranted" in text


def test_build_prompt_cancelled_framing() -> None:
    text = build_prompt(_report(status="cancelled", returncode=None))
    assert "cancelled" in text.lower()
    assert "exit code —" in text


def test_build_prompt_truncates_log_tail() -> None:
    lines = [f"line {i}" for i in range(100)]
    text = build_prompt(_report(log_tail=lines), tail=10)
    assert "line 99" in text
    assert "line 90" in text  # last 10 of 0..99 is 90..99
    assert "line 89" not in text  # earlier lines dropped
    assert "Last 10 line(s) of output:" in text


def test_build_prompt_no_output() -> None:
    text = build_prompt(_report(log_tail=[]))
    assert "(no output was captured)" in text


def test_build_prompt_unknown_kind_has_no_hint_but_still_valid() -> None:
    text = build_prompt(_report(kind="recolor", command="uv run streettracker recolor x"))
    assert "uv run streettracker recolor x" in text
    assert "CLAUDE.md" in text
