"""Build a ready-to-paste Claude Code prompt when a process needs attention.

The control panel runs StreetTracker's operational commands tokenlessly. When
one **fails** — or finishes but shows a sign of room for improvement — the
operator shouldn't have to hand-write a help request. This module turns a
finished job into a *complete, self-contained* prompt the operator can copy
straight into Claude Code: it names the project, the exact command, the exit
code, the captured output tail, and a command-specific pointer, then asks for a
diagnosis and fix.

Pure and deterministic (no I/O, no clock): a :class:`JobReport` in, a string
out, so the wording is fully unit-tested. :func:`summarize_issue` is the
matching pure check the job runner uses to decide *whether* there's something
worth surfacing a prompt for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["JobReport", "build_prompt", "summarize_issue"]

# Benign "N errors" / "N api errors" summary counts several commands print on a
# clean finish (e.g. dvsa-label's "…, 0 api errors …"). A ZERO count is not a
# problem, but the bare "error" catch-all below would otherwise match the word
# "errors" and flag every successful run. Strip zero-count summaries before the
# scan; a nonzero count ("3 api errors") is left intact so it still trips.
_ZERO_ERROR_COUNT_RE = re.compile(r"\b0 (?:\w+ )?errors?\b")

# Lines of captured output to embed. Enough to carry a full Python traceback
# without burying the ask.
DEFAULT_TAIL = 40

# Ordered strongest-first: the first pattern found names the issue. Patterns are
# matched case-insensitively against the captured output.
_ISSUE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cuda out of memory", "CUDA out of memory"),
    ("outofmemoryerror", "out of memory"),
    ("traceback (most recent call last)", "Python traceback in output"),
    ("configerror", "config validation error"),
    ("modulenotfounderror", "missing Python module"),
    ("permission denied", "permission denied"),
    ("connection refused", "connection refused"),
    ("no such file or directory", "missing file/path"),
    ("ssh: connect to host", "could not reach the device over SSH"),
    ("exception", "exception reported in output"),
    ("error", "error reported in output"),
)

# Command-specific pointer appended to the prompt so Claude starts in the right
# place. Keyed by job kind; unknown kinds get no extra pointer.
_KIND_HINTS: dict[str, str] = {
    "pull": (
        "This shells out to ssh/scp/sftp to the Orin. Check the device is reachable "
        "(`ssh -i ~/.ssh/streettracker streettracker@orin`), the key path, and the remote "
        "output dir."
    ),
    "alpr-run": (
        "This needs the `alpr` extra (`uv sync --extra alpr`) and reads the ghost mask + "
        "per-snap bboxes. Check the model files under analysis/alpr/models/ and the session's "
        "snaps."
    ),
    "dvsa-label": (
        "This calls the DVSA MOT API and needs configs/dvsa.json credentials. Check the config "
        "and the API responses."
    ),
    "dvsa-apply": "This folds DVSA labels onto per-track records; check the session's "
    "_dvsa_labels.json exists.",
    "vehicles": "This aggregates per-vehicle records from data.json + alpr_by_track + dvsa_labels.",
    "makemodel": (
        "This runs the make/model CNN on the session's snaps. Check the model checkpoint "
        "(analysis/makemodel/models/) and CUDA availability."
    ),
    "makemodel-build-uk": (
        "This extracts DVSA-labelled crops into a runs/uk_crops_* dataset. Check the labelled "
        "sessions and free disk space."
    ),
    "makemodel-train-uk": (
        "This trains the UK make classifier on the dev-box GPU. Check CUDA/torch, the corpus "
        "manifest, and runs/ output. Note the dev box idle-sleeps — long jobs need the "
        "wake-lock."
    ),
}


@dataclass
class JobReport:
    """The finished-job context a prompt is built from.

    ``command`` is the display command (e.g. ``uv run streettracker pull
    --session …``); ``log_tail`` is the most recent captured output lines.
    """

    kind: str
    command: str
    cwd: str
    status: str  # "failed" | "succeeded" | "cancelled"
    returncode: int | None = None
    elapsed_s: float | None = None
    log_tail: list[str] = field(default_factory=list)
    issue: str | None = None
    phase: str = ""
    summary: str = ""


def summarize_issue(returncode: int | None, log_lines: list[str]) -> str | None:
    """Short reason a job is worth a help prompt, or ``None`` if it's clean.

    A non-zero exit always qualifies; otherwise the captured output is scanned
    for known trouble markers (traceback, OOM, config error, …) so a command
    that "succeeds" but logged an exception still surfaces.
    """
    if returncode is not None and returncode != 0:
        return f"process exited with code {returncode}"
    haystack = _ZERO_ERROR_COUNT_RE.sub("", "\n".join(log_lines).lower())
    for needle, label in _ISSUE_PATTERNS:
        if needle in haystack:
            return label
    return None


def _fmt_dur(s: float | None) -> str:
    if s is None:
        return "unknown"
    s = max(0, int(s))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _opening(report: JobReport) -> str:
    if report.status == "failed":
        return (
            f"I'm running StreetTracker's `{report.kind}` step from the operator control panel "
            "and it failed. Please diagnose the root cause and fix it."
        )
    if report.status == "cancelled":
        return (
            f"I cancelled StreetTracker's `{report.kind}` step partway through the operator "
            "control panel. Here's where it was — please check whether anything needs cleaning "
            "up before I re-run it."
        )
    # succeeded, but summarize_issue flagged something
    return (
        f"StreetTracker's `{report.kind}` step finished from the operator control panel, but "
        "something looks off and there may be room for improvement. Please review and advise "
        "(and fix if warranted)."
    )


def build_prompt(report: JobReport, *, tail: int = DEFAULT_TAIL) -> str:
    """Render a full, self-contained Claude Code prompt for ``report``.

    The output stands alone — pasting it into a fresh session gives Claude the
    project, the exact command, the outcome, the captured output, and the ask.
    """
    lines: list[str] = [_opening(report), ""]

    lines.append("Command:")
    lines.append(f"    {report.command}")
    lines.append(f"    (working dir: {report.cwd})")
    lines.append("")

    rc = "—" if report.returncode is None else str(report.returncode)
    lines.append(f"Outcome: {report.status}, exit code {rc}, ran ~{_fmt_dur(report.elapsed_s)}.")
    if report.issue:
        lines.append(f"Detected issue: {report.issue}.")
    if report.phase:
        lines.append(f"Last phase: {report.phase}.")
    lines.append("")

    tail_lines = report.log_tail[-tail:] if tail > 0 else list(report.log_tail)
    if tail_lines:
        lines.append(f"Last {len(tail_lines)} line(s) of output:")
        lines.append("```")
        lines.extend(tail_lines)
        lines.append("```")
    else:
        lines.append("(no output was captured)")
    lines.append("")

    hint = _KIND_HINTS.get(report.kind)
    closing = (
        "This is the StreetTracker project; the conventions, deploy procedures and tuning "
        "history are in CLAUDE.md. "
    )
    if hint:
        closing += hint + " "
    if report.status == "failed":
        closing += "Investigate, fix the underlying problem, and re-run to confirm."
    else:
        closing += "Assess whether action is warranted and explain what you'd change."
    lines.append(closing)

    return "\n".join(lines)
