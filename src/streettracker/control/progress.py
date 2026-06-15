"""Per-command stdout progress parsers for the control-panel job runner.

Phase 2 runs the operational commands as subprocesses and streams their stdout
to the dashboard. These parsers turn that text into structured live progress —
a phase label, a current/total bar, per-epoch training metrics, and a final
summary — so the UI can show progress bars, ETAs and a live loss curve.

Design choices that keep this layer trustworthy and easy to test:

* **Pure text → numbers.** A parser is fed one line at a time and updates a
  :class:`Progress`. It holds **no clock and no I/O**, so every parser is
  deterministic under a canned transcript. Wall-clock concerns (ETA, the pull
  byte-bar that watches the local directory grow) live in the job runner, which
  calls :func:`estimate_eta` with its own timing.
* **One parser per command kind**, chosen by :func:`make_parser`. Unknown kinds
  fall back to :class:`GenericParser`, which still captures the final summary
  line — so a newly-wrapped command degrades gracefully rather than erroring.

The exact line formats parsed here are emitted (with ``flush=True``) by the
respective CLIs: ``cli/alpr_run.py``, ``analysis/makemodel/infer.py``,
``analysis/makemodel/uk_dataset.py`` (build + train), ``cli/pull.py`` and
``cli/dvsa_label.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Progress", "ProgressParser", "estimate_eta", "make_parser"]


@dataclass
class Progress:
    """Live state a parser maintains across a command's output.

    ``current``/``total`` drive a progress bar (``frac`` is derived);
    ``metrics`` carries structured extras (e.g. the per-epoch training rows for
    the live loss curve); ``summary`` is the last notable line, surfaced when
    the command finishes.
    """

    phase: str = ""
    current: int | None = None
    total: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    ok: bool | None = None  # set by finish(); True on clean exit
    summary: str = ""

    @property
    def frac(self) -> float | None:
        if self.total and self.total > 0 and self.current is not None:
            return max(0.0, min(1.0, self.current / self.total))
        return None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "current": self.current,
            "total": self.total,
            "frac": self.frac,
            "metrics": self.metrics,
            "done": self.done,
            "ok": self.ok,
            "summary": self.summary,
        }


def estimate_eta(current: int | None, total: int | None, elapsed_s: float) -> float | None:
    """Seconds remaining, extrapolated from the average rate so far.

    Returns ``0.0`` once complete, and ``None`` when there isn't enough signal
    to extrapolate (no total, nothing done yet, or zero elapsed time). Pure: the
    caller supplies ``elapsed_s`` from its own clock.
    """
    if total is None or current is None or total <= 0:
        return None
    if current >= total:
        return 0.0
    if current <= 0 or elapsed_s <= 0:
        return None
    rate = current / elapsed_s  # units/sec
    if rate <= 0:
        return None
    return (total - current) / rate


class ProgressParser:
    """Base parser: collects ``[tag] …`` lines as the rolling summary.

    Subclasses override :meth:`feed` for command-specific extraction; the base
    behaviour already gives a sensible result for short, summary-only commands
    (``dvsa-apply``, ``vehicles``).
    """

    kind = "generic"

    def __init__(self) -> None:
        self.progress = Progress()

    def feed(self, line: str) -> None:
        text = line.strip()
        if text.startswith("["):
            # Keep the most recent bracketed status line as the summary.
            self.progress.summary = text

    def finish(self, returncode: int) -> None:
        self.progress.done = True
        self.progress.ok = returncode == 0
        if returncode != 0 and not self.progress.phase:
            self.progress.phase = "failed"


class GenericParser(ProgressParser):
    """Explicit alias for the default behaviour (summary-only commands)."""


# ----------------------------------------------------------------------
# Counted batch commands: alpr-run + makemodel inference
# ----------------------------------------------------------------------

# "  [preferred] 120/1200 done" / "[bespoke] …" / "[batch] 120/1200 done"
_BATCH_RE = re.compile(r"\[(preferred|bespoke|batch)\]\s+(\d+)\s*/\s*(\d+)\s+done")
_PIPELINE_RE = re.compile(r"running pipeline:\s+(\S+)")
_WROTE_RE = re.compile(r"\bwrote\b\s+(.+)$")


class BatchParser(ProgressParser):
    """``alpr-run`` and ``makemodel`` both print ``[tag] N/total done``."""

    kind = "batch"

    def feed(self, line: str) -> None:
        text = line.strip()
        m = _BATCH_RE.search(text)
        if m:
            tag, cur, tot = m.group(1), int(m.group(2)), int(m.group(3))
            self.progress.current = cur
            self.progress.total = tot
            self.progress.phase = f"{tag} {cur}/{tot}"
            return
        mp = _PIPELINE_RE.search(text)
        if mp:
            self.progress.phase = f"pipeline {mp.group(1)}"
            return
        mw = _WROTE_RE.search(text)
        if mw:
            self.progress.summary = text


# ----------------------------------------------------------------------
# UK make-classifier training (the live loss curve)
# ----------------------------------------------------------------------

_TRAIN_PREAMBLE_RE = re.compile(
    r"device=(\S+)\s+amp=(\S+)\s+makes=(\d+)\s+train_crops=(\d+)\s+val_crops=(\d+)"
)
_TRAIN_EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s*/\s*(\d+)\s+loss=([\d.]+)\s+make@1=([\d.]+)\s+make@5=([\d.]+)\s+\((\d+)s\)"
)
_TRAIN_DONE_RE = re.compile(r"done:\s+best make@1=([\d.]+)\s+\(epoch\s+(\d+)\)\s+->\s+(.+)$")
_TRAIN_EARLYSTOP_RE = re.compile(r"early stop after\s+(\d+)\s+epochs")


class TrainParser(ProgressParser):
    """``makemodel-train-uk``: per-epoch loss / make@1 for the live chart.

    ``metrics["epochs"]`` accumulates one row per epoch
    (``{epoch, loss, make1, make5, sec}``); ``metrics`` also carries the
    pre-training facts (device, makes, train/val crop counts) and the running
    best so the training view can show before/after at a glance.
    """

    kind = "makemodel-train-uk"

    def __init__(self) -> None:
        super().__init__()
        self.progress.metrics["epochs"] = []

    def feed(self, line: str) -> None:
        text = line.strip()

        m = _TRAIN_PREAMBLE_RE.search(text)
        if m:
            self.progress.metrics.update(
                device=m.group(1),
                amp=(m.group(2).lower() == "true"),
                makes=int(m.group(3)),
                train_crops=int(m.group(4)),
                val_crops=int(m.group(5)),
            )
            self.progress.phase = "starting"
            return

        m = _TRAIN_EPOCH_RE.search(text)
        if m:
            epoch, total = int(m.group(1)), int(m.group(2))
            row = {
                "epoch": epoch,
                "loss": float(m.group(3)),
                "make1": float(m.group(4)),
                "make5": float(m.group(5)),
                "sec": int(m.group(6)),
            }
            self.progress.metrics["epochs"].append(row)
            self.progress.current = epoch
            self.progress.total = total
            self.progress.phase = f"epoch {epoch}/{total}"
            best = self.progress.metrics.get("best_make1")
            if best is None or row["make1"] > best:
                self.progress.metrics["best_make1"] = row["make1"]
                self.progress.metrics["best_epoch"] = epoch
            return

        m = _TRAIN_EARLYSTOP_RE.search(text)
        if m:
            self.progress.metrics["early_stopped_at"] = int(m.group(1))
            return

        m = _TRAIN_DONE_RE.search(text)
        if m:
            self.progress.metrics["best_make1"] = float(m.group(1))
            self.progress.metrics["best_epoch"] = int(m.group(2))
            self.progress.metrics["checkpoint"] = m.group(3).strip()
            self.progress.summary = text
            self.progress.phase = "done"


# ----------------------------------------------------------------------
# Pull (the stdout side; the byte-bar is the runner's dir watcher)
# ----------------------------------------------------------------------

_PULL_SESSION_RE = re.compile(r"\[pull\]\s+session:\s+(\S+)")
_PULL_TARGET_RE = re.compile(r"\[pull\]\s+target:\s+(.+?)\s*$")
_PULL_SIZEBYTES_RE = re.compile(r"size_bytes:\s+(\d+)")
_PULL_SIZE_RE = re.compile(r"size:\s+(.+?)\s+across\s+(\d+)\s+files")
_PULL_PATTERN_RE = re.compile(
    r"\[pull\]\s+(\S+):\s+(\d+)\s+remote\s*/\s*(\d+)\s+local\s*/\s*(\d+)\s+new"
)
_PULL_SKIP_RE = re.compile(r"skip-existing:\s+(\d+)\s+new image")
_PULL_DONE_RE = re.compile(r"\[pull\]\s+done\s+->")


class PullParser(ProgressParser):
    """``pull``: per-pattern new-file counts + the human size line.

    The real transfer bar/ETA comes from the runner watching the local session
    directory grow toward the known remote byte total; this captures the
    discrete signals the command does print (so the log panel is meaningful and
    ``new`` files tally as they land)."""

    kind = "pull"

    def __init__(self) -> None:
        super().__init__()
        self.progress.metrics["patterns"] = {}

    def _maybe_watch(self) -> None:
        """Once the session, local target and byte total are all known, expose a
        ``watch`` directive the runner uses to track the local dir filling up."""
        m = self.progress.metrics
        target, session, total = m.get("target"), m.get("session"), m.get("remote_bytes")
        if target and session and total:
            m["watch"] = {"path": f"{target}/{session}", "total_bytes": total}

    def feed(self, line: str) -> None:
        text = line.strip()

        m = _PULL_SESSION_RE.search(text)
        if m:
            self.progress.metrics["session"] = m.group(1)
            self._maybe_watch()
            return
        m = _PULL_TARGET_RE.search(text)
        if m:
            self.progress.metrics["target"] = m.group(1).strip()
            self._maybe_watch()
            return
        m = _PULL_SIZEBYTES_RE.search(text)
        if m:
            self.progress.metrics["remote_bytes"] = int(m.group(1))
            self._maybe_watch()
            return

        m = _PULL_SIZE_RE.search(text)
        if m:
            self.progress.metrics["remote_size_h"] = m.group(1).strip()
            self.progress.metrics["remote_files"] = int(m.group(2))
            self.progress.phase = "sizing"
            return

        m = _PULL_PATTERN_RE.search(text)
        if m:
            pat, remote, local, new = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            self.progress.metrics["patterns"][pat] = {
                "remote": remote,
                "local": local,
                "new": new,
            }
            self.progress.metrics["new_total"] = sum(
                p["new"] for p in self.progress.metrics["patterns"].values()
            )
            self.progress.phase = "fetching"
            return

        m = _PULL_SKIP_RE.search(text)
        if m:
            self.progress.metrics["new_fetched"] = int(m.group(1))
            self.progress.summary = text
            return

        if _PULL_DONE_RE.search(text):
            self.progress.phase = "done"
            self.progress.summary = text


# ----------------------------------------------------------------------
# DVSA label harvest
# ----------------------------------------------------------------------

_DVSA_TOTAL_RE = re.compile(r"\[dvsa-label\]\s+(\d+)\s+distinct high-conf plates")
_DVSA_DONE_RE = re.compile(r"(\d+)\s+labelled,\s+(\d+)\s+unknown")


class DvsaLabelParser(ProgressParser):
    """``dvsa-label``: total distinct plates + final labelled/unknown tally.

    Per-lookup progress isn't printed (the runner watches the labels JSON grow);
    this fixes the ``total`` and the closing summary."""

    kind = "dvsa-label"

    def feed(self, line: str) -> None:
        text = line.strip()
        m = _DVSA_TOTAL_RE.search(text)
        if m:
            self.progress.total = int(m.group(1))
            self.progress.phase = "looking up"
            return
        m = _DVSA_DONE_RE.search(text)
        if m:
            self.progress.metrics["labelled"] = int(m.group(1))
            self.progress.metrics["unknown"] = int(m.group(2))
            self.progress.current = int(m.group(1)) + int(m.group(2))
            self.progress.summary = text
            self.progress.phase = "done"


# ----------------------------------------------------------------------
# UK corpus build
# ----------------------------------------------------------------------

_BUILD_SESSIONS_RE = re.compile(r"\[makemodel-build-uk\]\s+(\d+)\s+session")
_BUILD_DONE_RE = re.compile(r"(\d+)\s+makes,\s+(\d+)\s+cars,\s+(\d+)\s+crops")


class BuildParser(ProgressParser):
    """``makemodel-build-uk``: session count + final makes/cars/crops totals.

    Extraction is I/O-bound and largely silent mid-run, so progress is coarse;
    the final line carries the corpus size the retrain decision turns on."""

    kind = "makemodel-build-uk"

    def feed(self, line: str) -> None:
        text = line.strip()
        m = _BUILD_SESSIONS_RE.search(text)
        if m:
            self.progress.metrics["sessions"] = int(m.group(1))
            self.progress.phase = "extracting"
            return
        m = _BUILD_DONE_RE.search(text)
        if m:
            self.progress.metrics["makes"] = int(m.group(1))
            self.progress.metrics["cars"] = int(m.group(2))
            self.progress.metrics["crops"] = int(m.group(3))
            self.progress.summary = text
            self.progress.phase = "done"


_PARSERS: dict[str, type[ProgressParser]] = {
    "alpr-run": BatchParser,
    "makemodel": BatchParser,
    "makemodel-train-uk": TrainParser,
    "pull": PullParser,
    "dvsa-label": DvsaLabelParser,
    "makemodel-build-uk": BuildParser,
}


def make_parser(kind: str) -> ProgressParser:
    """Return a fresh parser for a command ``kind`` (generic fallback)."""
    return _PARSERS.get(kind, GenericParser)()
