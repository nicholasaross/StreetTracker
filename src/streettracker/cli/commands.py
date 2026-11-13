"""The ``streettracker`` subcommands, in one table.

The top-level help text and the dispatcher are both built from
:data:`COMMANDS`, so they can't drift apart. Adding a subcommand is one line
here: its name, its one-line summary and where its ``main(argv)`` lives.

A command's module is imported only when that command runs. Several of them
pull in torch, Ultralytics or aiohttp at import, and ``streettracker --help``
(or any other command) must not pay for that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import cast

# A subcommand's entry point: takes the arguments after the command name and
# returns the process exit code.
CommandMain = Callable[[list[str]], int]


@dataclass(frozen=True, slots=True)
class Command:
    """One subcommand.

    ``target`` is ``"package.module:function"``. It is a string, not the
    function, so that building the table imports nothing.
    """

    name: str
    summary: str
    target: str

    def load(self) -> CommandMain:
        """Import the command's module and return its entry point."""
        module_name, _, function_name = self.target.partition(":")
        return cast("CommandMain", getattr(import_module(module_name), function_name))


# In the order `streettracker --help` lists them.
COMMANDS: tuple[Command, ...] = (
    Command("run", "live RTSP tracker (Orin only)", "streettracker.cli.run:main"),
    Command("batch", "batch process a video file", "streettracker.cli.batch:main"),
    Command("pull", "scp a session from a remote device", "streettracker.cli.pull:main"),
    Command(
        "recover-session",
        "rebuild a crashed session's data/meta/hourly JSON from its events log",
        "streettracker.cli.recover_session:main",
    ),
    Command(
        "recolor",
        "rerun color heuristic on a closed session",
        "streettracker.analysis.recolor:main",
    ),
    Command(
        "debug-color",
        "inspect HSV vote on one or more crop JPEGs",
        "streettracker.analysis.debug_color:main",
    ),
    Command(
        "vehicles",
        "plate-anchored per-vehicle aggregation",
        "streettracker.analysis.vehicles:main",
    ),
    Command(
        "people",
        "person-track activity enrichment (dog walkers, joggers, cyclists)",
        "streettracker.analysis.people:main",
    ),
    Command(
        "export-engine",
        "export .pt to .engine via Ultralytics",
        "streettracker.cli.export_engine:main",
    ),
    Command(
        "alpr-run",
        "run ALPR pipelines over a session's main snaps",
        "streettracker.cli.alpr_run:main",
    ),
    Command(
        "alpr-rescore",
        "recompute plate-read confidence from saved crops (2026-09-28 fix)",
        "streettracker.cli.alpr_rescore:main",
    ),
    Command(
        "alpr-colour",
        "plate-colour check on saved crops: drop other cars' plates (2026-10-04)",
        "streettracker.cli.alpr_colour:main",
    ),
    Command(
        "alpr-score",
        "score ALPR pipelines against labels",
        "streettracker.cli.alpr_score:main",
    ),
    Command("alpr-label", "interactive plate labeling", "streettracker.cli.alpr_label:main"),
    Command(
        "alpr-report",
        "render the ALPR comparison HTML",
        "streettracker.cli.alpr_report:main",
    ),
    Command(
        "dvsa-label",
        "look up make/model via DVSA MOT API per plate",
        "streettracker.cli.dvsa_label:main",
    ),
    Command(
        "dvsa-apply",
        "write harvested DVSA make/model onto a session's records",
        "streettracker.analysis.dvsa_apply:main",
    ),
    Command(
        "makemodel-train",
        "fine-tune the make/model CNN on the CompCars sv_data",
        "streettracker.analysis.makemodel.training:main",
    ),
    Command(
        "makemodel",
        "classify make/model on a session's snaps (writes _makemodel.json)",
        "streettracker.analysis.makemodel.infer:main",
    ),
    Command(
        "bodytype",
        "classify coarse body type (hatchback/suv/...) on a session's snaps",
        "streettracker.analysis.makemodel.bodytype_infer:main",
    ),
    Command(
        "colour",
        "classify vehicle colour (white/silver/black/...) on a session's snaps",
        "streettracker.analysis.makemodel.colour_infer:main",
    ),
    Command(
        "makemodel-build-uk",
        "extract DVSA-labelled UK make crops from sessions",
        "streettracker.analysis.makemodel.uk_dataset:build_main",
    ),
    Command(
        "makemodel-train-uk",
        "train the UK make classifier on extracted crops",
        "streettracker.analysis.makemodel.uk_dataset:train_main",
    ),
    Command(
        "makemodel-compare",
        "head-to-head: candidate vs production make model on shared held-out cars",
        "streettracker.analysis.makemodel.compare:main",
    ),
    Command(
        "model-release",
        "publish the production make model as a (public) GitHub Release",
        "streettracker.cli.model_release:main",
    ),
    Command(
        "showcase",
        "local website: enriched + recurring vehicles, with tagging",
        "streettracker.web.server:main",
    ),
    Command(
        "control",
        "operator control panel: live radiator + process control",
        "streettracker.control.server:main",
    ),
)

_BY_NAME = {command.name: command for command in COMMANDS}


def find_command(name: str) -> Command | None:
    """The command called ``name``, or ``None``."""
    return _BY_NAME.get(name)


def render_help() -> str:
    """The text ``streettracker --help`` prints: one aligned line per command."""
    width = max(len(command.name) for command in COMMANDS) + 2
    rows = "\n".join(f"  {command.name:<{width}}{command.summary}" for command in COMMANDS)
    return (
        "usage: streettracker [--version] <command> [<args>]\n"
        "\n"
        "commands:\n"
        f"{rows}\n"
        "\n"
        "Run ``streettracker <command> --help`` for per-command options.\n"
    )


__all__ = ["COMMANDS", "Command", "CommandMain", "find_command", "render_help"]
