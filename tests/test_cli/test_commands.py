"""The CLI command table (``cli/commands.py``).

The help text and the dispatcher are both built from ``COMMANDS``. These
tests check the table itself (every entry points at a real function, without
importing the heavy modules), that help and dispatch follow it, and that the
control panel only launches commands that exist.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

from streettracker import cli
from streettracker.cli import commands as commands_mod
from streettracker.cli.commands import COMMANDS, Command, find_command, render_help

_NAMES = [command.name for command in COMMANDS]


def _help_rows() -> list[str]:
    lines = render_help().splitlines()
    rows = lines[lines.index("commands:") + 1 :]
    return rows[: rows.index("")]


# ----------------------------------------------------------------------
# The table.


def test_command_names_are_unique() -> None:
    assert len(_NAMES) == len(set(_NAMES))


@pytest.mark.parametrize("command", COMMANDS, ids=_NAMES)
def test_command_entry_is_well_formed(command: Command) -> None:
    assert command.name == command.name.strip()
    assert " " not in command.name
    assert not command.name.startswith("-")  # would shadow --help / --version
    assert command.summary == command.summary.strip() != ""
    assert "\n" not in command.summary
    module_name, sep, function_name = command.target.partition(":")
    assert sep == ":" and module_name and function_name


@pytest.mark.parametrize("command", COMMANDS, ids=_NAMES)
def test_command_target_exists(command: Command) -> None:
    """The module is there and defines the function, checked from its source:
    importing it could need torch, Ultralytics or the alpr extra, which not
    every machine that runs the tests has."""
    module_name, _, function_name = command.target.partition(":")
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None, f"no module {module_name}"
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert function_name in defined, f"{module_name} defines no {function_name}()"


def test_find_command() -> None:
    assert find_command("vehicles") is COMMANDS[_NAMES.index("vehicles")]
    assert find_command("no-such-command") is None
    assert find_command("") is None


# ----------------------------------------------------------------------
# Help.


def test_help_lists_every_command_once_in_table_order() -> None:
    assert [row.split(None, 1) for row in _help_rows()] == [
        [command.name, command.summary] for command in COMMANDS
    ]


def test_help_summaries_line_up() -> None:
    column = 2 + max(len(name) for name in _NAMES) + 2
    for row, command in zip(_help_rows(), COMMANDS, strict=True):
        assert row[column:] == command.summary
        assert row[:column].strip() == command.name


def test_help_text_is_the_rendered_table() -> None:
    assert render_help() == cli._HELP
    assert cli._HELP.startswith("usage: streettracker [--version] <command> [<args>]\n")
    assert cli._HELP.endswith("for per-command options.\n")


@pytest.mark.parametrize("argv", [[], ["-h"], ["--help"]], ids=["no args", "-h", "--help"])
def test_help_is_printed(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out == render_help()
    assert captured.err == ""


# ----------------------------------------------------------------------
# Dispatch.


@pytest.mark.parametrize("command", COMMANDS, ids=_NAMES)
def test_each_command_dispatches_to_its_target(
    command: Command, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``streettracker <name> <args>`` imports the command's module, calls its
    function with the remaining arguments and returns its exit code."""
    module_name, _, function_name = command.target.partition(":")
    calls: list[tuple[str, list[str]]] = []

    def fake_import(name: str) -> types.SimpleNamespace:
        def entry(argv: list[str]) -> int:
            calls.append((name, argv))
            return 7

        return types.SimpleNamespace(**{function_name: entry})

    monkeypatch.setattr(commands_mod, "import_module", fake_import)
    assert cli.main([command.name, "one", "--two"]) == 7
    assert calls == [(module_name, ["one", "--two"])]


def test_unknown_command_exits_2_and_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["no-such-command"]) == 2
    captured = capsys.readouterr()
    assert "unknown subcommand: no-such-command" in captured.err
    assert captured.out == render_help()


def test_help_imports_no_command_module() -> None:
    """Printing the help must not import any command's module: several pull in
    torch or Ultralytics. Checked in a fresh interpreter, where nothing else
    has imported them first."""
    code = (
        "import sys\n"
        "from streettracker.cli import main\n"
        "from streettracker.cli.commands import COMMANDS\n"
        "assert main(['--help']) == 0\n"
        "modules = {c.target.partition(':')[0] for c in COMMANDS}\n"
        "print('LOADED', sorted(modules & set(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.splitlines()[-1] == "LOADED []"


# ----------------------------------------------------------------------
# The control panel launches jobs as ``streettracker <kind> ...``.


def test_the_panel_only_names_commands_that_exist() -> None:
    from streettracker.control import jobs, progress, prompts, server

    tables = {
        "server.ALLOWED_JOB_KINDS": server.ALLOWED_JOB_KINDS,
        "jobs._GPU_KINDS": jobs._GPU_KINDS,
        "progress._PARSERS": progress._PARSERS,
        "prompts._KIND_HINTS": prompts._KIND_HINTS,
    }
    for label, kinds in tables.items():
        unknown = sorted(set(kinds) - set(_NAMES))
        assert not unknown, f"{label} names commands the CLI doesn't have: {unknown}"
