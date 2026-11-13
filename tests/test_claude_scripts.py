"""The analysis scripts in ``.claude/`` still fit the package.

``CLAUDE.md`` and ``docs/history`` cite these scripts as the record of how a
number was produced, and about half of them import the package. Nothing else
exercises them, so a refactor could rename or move what they use without
anyone noticing.

They can't be run here (they need session data, a GPU or the camera), so they
are checked from their source: each one compiles, and every name it takes
from ``streettracker`` exists, whether it is imported by name
(``from streettracker.web.stats import build_stats``) or reached through a
module (``stats._tally_session``). Package modules are read as source too, so
this runs where torch and the ``alpr`` extra are not installed.

CI also runs pyflakes over the scripts (``ruff check --select F .claude/``).
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / ".claude"
PACKAGE = "streettracker"


def _tracked_scripts() -> list[Path]:
    """The scripts in the repository. Untracked ones on a dev box (scratch
    work, or scripts kept out of the repo on purpose) are not ours to judge."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", ".claude/*.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        tracked = [REPO_ROOT / name for name in out.split("\0") if name]
    except (OSError, subprocess.SubprocessError):
        tracked = []
    # Not a git checkout (an unpacked archive): take what is there.
    return sorted(tracked or SCRIPTS_DIR.glob("*.py"))


SCRIPTS = _tracked_scripts()


# ----------------------------------------------------------------------
# What a package module defines, read from its source.


def _module_source(module: str) -> Path | None:
    """The ``.py`` file of a package module, or ``None`` if there is none."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _is_module(module: str) -> bool:
    return _module_source(module) is not None


def _bound_names(target: ast.expr) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _nested_statements(node: ast.stmt) -> list[ast.stmt]:
    """The statements inside a compound statement's blocks (``if`` / ``try``
    / ``with`` / a loop): they run at module level too."""
    statements: list[ast.stmt] = []
    for field in ("body", "orelse", "finalbody", "handlers"):
        inner = getattr(node, field, None)
        if not isinstance(inner, list):
            continue
        for item in inner:
            statements.extend(item.body if isinstance(item, ast.ExceptHandler) else [item])
    return statements


def _statement_targets(node: ast.stmt) -> set[str]:
    """Names a single statement binds itself: an assignment's targets, a
    loop's variable, a ``with`` statement's ``as`` names."""
    if isinstance(node, ast.Assign):
        return set().union(*(_bound_names(target) for target in node.targets))
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
        return _bound_names(node.target)
    if isinstance(node, (ast.With, ast.AsyncWith)):
        bound = (item.optional_vars for item in node.items)
        return set().union(*(_bound_names(target) for target in bound if target is not None))
    return set()


def _top_level_names(body: list[ast.stmt]) -> tuple[set[str], bool]:
    """Names bound at module level by ``body``, and whether a star import
    makes the list incomplete. Looks inside ``if`` / ``try`` / ``with`` /
    loops, which run at module level, but not inside functions or classes."""
    names: set[str] = set()
    star = False
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            star = star or any(alias.name == "*" for alias in node.names)
            names |= {(a.asname or a.name).split(".")[0] for a in node.names if a.name != "*"}
            continue
        names |= _statement_targets(node)
        more, more_star = _top_level_names(_nested_statements(node))
        names |= more
        star = star or more_star
    return names, star


@cache
def _defines(module: str) -> frozenset[str] | None:
    """Module-level names of a package module. ``None`` when they can't be
    listed from source (a star import, or a module ``__getattr__``), in
    which case nothing is claimed about that module."""
    path = _module_source(module)
    if path is None:
        return None
    names, star = _top_level_names(ast.parse(path.read_text(encoding="utf-8")).body)
    if star or "__getattr__" in names:
        return None
    return frozenset(names)


def _has(module: str, name: str) -> bool:
    """Whether ``module.name`` exists: a name the module defines, or one of
    its submodules."""
    defined = _defines(module)
    return defined is None or name in defined or _is_module(f"{module}.{name}")


# ----------------------------------------------------------------------
# What a script uses from the package.


def _chain(node: ast.expr) -> list[str] | None:
    """``a.b.c`` as ``["a", "b", "c"]``; ``None`` for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def _rebound_names(tree: ast.Module) -> set[str]:
    """Names the script assigns, loops over or takes as a parameter. An
    import alias that is also one of these may not be a module where it is
    used, so nothing is claimed about its attributes."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


Problems = set[tuple[int, str]]  # (line, what is missing)
Aliases = dict[str, str]  # a script's local name -> the package module it is bound to


def _check_import(node: ast.Import, problems: Problems, aliases: Aliases) -> None:
    """``import streettracker.a.b [as m]``."""
    for alias in node.names:
        if alias.name.split(".")[0] != PACKAGE:
            continue
        if not _is_module(alias.name):
            problems.add((node.lineno, f"no module {alias.name}"))
        elif alias.asname:
            aliases[alias.asname] = alias.name
        else:
            aliases[PACKAGE] = PACKAGE


def _check_import_from(node: ast.ImportFrom, problems: Problems, aliases: Aliases) -> None:
    """``from streettracker.a import b [as c]``."""
    module = node.module
    if node.level != 0 or not module or module.split(".")[0] != PACKAGE:
        return
    if not _is_module(module):
        problems.add((node.lineno, f"no module {module}"))
        return
    for alias in node.names:
        if alias.name == "*":
            continue
        if not _has(module, alias.name):
            problems.add((node.lineno, f"{module} has no {alias.name}"))
        elif _is_module(f"{module}.{alias.name}"):
            aliases[alias.asname or alias.name] = f"{module}.{alias.name}"


def _check_attribute(node: ast.Attribute, problems: Problems, aliases: Aliases) -> None:
    """``m.name`` where ``m`` is bound to a package module."""
    parts = _chain(node)
    if parts is None or parts[0] not in aliases:
        return
    module = aliases[parts[0]]
    for attr in parts[1:]:
        if _is_module(f"{module}.{attr}"):
            module = f"{module}.{attr}"  # a.b: still walking down the package
            continue
        if not _has(module, attr):
            problems.add((node.lineno, f"{module} has no {attr}"))
        return  # past the module: an object's own attributes aren't checked


def unresolved(source: str) -> list[str]:
    """Every use of the package in ``source`` that names something missing,
    as ``"line N: what"`` strings."""
    tree = ast.parse(source)
    problems: Problems = set()
    aliases: Aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _check_import(node, problems, aliases)
        elif isinstance(node, ast.ImportFrom):
            _check_import_from(node, problems, aliases)
    for name in _rebound_names(tree):
        aliases.pop(name, None)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            _check_attribute(node, problems, aliases)
    return [f"line {line}: {what}" for line, what in sorted(problems)]


# ----------------------------------------------------------------------
# The scripts.


def test_there_are_scripts_to_check() -> None:
    assert len(SCRIPTS) >= 50, f"found only {len(SCRIPTS)} scripts under {SCRIPTS_DIR}"


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_script_compiles(script: Path) -> None:
    compile(script.read_text(encoding="utf-8"), str(script), "exec")


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_script_uses_only_names_the_package_has(script: Path) -> None:
    problems = unresolved(script.read_text(encoding="utf-8"))
    assert not problems, f"{script.name} uses names the package no longer has:\n" + "\n".join(
        f"  {p}" for p in problems
    )


# ----------------------------------------------------------------------
# The checker itself, on small made-up scripts.


def test_checker_accepts_real_names() -> None:
    source = """
import streettracker.web.stats as stats_mod
from streettracker.analysis.alpr import gate
from streettracker.cli import alpr_run, commands
from streettracker.common.fileio import atomic_write_json, read_json
from streettracker.web.stats import N_FASTEST, Stats, build_stats

print(stats_mod.build_stats, stats_mod._tally_session, stats_mod.N_FASTEST)
print(gate.load_plate_support, gate.PlateGate)
print(alpr_run._rollup_by_track, commands.COMMANDS)
"""
    assert unresolved(source) == []


def test_checker_reports_a_name_that_is_gone() -> None:
    source = """
from streettracker.web.stats import build_stats, build_statistics
from streettracker.common.fileio import atomic_write_yaml
"""
    assert unresolved(source) == [
        "line 2: streettracker.web.stats has no build_statistics",
        "line 3: streettracker.common.fileio has no atomic_write_yaml",
    ]


def test_checker_reports_a_module_that_is_gone() -> None:
    source = """
import streettracker.web.statistics
from streettracker.common.io import read_json
from streettracker.common import io
"""
    assert unresolved(source) == [
        "line 2: no module streettracker.web.statistics",
        "line 3: no module streettracker.common.io",
        "line 4: streettracker.common has no io",
    ]


def test_checker_reports_a_missing_name_reached_through_a_module() -> None:
    source = """
import streettracker
import streettracker.web.stats as stats_mod
from streettracker.analysis.alpr import gate
from streettracker.cli import alpr_run as run_mod

stats_mod._old_helper()
gate._SUPPORT_GLOB
run_mod._rollup_by_car([])
streettracker.web.stats.gone()
streettracker.web.stats.build_stats.cache_clear()   # past the module: not checked
"""
    assert unresolved(source) == [
        "line 7: streettracker.web.stats has no _old_helper",
        "line 8: streettracker.analysis.alpr.gate has no _SUPPORT_GLOB",
        "line 9: streettracker.cli.alpr_run has no _rollup_by_car",
        "line 10: streettracker.web.stats has no gone",
    ]


def test_checker_leaves_a_reused_alias_alone() -> None:
    """``stats`` is first a module, then a result. Its attributes can't be
    judged without following the script's control flow, so they aren't."""
    source = """
from streettracker.web import stats

stats = stats.build_stats("output")
print(stats.overall)
"""
    assert unresolved(source) == []


def test_checker_ignores_other_packages_and_relative_imports() -> None:
    source = """
import numpy as np
from collections import nothing_like_this
from . import sibling
from .sibling import thing

np.no_such_function()
"""
    assert unresolved(source) == []


def test_checker_sees_names_defined_under_a_module_level_condition() -> None:
    # analysis/__init__.py binds `sys` at the top and `_os` inside an `if`.
    source = """
import streettracker.analysis as analysis

print(analysis.sys, analysis.alpr)
"""
    assert unresolved(source) == []
