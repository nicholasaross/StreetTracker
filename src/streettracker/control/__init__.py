"""Operator control panel — a standalone web app for driving StreetTracker's
operational procedures (session pulls, ALPR/DVSA enrichment, corpus rebuilds,
CNN training, model promotion) and watching them live.

Runs off-device on the dev box, alongside (but separate from) the showcase
site (:mod:`streettracker.web`). Its whole reason for existing is that these
procedures otherwise need a human driving Claude Code through the
PowerShell/SSH steps by hand — the panel executes them as subprocesses, parses
their stdout for live progress, and renders a rich information radiator so the
operator can run everything **without** an assistant attached.

Layout (built in phases):

* :mod:`~streettracker.control.orin`     — Orin SSH status (reuses the
  ``cli.pull`` ssh/inventory helpers, made non-fatal + timeout-bounded for the
  background poller).
* :mod:`~streettracker.control.inspect`  — local introspection: session
  inventory + enrichment status, training-corpus stats, production-model
  metadata, and the retrain / promote recommendations derived from them.
* :mod:`~streettracker.control.server`   — the ``aiohttp`` app + routes + the
  ``streettracker control`` CLI entry point (mirrors
  :mod:`streettracker.web.server`).

Phase 2+ adds a subprocess job runner, per-command progress parsers, the
multi-step playbooks, and a localhost-only control middleware.
"""

from __future__ import annotations
