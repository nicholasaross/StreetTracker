"""StreetTracker CLI entry point.

Subcommands:
- ``streettracker run`` — live RTSP (Orin only)
- ``streettracker batch <video>`` — file input
- ``streettracker pull`` — scp session from device
- ``streettracker recolor <session>`` — rerun color heuristic
- ``streettracker debug-color <crop.jpg>`` — inspect a single crop
- ``streettracker vehicles <session>`` — plate-anchored per-vehicle aggregation
- ``streettracker people <session>`` — person-track activity enrichment (dog walkers, joggers)
- ``streettracker export-engine`` — ``.pt`` → ``.engine``
- ``streettracker alpr-run <session>`` — run ALPR pipelines on a session
- ``streettracker alpr-score <session>`` — score ALPR pipelines vs labels
- ``streettracker alpr-label <session>`` — interactive plate labeling
- ``streettracker alpr-report <session>`` — render comparison HTML
- ``streettracker dvsa-label <session>`` — DVSA MOT make/model lookup per plate
- ``streettracker dvsa-apply <session>`` — write DVSA make/model onto per-track records
- ``streettracker makemodel-train <sv_data>`` — fine-tune the make/model CNN on CompCars
- ``streettracker makemodel <session>`` — classify make/model on a session's snaps
- ``streettracker bodytype <session>`` — classify coarse body type on a session's snaps
- ``streettracker makemodel-build-uk <out>`` — extract DVSA-labelled UK make crops
- ``streettracker makemodel-train-uk <crops>`` — train the UK make classifier
- ``streettracker showcase`` — local website showcasing enriched + recurring vehicles
- ``streettracker control`` — operator control panel (live radiator + process control)

Each subcommand owns its own ``argparse.ArgumentParser`` in its module's
``main()`` — we dispatch on the first positional rather than using
``add_subparsers`` so subcommand parsers receive ``--help`` cleanly
instead of having the top-level parser intercept it.
"""

from __future__ import annotations

import sys

_HELP = """\
usage: streettracker [--version] <command> [<args>]

commands:
  run             live RTSP tracker (Orin only)
  batch           batch process a video file
  pull            scp a session from a remote device
  recolor         rerun color heuristic on a closed session
  debug-color     inspect HSV vote on one or more crop JPEGs
  vehicles        plate-anchored per-vehicle aggregation
  people          person-track activity enrichment (dog walkers, joggers, cyclists)
  export-engine   export .pt to .engine via Ultralytics
  alpr-run        run ALPR pipelines over a session's main snaps
  alpr-score      score ALPR pipelines against labels
  alpr-label      interactive plate labeling
  alpr-report     render the ALPR comparison HTML
  dvsa-label      look up make/model via DVSA MOT API per plate
  dvsa-apply      write harvested DVSA make/model onto a session's records
  makemodel-train fine-tune the make/model CNN on the CompCars sv_data
  makemodel        classify make/model on a session's snaps (writes _makemodel.json)
  bodytype        classify coarse body type (hatchback/suv/...) on a session's snaps
  colour          classify vehicle colour (white/silver/black/...) on a session's snaps
  makemodel-build-uk  extract DVSA-labelled UK make crops from sessions
  makemodel-train-uk  train the UK make classifier on extracted crops
  showcase        local website: enriched + recurring vehicles, with tagging
  control         operator control panel: live radiator + process control

Run ``streettracker <command> --help`` for per-command options.
"""


def _print_help() -> None:
    print(_HELP, end="")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_help()
        return 0

    head, rest = argv[0], argv[1:]

    if head in ("-h", "--help"):
        _print_help()
        return 0
    if head == "--version":
        from streettracker import __version__

        print(__version__)
        return 0

    if head == "recolor":
        from streettracker.analysis.recolor import main as recolor_main

        return recolor_main(rest)
    if head == "debug-color":
        from streettracker.analysis.debug_color import main as debug_color_main

        return debug_color_main(rest)
    if head == "vehicles":
        from streettracker.analysis.vehicles import main as vehicles_main

        return vehicles_main(rest)
    if head == "people":
        from streettracker.analysis.people import main as people_main

        return people_main(rest)
    if head == "pull":
        from streettracker.cli.pull import main as pull_main

        return pull_main(rest)
    if head == "export-engine":
        from streettracker.cli.export_engine import main as export_engine_main

        return export_engine_main(rest)
    if head == "alpr-run":
        from streettracker.cli.alpr_run import main as alpr_run_main

        return alpr_run_main(rest)
    if head == "alpr-score":
        from streettracker.cli.alpr_score import main as alpr_score_main

        return alpr_score_main(rest)
    if head == "alpr-label":
        from streettracker.cli.alpr_label import main as alpr_label_main

        return alpr_label_main(rest)
    if head == "alpr-report":
        from streettracker.cli.alpr_report import main as alpr_report_main

        return alpr_report_main(rest)
    if head == "dvsa-label":
        from streettracker.cli.dvsa_label import main as dvsa_label_main

        return dvsa_label_main(rest)
    if head == "dvsa-apply":
        from streettracker.analysis.dvsa_apply import main as dvsa_apply_main

        return dvsa_apply_main(rest)
    if head == "makemodel-train":
        from streettracker.analysis.makemodel.training import main as makemodel_train_main

        return makemodel_train_main(rest)
    if head == "makemodel":
        from streettracker.analysis.makemodel.infer import main as makemodel_main

        return makemodel_main(rest)
    if head == "bodytype":
        from streettracker.analysis.makemodel.bodytype_infer import main as bodytype_main

        return bodytype_main(rest)
    if head == "colour":
        from streettracker.analysis.makemodel.colour_infer import main as colour_main

        return colour_main(rest)
    if head == "makemodel-build-uk":
        from streettracker.analysis.makemodel.uk_dataset import (
            build_main as makemodel_build_uk_main,
        )

        return makemodel_build_uk_main(rest)
    if head == "makemodel-train-uk":
        from streettracker.analysis.makemodel.uk_dataset import (
            train_main as makemodel_train_uk_main,
        )

        return makemodel_train_uk_main(rest)

    if head == "showcase":
        from streettracker.web.server import main as showcase_main

        return showcase_main(rest)

    if head == "control":
        from streettracker.control.server import main as control_main

        return control_main(rest)

    if head == "run":
        from streettracker.cli.run import main as run_main

        return run_main(rest)
    if head == "batch":
        from streettracker.cli.batch import main as batch_main

        return batch_main(rest)

    print(f"[streettracker] unknown subcommand: {head}", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
