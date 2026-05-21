"""Run ALPR pipelines over a pulled session's main snaps.

    streettracker alpr-run <session_dir> [--pipeline both|bespoke|preferred]
                                         [--ablation]
                                         [--bespoke-model PATH]
                                         [--ocr-model NAME]
                                         [--detector-model NAME]
                                         [--gpu]
                                         [--limit N]

Writes ``<session>_alpr.json`` and ``<session>_alpr_by_track.json`` into
``session_dir``. Plate crops land under
``session_dir/alpr_crops/<pipeline>/``.

Ported from NanoTracker's ``alpr/cli/run.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from streettracker.analysis.alpr.base import (
    atomic_write_text,
    parse_snap_filename,
)
from streettracker.analysis.alpr.runner import PipelineRunner

# Default location for the bespoke detector weights. The repo's
# top-level ``.gitignore`` carries ``*.pt`` so anything under here stays
# untracked.
DEFAULT_BESPOKE_MODEL = (
    Path(__file__).resolve().parent.parent
    / "analysis" / "alpr" / "models" / "license_plate_detector.pt"
)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="streettracker alpr-run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_dir", type=Path)
    ap.add_argument(
        "--pipeline",
        choices=("both", "bespoke", "preferred"),
        default="both",
    )
    ap.add_argument(
        "--ablation",
        action="store_true",
        help="Also run bespoke detector + fast-plate-ocr OCR.",
    )
    ap.add_argument("--bespoke-model", type=Path, default=DEFAULT_BESPOKE_MODEL)
    ap.add_argument(
        "--detector-model",
        default="yolo-v9-t-384-license-plate-end2end",
        help="open-image-models alias for the preferred pipeline detector.",
    )
    ap.add_argument(
        "--ocr-model",
        default="global-plates-mobile-vit-v2-model",
        help="fast-plate-ocr model alias for the preferred pipeline OCR.",
    )
    ap.add_argument(
        "--gpu", action="store_true", help="Pass gpu=True to EasyOCR.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images (0 = all).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2

    snaps = _discover_snaps(session_dir)
    if not snaps:
        print(f"no vehicle_*_main_*.jpg snaps in {session_dir}", file=sys.stderr)
        return 1
    if args.limit > 0:
        snaps = snaps[: args.limit]
    print(f"[alpr] {len(snaps)} vehicle snaps")

    pipelines = _build_pipelines(args)
    if not pipelines:
        print("no pipelines selected", file=sys.stderr)
        return 2

    all_records: list[dict] = []
    crops_root = session_dir / "alpr_crops"
    for runner in pipelines:
        print(f"[alpr] running pipeline: {runner.name}")
        crop_dir = crops_root / runner.name
        for i, (image_path, tid, snap_index, cls) in enumerate(snaps, 1):
            result = runner.run(image_path, tid, snap_index, cls, crop_dir)
            all_records.append(result.to_json())
            if i % 10 == 0 or i == len(snaps):
                print(f"  [{runner.name}] {i}/{len(snaps)} done")
            if result.error:
                print(f"  [{runner.name}] {image_path.name}: ERROR {result.error}")

    session_label = session_dir.name
    out_path = session_dir / f"{session_label}_alpr.json"
    atomic_write_text(out_path, json.dumps(all_records, indent=2))
    print(f"[alpr] wrote {out_path}")

    rollup = _rollup_by_track(all_records)
    rollup_path = session_dir / f"{session_label}_alpr_by_track.json"
    atomic_write_text(rollup_path, json.dumps(rollup, indent=2))
    print(f"[alpr] wrote {rollup_path}")
    return 0


def _discover_snaps(session_dir: Path) -> list[tuple[Path, int, int, str]]:
    """Return ``[(path, track_id, snap_index, class_name)]`` for vehicle
    main snaps only — person crops aren't useful for ALPR."""
    out = []
    for p in sorted(session_dir.glob("*_main_*.jpg")):
        parsed = parse_snap_filename(p.name)
        if not parsed:
            continue
        cls, tid, n = parsed
        if cls != "vehicle":
            continue
        out.append((p, tid, n, cls))
    return out


def _build_pipelines(args: argparse.Namespace) -> list[PipelineRunner]:
    pipelines: list[PipelineRunner] = []
    bespoke_model = args.bespoke_model
    if not bespoke_model.is_absolute():
        bespoke_model = Path.cwd() / bespoke_model

    bespoke_det = None
    if args.pipeline in ("both", "bespoke") or args.ablation:
        from streettracker.analysis.alpr.bespoke import (
            BespokeDetector,
            EasyOcrRecognizer,
        )

        bespoke_det = BespokeDetector(bespoke_model)
        if args.pipeline in ("both", "bespoke"):
            pipelines.append(PipelineRunner(
                name="bespoke",
                detector=bespoke_det,
                recognizer=EasyOcrRecognizer(use_gpu=args.gpu),
            ))

    if args.pipeline in ("both", "preferred") or args.ablation:
        from streettracker.analysis.alpr.preferred import (
            FastPlateOcrRecognizer,
            OpenImageModelsDetector,
        )

        oim_det = OpenImageModelsDetector(args.detector_model)
        if args.pipeline in ("both", "preferred"):
            pipelines.append(PipelineRunner(
                name="preferred",
                detector=oim_det,
                recognizer=FastPlateOcrRecognizer(args.ocr_model),
            ))

    if args.ablation and bespoke_det is not None:
        from streettracker.analysis.alpr.preferred import FastPlateOcrRecognizer

        pipelines.append(PipelineRunner(
            name="ablation_bespokedet_fastocr",
            detector=bespoke_det,
            recognizer=FastPlateOcrRecognizer(args.ocr_model),
        ))

    return pipelines


def _rollup_by_track(records: list[dict]) -> dict:
    """Per-track best-of-N: for each ``(pipeline, track_id)``, pick the
    read with the highest ``ocr_conf``."""
    by_pipe_track: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in records:
        p = r["pipeline"]
        tid = r["track_id"]
        if not r.get("ocr_text"):
            continue
        cur_best = by_pipe_track[p].get(tid)
        if cur_best is None or (r.get("ocr_conf") or 0) > (cur_best.get("ocr_conf") or 0):
            by_pipe_track[p][tid] = {
                "track_id": tid,
                "snap_index": r["snap_index"],
                "image": r["image"],
                "ocr_text": r["ocr_text"],
                "ocr_conf": r.get("ocr_conf"),
                "det_conf": r.get("det_conf"),
            }

    tracks: dict[int, dict] = {}
    for pipe, by_tid in by_pipe_track.items():
        for tid, best in by_tid.items():
            tracks.setdefault(tid, {"track_id": tid})[f"best_{pipe}"] = best

    return {"tracks": sorted(tracks.values(), key=lambda r: r["track_id"])}


if __name__ == "__main__":
    raise SystemExit(main())
