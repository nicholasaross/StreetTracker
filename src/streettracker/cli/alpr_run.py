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
        default="yolo-v9-t-640-license-plate-end2end",
        help=(
            "open-image-models alias for the preferred pipeline detector. "
            "Default bumped 2026-05-25 from yolo-v9-t-384 (which finds 0%% "
            "of plates on full 4K snaps -- after the detector's internal "
            "10x downscale a 100 px plate becomes ~10 px, sub-detection)."
        ),
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
    ap.add_argument(
        "--pre-crop",
        action="store_true",
        help=(
            "Pre-crop the largest vehicle bbox (ultralytics YOLOv8n on COCO "
            "classes car/bus/truck) before feeding the plate detector. "
            "Recovers near-100%% of the no-read tracks from the 2026-05-25 "
            "soak's diagnostic sample, at the cost of ~+30%% per-image "
            "runtime. See "
            "src/streettracker/analysis/alpr/precrop.py for the docstring "
            "with the empirical comparison."
        ),
    )
    ap.add_argument(
        "--vehicle-model",
        default="yolov8n.pt",
        help=(
            "Ultralytics YOLO model used for the pre-crop vehicle stage. "
            "Defaults to yolov8n (auto-downloads on first use). Ignored "
            "without --pre-crop."
        ),
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

    # Load per-snap BotSORT bboxes if the session was recorded against a
    # runtime that persists them. ``None`` means we don't have a hint
    # and detectors fall back to whatever they do (PreCrop -> YOLO,
    # others -> full image). Sub-stream frame size from SessionMeta
    # lets us scale into the 4K snap's coord system at load time.
    bbox_index, sub_size = _load_bbox_index(session_dir)
    if bbox_index:
        print(
            f"[alpr] loaded {len(bbox_index)} per-snap bboxes "
            f"(sub-stream {sub_size}); pre-crop hints active"
        )
    elif args.pre_crop:
        print(
            "[alpr] no per-snap bboxes in data.json -- pre-crop will "
            "use the YOLO largest-vehicle-bbox fallback (older session?)"
        )

    all_records: list[dict] = []
    crops_root = session_dir / "alpr_crops"
    for runner in pipelines:
        print(f"[alpr] running pipeline: {runner.name}")
        crop_dir = crops_root / runner.name
        for i, (image_path, tid, snap_index, cls) in enumerate(snaps, 1):
            hint = _resolve_bbox_hint(image_path, tid, snap_index, bbox_index, sub_size)
            result = runner.run(
                image_path, tid, snap_index, cls, crop_dir,
                bbox_hint=hint,
            )
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


def _load_bbox_index(
    session_dir: Path,
) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], tuple[int, int] | None]:
    """Build ``(track_id, snap_index) -> sub_stream_bbox`` from data.json
    and pick the sub-stream frame size out of _meta.json. Returns empty
    dict + ``None`` size for sessions that pre-date the bbox-capture
    change (no error, downstream falls back to YOLO)."""
    session_label = session_dir.name
    data_path = session_dir / f"{session_label}_data.json"
    meta_path = session_dir / f"{session_label}_meta.json"

    bbox_index: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    if data_path.exists():
        try:
            records = json.loads(data_path.read_text())
        except (OSError, json.JSONDecodeError):
            records = []
        for r in records:
            tid = r.get("track_id")
            snaps = r.get("main_snaps") or []
            bboxes = r.get("main_snap_bboxes")
            if tid is None or not snaps or not bboxes:
                continue
            if len(bboxes) != len(snaps):
                # Schema violation: skip rather than misalign.
                continue
            for n, bb in zip(snaps, bboxes, strict=False):
                if bb is None:
                    continue
                try:
                    x1, y1, x2, y2 = (int(v) for v in bb)
                except (TypeError, ValueError):
                    continue
                bbox_index[(int(tid), int(n))] = (x1, y1, x2, y2)

    sub_size: tuple[int, int] | None = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        fs = meta.get("frame_size")
        if isinstance(fs, list) and len(fs) == 2:
            try:
                sub_size = (int(fs[0]), int(fs[1]))
            except (TypeError, ValueError):
                sub_size = None

    return bbox_index, sub_size


def _resolve_bbox_hint(
    image_path: Path,
    track_id: int,
    snap_index: int,
    bbox_index: dict[tuple[int, int], tuple[int, int, int, int]],
    sub_size: tuple[int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Look up the per-snap bbox and scale it from sub-stream coords
    into the 4K snap's pixel coords. Returns ``None`` if no bbox is
    known for this snap, or if we don't know the sub-stream frame
    size (analysis can't scale safely without it -- YOLO fallback)."""
    sub_bbox = bbox_index.get((track_id, snap_index))
    if sub_bbox is None or sub_size is None:
        return None
    sub_w, sub_h = sub_size
    if sub_w <= 0 or sub_h <= 0:
        return None
    # Read image dimensions from the JPEG header without decoding the
    # pixel data. PIL.Image.open is lazy: ``.size`` only parses the
    # header. PipelineRunner will cv2.imread the same path moments
    # later, so doing a full second decode here would double the I/O
    # cost per snap. Pillow is a transitive dep of both ultralytics
    # and open-image-models so it's already in the venv.
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except (OSError, ValueError):
        return None
    # Sub-stream and 4K may have slightly different aspect ratios
    # (Reolink's sub is 896:512 = 1.75 vs main 3840:2160 = 1.78), so
    # scale x and y independently rather than picking a single ratio.
    sx = w / sub_w
    sy = h / sub_h
    x1, y1, x2, y2 = sub_bbox
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def _build_pipelines(args: argparse.Namespace) -> list[PipelineRunner]:
    pipelines: list[PipelineRunner] = []
    bespoke_model = args.bespoke_model
    if not bespoke_model.is_absolute():
        bespoke_model = Path.cwd() / bespoke_model

    # Optional vehicle pre-crop wrapper. Shared across pipelines so the
    # ultralytics YOLO weights load once.
    def _wrap(detector, suffix: str):
        if not args.pre_crop:
            return detector
        from streettracker.analysis.alpr.precrop import PreCropDetector

        wrapped = PreCropDetector(
            plate_detector=detector,
            vehicle_model=args.vehicle_model,
        )
        # Override the wrapper name so pipeline labels stay terse.
        wrapped.name = f"precrop-{suffix}"
        return wrapped

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
                detector=_wrap(bespoke_det, "bespoke"),
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
                detector=_wrap(oim_det, "preferred"),
                recognizer=FastPlateOcrRecognizer(args.ocr_model),
            ))

    if args.ablation and bespoke_det is not None:
        from streettracker.analysis.alpr.preferred import FastPlateOcrRecognizer

        pipelines.append(PipelineRunner(
            name="ablation_bespokedet_fastocr",
            detector=_wrap(bespoke_det, "ablation_bespokedet_fastocr"),
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
