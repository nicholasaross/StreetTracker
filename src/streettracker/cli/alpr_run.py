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

from streettracker.analysis.alpr.base import atomic_write_text
from streettracker.analysis.alpr.runner import PipelineRunner
from streettracker.analysis.snap_assets import (
    discover_vehicle_snaps as _discover_snaps,
)
from streettracker.analysis.snap_assets import (
    load_bbox_index as _load_bbox_index,
)
from streettracker.analysis.snap_assets import (
    load_done_bbox_index as _load_done_bbox_index,
)
from streettracker.analysis.snap_assets import (
    resolve_bbox_hint as _resolve_bbox_hint,
)
from streettracker.analysis.snap_assets import (
    resolve_bbox_hint_window as _resolve_bbox_hint_window,
)

# Default location for the bespoke detector weights. The repo's
# top-level ``.gitignore`` carries ``*.pt`` so anything under here stays
# untracked.
DEFAULT_BESPOKE_MODEL = (
    Path(__file__).resolve().parent.parent
    / "analysis"
    / "alpr"
    / "models"
    / "license_plate_detector.pt"
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
        "--gpu",
        action="store_true",
        help="Pass gpu=True to EasyOCR.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images (0 = all).",
    )
    ap.add_argument(
        "--plate-pad-frac",
        type=float,
        default=0.10,
        help=(
            "Padding (fraction of plate-bbox side length) added around the "
            "detected plate before the OCR crop. Default 0.10. The "
            "2026-05-30 garbage analysis found 74%% of high-conf misreads "
            "were 6 chars (one short of a 7-char UK plate) -- the OCR fed a "
            "plate with an edge character clipped. A larger pad (try 0.25-"
            "0.35) gives the OCR the whole plate to segment."
        ),
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
        default=None,
        help=(
            "Ultralytics YOLO model used for the vehicle stage. Defaults "
            "to yolov8m for --crop-mode fullframe (full-frame detection "
            "needs the accuracy) and yolov8n for the legacy hint path. "
            "Auto-downloads on first use."
        ),
    )
    ap.add_argument(
        "--crop-mode",
        choices=("fullframe", "hint"),
        default="fullframe",
        help=(
            "How the plate detector's crop is targeted. 'fullframe' "
            "(default since 2026-07-28): detect vehicles on the whole "
            "snap, keep on-road candidates, rank by distance to the "
            "bbox hint and take the best plate detection -- the "
            "offline E1 re-score showed the recorded bboxes sit a "
            "median ~510 px from the car in the image, and this path "
            "lifted the R->L per-car canonical rate from 7-10%% to "
            "75-93%% on the same images. 'hint' restores the legacy "
            "hint-window crop behaviour (with --pre-crop as before)."
        ),
    )
    ap.add_argument(
        "--road-polygon",
        type=Path,
        default=Path(".claude/triggers_proposal.json"),
        help=(
            "JSON with the operator-traced road outline as fractional "
            "vertices (triggers_proposal.json schema: vertices_frac). "
            "Used by --crop-mode fullframe to reject off-road vehicles "
            "(driveways, parked forecourts). Missing file = no on-road "
            "filter, with a notice."
        ),
    )
    ap.add_argument(
        "--hint-lookahead",
        type=int,
        default=3,
        help=(
            "Build each snap's pre-crop hint as the UNION of the track's "
            "fire-time bboxes for snaps i..i+N (default 3). The 4K HTTP "
            "snap lands ~0.7-1.3s after the fire decision (p50 710ms) and "
            "a moving car exits its fire-time bbox in that gap -- the "
            "2026-06-10 triage of 99 R->L failures attributed 96%% of "
            "them to exactly this. Consecutive fires are 300-400ms apart, "
            "so the i..i+3 union covers the car's real observed positions "
            "over the latency window; at end-of-track the window is "
            "extrapolated from the last observed step instead. 0 restores "
            "the single-bbox legacy hint."
        ),
    )
    ap.add_argument(
        "--no-static-filter",
        action="store_true",
        help=(
            "Disable the automatic static-plate filter. By default, "
            "detections that recur at a fixed 4K position (while the "
            "tracked car moves, or across many distinct tracks with "
            "near-identical box size) are marked static_suspect in "
            "<session>_alpr.json, excluded from the per-track rollup, "
            "and summarised in <session>_static_plates.json. This is "
            "the dynamic successor to --ghost-mask: parked cars move "
            "between sessions, so a hand-traced rect can't keep up -- "
            "e.g. FD61PVX re-parked outside its masked spot and "
            "contributed 49 canonical 'reads' to passing R->L tracks "
            "across the 2026-06/07 soaks."
        ),
    )
    ap.add_argument(
        "--ghost-mask",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file with parked-car/no-go rects in 4K snap "
            'coords. Schema: {"source_size": [w, h], "rects_4k": '
            "[[x1, y1, x2, y2], ...]}. Each rect is zero-filled on the "
            "image before detection so the plate detector cannot see "
            "stationary plates that aren't from passing traffic. "
            "Eliminates the parked-car aliasing surfaced by Step 9 of "
            "the ANPR tuning loop in CLAUDE.md."
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
    done_index = _load_done_bbox_index(session_dir)
    if done_index:
        print(
            f"[alpr] {len(done_index)} completion-time bboxes available "
            f"(exact car positions; window fallback for the rest)"
        )
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

    ghost_rects, ghost_src = _load_ghost_mask(args.ghost_mask)
    if ghost_rects:
        print(f"[alpr] ghost mask: {len(ghost_rects)} rect(s) loaded (source_size={ghost_src})")

    all_records: list[dict] = []
    crops_root = session_dir / "alpr_crops"
    for runner in pipelines:
        print(f"[alpr] running pipeline: {runner.name}")
        crop_dir = crops_root / runner.name
        for i, (image_path, tid, snap_index, cls) in enumerate(snaps, 1):
            if args.hint_lookahead > 0:
                hint = _resolve_bbox_hint_window(
                    image_path,
                    tid,
                    snap_index,
                    bbox_index,
                    sub_size,
                    lookahead=args.hint_lookahead,
                    done_index=done_index,
                )
            else:
                hint = _resolve_bbox_hint(image_path, tid, snap_index, bbox_index, sub_size)
            result = runner.run(
                image_path,
                tid,
                snap_index,
                cls,
                crop_dir,
                bbox_hint=hint,
                ghost_rects=ghost_rects,
                ghost_source_size=ghost_src,
            )
            all_records.append(result.to_json())
            if i % 10 == 0 or i == len(snaps):
                print(f"  [{runner.name}] {i}/{len(snaps)} done")
            if result.error:
                print(f"  [{runner.name}] {image_path.name}: ERROR {result.error}")

    session_label = session_dir.name

    if not args.no_static_filter:
        from streettracker.analysis.alpr.staticfilter import (
            find_static_spots,
            mark_static_suspects,
        )

        # Completion-time bboxes are the car's true position; fall back
        # to fire-time bboxes for older sessions so the affine-
        # consistency check still has something to work with.
        car_index = {**bbox_index, **done_index}
        img_size = _first_snap_size(snaps)
        spots, consistent = find_static_spots(all_records, car_index, sub_size, img_size)
        n_suspect = mark_static_suspects(all_records, spots, consistent)
        spots_path = session_dir / f"{session_label}_static_plates.json"
        from datetime import datetime, timezone

        atomic_write_text(
            spots_path,
            json.dumps(
                {
                    # Provenance stamp: which crop path produced this
                    # session's ALPR results. The control panel reads it
                    # to badge sessions as re-enriched (fullframe) vs
                    # needing a re-run -- see introspect.session_info.
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "crop_mode": args.crop_mode,
                    "spots": spots,
                    "n_suspect_reads": n_suspect,
                },
                indent=2,
            ),
        )
        print(
            f"[alpr] static-plate filter: {len(spots)} static spot(s), "
            f"{n_suspect} detection(s) marked static_suspect "
            f"(excluded from the by-track rollup); map -> {spots_path.name}"
        )

    out_path = session_dir / f"{session_label}_alpr.json"
    atomic_write_text(out_path, json.dumps(all_records, indent=2))
    print(f"[alpr] wrote {out_path}")

    rollup = _rollup_by_track(all_records)
    rollup_path = session_dir / f"{session_label}_alpr_by_track.json"
    atomic_write_text(rollup_path, json.dumps(rollup, indent=2))
    print(f"[alpr] wrote {rollup_path}")
    return 0


def _first_snap_size(
    snaps: list[tuple[Path, int, int, str]],
) -> tuple[int, int] | None:
    """Pixel size of the first readable snap (header-only read).

    The static filter needs it to scale sub-stream car bboxes into
    snap coords. All snaps in a session share one main-stream
    resolution, so the first is representative.
    """
    for image_path, _tid, _snap, _cls in snaps:
        try:
            from PIL import Image
        except ImportError:
            return None
        try:
            with Image.open(image_path) as im:
                return (int(im.size[0]), int(im.size[1]))
        except (OSError, ValueError):
            continue
    return None


def _load_ghost_mask(
    path: Path | None,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int] | None]:
    """Load parked-car/no-go rects from a per-install JSON file.

    Schema: ``{"source_size": [w, h], "rects_4k": [[x1,y1,x2,y2], ...]}``.
    ``source_size`` is the pixel resolution the rect coords were
    captured in (typically the actual snap dimensions at the time of
    capture). The runner scales each rect to the current snap's
    dimensions at apply time, so the mask survives resolution changes
    (e.g. Reolink firmware that changes the main-stream resolution).
    Reolink snaps on this install are 4512x2512 despite the colloquial
    "4K" label, so ``source_size`` for the live mask is [4512, 2512],
    not [3840, 2160]. Empty / missing returns ``([], None)``.
    """
    if path is None:
        return [], None
    if not path.exists():
        print(f"[alpr] --ghost-mask {path}: not found, ignoring", file=sys.stderr)
        return [], None
    try:
        spec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"[alpr] --ghost-mask {path}: parse error ({e}), ignoring", file=sys.stderr)
        return [], None
    rects = spec.get("rects_4k") or []
    out: list[tuple[int, int, int, int]] = []
    for r in rects:
        try:
            x1, y1, x2, y2 = (int(v) for v in r)
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    src = spec.get("source_size")
    src_size: tuple[int, int] | None = None
    if isinstance(src, list) and len(src) == 2:
        try:
            src_size = (int(src[0]), int(src[1]))
        except (TypeError, ValueError):
            src_size = None
    return out, src_size


def _load_road_polygon(path: Path | None) -> list[tuple[float, float]] | None:
    """Fractional road-outline vertices for the fullframe crop path.

    Reads the ``vertices_frac`` field of a triggers_proposal-schema
    JSON. Missing/unreadable file returns ``None`` (no on-road filter)
    with a notice rather than an error -- the fullframe path still
    works, it just can't reject driveway/forecourt vehicles.
    """
    if path is None or not path.exists():
        if path is not None:
            print(
                f"[alpr] --road-polygon {path}: not found; fullframe crop "
                f"runs without the on-road filter",
                file=sys.stderr,
            )
        return None
    try:
        spec = json.loads(path.read_text())
        verts = spec.get("vertices_frac") or []
        out = [(float(x), float(y)) for x, y in verts]
        return out if len(out) >= 3 else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"[alpr] --road-polygon {path}: parse error ({e}), ignoring", file=sys.stderr)
        return None


def _build_pipelines(args: argparse.Namespace) -> list[PipelineRunner]:
    pipelines: list[PipelineRunner] = []
    bespoke_model = args.bespoke_model
    if not bespoke_model.is_absolute():
        bespoke_model = Path.cwd() / bespoke_model

    poly_frac = _load_road_polygon(args.road_polygon) if args.crop_mode == "fullframe" else None

    # Crop-targeting wrapper. Shared across pipelines so the ultralytics
    # YOLO weights load once per wrapper class.
    def _wrap(detector, suffix: str):
        if args.crop_mode == "fullframe":
            from streettracker.analysis.alpr.fullframe import TrajectoryCropDetector

            ff_wrapped = TrajectoryCropDetector(
                plate_detector=detector,
                vehicle_model=args.vehicle_model or "yolov8m.pt",
                road_polygon_frac=poly_frac,
            )
            ff_wrapped.name = f"fullframe-{suffix}"
            return ff_wrapped
        if not args.pre_crop:
            return detector
        from streettracker.analysis.alpr.precrop import PreCropDetector

        wrapped = PreCropDetector(
            plate_detector=detector,
            vehicle_model=args.vehicle_model or "yolov8n.pt",
            # Motion-window hints are wide; restore tight-crop resolution
            # by vehicle-detecting INSIDE the window before plate-detection.
            vehicle_stage_in_hint=args.hint_lookahead > 0,
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
            pipelines.append(
                PipelineRunner(
                    name="bespoke",
                    detector=_wrap(bespoke_det, "bespoke"),
                    recognizer=EasyOcrRecognizer(use_gpu=args.gpu),
                    plate_pad_frac=args.plate_pad_frac,
                )
            )

    if args.pipeline in ("both", "preferred") or args.ablation:
        from streettracker.analysis.alpr.preferred import (
            FastPlateOcrRecognizer,
            OpenImageModelsDetector,
        )

        oim_det = OpenImageModelsDetector(args.detector_model)
        if args.pipeline in ("both", "preferred"):
            pipelines.append(
                PipelineRunner(
                    name="preferred",
                    detector=_wrap(oim_det, "preferred"),
                    recognizer=FastPlateOcrRecognizer(args.ocr_model),
                    plate_pad_frac=args.plate_pad_frac,
                )
            )

    if args.ablation and bespoke_det is not None:
        from streettracker.analysis.alpr.preferred import FastPlateOcrRecognizer

        pipelines.append(
            PipelineRunner(
                name="ablation_bespokedet_fastocr",
                detector=_wrap(bespoke_det, "ablation_bespokedet_fastocr"),
                recognizer=FastPlateOcrRecognizer(args.ocr_model),
                plate_pad_frac=args.plate_pad_frac,
            )
        )

    return pipelines


def _rollup_by_track(records: list[dict]) -> dict:
    """Per-track best-of-N + consensus rollup.

    For each ``(pipeline, track_id)``:
    * ``best_<pipeline>``: the single read with the highest
      ``ocr_conf`` (legacy behaviour, kept for backward compat).
    * ``consensus_<pipeline>``: the confidence-weighted character-vote
      consensus across all of the track's reads above
      :data:`consensus.MIN_INPUT_CONF`. Often outperforms the best-of-N
      pick when individual reads are low-conf but agree -- see
      :func:`streettracker.analysis.alpr.consensus.consensus_plate`.

    Both fields land in the per-track rollup so downstream consumers
    can pick whichever is appropriate for their use case.
    """
    from streettracker.analysis.alpr.consensus import consensus_plate

    by_pipe_track: dict[str, dict[int, dict]] = defaultdict(dict)
    by_pipe_track_all: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        p = r["pipeline"]
        tid = r["track_id"]
        if not r.get("ocr_text"):
            continue
        if r.get("static_suspect"):
            # Static-plate reads (parked cars / fixed scene objects
            # swept into the crop window) must not become a track's
            # best read or vote in its consensus -- that's exactly the
            # mis-attribution the filter exists to stop.
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
                # Persist the canonical-UK-plate annotation through to the
                # best-per-track rollup so downstream aggregators
                # (vehicles.py, dvsa-label) can filter without re-running
                # the regex.
                "canonical_uk_shape": r.get("canonical_uk_shape"),
            }
        by_pipe_track_all[p][tid].append(r)

    tracks: dict[int, dict] = {}
    for pipe, by_tid in by_pipe_track.items():
        for tid, best in by_tid.items():
            tracks.setdefault(tid, {"track_id": tid})[f"best_{pipe}"] = best
    for pipe, by_tid_reads in by_pipe_track_all.items():
        for tid, reads in by_tid_reads.items():
            c = consensus_plate(reads)
            if c is not None:
                tracks.setdefault(tid, {"track_id": tid})[f"consensus_{pipe}"] = c

    return {"tracks": sorted(tracks.values(), key=lambda r: r["track_id"])}


if __name__ == "__main__":
    raise SystemExit(main())
