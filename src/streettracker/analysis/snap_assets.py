"""Shared session-asset helpers for off-device post-processing passes.

Both ``alpr-run`` and ``makemodel`` consume the same per-session
artifacts: the ``*_main_*.jpg`` 4K snaps, the per-snap BotSORT bboxes
persisted in ``<session>_data.json`` (``main_snap_bboxes``, parallel to
``main_snaps``), and the sub-stream frame size in ``<session>_meta.json``
needed to scale those bboxes into the snap's pixel coords.

These helpers were originally private to ``cli/alpr_run.py``; they're
lifted here verbatim (made public) so the make/model classifier can
target the *same* tracked-vehicle crop the plate detector does without
duplicating the scaling logic. ``cli/alpr_run.py`` re-imports them.
"""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.analysis.alpr.base import parse_snap_filename


def discover_vehicle_snaps(session_dir: Path) -> list[tuple[Path, int, int, str]]:
    """Return ``[(path, track_id, snap_index, class_name)]`` for vehicle
    main snaps only -- person crops aren't useful for ALPR or make/model."""
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


def load_bbox_index(
    session_dir: Path,
) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], tuple[int, int] | None]:
    """Build ``(track_id, snap_index) -> sub_stream_bbox`` from data.json
    and pick the sub-stream frame size out of _meta.json. Returns empty
    dict + ``None`` size for sessions that pre-date the bbox-capture
    change (no error, callers fall back to no-hint)."""
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


def resolve_bbox_hint(
    image_path: Path,
    track_id: int,
    snap_index: int,
    bbox_index: dict[tuple[int, int], tuple[int, int, int, int]],
    sub_size: tuple[int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Look up the per-snap bbox and scale it from sub-stream coords
    into the snap's pixel coords. Returns ``None`` if no bbox is known
    for this snap, or if we don't know the sub-stream frame size
    (can't scale safely without it)."""
    sub_bbox = bbox_index.get((track_id, snap_index))
    if sub_bbox is None or sub_size is None:
        return None
    sub_w, sub_h = sub_size
    if sub_w <= 0 or sub_h <= 0:
        return None
    # Read image dimensions from the JPEG header without decoding pixels.
    # PIL.Image.open is lazy: ``.size`` only parses the header.
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except (OSError, ValueError):
        return None
    # Sub-stream and main may have slightly different aspect ratios, so
    # scale x and y independently rather than picking a single ratio.
    sx = w / sub_w
    sy = h / sub_h
    x1, y1, x2, y2 = sub_bbox
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
