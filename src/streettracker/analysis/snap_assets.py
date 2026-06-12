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


def load_done_bbox_index(
    session_dir: Path,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """``(track_id, snap_index) -> completion-time bbox`` from
    ``main_snap_bboxes_done`` (where the car was when the 4K snap
    actually landed). Empty for sessions recorded before the field
    existed -- callers fall back to fire-time windows."""
    session_label = session_dir.name
    data_path = session_dir / f"{session_label}_data.json"
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    if not data_path.exists():
        return out
    try:
        records = json.loads(data_path.read_text())
    except (OSError, json.JSONDecodeError):
        return out
    for r in records:
        tid = r.get("track_id")
        snaps = r.get("main_snaps") or []
        bboxes = r.get("main_snap_bboxes_done")
        if tid is None or not snaps or not bboxes or len(bboxes) != len(snaps):
            continue
        for n, bb in zip(snaps, bboxes, strict=False):
            if bb is None:
                continue
            try:
                x1, y1, x2, y2 = (int(v) for v in bb)
            except (TypeError, ValueError):
                continue
            out[(int(tid), int(n))] = (x1, y1, x2, y2)
    return out


def _scale_bbox_to_image(
    image_path: Path,
    bbox: tuple[float, float, float, float],
    sub_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Scale a sub-stream-coord bbox into ``image_path``'s pixel coords.
    Returns ``None`` when the image header can't be read."""
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
    x1, y1, x2, y2 = bbox
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


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
    return _scale_bbox_to_image(image_path, sub_bbox, sub_size)


# Cap on how far the end-of-track extrapolation may shift the window,
# as a multiple of the bbox's own dimension per axis. Guards against a
# BotSORT glitch step (huge bbox jump) blowing the window up to most of
# the frame -- which would resurrect the parked-car aliasing the bbox
# hints exist to prevent.
_WINDOW_MAX_SHIFT_BBOXES = 2.5


def resolve_bbox_hint_window(
    image_path: Path,
    track_id: int,
    snap_index: int,
    bbox_index: dict[tuple[int, int], tuple[int, int, int, int]],
    sub_size: tuple[int, int] | None,
    *,
    lookahead: int = 3,
    done_index: dict[tuple[int, int], tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int] | None:
    """Motion-aware crop window: the union of this snap's bbox with the
    track's next ``lookahead`` fire-time bboxes, scaled to snap pixels.

    The 2026-06-10 R->L failure triage (99 hand-labelled failures) found
    96 % of no-read snaps were caused by snap latency: the 4K HTTP snap
    lands ~0.7-1.3 s after the fire decision (p50 710 ms, p99 1.2 s),
    by which time the car has left its fire-time bbox entirely -- the
    plate detector was being shown empty road. Consecutive pipeline
    fires are 300-400 ms apart, so the union of bboxes ``i..i+3`` spans
    the car's real observed positions over ~0-1.2 s -- exactly the
    window the image could have been captured in. No velocity model,
    just positions BotSORT actually measured.

    At end-of-track (no later bboxes -- the worst case: the car is
    leaving the band at speed) the window is instead extrapolated
    linearly from the last observed inter-snap step, capped at
    ``_WINDOW_MAX_SHIFT_BBOXES`` bbox-dimensions per axis.

    ``lookahead=0`` degrades to :func:`resolve_bbox_hint` exactly.
    Returns ``None`` under the same conditions as the single-bbox
    resolver (no bbox for this snap / unknown sub-stream size).

    ``done_index`` (see :func:`load_done_bbox_index`) supplies
    completion-time bboxes where the runtime recorded them. When one
    exists for this snap, the window is simply
    ``union(fire bbox, done bbox)`` -- the car's true position in the
    saved image, no lookahead or extrapolation needed.
    """
    base = bbox_index.get((track_id, snap_index))
    if base is None or sub_size is None:
        return None

    done = (done_index or {}).get((track_id, snap_index))
    if done is not None:
        window_exact = (
            min(base[0], done[0]),
            min(base[1], done[1]),
            max(base[2], done[2]),
            max(base[3], done[3]),
        )
        return _scale_bbox_to_image(image_path, window_exact, sub_size)

    boxes: list[tuple[float, float, float, float]] = [base]
    for k in range(1, max(0, lookahead) + 1):
        nxt = bbox_index.get((track_id, snap_index + k))
        if nxt is not None:
            boxes.append(nxt)

    if len(boxes) == 1 and lookahead > 0:
        prev = bbox_index.get((track_id, snap_index - 1))
        if prev is not None:
            # Last snap of the track: extrapolate the window along the
            # final observed step instead of giving up.
            bw, bh = base[2] - base[0], base[3] - base[1]
            dx = float(base[0] - prev[0] + base[2] - prev[2]) / 2.0
            dy = float(base[1] - prev[1] + base[3] - prev[3]) / 2.0
            sx = max(
                -_WINDOW_MAX_SHIFT_BBOXES * bw, min(_WINDOW_MAX_SHIFT_BBOXES * bw, dx * lookahead)
            )
            sy = max(
                -_WINDOW_MAX_SHIFT_BBOXES * bh, min(_WINDOW_MAX_SHIFT_BBOXES * bh, dy * lookahead)
            )
            boxes.append((base[0] + sx, base[1] + sy, base[2] + sx, base[3] + sy))

    window = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
    return _scale_bbox_to_image(image_path, window, sub_size)
