"""Per-track state buffer used by the live runtime.

UltralyticsTracker (see ``streettracker.inference.ultralytics_runner``)
hands us a flat ``list[Detection]`` per frame, with ``track_id`` already
populated by BotSORT. The runtime, however, needs more state than
detections carry: a rolling history of motion samples (for direction /
speed / lane), a buffered list of crops (for color voting at
finalize), a per-track HQ-best slot (the cleanest non-edge crop seen),
and snapshot bookkeeping (how many fires we've committed, which
``_main_<N>.jpg`` files actually landed).

NanoTracker carried all of that on the ``Track`` objects owned by its
hand-rolled ``IoUTracker``. StreetTracker delegates tracking to
Ultralytics, so we need a thin wrapper that re-creates that state.
:class:`TrackBuffer` is that wrapper: feed it detections each frame,
and it returns the expired tracks (those that haven't been seen for
``max_misses`` consecutive frames) so the runtime can finalize them.

Pure logic; no I/O. The single side-effecting helper is
:func:`save_thumbnail`, which writes a JPEG via OpenCV. The rest --
:func:`safe_crop`, :func:`sharpness_score`, :func:`compute_attributes`
-- are pure functions of their inputs and are unit-tested in isolation.

Ported from NanoTracker (``nano_tracker.py``):

* ``Track`` + ``CropSample`` + ``TrackPoint`` (lines 80-160)
* ``_safe_crop``, ``_sharpness_score``, ``_update_hq_best``
  (lines 317-368)
* ``IoUTracker._append_point`` (lines 266-285) -- the per-detection
  state update is what we keep; the IoU matching itself is gone because
  Ultralytics already did the work.
* ``compute_attributes`` (lines 489-530)
* ``save_thumbnail`` (lines 537-544)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from streettracker.common.coco import class_name
from streettracker.common.output import format_wall
from streettracker.common.schema import TrackRecord, asset_prefix_for_class
from streettracker.common.types import Detection

if TYPE_CHECKING:
    import numpy as np


# ----------------------------------------------------------------------
# Defaults match NanoTracker (nano_tracker.py:102-106). Bumping any of
# these has cross-cutting implications -- e.g. lowering CROP_BUFFER_MAX
# reduces memory per track but may rob the color voter of large clean
# samples. Tune carefully and only via deliberate config later.

_CROP_BUFFER_MAX = 12  # halve to ~6 via [::2] on overflow
_CROP_PAD_FRAC = 0.2  # 20% bbox padding gives ALPR / OCR some context
_HQ_EDGE_MARGIN_PX = 4  # bboxes touching the frame edge are clipped
_HQ_AREA_KEEP_FRAC = 0.7  # re-score sharpness only if area >= this * current best
_HQ_SHARPNESS_TARGET_PX = 128  # downsample for sub-millisecond Laplacian


# ----------------------------------------------------------------------
# Per-track state.


@dataclass(slots=True)
class MotionPoint:
    """One bbox sample. Per-frame; persists for the life of the track."""

    frame_idx: int
    t: float
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass(slots=True)
class CropSample:
    """Buffered crop used by the finalize-time color voter."""

    t: float
    score: float
    crop: np.ndarray
    on_edge: bool


@dataclass(slots=True)
class BufferedTrack:
    """Mutable per-track state -- one per active BotSORT track id.

    The runtime ingests detections into the buffer; on finalize the
    buffer hands the track back so :func:`compute_attributes` can turn
    it into a :class:`TrackRecord`.
    """

    id: int
    class_id: int
    points: list[MotionPoint] = field(default_factory=list)
    crops: list[CropSample] = field(default_factory=list)
    hq_best_crop: np.ndarray | None = None
    hq_best_score: float = 0.0
    hq_best_area: int = 0
    snap_count: int = 0
    snap_saved_indexes: set[int] = field(default_factory=set)
    # Per-snap-index record of the asset prefix used to build the on-disk
    # path at fire time. BotSORT can reassign a track's class as evidence
    # accumulates (see ``ingest`` below), so a track may fire its first
    # snap as "person" and finalize as "vehicle" -- the on-disk filename
    # would then mismatch ``record.asset_prefix``. ``finalize_track``
    # consults this dict to rename any mismatched files; the snap
    # ``_on_done`` callback consults ``final_prefix`` for in-flight
    # fires that complete after finalize.
    snap_fire_prefixes: dict[int, str] = field(default_factory=dict)
    # Set in ``finalize_track`` to lock the prefix used for the
    # TrackRecord. ``None`` while the track is still active.
    final_prefix: str | None = None
    misses: int = 0  # consecutive frames the tracker did not see this id
    # ``time.monotonic()`` of the last blur-skip log line emitted for
    # this track. The runtime's snap walk throttles the per-track log
    # to once every few seconds so a fast-moving (and therefore
    # consistently blurred) vehicle doesn't flood the journal.
    last_blur_skip_logged_mono: float = 0.0

    @property
    def last_bbox(self) -> tuple[float, float, float, float] | None:
        if not self.points:
            return None
        p = self.points[-1]
        return (p.x1, p.y1, p.x2, p.y2)


# ----------------------------------------------------------------------
# Pure helpers (no module state -- safe to unit-test in isolation).


def safe_crop(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    pad_frac: float = 0.0,
) -> np.ndarray | None:
    """Crop ``frame`` at the bbox, optionally padded by ``pad_frac`` of
    its size on each side. Clamped to frame bounds; returns ``None`` if
    the resulting region is empty.
    """
    h, w = frame.shape[:2]
    px = (x2 - x1) * pad_frac
    py = (y2 - y1) * pad_frac
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()


def sharpness_score(crop: np.ndarray) -> float:
    """Variance-of-Laplacian sharpness score (higher = sharper).

    Downsamples to ``_HQ_SHARPNESS_TARGET_PX`` longest side so the
    Laplacian itself is sub-millisecond on a Jetson Orin even for big
    crops. Relative sharpness ranking is preserved by the downsample.
    """
    import cv2

    h, w = crop.shape[:2]
    longest = max(h, w)
    if longest > _HQ_SHARPNESS_TARGET_PX:
        scale = _HQ_SHARPNESS_TARGET_PX / float(longest)
        small = cv2.resize(
            crop,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = crop
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def save_thumbnail(crop: np.ndarray, path: Path, *, quality: int = 85) -> bool:
    """Write ``crop`` to ``path`` as a JPEG. Returns ``True`` on success.

    Failures are non-fatal at the call site -- the runtime keeps going
    and the missing tile just renders blank in the dashboard.
    """
    try:
        import cv2

        return bool(cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, quality]))
    except Exception:
        return False


def total_displacement(points: list[MotionPoint]) -> float:
    """Path length (sum of Euclidean step distances)."""
    if len(points) < 2:
        return 0.0
    tot = 0.0
    for i in range(1, len(points)):
        dx = points[i].cx - points[i - 1].cx
        dy = points[i].cy - points[i - 1].cy
        tot += (dx * dx + dy * dy) ** 0.5
    return tot


def update_hq_best(track: BufferedTrack, crop: np.ndarray, on_edge: bool) -> None:
    """Refresh the per-track HQ best slot. See module docstring.

    Score = area * sharpness, restricted to non-edge crops. The
    sharpness Laplacian is skipped when the area can't beat the
    current best on its own (most frames after the peak), keeping
    per-detection cost near zero on a hot pass.
    """
    if on_edge:
        return
    area = int(crop.shape[0]) * int(crop.shape[1])
    if track.hq_best_crop is not None and area < _HQ_AREA_KEEP_FRAC * track.hq_best_area:
        return
    sharp = sharpness_score(crop)
    score = area * sharp
    if track.hq_best_crop is None or score > track.hq_best_score:
        track.hq_best_crop = crop
        track.hq_best_score = score
        track.hq_best_area = area


# ----------------------------------------------------------------------
# TrackBuffer -- the runtime's view of in-flight tracks.


@dataclass(slots=True)
class TrackBuffer:
    """Per-id rolling state, fed one frame's detections at a time.

    Detections with ``track_id is None`` (BotSORT didn't promote them
    to a track) are ignored. Tracks not seen in a frame have their
    ``misses`` counter incremented; once a track's misses exceed
    ``max_misses`` it is finalized and returned from :meth:`ingest`.
    """

    max_misses: int = 15
    _active: dict[int, BufferedTrack] = field(default_factory=dict)

    def active(self) -> list[BufferedTrack]:
        """Snapshot of the currently-live tracks. Live view across
        :meth:`ingest` calls; do not mutate.
        """
        return list(self._active.values())

    def ingest(
        self,
        detections: list[Detection],
        frame_idx: int,
        t: float,
        frame: np.ndarray,
    ) -> list[BufferedTrack]:
        """Update state from one frame's detections; return expired tracks.

        ``frame`` is the BGR frame that produced ``detections``; we
        sample crops from it for the color voter + HQ-best slot.

        Returns the list of tracks whose ``misses`` exceeded
        ``max_misses`` *this* frame -- the runtime should finalize them
        and stop referencing the returned objects (the buffer no longer
        owns them).
        """
        seen_this_frame: set[int] = set()
        h, w = frame.shape[:2]

        for d in detections:
            if d.track_id is None:
                continue
            seen_this_frame.add(d.track_id)
            tr = self._active.get(d.track_id)
            if tr is None:
                tr = BufferedTrack(id=d.track_id, class_id=d.class_id)
                self._active[d.track_id] = tr
            else:
                # Update class_id to most recent (BotSORT occasionally
                # reassigns a track's class as evidence accumulates).
                tr.class_id = d.class_id
            self._append_point(tr, d, frame_idx, t, frame, w, h)

        # Tick misses on tracks not seen this frame; expire those over the cap.
        expired: list[BufferedTrack] = []
        for tid in list(self._active.keys()):
            if tid in seen_this_frame:
                self._active[tid].misses = 0
            else:
                tr = self._active[tid]
                tr.misses += 1
                if tr.misses > self.max_misses:
                    expired.append(tr)
                    del self._active[tid]
        return expired

    def flush(self) -> list[BufferedTrack]:
        """Finalize all remaining active tracks (shutdown / IR transition)."""
        out = list(self._active.values())
        self._active.clear()
        return out

    @staticmethod
    def _append_point(
        tr: BufferedTrack,
        d: Detection,
        frame_idx: int,
        t: float,
        frame: np.ndarray,
        w: int,
        h: int,
    ) -> None:
        tr.points.append(
            MotionPoint(
                frame_idx=frame_idx,
                t=t,
                cx=d.cx,
                cy=d.cy,
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                score=d.score,
            )
        )
        on_edge = (
            d.x1 < _HQ_EDGE_MARGIN_PX
            or d.y1 < _HQ_EDGE_MARGIN_PX
            or d.x2 > w - _HQ_EDGE_MARGIN_PX
            or d.y2 > h - _HQ_EDGE_MARGIN_PX
        )
        crop = safe_crop(frame, d.x1, d.y1, d.x2, d.y2, pad_frac=_CROP_PAD_FRAC)
        if crop is None:
            return
        tr.crops.append(CropSample(t=t, score=d.score, crop=crop, on_edge=on_edge))
        if len(tr.crops) > _CROP_BUFFER_MAX:
            # Decimate by 2 to keep the time-spread; alternative would
            # be FIFO eviction but that biases toward late-track crops
            # which often have poor lighting on departure.
            tr.crops = tr.crops[::2]
        update_hq_best(tr, crop, on_edge)


# ----------------------------------------------------------------------
# compute_attributes -- turns a finalized BufferedTrack into a TrackRecord.


def compute_attributes(
    track: BufferedTrack,
    frame_h: int,
    *,
    min_duration_s: float,
    parked_disp_px: float,
    color: str,
    t_start_wall: float,
    main_snaps: list[int] | None = None,
) -> TrackRecord | None:
    """Compute attributes for one finalized track; return ``None`` if filtered.

    Filters:
    * fewer than 2 motion points -- nothing to measure motion against;
    * total duration below ``min_duration_s`` -- too brief to be reliable;
    * net displacement below ``parked_disp_px`` -- track was parked
      (or BotSORT held it across a stop), not relevant for road traffic.

    ``main_snaps`` is the list of N values whose ``_main_N.jpg`` actually
    landed on disk. Passed in (rather than read off the track) because
    snap writes complete asynchronously and the runtime is the source
    of truth for which fires succeeded.

    Mirrors NanoTracker's ``compute_attributes`` (nano_tracker.py:489-530)
    field-by-field, but emits a :class:`TrackRecord` dataclass instead
    of a plain dict.
    """
    if len(track.points) < 2:
        return None
    duration = track.points[-1].t - track.points[0].t
    if duration < min_duration_s:
        return None

    p0 = track.points[0]
    pN = track.points[-1]
    net_disp = ((pN.cx - p0.cx) ** 2 + (pN.cy - p0.cy) ** 2) ** 0.5
    if net_disp < parked_disp_px:
        return None

    disp = total_displacement(track.points)
    speed_px_s = net_disp / duration if duration > 0 else 0.0
    direction = "left to right" if pN.cx > p0.cx else "right to left"

    avg_y = sum(p.cy for p in track.points) / len(track.points)
    third = frame_h / 3.0
    lane = "top" if avg_y < third else ("middle" if avg_y < 2 * third else "bottom")

    avg_conf = sum(p.score for p in track.points) / len(track.points)
    return TrackRecord(
        track_id=track.id,
        class_id=track.class_id,
        class_name=class_name(track.class_id),
        time_start=format_wall(t_start_wall + p0.t),
        time_end=format_wall(t_start_wall + pN.t),
        time_start_unix=round(t_start_wall + p0.t, 2),
        time_end_unix=round(t_start_wall + pN.t, 2),
        time_start_s=round(p0.t, 2),
        time_end_s=round(pN.t, 2),
        duration_visible=round(duration, 2),
        direction=direction,
        speed_px_s=round(speed_px_s, 1),
        color=color,
        lane=lane,
        avg_confidence=round(avg_conf, 3),
        displacement_px=round(disp, 1),
        net_displacement_px=round(net_disp, 1),
        num_detections=len(track.points),
        asset_prefix=asset_prefix_for_class(track.class_id),
        main_snaps=sorted(main_snaps) if main_snaps else [],
    )
