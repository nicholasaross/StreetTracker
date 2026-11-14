"""Locate the tracked vehicle in a 4K snap for the CNN heads.

The make, colour and body-type classifiers (and the corpus they train on)
crop each 4K snap to "the tracked car". Historically that crop came from
the FIRE-TIME sub-stream bbox (:func:`snap_assets.resolve_bbox_hint`),
but the 4K snap lands ~0.7 s after the fire decision, so the car has
moved on. A 2026-09-24 audit (``.claude/makemodel_crop_audit.py``, 799
plate-anchored snaps of ``uk_crops_0730_576``) found only 7.6 % of the
training crops held >=80 % of the labelled car and 66.5 % held <30 % --
mostly empty road or a different (often parked) vehicle. Same root cause
as the R->L ANPR "geometry cap" fixed by the fullframe crop path on
2026-07-28 (:mod:`streettracker.analysis.alpr.fullframe`).

This module locates the car on the full frame instead. Modes:

* ``"hint"`` -- legacy: the stale fire-time bbox, scaled to the snap.
  Kept so models trained on hint crops keep their exact behaviour.
* ``"plate"`` -- training: the vehicle box containing a read of the
  car's OWN plate on this snap (``<session>_alpr.json``). Certain
  identity; a snap that didn't read the plate yields nothing.
* ``"fullframe"`` -- inference: plate-anchored when this snap carries a
  non-static canonical plate read, else the nearest on-road vehicle to
  the stale hint (``fullframe.rank_vehicle_candidates`` rank 0).

In the last two modes a vehicle stopped in the road is not a candidate for
a track that is only passing it, and nor is a box hidden behind one
(:mod:`streettracker.analysis.stationary`). Which vehicles are stopped is
read from every snap's boxes, so it is only fully known once the whole
session has been through the detector: see :meth:`SnapVehicleLocator.settle`.

Full-frame vehicle detection is the expensive step (YOLOv8m @1920, ~0.1 s
per 4K snap), so boxes are cached per session in
``<session>_vehicle_boxes.json`` and shared by the corpus builder and all
three inference commands.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from streettracker.analysis.alpr.base import parse_snap_filename
from streettracker.analysis.alpr.fullframe import (
    DEFAULT_MIN_VEHICLE_H_PX,
    DEFAULT_VEHICLE_IMGSZ,
    rank_vehicle_candidates,
)
from streettracker.analysis.snap_assets import (
    load_bbox_index,
    load_done_bbox_index,
    load_snap_times,
    resolve_bbox_hint,
)
from streettracker.analysis.stationary import (
    HIDDEN_FRAC,
    MAX_GAP_S,
    MIN_GAP_S,
    MIN_TRACKS,
    NEIGHBOUR_TRACKS,
    OWN_BOX_IOU,
    SAME_BOX_IOU,
    SAME_VEHICLE_IOU,
    SnapSighting,
    find_stationary_vehicles,
    group_refused,
    same_vehicle,
    sits_on,
    taken_by,
)
from streettracker.common.fileio import atomic_write_json, atomic_write_text
from streettracker.common.session_files import SessionFiles

if TYPE_CHECKING:
    import numpy as np

Box = tuple[float, float, float, float]
IntBox = tuple[int, int, int, int]

CROP_MODES = ("hint", "plate", "fullframe")

# COCO car, motorcycle, bus, truck. Motorcycle is included (unlike the
# ALPR fullframe path) because the make corpus carries motorbike makes
# (KAWASAKI / YAMAHA / TRIUMPH / PIAGGIO) whose plates sit on a bike box.
VEHICLE_BOX_CLASSES = (2, 3, 5, 7)
VEHICLE_BOX_CONF = 0.2
DEFAULT_VEHICLE_MODEL = "yolov8m.pt"
CACHE_SUFFIX = "_vehicle_boxes.json"
_CACHE_FLUSH_EVERY = 500

# Same-length OCR variants of one plate score >= 85 (one char off on 7).
_PLATE_MATCH_RATIO = 85


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def plate_matches(read: str | None, plate: str) -> bool:
    """``read`` is ``plate`` or a same-length one-character OCR variant."""
    if not read:
        return False
    read = read.replace(" ", "").upper()
    plate = plate.replace(" ", "").upper()
    if read == plate:
        return True
    if len(read) != len(plate):
        return False
    from rapidfuzz import fuzz

    return fuzz.ratio(read, plate) >= _PLATE_MATCH_RATIO


def load_alpr_reads(session_dir: Path) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """``(track_id, snap_index) -> [alpr record, ...]`` from ``_alpr.json``
    (empty when the session hasn't been through ``alpr-run``)."""
    path = session_dir / f"{session_dir.name}_alpr.json"
    out: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for r in records:
        tid, n = r.get("track_id"), r.get("snap_index")
        if tid is None or n is None:
            continue
        out[(int(tid), int(n))].append(r)
    return out


def anchor_plate_bbox(records: list[dict[str, Any]], plate: str | None) -> Box | None:
    """Plate bbox (full-image px) of the best usable read on one snap.

    With ``plate`` given, only reads of that plate (or a one-char OCR
    variant) count -- the training anchor. With ``plate=None``, any
    canonical-shape read does -- the inference anchor. Static-suspect
    reads (parked-car beacons) never anchor."""
    best: dict[str, Any] | None = None
    for r in records:
        if r.get("static_suspect") or not r.get("det_bbox") or not r.get("ocr_text"):
            continue
        if plate is not None:
            if not plate_matches(r["ocr_text"], plate):
                continue
        elif not r.get("canonical_uk_shape"):
            continue
        if best is None or (r.get("ocr_conf") or 0.0) > (best.get("ocr_conf") or 0.0):
            best = r
    if best is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in best["det_bbox"])
    return (x1, y1, x2, y2)


def plate_anchored_box(boxes: list[Box], plate_bbox: Box) -> Box | None:
    """The tightest vehicle box containing the plate's centre."""
    pcx, pcy = (plate_bbox[0] + plate_bbox[2]) / 2.0, (plate_bbox[1] + plate_bbox[3]) / 2.0
    containing = [b for b in boxes if b[0] <= pcx <= b[2] and b[1] <= pcy <= b[3]]
    return min(containing, key=_area) if containing else None


def _to_int_box(b: Box) -> IntBox:
    return (int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3])))


def yolo_vehicle_detector(
    model: str = DEFAULT_VEHICLE_MODEL,
    *,
    imgsz: int = DEFAULT_VEHICLE_IMGSZ,
    conf: float = VEHICLE_BOX_CONF,
) -> Callable[[np.ndarray], list[tuple[float, float, float, float, float]]]:
    """Lazy ultralytics full-frame vehicle detector: image -> [(x1,y1,x2,y2,conf)]."""
    yolo: Any = None

    def detect(image: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        nonlocal yolo
        if yolo is None:
            from ultralytics import YOLO

            yolo = YOLO(model)
        res = yolo.predict(
            image, classes=list(VEHICLE_BOX_CLASSES), conf=conf, imgsz=imgsz, verbose=False
        )[0]
        xy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        return [
            (float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(c))
            for r, c in zip(xy, cf, strict=True)
        ]

    return detect


class VehicleBoxCache:
    """Per-session cache of full-frame vehicle boxes keyed by snap filename.

    The file records the detector parameters; a cache written with
    different ones is ignored (rebuilt), so changing the model or imgsz
    never serves stale boxes. Writes are atomic and flushed every
    ``_CACHE_FLUSH_EVERY`` new entries so a killed job keeps its work.
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        detector: Callable[[np.ndarray], list[tuple[float, float, float, float, float]]]
        | None = None,
        model: str = DEFAULT_VEHICLE_MODEL,
        imgsz: int = DEFAULT_VEHICLE_IMGSZ,
        conf: float = VEHICLE_BOX_CONF,
    ) -> None:
        self.path = session_dir / f"{session_dir.name}{CACHE_SUFFIX}"
        self._params = {
            "model": Path(model).name,
            "imgsz": imgsz,
            "conf": conf,
            "classes": list(VEHICLE_BOX_CLASSES),
        }
        self._detect = detector or yolo_vehicle_detector(model, imgsz=imgsz, conf=conf)
        self._boxes: dict[str, list[list[float]]] = {}
        self._dirty = 0
        self.n_hits = 0
        self.n_detected = 0
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if data.get("params") == self._params:
                self._boxes = data.get("boxes", {})

    def boxes(self, image_name: str, image: np.ndarray) -> list[Box]:
        cached = self._boxes.get(image_name)
        if cached is not None:
            self.n_hits += 1
        else:
            cached = [[round(v, 1) for v in det] for det in self._detect(image)]
            self._boxes[image_name] = cached
            self.n_detected += 1
            self._dirty += 1
            if self._dirty >= _CACHE_FLUSH_EVERY:
                self.save()
        return [(b[0], b[1], b[2], b[3]) for b in cached]

    def known(self) -> dict[str, list[Box]]:
        """Every snap's boxes held so far, by snap filename."""
        return {
            name: [(b[0], b[1], b[2], b[3]) for b in boxes] for name, boxes in self._boxes.items()
        }

    def save(self) -> None:
        if not self._dirty:
            return
        atomic_write_text(self.path, json.dumps({"params": self._params, "boxes": self._boxes}))
        self._dirty = 0


@dataclass(frozen=True, slots=True)
class _Located:
    """What :meth:`SnapVehicleLocator.locate` was given for one snap, and the
    box it returned: enough to choose again without the pixels."""

    name: str
    size: tuple[int, int]
    hint: IntBox | None
    plate: str | None
    box: IntBox | None


class SnapVehicleLocator:
    """Where to crop the tracked vehicle in one session's 4K snaps.

    ``locate`` returns ``(box, source)``: the integer box in snap pixel
    coords (or ``None`` -- skip this snap) and which rule produced it
    (``"hint"`` / ``"plate"`` / ``"trajectory"``).

    Outside ``hint`` mode, a stationary vehicle is not a candidate for a
    track that is passing it, and nor is a box hidden behind one. A locator
    starts out knowing the stationary vehicles in whatever the box cache
    already holds; :meth:`settle` works them out again from every snap
    located and says which crops change. ``stationary_guard=False`` turns
    this off.
    """

    def __init__(
        self,
        session_dir: Path,
        mode: str,
        *,
        road_polygon_frac: list[tuple[float, float]] | None = None,
        cache: VehicleBoxCache | None = None,
        min_vehicle_h_px: int = DEFAULT_MIN_VEHICLE_H_PX,
        stationary_guard: bool = True,
    ) -> None:
        if mode not in CROP_MODES:
            raise ValueError(f"crop mode must be one of {CROP_MODES}, got {mode!r}")
        self.mode = mode
        self._session_dir = session_dir
        self._bbox_index, self._sub_size = load_bbox_index(session_dir)
        self._poly = road_polygon_frac
        self._min_h = min_vehicle_h_px
        self._reads = load_alpr_reads(session_dir) if mode != "hint" else {}
        self._cache = cache if cache is not None or mode == "hint" else VehicleBoxCache(session_dir)
        self.source_counts: dict[str, int] = defaultdict(int)
        self._sources: dict[tuple[int, int], str] = {}

        self._guard = stationary_guard and mode != "hint"
        self._done_index = load_done_bbox_index(session_dir) if self._guard else {}
        self._snap_times = load_snap_times(session_dir) if self._guard else {}
        self._located: dict[tuple[int, int], _Located] = {}
        self._stationary: dict[tuple[int, int], list[Box]] = {}
        #: Set by :meth:`settle`: the crops a stationary vehicle would have
        #: taken, and how many of those snaps were left with no crop at all.
        self.n_refused = 0
        self.n_left_without = 0
        self._find_stationary()

    @property
    def n_bboxes(self) -> int:
        return len(self._bbox_index)

    def has_plate_read(self, track_id: int, snap_index: int, plate: str) -> bool:
        """Cheap (no decode) pre-filter for ``plate`` mode."""
        return anchor_plate_bbox(self._reads.get((track_id, snap_index), []), plate) is not None

    def plate_bbox(self, track_id: int, snap_index: int, plate: str | None) -> Box | None:
        return anchor_plate_bbox(self._reads.get((track_id, snap_index), []), plate)

    def locate(
        self,
        path: Path,
        track_id: int,
        snap_index: int,
        image: np.ndarray | None,
        *,
        plate: str | None = None,
    ) -> tuple[IntBox | None, str | None]:
        key = (track_id, snap_index)
        hint = resolve_bbox_hint(path, track_id, snap_index, self._bbox_index, self._sub_size)
        if self.mode == "hint":
            # The legacy path never needed pixels (callers skip unreadable
            # images themselves), so keep that behaviour exactly.
            return self._count(key, hint, "hint")

        if self.mode == "plate" and plate is None:
            raise ValueError("plate mode needs the car's plate")
        if image is None:
            return self._count(key, None, None)
        assert self._cache is not None  # noqa: S101 - set for every non-hint mode
        h, w = image.shape[:2]
        boxes = self._cache.boxes(path.name, image)
        box, source = self._choose(key, boxes, (w, h), hint, plate)
        self._located[key] = _Located(path.name, (w, h), hint, plate, box)
        return self._count(key, box, source)

    def _choose(
        self,
        key: tuple[int, int],
        boxes: list[Box],
        size: tuple[int, int],
        hint: IntBox | None,
        plate: str | None,
        *,
        guard: bool = True,
    ) -> tuple[IntBox | None, str | None]:
        """The tracked vehicle among one snap's ``boxes``. ``guard=False``
        is the choice with stationary vehicles left in."""
        others = self._stationary_others(key, size, hint) if guard else []
        anchor = self.plate_bbox(key[0], key[1], plate)
        if anchor is not None:
            # A plate inside the box says the car is in view, so a box partly
            # behind a stationary vehicle stays. One that *is* a stationary
            # vehicle does not: it is often the tighter box round the plate
            # of a car passing in front of it.
            mine = [b for b in boxes if same_vehicle(b, others) is None]
            box = plate_anchored_box(mine, anchor)
            if box is not None:
                return _to_int_box(box), "plate"
        if self.mode == "plate":
            return None, None

        ranked = rank_vehicle_candidates(
            [b for b in boxes if taken_by(b, others) is None],
            size[0],
            size[1],
            bbox_hint=hint,
            road_polygon_frac=self._poly,
            min_vehicle_h_px=self._min_h,
        )
        # No hint means no anchor for "which car" -- the largest on-road
        # vehicle is a guess, so skip rather than mislabel the track.
        if not ranked or hint is None:
            return None, None
        return _to_int_box(ranked[0]), "trajectory"

    def _stationary_others(
        self, key: tuple[int, int], size: tuple[int, int], hint: IntBox | None
    ) -> list[Box]:
        """The stationary vehicles around a snap that its track is not on."""
        vehicles = self._stationary.get(key)
        if not vehicles:
            return []
        own: list[Box | IntBox | None] = [hint]
        done = self._done_index.get(key)
        if done is not None and self._sub_size is not None:
            sx, sy = size[0] / self._sub_size[0], size[1] / self._sub_size[1]
            own.append((done[0] * sx, done[1] * sy, done[2] * sx, done[3] * sy))
        return [v for v in vehicles if not sits_on(v, own)]

    def _find_stationary(self) -> None:
        """Work out the stationary vehicles from the boxes cached so far."""
        if not self._guard or self._cache is None:
            return
        sightings = []
        for name, boxes in self._cache.known().items():
            parsed = parse_snap_filename(name)
            if parsed is None:
                continue
            key = (parsed[1], parsed[2])
            when = self._snap_times.get(key)
            if when is not None:
                sightings.append(SnapSighting(key[0], key[1], when, tuple(boxes)))
        self._stationary = find_stationary_vehicles(sightings)

    def settle(self) -> list[tuple[int, int]]:
        """Call once every snap has been through :meth:`locate`.

        A stopped vehicle shows as one only against the snaps around it, and
        on a session's first pass the later ones had not been through the
        detector when the earlier ones were located. This works the
        stationary vehicles out again from every snap and returns the
        ``(track_id, snap_index)`` of each snap whose crop is now a
        different box. Locate those again; the answer is then final.

        It also counts what the guard did (:attr:`n_refused`,
        :attr:`n_left_without`) and writes the vehicles that would have
        taken a crop to ``<session>_stationary_vehicles.json``.
        """
        if not self._guard or self._cache is None:
            return []
        self._find_stationary()
        known = self._cache.known()
        changed: list[tuple[int, int]] = []
        refused: list[tuple[float, int, Sequence[float]]] = []
        self.n_left_without = 0
        for key, was in self._located.items():
            boxes = known[was.name]
            box, _source = self._choose(key, boxes, was.size, was.hint, was.plate)
            if box != was.box:
                changed.append(key)
            plain, _source = self._choose(key, boxes, was.size, was.hint, was.plate, guard=False)
            if plain is not None and plain != box:
                others = self._stationary_others(key, was.size, was.hint)
                vehicle = taken_by(plain, others) or plain
                refused.append((self._snap_times[key], key[0], vehicle))
                self.n_left_without += box is None
        self.n_refused = len(refused)
        self._write_stationary(refused)
        return changed

    def _write_stationary(self, refused: list[tuple[float, int, Sequence[float]]]) -> None:
        """One entry per stationary vehicle that would have taken a crop:
        where it stood, when, and how many crops and tracks."""

        def iso(unix: float) -> str:
            return datetime.fromtimestamp(unix, timezone.utc).isoformat(timespec="seconds")

        vehicles = [
            {
                "box": [int(round(v)) for v in group["box"]],
                "first": iso(group["first_unix"]),
                "last": iso(group["last_unix"]),
                "n_crops": group["n_crops"],
                "n_tracks": group["n_tracks"],
            }
            for group in group_refused(refused)
        ]
        atomic_write_json(
            SessionFiles(self._session_dir).stationary_vehicles,
            {
                "rule": {
                    "same_box_iou": SAME_BOX_IOU,
                    "min_gap_s": MIN_GAP_S,
                    "neighbour_tracks": NEIGHBOUR_TRACKS,
                    "max_gap_s": MAX_GAP_S,
                    "min_tracks": MIN_TRACKS,
                    "same_vehicle_iou": SAME_VEHICLE_IOU,
                    "hidden_frac": HIDDEN_FRAC,
                    "own_box_iou": OWN_BOX_IOU,
                },
                "n_snaps": len(self._located),
                "n_crops_refused": self.n_refused,
                "n_left_without_crop": self.n_left_without,
                "vehicles": vehicles,
            },
        )

    def _count(
        self, key: tuple[int, int], box: IntBox | None, source: str | None
    ) -> tuple[IntBox | None, str | None]:
        # A snap located a second time (after `settle`) counts once, under
        # the source it ended with.
        before = self._sources.pop(key, None)
        if before is not None:
            self.source_counts[before] -= 1
            if not self.source_counts[before]:
                del self.source_counts[before]
        name = source or "none"
        self._sources[key] = name
        self.source_counts[name] += 1
        return box, source

    def close(self) -> None:
        """Persist any newly detected vehicle boxes."""
        if self._cache is not None:
            self._cache.save()


def resolve_crop_settings(
    meta: dict[str, Any],
    crop_mode: str,
    pad_frac: float | None,
    legacy_pad_frac: float,
) -> tuple[str, float]:
    """Inference crop mode + pad for a checkpoint.

    ``crop_mode="auto"`` follows the checkpoint's training corpus: a model
    trained on plate-anchored crops infers with ``fullframe``; a model
    with no recorded crop mode (every pre-2026-09 checkpoint) keeps the
    legacy ``hint`` path so its predictions don't change. ``pad_frac=None``
    likewise uses the checkpoint's recorded training pad, else the
    command's legacy default.
    """
    if crop_mode == "auto":
        trained = meta.get("crop_mode")
        mode = "fullframe" if trained in ("plate", "fullframe") else "hint"
    else:
        mode = crop_mode
    if pad_frac is None:
        recorded = meta.get("crop_pad_frac")
        pad_frac = float(recorded) if recorded is not None else legacy_pad_frac
    return mode, pad_frac
