"""Stationary vehicles in a session's 4K snaps: the guard the CNN heads use.

The make, colour and body-type heads crop "the tracked car" from each 4K
snap. Where the snap carries no plate read, the crop is the on-road vehicle
nearest the track's stale sub-stream box (:mod:`vehicle_locator`). A vehicle
stopped in the road takes that crop from the cars that pass it: on
2026-11-13 a Citroen stopped for 84 minutes was returned as the make, colour
and body type of about 130 passing cars. The plate side has had a filter for
this since 2026-07 (:mod:`alpr.staticfilter`); this is the same idea for
vehicle boxes.

**Which vehicles are stationary.** A vehicle is standing still around a snap
when other tracks' snaps show the same box both before and after it: the
vehicle was there before this track came and was still there once it had
gone, so it cannot be the car the track followed. "The same box" is an IoU
of at least :data:`SAME_BOX_IOU`. Only snaps of the
:data:`NEIGHBOUR_TRACKS` nearest other tracks on each side count, and only
those at least :data:`MIN_GAP_S` away in time. At least
:data:`MIN_TRACKS` tracks must show the box, the snap's own track among them
if it does (:func:`find_stationary_vehicles`).

Each part answers something measured on the eleven sessions of 2026-10-01
to 2026-11-14 (``.claude/stationary_vehicle_check.py``):

* *Before and after*, not a count. A car that drives in and parks, or one
  that pulls away, is seen at rest on one side of its own track only, and
  the box in that track's snap is the tracked car. Asking for both sides
  leaves those alone.
* *The nearest few tracks*, not every track. Snaps fire at fixed places
  along the road, so two moving cars put near-identical boxes in snaps
  hours apart (about 0.1 % of snap pairs). A stopped vehicle is in nearly
  every snap around it; a coincidence is in one. Looking only at the
  neighbours keeps the chance of a false match the same on a busy day as on
  a quiet one.
* *Three tracks*. With the snap's own track that is one sighting before
  and one after. Where the snap itself does not show the vehicle cleanly (a
  pedestrian beside it, or the tracked car in front of it), two tracks on
  one side and one on the other say as much. Two sightings alone are not
  enough: one before and one after are, often enough, two moving cars.
* *A gap in time*. One car is sometimes two tracks (an ID switch), and at
  the far end of the road a moving car holds much the same box for a second
  or two. Ten seconds is longer than a car spends in frame.

**Which crops that rules out.** Two cases. A candidate box *is* a stationary
vehicle (:func:`same_vehicle`): the detector's box for one moves a little
from snap to snap (a pedestrian beside it, a car behind it), so this match
is looser, :data:`SAME_VEHICLE_IOU`. Or it is *hidden behind* one
(:func:`hidden_behind`): the stationary vehicle stands nearer the camera and
covers at least :data:`HIDDEN_FRAC` of the box, so the crop would show the
stationary vehicle across the car it is meant to hold. On 13 November that
second case was as common as the first: the passing car's own box, correctly
chosen, with the Citroen in front of it.

The second case is for boxes chosen without a plate read. Where the car's
plate was read inside the box, the car is in view: on DVSA-labelled tracks
the heads were right on 12 to 15 of 19 such crops, against 3 or 4 of 18
hidden boxes chosen without one.

A track can also sit on a stationary vehicle: the vehicle's own track while
it waits, or a second track the tracker opened on it. :func:`sits_on` says
so from the track's own sub-stream boxes, and for that track the vehicle
rules nothing out.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Box = tuple[float, float, float, float]
SnapKey = tuple[int, int]

# Two detections of one stopped vehicle agree to 0.97-0.99; 0.9 leaves room
# for a passer clipping its edge. At 0.8 chance matches between moving cars
# are about six times as common.
SAME_BOX_IOU = 0.9

# Other tracks' snaps closer in time than this are not evidence.
MIN_GAP_S = 10.0

# How many other tracks, on each side in time, may give evidence.
NEIGHBOUR_TRACKS = 3

# ...and how far away in time they may be. A vehicle no track passed for
# half an hour is not known to have stayed.
MAX_GAP_S = 1800.0

# How many tracks must show a box, the snap's own included, with at least
# one before and one after.
MIN_TRACKS = 3

# A box in a snap overlapping a known stationary vehicle this much is that
# vehicle. A car beside or behind one overlaps it by under 0.5.
SAME_VEHICLE_IOU = 0.7

# A box this far covered by a stationary vehicle standing in front of it is
# hidden behind that vehicle. On DVSA-labelled tracks the heads were right on
# 80 % of crops covered by less than a quarter (59 crops), as on any other
# crop, and on 4 of 18 covered by more.
HIDDEN_FRAC = 0.25

# A track whose own box overlaps a stationary vehicle this much is on that
# vehicle. Measured overlaps fall in two groups, under 0.5 (passing cars)
# and over 0.7 (the vehicle's own tracks).
OWN_BOX_IOU = 0.5

# Refused boxes this alike, seen within MAX_GAP_S of each other, are one
# vehicle in the summary.
_EPISODE_IOU = 0.7


@dataclass(frozen=True, slots=True)
class SnapSighting:
    """The vehicle boxes of one 4K snap, in snap pixels, and when it fired."""

    track_id: int
    snap_index: int
    time: float
    boxes: tuple[Box, ...]

    @property
    def key(self) -> SnapKey:
        return (self.track_id, self.snap_index)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def covered_frac(box: Sequence[float], by: Sequence[float]) -> float:
    """The share of ``box`` that lies inside ``by``."""
    ix = min(box[2], by[2]) - max(box[0], by[0])
    iy = min(box[3], by[3]) - max(box[1], by[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    if ix <= 0 or iy <= 0 or area <= 0:
        return 0.0
    return ix * iy / area


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` (n, 4) with every box in ``b`` (m, 4)."""
    ix = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    iy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _neighbours(
    order: list[SnapSighting],
    i: int,
    step: int,
    *,
    min_gap_s: float,
    max_gap_s: float,
    neighbour_tracks: int,
) -> list[int]:
    """Positions in ``order`` of the snaps of the nearest other tracks on one
    side of ``order[i]`` (``step`` -1 for earlier, +1 for later)."""
    here = order[i]
    tracks: set[int] = set()
    found: list[int] = []
    j = i + step
    while 0 <= j < len(order):
        other = order[j]
        gap = abs(other.time - here.time)
        if gap > max_gap_s:
            break
        if other.track_id != here.track_id and gap >= min_gap_s:
            if other.track_id not in tracks:
                if len(tracks) >= neighbour_tracks:
                    break
                tracks.add(other.track_id)
            found.append(j)
        j += step
    return found


def find_stationary_vehicles(
    sightings: Iterable[SnapSighting],
    *,
    same_box_iou: float = SAME_BOX_IOU,
    min_gap_s: float = MIN_GAP_S,
    neighbour_tracks: int = NEIGHBOUR_TRACKS,
    max_gap_s: float = MAX_GAP_S,
    min_tracks: int = MIN_TRACKS,
) -> dict[SnapKey, list[Box]]:
    """The vehicles standing still around each snap: boxes that at least
    ``min_tracks`` of the tracks nearest in time show, one or more of them
    before the snap and one or more after (see the module docstring). Snaps
    with none are left out of the result.

    A vehicle is given as the snap's own box for it where the snap shows it,
    else as the box nearest in time before the snap. So it is found even
    where this snap's detection of it is off or missing.
    """
    order = sorted(sightings, key=lambda s: (s.time, s.track_id, s.snap_index))
    arrays = [np.asarray(s.boxes, dtype=np.float64).reshape(-1, 4) for s in order]

    def neighbours(i: int, step: int) -> list[int]:
        return _neighbours(
            order,
            i,
            step,
            min_gap_s=min_gap_s,
            max_gap_s=max_gap_s,
            neighbour_tracks=neighbour_tracks,
        )

    found: dict[SnapKey, list[Box]] = {}
    for i, snap in enumerate(order):
        before = neighbours(i, -1)
        if not before:
            continue
        after = neighbours(i, 1)
        if not after:
            continue
        # This snap first, then earlier snaps (nearest first), then later ones.
        snaps = [i, *before, *after]
        boxes = np.concatenate([arrays[j] for j in snaps])
        if not len(boxes):
            continue
        counts = [len(arrays[j]) for j in snaps]
        track_ids = sorted({order[j].track_id for j in snaps})
        column = np.repeat([track_ids.index(order[j].track_id) for j in snaps], counts)
        is_before = np.repeat([0 < k <= len(before) for k in range(len(snaps))], counts)
        is_after = np.repeat([k > len(before) for k in range(len(snaps))], counts)

        same = _iou_matrix(boxes, boxes) >= same_box_iou
        shown_by = np.zeros((len(boxes), len(track_ids)), dtype=bool)
        for t in range(len(track_ids)):
            shown_by[:, t] = same[:, column == t].any(axis=1)
        stays = (
            (shown_by.sum(axis=1) >= min_tracks)
            & same[:, is_before].any(axis=1)
            & same[:, is_after].any(axis=1)
        )
        vehicles: list[Box] = []
        for row in boxes[stays]:
            box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            # One vehicle shows in several of these snaps: keep it once.
            if all(iou(box, kept) < same_box_iou for kept in vehicles):
                vehicles.append(box)
        if vehicles:
            found[snap.key] = vehicles
    return found


def sits_on(
    vehicle: Sequence[float],
    own_boxes: Iterable[Sequence[float] | None],
    *,
    min_iou: float = OWN_BOX_IOU,
) -> bool:
    """Whether a track is on ``vehicle``: one of the track's own boxes for
    this snap (where it was when the snap fired, and when the snap landed)
    overlaps it. A passing car's own box is somewhere else."""
    return any(own is not None and iou(vehicle, own) >= min_iou for own in own_boxes)


def same_vehicle(
    box: Sequence[float],
    vehicles: Iterable[Sequence[float]],
    *,
    min_iou: float = SAME_VEHICLE_IOU,
) -> Sequence[float] | None:
    """The one of ``vehicles`` that ``box`` is a detection of, or ``None``."""
    for vehicle in vehicles:
        if iou(box, vehicle) >= min_iou:
            return vehicle
    return None


def hidden_behind(
    box: Sequence[float],
    vehicles: Iterable[Sequence[float]],
    *,
    min_frac: float = HIDDEN_FRAC,
) -> Sequence[float] | None:
    """The one of ``vehicles`` standing in front of ``box`` and covering at
    least ``min_frac`` of it, or ``None``. In front means nearer the camera,
    which looks down on the road: the vehicle whose box reaches lower in the
    frame."""
    for vehicle in vehicles:
        if vehicle[3] > box[3] and covered_frac(box, vehicle) >= min_frac:
            return vehicle
    return None


def taken_by(
    box: Sequence[float],
    vehicles: Iterable[Sequence[float]],
) -> Sequence[float] | None:
    """The stationary vehicle that rules ``box`` out as a crop chosen
    without a plate read, or ``None``: the one it is, or is hidden behind."""
    vehicles = list(vehicles)
    return same_vehicle(box, vehicles) or hidden_behind(box, vehicles)


def group_refused(
    refused: Iterable[tuple[float, int, Sequence[float]]],
    *,
    max_gap_s: float = MAX_GAP_S,
) -> list[dict[str, Any]]:
    """Group the crops a session's guard refused by the stationary vehicle
    that took them, for the summary file. ``refused`` is one
    ``(snap time, track_id, vehicle box)`` per crop.

    One entry per vehicle and stretch of time: its box as first seen, the
    first and last snap times it took a crop, and how many crops and tracks.
    The first and last tracks to pass a stopped vehicle have no evidence on
    one side, so an entry starts and ends a track late and early.
    """
    groups: list[dict[str, Any]] = []
    tracks: list[set[int]] = []
    for time, track_id, box in sorted(refused, key=lambda r: (r[0], r[1])):
        for k in range(len(groups) - 1, -1, -1):
            group = groups[k]
            if time - group["last_unix"] <= max_gap_s and iou(box, group["box"]) >= _EPISODE_IOU:
                group["last_unix"] = time
                group["n_crops"] += 1
                tracks[k].add(track_id)
                break
        else:
            groups.append(
                {
                    "box": [round(float(v), 1) for v in box],
                    "first_unix": time,
                    "last_unix": time,
                    "n_crops": 1,
                }
            )
            tracks.append({track_id})
    for group, seen_by in zip(groups, tracks, strict=True):
        group["n_tracks"] = len(seen_by)
    return groups
