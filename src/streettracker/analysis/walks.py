"""Door-origin walk detection.

Answers "which walks were mine?" without recognising anyone: a walk that
starts or ends inside an **operator-marked door zone** is one of the
household's own trips, identified by *where it enters/leaves the frame*
rather than by any appearance or biometric signature. Strangers don't
emerge from your front door, so none of them are identified or evaluated
-- this is behavioural, the same class of signal as the footfall stats.

Inputs:

- A track's ``entry_point_frac`` / ``exit_point_frac`` (fractional
  ``[x, y]``), recorded at finalize by ``track_buffer.compute_attributes``
  for sessions captured after that change landed. Tracks without them
  (older sessions) can't be classified and are reported as ``unknown``.
- A **door zone**: a closed polygon of fractional ``[x, y]`` vertices the
  operator traces on a frame (``configs/door_zone.json``), same
  resolution-independent scheme as the road polygon / ghost mask. There
  is no default -- the door's position is per-install operator knowledge.

A walk is:

- ``originated`` -- entry point in the door zone (left the door),
- ``returned`` -- exit point in the door zone (arrived at the door),
- ``round_trip`` -- both (a there-and-back within one track; rare, since
  BotSORT usually splits the out and back legs -- the cross-track
  out-and-back pairing in ``analysis/people.py`` handles the common case),
- ``passing`` -- neither (through-traffic on the pavement),
- ``unknown`` -- no entry/exit points recorded.

Deliberately NO identity: this says a trip started at *your* door, not
*who* took it. If several people share the door it can't tell them apart
-- that's the household's trips, by design.
"""

from __future__ import annotations

from typing import Any

# DoorZone lives in common/ because the live runtime needs it too (it
# exempts door-touching tracks from the parked-track filter); re-exported
# here so the analysis-side import path is unchanged.
from streettracker.common.door_zone import (
    DEFAULT_DOOR_ZONE_PATH,
    DoorZone,
    _point_in_polygon,
)

__all__ = [
    "DEFAULT_DOOR_ZONE_PATH",
    "ORIGINATED",
    "PASSING",
    "RETURNED",
    "ROUND_TRIP",
    "UNKNOWN",
    "DoorZone",
    "classify_walk_origin",
    "door_origin_for_records",
    "is_own_trip",
    "_point_in_polygon",
]

# A walk's door relationship, from its entry/exit points.
ORIGINATED = "originated"  # entry in the door zone
RETURNED = "returned"  # exit in the door zone
ROUND_TRIP = "round_trip"  # both
PASSING = "passing"  # neither
UNKNOWN = "unknown"  # no entry/exit points recorded


def classify_walk_origin(
    entry_point_frac: list[float] | None,
    exit_point_frac: list[float] | None,
    zone: DoorZone,
) -> str:
    """Door relationship for one track's entry/exit points.

    Returns one of ``ORIGINATED`` / ``RETURNED`` / ``ROUND_TRIP`` /
    ``PASSING`` / ``UNKNOWN`` (the last when neither point was recorded).
    """
    if entry_point_frac is None and exit_point_frac is None:
        return UNKNOWN
    at_start = zone.contains(entry_point_frac)
    at_end = zone.contains(exit_point_frac)
    if at_start and at_end:
        return ROUND_TRIP
    if at_start:
        return ORIGINATED
    if at_end:
        return RETURNED
    return PASSING


def door_origin_for_records(
    records: list[dict[str, Any]],
    zone: DoorZone,
) -> dict[int, str]:
    """Classify every person track's door relationship.

    ``records`` is the parsed ``data.json`` array. Returns
    ``{track_id: classification}`` for person tracks only. A walk counts
    as one of the household's own trips when its classification is
    ``ORIGINATED``, ``RETURNED`` or ``ROUND_TRIP``.
    """
    out: dict[int, str] = {}
    for r in records:
        if r.get("class_name") != "person" or r.get("class_suspect"):
            continue
        tid = r.get("track_id")
        if tid is None:
            continue
        out[int(tid)] = classify_walk_origin(
            r.get("entry_point_frac"), r.get("exit_point_frac"), zone
        )
    return out


def is_own_trip(classification: str) -> bool:
    """Whether a classification marks a door-touching (household) trip."""
    return classification in (ORIGINATED, RETURNED, ROUND_TRIP)
