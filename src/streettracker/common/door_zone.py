"""The operator-traced door zone -- shared by the live runtime and the
off-device analysis.

It lives in ``common/`` rather than ``analysis/`` because both sides need
it: ``analysis/walks.py`` classifies each finished walk against it, and
the runtime consults it at finalize so a track that touches the door
isn't discarded by the parked/short-track filters (a walk that steps out
and comes straight back has almost no *net* displacement, which is
exactly what those filters delete -- see
``device/track_buffer.compute_attributes``).

The zone is a closed polygon of fractional ``[x, y]`` vertices in
``configs/door_zone.json``, the same resolution-independent scheme as the
road polygon and ghost mask. There is no default: the door's position is
per-install operator knowledge, and when the file is absent everything
here degrades to "no door zone configured".

Fractional coords are stream-independent -- the sub-stream the tracker
runs on and the main stream the operator traces share a field of view
and differ only by an anisotropic squash, the same assumption
``analysis/snap_assets._scale_bbox_to_image`` makes for every ALPR
pre-crop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DOOR_ZONE_PATH = Path("configs/door_zone.json")


@dataclass(slots=True)
class DoorZone:
    """An operator-traced door region in fractional frame coords."""

    polygon_frac: list[list[float]]

    @classmethod
    def load(cls, path: Path = DEFAULT_DOOR_ZONE_PATH) -> DoorZone | None:
        """Load the door zone, or ``None`` if the file is absent/invalid
        (door-origin analysis then simply doesn't run)."""
        if not path.exists():
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        poly = doc.get("polygon_frac")
        if not isinstance(poly, list) or len(poly) < 3:
            return None
        try:
            verts = [[float(x), float(y)] for x, y in poly]
        except (TypeError, ValueError):
            return None
        return cls(polygon_frac=verts)

    def contains(self, point: list[float] | None) -> bool:
        """True if a fractional ``[x, y]`` point is inside the zone."""
        if not point or len(point) < 2:
            return False
        return _point_in_polygon(float(point[0]), float(point[1]), self.polygon_frac)


def _point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon (even-odd rule). Points exactly on an
    edge are treated as inside consistently enough for a hand-traced zone
    -- the door polygon is drawn with margin, so boundary precision is
    not load-bearing."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside
