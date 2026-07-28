"""Static-plate suppression for per-image ALPR results.

The pre-crop hint windows are wide (fire->completion union, motion
lookahead), so crops routinely include *stationary* plates that aren't
the tracked car's: parked cars in driveways/roadside and plate-shaped
fixed scene objects. Those produce two corruptions:

* confident garbage reads from tiny background plates (the 2026-07-27
  R->L decomposition found ~half of failed R->L detections recur at a
  fixed 4K position while the tracked car moves), and
* *canonical* reads of a parked car credited to whichever track was
  passing (e.g. FD61PVX contributed 49 canonical "reads" to moving
  R->L tracks after it re-parked outside the static ghost-mask rect).

The single operator-traced ghost-mask rect can't keep up with where
residents actually park, so this module derives the static map from the
session's own evidence. The discriminator is **affine consistency**,
not raw image motion -- near the road's vanishing point a *genuine*
receding plate barely moves in pixel space, so "plate stayed put while
the car moved" would flag half the legitimate reads. Instead, for two
snaps of one track we predict where the plate *should* be at snap B by
holding its position fixed relative to the car bbox (translation +
scale), and compare:

* genuine plate: actual motion tracks the prediction (it is attached
  to the car);
* parked/background plate: the prediction moves with the car, the
  actual detection doesn't.

Pairs whose predicted motion is small (vanishing zone, crawling car)
are treated as *unknown*, never static -- conservative by design.
A cross-track rule additionally catches spots dominated by single-snap
tracks: many distinct tracks detecting a plate at one tight position
with near-identical box width is a parked-plate signature.

Detections near a static spot are annotated ``static_suspect`` (kept in
``<session>_alpr.json`` for inspection) and excluded from the per-track
rollup, so downstream consumers (``vehicles``, ``dvsa-label``, the
showcase) never see them. Detections that *pass* the affine-consistency
test with their own track are immune from marking even inside a spot --
a moving car legitimately drives its plate through the pixels where
someone else is parked. The spot map itself is persisted as
``<session>_static_plates.json``.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

# A within-track pair is informative only when the affine prediction
# says the plate should have moved at least this far (4K px). Below
# this the pair is ambiguous (vanishing zone / crawling car) and never
# seeds a spot.
DEFAULT_MIN_PREDICTED_PX = 30.0

# Actual motion at most this fraction of predicted motion => static.
DEFAULT_STATIC_MOTION_FRAC = 0.35

# Actual-vs-predicted residual within this fraction of predicted
# (plus a small pixel floor) => the det provably moves with its car
# and becomes immune from marking.
DEFAULT_CONSISTENT_RESID_FRAC = 0.35
DEFAULT_CONSISTENT_RESID_FLOOR_PX = 10.0

# Cross-track spot thresholds (parked plates seen mostly by
# single-snap tracks): many distinct tracks, tight position, near-
# identical detected width. Deliberately strict -- near the vanishing
# point every car's plate passes through almost the same pixels.
DEFAULT_CROSS_MIN_TRACKS = 8
DEFAULT_CROSS_POS_SD_PX = 12.0
DEFAULT_CROSS_WIDTH_RSD = 0.12

# Spatial cluster cell size / spot match radius (4K px).
DEFAULT_RADIUS_PX = 28.0
DEFAULT_MARK_RADIUS_PX = 35.0

# Clamp on the det centre's bbox-relative coords used by the affine
# prediction, so a detection far outside the car bbox can't produce a
# wildly extrapolated prediction when the bbox rescales.
_UV_CLAMP = 3.0


def _center(bbox: list[int] | tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _predicted_motion(
    det_c: tuple[float, float],
    car_a: tuple[float, float, float, float],
    car_b: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Where should a car-attached point at ``det_c`` (snap A) move by
    snap B, given the car bbox went from ``car_a`` to ``car_b``?

    Holds the point's bbox-relative position (u, v) fixed across the
    bbox's translation + scale change. Returns the predicted (dx, dy).
    """
    aw = max(car_a[2] - car_a[0], 1.0)
    ah = max(car_a[3] - car_a[1], 1.0)
    u = (det_c[0] - car_a[0]) / aw
    v = (det_c[1] - car_a[1]) / ah
    u = max(-_UV_CLAMP, min(_UV_CLAMP, u))
    v = max(-_UV_CLAMP, min(_UV_CLAMP, v))
    bw = car_b[2] - car_b[0]
    bh = car_b[3] - car_b[1]
    px = car_b[0] + u * bw
    py = car_b[1] + v * bh
    return (px - det_c[0], py - det_c[1])


def find_static_spots(
    records: list[dict],
    car_index: dict[tuple[int, int], tuple[int, int, int, int]] | None = None,
    sub_size: tuple[int, int] | None = None,
    img_size: tuple[int, int] | None = None,
    *,
    min_predicted_px: float = DEFAULT_MIN_PREDICTED_PX,
    static_motion_frac: float = DEFAULT_STATIC_MOTION_FRAC,
    cross_min_tracks: int = DEFAULT_CROSS_MIN_TRACKS,
    cross_pos_sd_px: float = DEFAULT_CROSS_POS_SD_PX,
    cross_width_rsd: float = DEFAULT_CROSS_WIDTH_RSD,
    radius_px: float = DEFAULT_RADIUS_PX,
) -> tuple[list[dict], set[tuple[int, int]]]:
    """Derive static-plate spots from a session's per-image ALPR records.

    ``car_index`` maps ``(track_id, snap_index)`` to the tracked car's
    **sub-stream** bbox (completion-time where available); ``sub_size``
    and ``img_size`` scale it into the snap's pixel space. Without
    them the affine test is skipped and only the cross-track rule runs.

    Returns ``(spots, consistent_keys)``: the spot list (``cx``/``cy``
    centroid in 4K px, ``n_dets``, ``n_tracks``, ``source``, sample
    ``texts``) sorted by ``n_dets`` descending, and the set of
    ``(track_id, snap_index)`` whose detection provably moves with its
    car (immune from marking).
    """
    car_index = car_index or {}
    scale: tuple[float, float] | None = None
    if sub_size and img_size and sub_size[0] > 0 and sub_size[1] > 0:
        scale = (img_size[0] / sub_size[0], img_size[1] / sub_size[1])

    # cx, cy, width, track_id, snap_index, text -- one row per detection
    dets: list[tuple[float, float, float, int, int, str]] = []
    by_track: dict[int, list[int]] = defaultdict(list)
    for r in records:
        bb = r.get("det_bbox")
        if not bb:
            continue
        cx, cy = _center(bb)
        tid = int(r["track_id"])
        dets.append(
            (cx, cy, float(bb[2] - bb[0]), tid, int(r["snap_index"]), r.get("ocr_text") or "")
        )
        by_track[tid].append(len(dets) - 1)

    def car_img_bbox(tid: int, snap: int) -> tuple[float, float, float, float] | None:
        if scale is None:
            return None
        bb = car_index.get((tid, snap))
        if bb is None:
            return None
        sx, sy = scale
        return (bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy)

    seed_idx: set[int] = set()
    consistent_keys: set[tuple[int, int]] = set()
    for tid, idxs in by_track.items():
        for a in range(len(idxs)):
            ia = idxs[a]
            da = dets[ia]
            car_a = car_img_bbox(tid, da[4])
            if car_a is None:
                continue
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                db = dets[ib]
                car_b = car_img_bbox(tid, db[4])
                if car_b is None:
                    continue
                pred = _predicted_motion((da[0], da[1]), car_a, car_b)
                pred_mag = math.hypot(*pred)
                if pred_mag < min_predicted_px:
                    continue  # ambiguous: vanishing zone / crawling car
                act = (db[0] - da[0], db[1] - da[1])
                act_mag = math.hypot(*act)
                resid = math.hypot(act[0] - pred[0], act[1] - pred[1])
                if act_mag <= static_motion_frac * pred_mag:
                    seed_idx.add(ia)
                    seed_idx.add(ib)
                elif resid <= (
                    DEFAULT_CONSISTENT_RESID_FRAC * pred_mag + DEFAULT_CONSISTENT_RESID_FLOOR_PX
                ):
                    consistent_keys.add((tid, da[4]))
                    consistent_keys.add((tid, db[4]))

    # Spatial clustering: a 3x3-neighbourhood cluster becomes a spot
    # when it contains affine-static seeds, or independently satisfies
    # the (strict) cross-track parked-plate signature.
    cell = radius_px
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, d in enumerate(dets):
        grid[(int(d[0] // cell), int(d[1] // cell))].append(i)

    spots: list[dict] = []
    for key in grid:
        members: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                members.extend(grid.get((key[0] + dx, key[1] + dy), ()))
        if not members:
            continue
        n_seeds = sum(1 for i in members if i in seed_idx)
        has_seed = n_seeds >= 2
        cross = False
        tids = {dets[i][3] for i in members}
        if len(tids) >= cross_min_tracks:
            xs = [dets[i][0] for i in members]
            ys = [dets[i][1] for i in members]
            ws = [dets[i][2] for i in members]
            pos_sd = max(statistics.pstdev(xs), statistics.pstdev(ys))
            mean_w = statistics.fmean(ws)
            w_rsd = statistics.pstdev(ws) / mean_w if mean_w > 0 else 1.0
            cross = pos_sd <= cross_pos_sd_px and w_rsd <= cross_width_rsd
        if not (has_seed or cross):
            continue
        # Spot geometry from the seed dets when present (they are the
        # proven-static ones); otherwise from the whole cluster.
        core = [i for i in members if i in seed_idx] if has_seed else members
        texts = sorted({dets[i][5] for i in core if dets[i][5]})[:8]
        spots.append(
            {
                "cx": round(statistics.fmean(dets[i][0] for i in core), 1),
                "cy": round(statistics.fmean(dets[i][1] for i in core), 1),
                "n_dets": len(core),
                "n_tracks": len({dets[i][3] for i in core}),
                "width_px": round(statistics.median(dets[i][2] for i in core), 1),
                "source": "within_track" if has_seed else "cross_track",
                "texts": texts,
            }
        )

    # collapse near-duplicate spots from overlapping neighbourhoods
    spots.sort(key=lambda s: (-s["n_dets"], s["cx"], s["cy"]))
    kept: list[dict] = []
    for s in spots:
        if all(math.dist((s["cx"], s["cy"]), (k["cx"], k["cy"])) > radius_px for k in kept):
            kept.append(s)
    return kept, consistent_keys


def mark_static_suspects(
    records: list[dict],
    spots: list[dict],
    consistent_keys: set[tuple[int, int]] | None = None,
    *,
    mark_radius_px: float = DEFAULT_MARK_RADIUS_PX,
) -> int:
    """Annotate records whose detection sits on a static spot.

    Adds ``static_suspect: True`` in place; returns the count marked.
    Records without a detection, and records whose detection passed the
    affine-consistency test with their own track (``consistent_keys``),
    are never marked -- a moving car may legitimately carry its plate
    through a parked car's pixels.
    """
    if not spots:
        return 0
    consistent_keys = consistent_keys or set()
    n = 0
    centres = [(s["cx"], s["cy"]) for s in spots]
    for r in records:
        bb = r.get("det_bbox")
        if not bb:
            continue
        if (int(r["track_id"]), int(r["snap_index"])) in consistent_keys:
            continue
        c = _center(bb)
        if any(math.dist(c, sc) <= mark_radius_px for sc in centres):
            r["static_suspect"] = True
            n += 1
    return n
