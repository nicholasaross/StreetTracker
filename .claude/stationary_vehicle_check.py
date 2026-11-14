"""What the stationary-vehicle guard does to the CNN heads' crops (E2.6 f).

    uv run python .claude/stationary_vehicle_check.py output/session_A output/session_B ...

Reads only saved files: each session's ``_vehicle_boxes.json`` (so the
session must have been through a CNN command), ``_data.json``, ``_alpr.json``,
``_dvsa_labels.json`` and the per-image ``_colour.json`` / ``_makemodel.json``.
No pixels and no GPU; about a minute per week-long session.

Three measurements, each per session and in total:

1. **Crops changed.** Every vehicle snap is located twice, with the guard
   and without, exactly as the commands do it. Counted: the crops the guard
   changes, what they become (another box, or none), and the tracks touched.

2. **Plate-blind check of the choice.** On a snap where the track's own
   gated plate was read, the vehicle holding that plate is known. Ignore the
   plate, let the trajectory rule choose, and see whether it picks that
   vehicle, with the guard and without. The guard refuses a box partly
   hidden behind a stationary vehicle, and on these snaps that box is
   sometimes the right one; the commands keep it there, because the plate
   anchors the crop, and the report says how many. The last line of this
   part counts plate-anchored crops the guard moves off a stationary
   vehicle: mostly the tightest-box rule having anchored a moving car's
   plate to a parked car behind it.

3. **The heads on those crops.** For DVSA-labelled tracks, the saved
   per-image colour and make predictions against the register, split into
   crops the guard changes and the rest. On a session inferred before the
   guard this is the accuracy of crops taken from a stopped vehicle; on one
   inferred with it the split is empty, because the files already hold the
   guarded crops.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from streettracker.analysis.alpr.fullframe import load_road_polygon
from streettracker.analysis.snap_assets import discover_vehicle_snaps, image_size
from streettracker.analysis.stationary import iou
from streettracker.analysis.vehicle_locator import (
    SnapVehicleLocator,
    VehicleBoxCache,
    anchor_plate_bbox,
    load_alpr_reads,
    plate_anchored_box,
)

ROAD_POLYGON = Path(".claude/triggers_proposal.json")


class _PlateBlind(SnapVehicleLocator):
    """A locator that never sees a plate read: the trajectory rule alone."""

    def plate_bbox(self, track_id: int, snap_index: int, plate: str | None) -> None:
        return None


def _no_detector(_image: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    raise RuntimeError("snap missing from _vehicle_boxes.json: run a CNN command first")


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _per_image(session: Path, suffix: str) -> dict[tuple[int, int], dict]:
    rows = _load(session / f"{session.name}_{suffix}.json") or []
    return {(r["track_id"], r["snap_index"]): r for r in rows}


def check(session: Path, poly: list[tuple[float, float]] | None) -> Counter[str]:
    n: Counter[str] = Counter()
    cache = VehicleBoxCache(session, detector=_no_detector)
    cached = cache.known()
    snaps = [s for s in discover_vehicle_snaps(session) if s[0].name in cached]
    if not snaps:
        print(f"{session.name}: no cached vehicle boxes, skipped")
        return n
    width, height = image_size(snaps[0][0]) or (4512, 2512)
    frame = np.broadcast_to(np.zeros((), dtype=np.uint8), (height, width, 3))

    def locator(cls: type[SnapVehicleLocator], guard: bool) -> SnapVehicleLocator:
        return cls(
            session, "fullframe", road_polygon_frac=poly, cache=cache, stationary_guard=guard
        )

    guarded, plain = locator(SnapVehicleLocator, True), locator(SnapVehicleLocator, False)
    blind, blind_plain = locator(_PlateBlind, True), locator(_PlateBlind, False)

    labels = (_load(session / f"{session.name}_dvsa_labels.json") or {}).get("labels", {})
    label_of = {tid: row for row in labels.values() for tid in row.get("track_ids", [])}
    reads = load_alpr_reads(session)
    colour, make = _per_image(session, "colour"), _per_image(session, "makemodel")
    by_track: dict[int, list[bool]] = defaultdict(list)

    for path, tid, snap, _cls in snaps:
        key = (tid, snap)
        was, was_source = plain.locate(path, tid, snap, frame)
        now, _ = guarded.locate(path, tid, snap, frame)
        changed = was != now
        if was_source:
            n[f"{was_source} crops"] += 1
            n[f"{was_source} crops changed"] += changed
            n[f"{was_source} crops changed to none"] += changed and now is None
            if was_source == "trajectory":
                by_track[tid].append(changed)

        label = label_of.get(tid)
        if label is None:
            continue

        # 2. plate-blind: is the trajectory rule's pick the plate's vehicle?
        anchor = anchor_plate_bbox(reads.get(key, []), label["plate"])
        truth = plate_anchored_box(cached[path.name], anchor) if anchor else None
        if truth is not None:
            pick_was, _ = blind_plain.locate(path, tid, snap, frame)
            pick_now, _ = blind.locate(path, tid, snap, frame)
            right_was = pick_was is not None and iou(pick_was, truth) >= 0.5
            right_now = pick_now is not None and iou(pick_now, truth) >= 0.5
            n["truth snaps"] += 1
            n["truth: right without guard"] += right_was
            n["truth: right with guard"] += right_now
            if pick_was != pick_now:
                n["truth: guard acted"] += 1
                n["truth: guard acted, right before"] += right_was
                # ...of which the real choice, which has the plate, keeps the box.
                n["truth: guard acted, right before, kept by the plate"] += (
                    right_was and now is not None and iou(now, truth) >= 0.5
                )
                n["truth: guard acted, right after"] += right_now
                n["truth: guard acted, none after"] += pick_now is None
            if was_source == "plate" and changed:
                n["truth refused (plate anchored to a stationary vehicle)"] += 1

        # 3. the saved predictions on trajectory crops of labelled tracks
        if was_source != "trajectory":
            continue
        group = "crops the guard changes" if changed else "other crops"
        row = colour.get(key)
        if row and row.get("crop_source") == "trajectory" and row.get("colour"):
            n[f"colour, {group}"] += 1
            n[f"colour right, {group}"] += (
                row["colour"].lower() == str(label.get("primary_colour")).lower()
            )
        row = make.get(key)
        top = (row.get("top_k") or [{}])[0].get("make") if row else None
        if row and row.get("crop_source") == "trajectory" and top:
            n[f"make, {group}"] += 1
            n[f"make right, {group}"] += top.upper() == str(label.get("make")).upper()

    n["tracks with trajectory crops"] = len(by_track)
    n["tracks with a changed crop"] = sum(any(v) for v in by_track.values())
    n["tracks with every crop changed"] = sum(all(v) for v in by_track.values())
    return n


def report(title: str, n: Counter[str]) -> None:
    def share(part: str, whole: str) -> str:
        return (
            f"{n[part]:6d} of {n[whole]:6d} ({100 * n[part] / n[whole]:.1f} %)" if n[whole] else "-"
        )

    print(f"\n== {title}")
    for source in ("trajectory", "plate"):
        print(
            f"  {source} crops changed      {share(f'{source} crops changed', f'{source} crops')}"
        )
        print(f"    ...left with no crop      {n[f'{source} crops changed to none']:6d}")
    print(
        "  tracks with a changed crop    "
        + share("tracks with a changed crop", "tracks with trajectory crops")
    )
    print(f"    ...with every crop changed  {n['tracks with every crop changed']:6d}")
    print(f"  plate-blind, right without    {share('truth: right without guard', 'truth snaps')}")
    print(f"  plate-blind, right with       {share('truth: right with guard', 'truth snaps')}")
    acted = "truth: guard acted"
    print(
        f"    where the guard acted ({n[acted]}): right before {n[acted + ', right before']}, "
        f"right after {n[acted + ', right after']}, none after {n[acted + ', none after']}"
    )
    print(
        "    ...of the right-before, kept anyway where the plate anchors the crop: "
        f"{n[acted + ', right before, kept by the plate']}"
    )
    key = "truth refused (plate anchored to a stationary vehicle)"
    print(f"  plate crops moved off a stationary vehicle on labelled tracks: {n[key]}")
    for head in ("colour", "make"):
        for group in ("crops the guard changes", "other crops"):
            print(
                f"  {head:6s} right, {group:24s}{share(f'{head} right, {group}', f'{head}, {group}')}"
            )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    poly = load_road_polygon(ROAD_POLYGON)
    total: Counter[str] = Counter()
    for arg in argv:
        session = Path(arg)
        n = check(session, poly)
        if n:
            report(session.name, n)
            total.update(n)
    if len(argv) > 1:
        report(f"all {len(argv)} sessions", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
