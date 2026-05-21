"""SnapPlanner spatial-gate behaviour + legacy-mode regression coverage.

The trajectories below are synthetic but mirror the failure mode that
motivated the right-half gate: live snaps were firing while bboxes were
in the left half, capturing plates at unreadable angles.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from streettracker.device.snap_planner import (
    RoadGate,
    RoadGateConfig,
    SnapDecision,
    SnapPlanner,
    SnapPlannerConfig,
    compute_quality_score,
    is_exit_imminent,
    right_half_zone_index,
)

FW, FH = 896, 512
MID = FW * 0.5
BBOX_W, BBOX_H = 200, 160  # 200*160 / (896*512) ~= 7 % area, above 5 % gate


def _bbox(cx: float) -> tuple[float, float, float, float]:
    """Centred bbox with constant size; only x position varies."""
    return (cx - BBOX_W / 2, 200, cx + BBOX_W / 2, 200 + BBOX_H)


def _walk(
    planner: SnapPlanner, tid: int, xs: list[float]
) -> Iterator[tuple[int, float, SnapDecision]]:
    for f, cx in enumerate(xs):
        yield f, cx, planner.consider(tid, _bbox(cx), f, sharpness=200.0)


# ----------------------------------------------------------------------
# Spatial-gate behaviour (right_half_only=True, the default)


def test_left_to_right_fires_just_after_crossing_midline() -> None:
    """A vehicle moving left to right should snap on the first frame
    after its bbox centre crosses x = frame_width / 2."""
    planner = SnapPlanner(FW, FH)
    xs = [100 + f * 20 for f in range(40)]  # 100 -> 880
    fires = [(f, cx, d) for f, cx, d in _walk(planner, 1, xs) if d.should_fire]

    assert len(fires) == 1
    f, cx, d = fires[0]
    assert cx >= MID, f"fired before midline at x={cx}"
    assert cx - MID < 25, f"fired well past midline (delta={cx - MID})"
    assert d.reason == "right_half_entry"
    assert d.snap_index == 1


def test_right_to_left_fires_immediately_on_entry() -> None:
    """A vehicle entering from the right is already in the gate on
    frame 0 and should snap on that very frame."""
    planner = SnapPlanner(FW, FH)
    xs = [800 - f * 20 for f in range(40)]  # 800 -> 20
    fires = [(f, cx, d) for f, cx, d in _walk(planner, 2, xs) if d.should_fire]

    assert len(fires) == 1
    f, cx, d = fires[0]
    assert f == 0
    assert cx >= MID
    assert d.reason == "right_half_entry"


def test_left_half_only_track_yields_no_live_fire_and_no_finalize_fire() -> None:
    """A track that never crosses to the right half must not snap.
    The finalize fallback is also gated -- otherwise it would re-emit
    exactly the left-half snap the spatial gate exists to prevent."""
    planner = SnapPlanner(FW, FH)
    xs = [50 + f * 5 for f in range(20)]  # 50 -> 145, stays in left half
    fires = [d for _, _, d in _walk(planner, 3, xs) if d.should_fire]
    assert fires == []

    fin = planner.on_track_finalize(3)
    assert fin.should_fire is False
    assert fin.reason == "out_of_gate"


def test_below_area_threshold_in_right_half_does_not_fire() -> None:
    """A tiny bbox in the right half is below the area gate; it should
    not fire even though it is geometrically inside the spatial gate."""
    planner = SnapPlanner(FW, FH)
    # 50*50 / (896*512) ~= 0.5 % -- well under the 5 % threshold.
    tiny = (700, 200, 750, 250)
    d = planner.consider(4, tiny, 0, sharpness=200.0)
    assert d.should_fire is False
    assert d.reason == "ineligible"


def test_blur_gate_suppresses_fire_without_consuming_budget() -> None:
    """A sharp-enough later frame must still be able to fire."""
    cfg = SnapPlannerConfig(min_sharpness=80.0)
    planner = SnapPlanner(FW, FH, cfg)

    # Frame 0: in gate but too blurry.
    d0 = planner.consider(5, _bbox(700), 0, sharpness=20.0)
    assert d0.should_fire is False
    assert d0.reason == "blur_gate"

    # Frame 1: in gate, sharp -- should fire.
    d1 = planner.consider(5, _bbox(720), 1, sharpness=200.0)
    assert d1.should_fire is True
    assert d1.reason == "right_half_entry"
    assert d1.snap_index == 1


def test_max_per_track_budget_is_respected() -> None:
    planner = SnapPlanner(FW, FH, SnapPlannerConfig(max_per_track=1))
    fired = 0
    # 60 frames, all in right half, well past cooldown after first fire.
    for f in range(60):
        d = planner.consider(6, _bbox(700 + (f % 5) * 10), f, sharpness=200.0)
        if d.should_fire:
            fired += 1
    assert fired == 1


def test_post_fire_cooldown_suppresses_immediate_refire() -> None:
    """With max_per_track=3, a second snap should not fire during
    the cooldown window even though the bbox is still in the gate."""
    cfg = SnapPlannerConfig(max_per_track=3, post_fire_cooldown_frames=10)
    planner = SnapPlanner(FW, FH, cfg)

    fire_frames = []
    for f in range(40):
        d = planner.consider(7, _bbox(700), f, sharpness=200.0)
        if d.should_fire:
            fire_frames.append(f)
    # First fire at f=0; subsequent fires must each be >= cooldown apart.
    assert fire_frames[0] == 0
    for prev, cur in zip(fire_frames, fire_frames[1:], strict=False):
        assert cur - prev >= cfg.post_fire_cooldown_frames


def test_finalize_fires_when_track_visited_right_half_but_never_committed() -> None:
    """Construct a track that hits the right half briefly (raising
    peak_score and ever_in_right_half) but every right-half frame is
    blur-gated. Finalize should then succeed."""
    cfg = SnapPlannerConfig(min_sharpness=80.0)
    planner = SnapPlanner(FW, FH, cfg)

    # Two right-half frames, both blurry -> no live fire, but
    # peak_score and ever_in_right_half are now set.
    planner.consider(8, _bbox(700), 0, sharpness=10.0)
    planner.consider(8, _bbox(710), 1, sharpness=10.0)

    fin = planner.on_track_finalize(8)
    assert fin.should_fire is True
    assert fin.reason == "finalize_last_chance"
    assert fin.snap_index == 1


def test_forget_removes_state() -> None:
    planner = SnapPlanner(FW, FH)
    planner.consider(9, _bbox(700), 0, sharpness=200.0)
    assert 9 in planner.state_snapshot()
    planner.forget(9)
    assert 9 not in planner.state_snapshot()


def test_state_snapshot_includes_ever_in_right_half_flag() -> None:
    planner = SnapPlanner(FW, FH)
    planner.consider(10, _bbox(100), 0, sharpness=200.0)  # left half
    planner.consider(11, _bbox(700), 0, sharpness=200.0)  # right half
    snap = planner.state_snapshot()
    assert snap[10]["ever_in_right_half"] is False
    assert snap[11]["ever_in_right_half"] is True


# ----------------------------------------------------------------------
# Zone-thirds firing (max_per_track > 1 splits the right half)


def test_zone_index_helper_partitions_right_half_evenly() -> None:
    """Three zones across the right half [FW/2, FW]: each spans FW/6.

    Mid-zone points used here to avoid FP boundary ambiguity (a centre
    falling exactly on the zone 0/1 boundary at rel=1/3 lands on
    whichever side FP rounding picks; mid-zone points are stable).
    """
    assert right_half_zone_index(FW * 0.55, FW, 3) == 0  # rel=0.1, zone 0
    assert right_half_zone_index(FW * 0.75, FW, 3) == 1  # rel=0.5, zone 1
    assert right_half_zone_index(FW * 0.95, FW, 3) == 2  # rel=0.9, zone 2
    # Midline maps to zone 0; right edge clamps into the last zone.
    assert right_half_zone_index(FW * 0.5, FW, 3) == 0
    assert right_half_zone_index(FW - 1, FW, 3) == 2
    # Pathological inputs (centre in left half, num_zones <= 0) clamp safely.
    assert right_half_zone_index(0.0, FW, 3) == 0
    assert right_half_zone_index(FW * 0.75, FW, 0) == 0


def test_zone_index_collapses_to_zero_when_one_zone() -> None:
    """max_per_track=1 reverts to the original single-zone behaviour."""
    for cx in (FW * 0.5, FW * 0.7, FW - 1):
        assert right_half_zone_index(cx, FW, 1) == 0


def test_left_to_right_fires_three_spatially_distinct_snaps() -> None:
    """A vehicle traversing left to right with budget=3 must fire once
    per right-half zone, in zone order 0 -> 1 -> 2."""
    cfg = SnapPlannerConfig(max_per_track=3, post_fire_cooldown_frames=2)
    planner = SnapPlanner(FW, FH, cfg)

    xs = [50 + f * 18 for f in range(50)]  # 50 -> ~932
    fires = [(f, cx, d) for f, cx, d in _walk(planner, 20, xs) if d.should_fire]

    assert len(fires) == 3
    zones = [right_half_zone_index(cx, FW, 3) for _, cx, _ in fires]
    assert zones == [0, 1, 2]
    indices = [d.snap_index for _, _, d in fires]
    assert indices == [1, 2, 3]
    assert all(d.reason == "right_half_entry" for _, _, d in fires)


def test_right_to_left_fires_three_spatially_distinct_snaps() -> None:
    """R-to-L enters at the outer zone first, then mid, then inner."""
    cfg = SnapPlannerConfig(max_per_track=3, post_fire_cooldown_frames=2)
    planner = SnapPlanner(FW, FH, cfg)

    xs = [870 - f * 18 for f in range(50)]
    fires = [(f, cx, d) for f, cx, d in _walk(planner, 21, xs) if d.should_fire]

    assert len(fires) == 3
    zones = [right_half_zone_index(cx, FW, 3) for _, cx, _ in fires]
    assert zones == [2, 1, 0]


def test_oscillating_track_does_not_refire_same_zone() -> None:
    """Vehicle that stops mid-right-half and jitters around its zone
    boundary must not produce duplicate snaps for the same zone."""
    cfg = SnapPlannerConfig(max_per_track=3, post_fire_cooldown_frames=2)
    planner = SnapPlanner(FW, FH, cfg)

    # Park the bbox centre near the zone 1 / zone 2 boundary and
    # wiggle by a few px for 50 frames.
    boundary = FW * (0.5 + 1 / 3)
    xs = [boundary + ((-1) ** f) * 3 for f in range(50)]
    fires = [d for _, _, d in _walk(planner, 22, xs) if d.should_fire]

    # We expect at most one fire per side of the boundary visited.
    zones_hit = {right_half_zone_index(cx, FW, 3) for cx in xs}
    assert len(fires) == len(zones_hit), (
        f"oscillation produced {len(fires)} fires across {zones_hit}"
    )


def test_only_rightmost_zone_visited_fires_once() -> None:
    """A very brief R-to-L track that exits the gate before crossing
    a zone boundary should yield exactly one snap."""
    cfg = SnapPlannerConfig(max_per_track=3, post_fire_cooldown_frames=2)
    planner = SnapPlanner(FW, FH, cfg)

    # Stay in the outermost zone (zone 2: cx in [FW * 5/6, FW]).
    xs = [870, 868, 866, 864, 862, 860, 858, 856, 854, 852]
    fires = [d for _, _, d in _walk(planner, 23, xs) if d.should_fire]
    assert len(fires) == 1
    assert fires[0].snap_index == 1


def test_finalize_fallback_skipped_when_at_least_one_zone_fired() -> None:
    """Track that fires in one zone then exits gate should NOT get a
    finalize-fallback fire (no budget exhaustion check; budget_exhausted
    is the reason since fires_committed > 0)."""
    cfg = SnapPlannerConfig(max_per_track=3, post_fire_cooldown_frames=2)
    planner = SnapPlanner(FW, FH, cfg)
    xs = [870, 860, 850, 840, 830]  # one zone visited then track ends
    for _, _, _ in _walk(planner, 24, xs):
        pass
    fin = planner.on_track_finalize(24)
    assert fin.should_fire is False
    assert fin.reason == "budget_exhausted"


# ----------------------------------------------------------------------
# Legacy mode (right_half_only=False) still works


def test_legacy_mode_uses_peak_decay_path() -> None:
    """A left to right vehicle in legacy mode should fire via one of
    the score-based triggers, not the spatial-gate code path."""
    cfg = SnapPlannerConfig(right_half_only=False)
    planner = SnapPlanner(FW, FH, cfg)

    xs = [100 + f * 20 for f in range(40)]
    reasons = [d.reason for _, _, d in _walk(planner, 12, xs) if d.should_fire]
    assert reasons, "legacy mode did not fire on a full-frame traversal"
    assert all(r != "right_half_entry" for r in reasons)


def test_legacy_finalize_does_not_apply_spatial_gate() -> None:
    cfg = SnapPlannerConfig(right_half_only=False)
    planner = SnapPlanner(FW, FH, cfg)
    # Track stays in left half but is above area threshold -- legacy
    # mode should still allow finalize fallback to fire.
    for f in range(10):
        planner.consider(13, _bbox(100 + f * 5), f, sharpness=200.0)
    fin = planner.on_track_finalize(13)
    assert fin.should_fire in (True, False)  # may have already fired live
    assert fin.reason != "out_of_gate"


# ----------------------------------------------------------------------
# Pure helpers


def test_compute_quality_score_zero_for_degenerate_bbox() -> None:
    assert compute_quality_score((10, 10, 10, 10), (FW, FH)) == 0.0
    assert compute_quality_score((10, 10, 10, 20), (FW, FH)) == 0.0


def test_compute_quality_score_saturates_above_25pct_area() -> None:
    """Area term should cap at 1.0; sharpness=None reads as fully sharp."""
    # 600 * 400 / (896 * 512) ~= 52 %, well above the 25 % saturation knee.
    s = compute_quality_score((0, 0, 600, 400), (FW, FH))
    assert s == pytest.approx(1.0)


def test_is_exit_imminent_without_prev_is_pessimistic() -> None:
    """No prev_bbox -> any edge within 2 * margin counts as imminent."""
    margin = 0.04
    assert is_exit_imminent((5, 100, 100, 200), None, (FW, FH), margin) is True
    # A bbox safely inside the frame is not imminent.
    assert is_exit_imminent((400, 200, 500, 300), None, (FW, FH), margin) is False


def test_is_exit_imminent_uses_motion_direction_with_prev() -> None:
    margin = 0.04
    # Moving right and near the right edge -> imminent.
    cur = (FW - 10, 200, FW - 5, 300)
    prev = (FW - 50, 200, FW - 45, 300)
    assert is_exit_imminent(cur, prev, (FW, FH), margin) is True
    # Same position but moving LEFT -> not imminent on the right edge.
    prev_moving_left = (FW - 3, 200, FW + 2, 300)
    assert is_exit_imminent(cur, prev_moving_left, (FW, FH), margin) is False


# ----------------------------------------------------------------------
# Road-gate mode (polygon + axis trigger crossings)
#
# Synthetic polygon: a horizontal stripe across the centre of the frame.
# Principal axis runs along +x ; the y-axis flip rule picks the +x axis
# because the stripe's variance in x dominates and there is no y-vs--y
# asymmetry. The triggers below sit at t'=0.25, 0.50, 0.75 along the
# usable t-band [0.0, 1.0].
ROAD_POLY = [(0.10, 0.45), (0.90, 0.45), (0.90, 0.55), (0.10, 0.55)]
ROAD_TRIGGERS = [0.25, 0.50, 0.75]


def _road_planner(triggers=None, t_usable=(0.0, 1.0), cooldown=2) -> SnapPlanner:
    cfg = SnapPlannerConfig(
        max_per_track=10,  # not the limiting factor in road-gate mode
        post_fire_cooldown_frames=cooldown,
        road_gate=RoadGateConfig(
            polygon_frac=ROAD_POLY,
            trigger_t_prime=triggers if triggers is not None else ROAD_TRIGGERS,
            t_usable_frac=t_usable,
        ),
    )
    return SnapPlanner(FW, FH, cfg)


def _bbox_at(
    cx: float, cy: float, w: float = 300.0, h: float = 160.0
) -> tuple[float, float, float, float]:
    """Bbox sized for ~10% area-frac at 896x512 -- safely above the
    default 5% area_threshold_frac so it doesn't get gated away."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_road_gate_geometry_smoke() -> None:
    """Sanity: the polygon centroid is in the middle of the frame and
    the principal axis points along +x because the stripe is wide."""
    cfg = RoadGateConfig(polygon_frac=ROAD_POLY, trigger_t_prime=ROAD_TRIGGERS)
    gate = RoadGate.from_config(cfg, FW, FH)
    assert abs(gate.axis_x) > abs(gate.axis_y)
    # Centroid roughly at frame centre.
    assert abs(gate.centroid_x - FW * 0.5) < 5
    assert abs(gate.centroid_y - FH * 0.5) < 5
    # Polygon point on the centre of the stripe sits at t' ~= 0.5.
    tp_mid = gate.t_prime(FW * 0.5, FH * 0.5)
    assert tp_mid is not None
    assert abs(tp_mid - 0.5) < 0.02


def test_road_gate_rejects_point_outside_polygon() -> None:
    cfg = RoadGateConfig(polygon_frac=ROAD_POLY, trigger_t_prime=ROAD_TRIGGERS)
    gate = RoadGate.from_config(cfg, FW, FH)
    # Top-left corner of frame -- well outside the central stripe.
    assert not gate.contains(20.0, 20.0)


def test_left_to_right_traversal_fires_three_trigger_crossings() -> None:
    """A track moving along +x through the road stripe should fire
    each trigger exactly once, in order T0 -> T1 -> T2."""
    p = _road_planner()
    fires: list[tuple[int, float, SnapDecision]] = []
    # Cross from cx ~ left edge of road to cx ~ right edge of road.
    # 30 frames of 25 px steps covers x=150..875 -- well across the stripe.
    for f in range(30):
        cx = 150 + f * 25
        cy = FH * 0.5
        d = p.consider(1, _bbox_at(cx, cy), f, sharpness=200.0)
        if d.should_fire:
            fires.append((f, cx, d))

    assert len(fires) == 3
    assert [d.reason for _, _, d in fires] == ["trigger_crossing"] * 3
    assert [d.snap_index for _, _, d in fires] == [1, 2, 3]
    # Fire x-coords should be monotonically increasing.
    xs = [cx for _, cx, _ in fires]
    assert xs == sorted(xs)


def test_right_to_left_traversal_also_fires_three_crossings() -> None:
    """Same stripe, opposite direction: triggers fire T2 -> T1 -> T0."""
    p = _road_planner()
    fires = []
    for f in range(30):
        cx = 870 - f * 25
        d = p.consider(2, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires.append((f, cx, d))

    assert len(fires) == 3
    xs = [cx for _, cx, _ in fires]
    assert xs == sorted(xs, reverse=True)


def test_track_outside_polygon_never_fires() -> None:
    """Bbox centre stays in the top strip of the frame, well above the
    road polygon. No triggers should fire, and finalize is suppressed."""
    p = _road_planner()
    for f in range(30):
        cx = 150 + f * 25
        d = p.consider(3, _bbox_at(cx, FH * 0.1), f, sharpness=200.0)
        assert d.should_fire is False
        assert d.reason == "outside_road_polygon"
    fin = p.on_track_finalize(3)
    assert fin.should_fire is False
    assert fin.reason == "out_of_gate"


def test_repeat_crossings_of_same_trigger_only_fire_once() -> None:
    """A track that walks past trigger T1, reverses, walks back, then
    forward again should only fire T1 a single time."""
    p = _road_planner()
    # Forward across all three triggers -> fires T0, T1, T2.
    xs_fwd = [150 + f * 25 for f in range(30)]
    # Then reverse back through them.
    xs_back = [875 - f * 25 for f in range(30)]
    xs_fwd2 = [150 + f * 25 for f in range(30)]
    fires = 0
    for f, cx in enumerate(xs_fwd + xs_back + xs_fwd2):
        d = p.consider(4, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires += 1
    assert fires == 3, f"expected exactly 3 fires across out-back-out, got {fires}"


def test_track_inside_polygon_but_never_crossing_any_trigger() -> None:
    """A track that wiggles between T0 and T1 (never crossing either)
    should produce zero fires."""
    p = _road_planner()
    # cx oscillates between t'=0.30 and t'=0.45 -- both inside the
    # gate but strictly between trigger 0 (0.25) and trigger 1 (0.50).
    gate = p.road_gate
    assert gate is not None
    # Convert target t' values back to image x by inverting the linear map.
    def x_for_tp(tp: float) -> float:
        return FW * (0.10 + 0.80 * tp)
    fires = 0
    for f in range(40):
        cx = x_for_tp(0.30 if f % 2 == 0 else 0.45)
        d = p.consider(5, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires += 1
    assert fires == 0


def test_t_usable_clip_excludes_distant_tip() -> None:
    """With t_usable_frac=(0.2, 0.8), a track entering the polygon
    while still in the clipped distant-tip band should not register
    any crossings until it enters the usable band."""
    p = _road_planner(triggers=[0.5], t_usable=(0.2, 0.8))
    fires = 0
    # cx sweeps from the polygon's left edge (at orig t'=0) into the
    # usable band -- the single trigger at usable-t'=0.5 sits at
    # orig-t'=0.5 (the polygon centre), unchanged.
    for f in range(30):
        cx = 150 + f * 25
        d = p.consider(6, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires += 1
    assert fires == 1


def test_blur_gate_in_road_mode_holds_trigger_for_later_fire() -> None:
    """If sharpness fails the gate when a trigger would otherwise fire,
    the trigger stays un-fired so a subsequent sharper frame can take it."""
    cfg = SnapPlannerConfig(
        post_fire_cooldown_frames=0,
        min_sharpness=80.0,
        road_gate=RoadGateConfig(polygon_frac=ROAD_POLY, trigger_t_prime=[0.5]),
    )
    p = SnapPlanner(FW, FH, cfg)
    # First sample inside the gate (no crossing possible -- prev=None).
    d0 = p.consider(7, _bbox_at(200, FH * 0.5), 0, sharpness=200.0)
    assert d0.should_fire is False
    # Cross the trigger but with bad sharpness -> blur_gate.
    d1 = p.consider(7, _bbox_at(700, FH * 0.5), 1, sharpness=10.0)
    assert d1.reason == "blur_gate"
    # Step further along, still in gate, sharp this time. Even though
    # the trigger was technically already passed, the previous sample
    # was rejected so prev_t_prime was never advanced past it -- the
    # crossing remains detectable.
    d2 = p.consider(7, _bbox_at(750, FH * 0.5), 2, sharpness=200.0)
    assert d2.should_fire is True
    assert d2.reason == "trigger_crossing"


def test_road_gate_finalize_never_fires_fallback() -> None:
    """Road-gate mode has no last-chance fire -- a track that visited
    the gate but never crossed a trigger gets zero snaps."""
    p = _road_planner()
    # Hold position between T0 and T1; never crosses a trigger.
    for f in range(20):
        p.consider(8, _bbox_at(FW * 0.30, FH * 0.5), f, sharpness=200.0)
    fin = p.on_track_finalize(8)
    assert fin.should_fire is False


# ----------------------------------------------------------------------
# Asymmetric (direction-tagged) triggers
#
# The synthetic ROAD_POLY is a horizontal stripe, so the principal axis
# points along +x and "forward" motion (increasing t') corresponds to
# left-to-right travel in image coords. "reverse" is right-to-left.

ASYM_TRIGGERS = [0.25, 0.50, 0.75]


def _asym_planner(
    directions: list, triggers: list[float] | None = None, cooldown: int = 2
) -> SnapPlanner:
    cfg = SnapPlannerConfig(
        max_per_track=10,
        post_fire_cooldown_frames=cooldown,
        road_gate=RoadGateConfig(
            polygon_frac=ROAD_POLY,
            trigger_t_prime=triggers if triggers is not None else ASYM_TRIGGERS,
            t_usable_frac=(0.0, 1.0),
            trigger_directions=directions,
        ),
    )
    return SnapPlanner(FW, FH, cfg)


def test_forward_only_trigger_fires_on_left_to_right_motion() -> None:
    """All triggers tagged "forward". L-to-R traversal fires all three;
    R-to-L traversal fires none."""
    p_lr = _asym_planner(["forward", "forward", "forward"])
    fires_lr = 0
    for f in range(30):
        cx = 150 + f * 25
        d = p_lr.consider(100, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires_lr += 1
    assert fires_lr == 3

    p_rl = _asym_planner(["forward", "forward", "forward"])
    fires_rl = 0
    for f in range(30):
        cx = 870 - f * 25
        d = p_rl.consider(101, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires_rl += 1
    assert fires_rl == 0


def test_reverse_only_trigger_fires_on_right_to_left_motion() -> None:
    """Symmetric to the forward-only case but for "reverse" tags."""
    p_lr = _asym_planner(["reverse", "reverse", "reverse"])
    fires_lr = 0
    for f in range(30):
        cx = 150 + f * 25
        d = p_lr.consider(102, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires_lr += 1
    assert fires_lr == 0

    p_rl = _asym_planner(["reverse", "reverse", "reverse"])
    fires_rl = 0
    for f in range(30):
        cx = 870 - f * 25
        d = p_rl.consider(103, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires_rl += 1
    assert fires_rl == 3


def test_mixed_directions_each_direction_picks_up_only_its_own_plus_both() -> None:
    """Mix one forward, one both, one reverse. L-to-R sees fwd + both;
    R-to-L sees rev + both. Neither direction can fire the other's."""
    directions = ["forward", "both", "reverse"]

    p_lr = _asym_planner(directions)
    fires_lr: list[tuple[float, int]] = []
    for f in range(30):
        cx = 150 + f * 25
        d = p_lr.consider(104, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            assert d.snap_index is not None
            fires_lr.append((cx, d.snap_index))
    # Forward sweep: should fire trigger 0 (forward, t'=0.25) then
    # trigger 1 (both, t'=0.50). Trigger 2 is reverse-only -> skipped.
    assert len(fires_lr) == 2
    assert [idx for _, idx in fires_lr] == [1, 2]

    p_rl = _asym_planner(directions)
    fires_rl: list[tuple[float, int]] = []
    for f in range(30):
        cx = 870 - f * 25
        d = p_rl.consider(105, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            assert d.snap_index is not None
            fires_rl.append((cx, d.snap_index))
    # Reverse sweep: trigger 2 (reverse, t'=0.75) then trigger 1 (both,
    # t'=0.50). Trigger 0 is forward-only -> skipped.
    assert len(fires_rl) == 2
    assert [idx for _, idx in fires_rl] == [1, 2]


def test_directions_default_to_both_when_omitted() -> None:
    """Omitting trigger_directions must preserve pre-feature behaviour:
    every trigger fires in either direction. Exercises the default path
    through RoadGate.from_config without breaking existing configs."""
    cfg = RoadGateConfig(
        polygon_frac=ROAD_POLY,
        trigger_t_prime=ASYM_TRIGGERS,
        # trigger_directions deliberately omitted
    )
    gate = RoadGate.from_config(cfg, FW, FH)
    assert gate.trigger_directions == ("both", "both", "both")


def test_invalid_direction_string_raises() -> None:
    cfg = RoadGateConfig(
        polygon_frac=ROAD_POLY,
        trigger_t_prime=[0.5],
        trigger_directions=["sideways"],  # type: ignore[list-item]
    )
    with pytest.raises(ValueError, match="invalid trigger direction"):
        RoadGate.from_config(cfg, FW, FH)


def test_trigger_directions_length_mismatch_raises() -> None:
    cfg = RoadGateConfig(
        polygon_frac=ROAD_POLY,
        trigger_t_prime=[0.25, 0.5, 0.75],
        trigger_directions=["forward", "both"],
    )
    with pytest.raises(ValueError, match="length must match"):
        RoadGate.from_config(cfg, FW, FH)


def test_stationary_frames_never_fire_any_trigger() -> None:
    """When cur_tp == prev_tp (motion has neither forward nor reverse
    sign), no trigger -- including "both" -- should fire. This avoids
    classifying a stationary jitter as either direction."""
    cfg = RoadGateConfig(polygon_frac=ROAD_POLY, trigger_t_prime=[0.5])
    gate = RoadGate.from_config(cfg, FW, FH)
    # prev_tp == cur_tp == trigger value: motion has no direction.
    assert gate.crossings(0.5, 0.5, set()) == []
    # prev_tp == cur_tp slightly past trigger: still no motion, no fire.
    assert gate.crossings(0.6, 0.6, set()) == []


def test_asymmetric_trigger_at_extreme_t_prime_fires_only_in_intended_direction() -> None:
    """Operator's real-world use case: a forward trigger near t'=0 for
    R-to-L *approach* captures (front plate) and a reverse trigger near
    t'=1 for L-to-R *departure* captures (rear plate). Each direction's
    earliest snap must land at the small-bbox / distant end of its own
    motion -- which here means the trigger at the *start* of the motion
    in t' space."""
    # In our test polygon (axis along +x), "forward" = L-to-R, so a
    # forward trigger near t'=0 sits near the *left* edge -- the start
    # of an L-to-R track. A reverse trigger near t'=1 sits near the
    # *right* edge -- the start of an R-to-L track.
    directions = ["forward", "reverse"]
    triggers = [0.10, 0.90]
    # L-to-R: should fire trigger 0 early in the track and skip trigger 1.
    p_lr = _asym_planner(directions, triggers=triggers)
    lr_fires: list[float] = []
    for f in range(30):
        cx = 150 + f * 25
        d = p_lr.consider(106, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            lr_fires.append(cx)
    assert len(lr_fires) == 1
    # Fired near the left of the track.
    assert lr_fires[0] < FW * 0.35, f"forward fire too late: x={lr_fires[0]}"

    # R-to-L: should fire trigger 1 early in the track and skip trigger 0.
    p_rl = _asym_planner(directions, triggers=triggers)
    rl_fires: list[float] = []
    for f in range(30):
        cx = 870 - f * 25
        d = p_rl.consider(107, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            rl_fires.append(cx)
    assert len(rl_fires) == 1
    assert rl_fires[0] > FW * 0.65, f"reverse fire too late: x={rl_fires[0]}"


def test_backward_compat_no_road_gate_uses_right_half_only() -> None:
    """SnapPlanner with no road_gate config still runs the existing
    right_half_only zone-thirds path."""
    cfg = SnapPlannerConfig(max_per_track=3, post_fire_cooldown_frames=2)
    p = SnapPlanner(FW, FH, cfg)
    assert p.road_gate is None
    fires = []
    for f in range(50):
        cx = 50 + f * 18
        d = p.consider(9, _bbox_at(cx, FH * 0.5), f, sharpness=200.0)
        if d.should_fire:
            fires.append(d)
    assert len(fires) == 3
    assert all(d.reason == "right_half_entry" for d in fires)
