"""Coverage for ``streettracker.device.track_buffer``.

The buffer is pure-Python + numpy + (for sharpness) OpenCV. Tests use
small synthetic BGR frames and stubbed ``Detection`` objects so they
run quickly and without a GPU. The compute_attributes path is exercised
end-to-end against a constructed BufferedTrack; the field-by-field
assertions document the field contract that downstream consumers
(dashboard, hourly rollup, ALPR pipelines) rely on.
"""

from __future__ import annotations

import numpy as np
import pytest

from streettracker.common.schema import TrackRecord
from streettracker.common.types import Detection
from streettracker.device.track_buffer import (
    BufferedTrack,
    CropSample,
    MotionPoint,
    TrackBuffer,
    compute_attributes,
    median_bbox_aspect,
    safe_crop,
    sharpness_score,
    total_displacement,
    update_hq_best,
)

# 2026-05-22T12:00Z, well clear of Windows' fromtimestamp lower bound.
_T0_WALL = 1_779_792_000.0

# Frames are 360x640 BGR (slightly more realistic than a tiny test
# pixel). Cheap to allocate; ~700 KB per frame.
_FRAME_H, _FRAME_W = 360, 640


def _frame(value: int = 100) -> np.ndarray:
    """Uniform BGR frame; lets us see crops as distinct uint8 regions
    by writing a band of a different value into them in tests."""
    return np.full((_FRAME_H, _FRAME_W, 3), value, dtype=np.uint8)


def _det(
    track_id: int | None,
    *,
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 200.0,
    score: float = 0.9,
    class_id: int = 2,
) -> Detection:
    return Detection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=score,
        class_id=class_id,
        track_id=track_id,
    )


# ----------------------------------------------------------------------
# safe_crop


def test_safe_crop_returns_inner_pixels() -> None:
    f = _frame(50)
    f[100:200, 100:200] = 200
    crop = safe_crop(f, 100.0, 100.0, 200.0, 200.0)
    assert crop is not None
    assert crop.shape == (100, 100, 3)
    assert (crop == 200).all()


def test_safe_crop_clamps_to_frame_bounds() -> None:
    f = _frame()
    crop = safe_crop(f, -10.0, -5.0, _FRAME_W + 50, _FRAME_H + 50)
    assert crop is not None
    assert crop.shape == (_FRAME_H, _FRAME_W, 3)


def test_safe_crop_returns_none_for_empty_region() -> None:
    f = _frame()
    assert safe_crop(f, 100.0, 100.0, 100.0, 100.0) is None
    assert safe_crop(f, 200.0, 200.0, 100.0, 100.0) is None


def test_safe_crop_pad_frac_extends_bbox() -> None:
    f = _frame()
    base = safe_crop(f, 100.0, 100.0, 200.0, 200.0, pad_frac=0.0)
    padded = safe_crop(f, 100.0, 100.0, 200.0, 200.0, pad_frac=0.2)
    assert base is not None and padded is not None
    # 20% padding on a 100px bbox => +20 each side => 140x140.
    assert padded.shape == (140, 140, 3)
    assert base.shape == (100, 100, 3)


# ----------------------------------------------------------------------
# sharpness_score -- relative ranking is what we actually care about.


def test_sharpness_uniform_crop_is_near_zero() -> None:
    crop = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert sharpness_score(crop) == pytest.approx(0.0, abs=1e-6)


def test_sharpness_noisy_crop_is_higher_than_uniform() -> None:
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    uniform = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert sharpness_score(noisy) > sharpness_score(uniform)


# ----------------------------------------------------------------------
# total_displacement


def test_total_displacement_zero_for_empty_or_single_point() -> None:
    assert total_displacement([]) == 0.0
    assert total_displacement([_motion_point(0.0, 100, 100)]) == 0.0


def test_total_displacement_sums_step_distances() -> None:
    pts = [
        _motion_point(0.0, 0, 0),
        _motion_point(1.0, 3, 4),  # +5
        _motion_point(2.0, 6, 8),  # +5
    ]
    assert total_displacement(pts) == pytest.approx(10.0)


def _motion_point(t: float, cx: float, cy: float) -> MotionPoint:
    return MotionPoint(
        frame_idx=int(t * 10),
        t=t,
        cx=cx,
        cy=cy,
        x1=cx - 10,
        y1=cy - 10,
        x2=cx + 10,
        y2=cy + 10,
        score=0.9,
    )


# ----------------------------------------------------------------------
# update_hq_best


def test_update_hq_best_skips_on_edge_crops() -> None:
    tr = BufferedTrack(id=1, class_id=2)
    crop = np.full((100, 100, 3), 200, dtype=np.uint8)
    update_hq_best(tr, crop, on_edge=True)
    assert tr.hq_best_crop is None


def test_update_hq_best_keeps_higher_score() -> None:
    tr = BufferedTrack(id=1, class_id=2)
    rng = np.random.default_rng(0)
    small = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    big = rng.integers(0, 256, size=(150, 150, 3), dtype=np.uint8)
    update_hq_best(tr, small, on_edge=False)
    first_score = tr.hq_best_score
    update_hq_best(tr, big, on_edge=False)
    # Bigger crop with similar noise level should win on area * sharpness.
    assert tr.hq_best_score > first_score
    assert tr.hq_best_crop is big


def test_update_hq_best_skips_sharpness_when_area_below_threshold() -> None:
    """If a new crop can't beat the current best on area alone, the
    Laplacian shouldn't run -- the second update_hq_best call should
    leave the state unchanged regardless of the crop's actual sharpness.
    """
    tr = BufferedTrack(id=1, class_id=2)
    rng = np.random.default_rng(0)
    big = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    tiny_but_crisp = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    update_hq_best(tr, big, on_edge=False)
    score_before = tr.hq_best_score
    update_hq_best(tr, tiny_but_crisp, on_edge=False)
    assert tr.hq_best_score == score_before
    assert tr.hq_best_crop is big


# ----------------------------------------------------------------------
# TrackBuffer


def test_buffer_starts_empty() -> None:
    buf = TrackBuffer()
    assert buf.active() == []


def test_buffer_ignores_detections_without_track_id() -> None:
    buf = TrackBuffer()
    expired = buf.ingest([_det(track_id=None)], frame_idx=0, t=0.0, frame=_frame())
    assert expired == []
    assert buf.active() == []


def test_buffer_creates_track_on_first_sighting() -> None:
    buf = TrackBuffer()
    buf.ingest([_det(track_id=7)], frame_idx=0, t=0.0, frame=_frame())
    [tr] = buf.active()
    assert tr.id == 7
    assert tr.class_id == 2  # default
    assert len(tr.points) == 1
    assert tr.misses == 0


def test_buffer_appends_subsequent_points_to_same_track() -> None:
    buf = TrackBuffer()
    buf.ingest(
        [_det(7, x1=100, y1=100, x2=200, y2=200)],
        frame_idx=0,
        t=0.0,
        frame=_frame(),
    )
    buf.ingest(
        [_det(7, x1=120, y1=100, x2=220, y2=200)],
        frame_idx=1,
        t=0.05,
        frame=_frame(),
    )
    [tr] = buf.active()
    assert len(tr.points) == 2
    assert tr.points[1].cx == pytest.approx(170.0)


def test_buffer_class_id_is_confidence_weighted_majority() -> None:
    """BotSORT occasionally reclassifies a track mid-life. The buffer
    accumulates per-class confidence votes and exposes
    ``argmax(class_votes)`` as ``class_id``. With two equal-confidence
    detections of different classes, the first inserted wins
    (deterministic tie-break via dict insertion order)."""
    buf = TrackBuffer()
    buf.ingest([_det(7, class_id=2, score=0.9)], frame_idx=0, t=0.0, frame=_frame())
    buf.ingest([_det(7, class_id=7, score=0.9)], frame_idx=1, t=0.05, frame=_frame())
    [tr] = buf.active()
    assert tr.class_votes == {2: pytest.approx(0.9), 7: pytest.approx(0.9)}
    assert tr.class_id == 2  # first-inserted on tie


def test_buffer_class_id_resists_late_single_frame_flip() -> None:
    """A single late-frame detection of a different class with similar
    confidence cannot flip a track that has accumulated several frames
    of the prior class. This is the fix for the ``person #500 / car #976``
    misclassifications observed 2026-05-25 in the live deployment, where
    the previous "most-recent-detection wins" rule let a single end-of-life
    YOLO slip corrupt the entire track's class_name."""
    buf = TrackBuffer()
    # Five frames of class 2 (car) at moderate confidence.
    for i in range(5):
        buf.ingest(
            [_det(7, class_id=2, score=0.85)],
            frame_idx=i,
            t=i * 0.05,
            frame=_frame(),
        )
    # One trailing frame of class 0 (person) at the same confidence.
    buf.ingest(
        [_det(7, class_id=0, score=0.85)],
        frame_idx=5,
        t=0.25,
        frame=_frame(),
    )
    [tr] = buf.active()
    assert tr.class_id == 2  # majority vote, not the last detection
    assert tr.class_votes[2] == pytest.approx(5 * 0.85)
    assert tr.class_votes[0] == pytest.approx(0.85)


def test_buffer_class_id_flips_when_cumulative_evidence_outweighs() -> None:
    """Voting still flips the class if the new class accumulates more
    confidence than the prior class. The fix doesn't lock in the first
    class -- it just requires more than a single frame of disagreement."""
    buf = TrackBuffer()
    # One low-confidence frame of class 2.
    buf.ingest([_det(7, class_id=2, score=0.2)], frame_idx=0, t=0.0, frame=_frame())
    # Four high-confidence frames of class 0.
    for i in range(1, 5):
        buf.ingest(
            [_det(7, class_id=0, score=0.9)],
            frame_idx=i,
            t=i * 0.05,
            frame=_frame(),
        )
    [tr] = buf.active()
    assert tr.class_id == 0  # 4 * 0.9 = 3.6 > 0.2
    assert tr.class_votes[0] == pytest.approx(4 * 0.9)
    assert tr.class_votes[2] == pytest.approx(0.2)


def test_buffer_expires_tracks_after_max_misses() -> None:
    buf = TrackBuffer(max_misses=3)
    buf.ingest([_det(7)], frame_idx=0, t=0.0, frame=_frame())
    # Three frames with no detections -- exactly at the cap, not yet expired.
    for i in range(1, 4):
        expired = buf.ingest([], frame_idx=i, t=i * 0.05, frame=_frame())
        assert expired == []
    # Fourth missed frame trips the expiration.
    expired = buf.ingest([], frame_idx=4, t=0.2, frame=_frame())
    assert len(expired) == 1
    assert expired[0].id == 7
    assert buf.active() == []


def test_buffer_resets_misses_when_track_reappears() -> None:
    buf = TrackBuffer(max_misses=3)
    buf.ingest([_det(7)], frame_idx=0, t=0.0, frame=_frame())
    buf.ingest([], frame_idx=1, t=0.05, frame=_frame())
    buf.ingest([], frame_idx=2, t=0.10, frame=_frame())
    # Reappearance resets the counter.
    buf.ingest([_det(7)], frame_idx=3, t=0.15, frame=_frame())
    [tr] = buf.active()
    assert tr.misses == 0
    # The track must now survive another full max_misses-length gap.
    for i in range(4, 7):
        buf.ingest([], frame_idx=i, t=i * 0.05, frame=_frame())
    assert len(buf.active()) == 1


def test_buffer_flush_returns_all_active_and_empties() -> None:
    buf = TrackBuffer()
    buf.ingest([_det(7), _det(8)], frame_idx=0, t=0.0, frame=_frame())
    flushed = buf.flush()
    assert {tr.id for tr in flushed} == {7, 8}
    assert buf.active() == []


def test_buffer_crops_grow_until_capped_then_decimate() -> None:
    buf = TrackBuffer()
    f = _frame()
    for i in range(30):
        buf.ingest([_det(7)], frame_idx=i, t=i * 0.05, frame=f)
    [tr] = buf.active()
    # Buffer caps at 12, decimates by ::2 on overflow -- under 30 ingests
    # the survivor count should stay in single digits to low teens.
    assert 5 <= len(tr.crops) <= 12


def test_buffer_marks_edge_crops_on_edge() -> None:
    """A bbox at the frame edge should produce a crop flagged on_edge,
    which excludes it from HQ-best."""
    buf = TrackBuffer()
    # Right at the right edge of the frame.
    buf.ingest(
        [_det(7, x1=_FRAME_W - 50, y1=100, x2=_FRAME_W - 1, y2=200)],
        frame_idx=0,
        t=0.0,
        frame=_frame(),
    )
    [tr] = buf.active()
    assert tr.crops[0].on_edge is True
    # And so HQ-best never got populated.
    assert tr.hq_best_crop is None


# ----------------------------------------------------------------------
# compute_attributes


def _track_with_motion(
    ids_and_cxs: list[tuple[float, float, float]],
) -> BufferedTrack:
    """Build a BufferedTrack with the given (t, cx, cy) sequence.

    Other bbox fields are derived from cx/cy via a constant 100x100 box.
    """
    tr = BufferedTrack(id=42, class_id=2)
    for t, cx, cy in ids_and_cxs:
        tr.points.append(
            MotionPoint(
                frame_idx=int(t * 20),
                t=t,
                cx=cx,
                cy=cy,
                x1=cx - 50,
                y1=cy - 50,
                x2=cx + 50,
                y2=cy + 50,
                score=0.9,
            )
        )
    return tr


def test_compute_attributes_filters_short_tracks() -> None:
    tr = _track_with_motion([(0.0, 100, 100), (0.5, 200, 100)])
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=1.0,
        parked_disp_px=10.0,
        color="red",
        t_start_wall=_T0_WALL,
    )
    assert result is None


def test_compute_attributes_filters_single_point_tracks() -> None:
    tr = _track_with_motion([(0.0, 100, 100)])
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=0.0,
        parked_disp_px=0.0,
        color="red",
        t_start_wall=_T0_WALL,
    )
    assert result is None


def test_compute_attributes_filters_parked_tracks() -> None:
    """A long-lived track that never moved more than the parked threshold
    should be dropped -- it's a stationary vehicle, not road traffic."""
    tr = _track_with_motion([(0.0, 100, 100), (1.0, 105, 100), (2.0, 103, 100), (5.0, 101, 100)])
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=1.0,
        parked_disp_px=50.0,
        color="red",
        t_start_wall=_T0_WALL,
    )
    assert result is None


def test_compute_attributes_returns_record_for_moving_track() -> None:
    tr = _track_with_motion([(0.0, 100, 100), (1.0, 300, 100), (2.0, 500, 100)])
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=1.0,
        parked_disp_px=50.0,
        color="red",
        t_start_wall=_T0_WALL,
    )
    assert isinstance(result, TrackRecord)
    assert result.track_id == 42
    assert result.class_id == 2
    assert result.class_name == "car"
    assert result.color == "red"
    assert result.direction == "left to right"
    assert result.lane == "top"  # cy=100, frame_h/3 ~= 120
    assert result.duration_visible == pytest.approx(2.0)
    assert result.net_displacement_px == pytest.approx(400.0)
    assert result.speed_px_s == pytest.approx(200.0)
    assert result.num_detections == 3
    assert result.asset_prefix == "vehicle"
    assert result.main_snaps == []


def test_compute_attributes_entry_exit_points_need_frame_w() -> None:
    """Entry/exit fracs fill only when frame_w is supplied; otherwise
    they stay None (older callers, no half-scaled x)."""
    tr = _track_with_motion([(0.0, 100, 90), (1.0, 300, 180), (2.0, 500, 270)])
    without = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=1.0,
        parked_disp_px=50.0,
        color="red",
        t_start_wall=_T0_WALL,
    )
    assert without is not None
    assert without.entry_point_frac is None and without.exit_point_frac is None

    withw = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=1.0,
        parked_disp_px=50.0,
        color="red",
        t_start_wall=_T0_WALL,
        frame_w=_FRAME_W,
    )
    assert withw is not None
    # first point (100, 90) and last (500, 270) over a 640x360 frame,
    # rounded to 4 dp as stored.
    assert withw.entry_point_frac == [round(100 / 640, 4), round(90 / 360, 4)]
    assert withw.exit_point_frac == [round(500 / 640, 4), round(270 / 360, 4)]


def test_compute_attributes_direction_right_to_left() -> None:
    tr = _track_with_motion([(0.0, 500, 100), (1.0, 100, 100)])
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=0.0,
        parked_disp_px=10.0,
        color="white",
        t_start_wall=_T0_WALL,
    )
    assert result is not None
    assert result.direction == "right to left"


def test_compute_attributes_lane_bucketing() -> None:
    third = _FRAME_H / 3.0
    cases = [
        (third * 0.5, "top"),
        (third * 1.5, "middle"),
        (third * 2.5, "bottom"),
    ]
    for cy, expected_lane in cases:
        tr = _track_with_motion([(0.0, 100, cy), (1.0, 500, cy)])
        result = compute_attributes(
            tr,
            frame_h=_FRAME_H,
            min_duration_s=0.0,
            parked_disp_px=10.0,
            color="black",
            t_start_wall=_T0_WALL,
        )
        assert result is not None
        assert result.lane == expected_lane


def test_compute_attributes_person_class_uses_person_prefix() -> None:
    tr = BufferedTrack(id=99, class_id=0)
    tr.points = _track_with_motion([(0.0, 100, 100), (1.0, 300, 100)]).points
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=0.0,
        parked_disp_px=10.0,
        color="unknown",
        t_start_wall=_T0_WALL,
    )
    assert result is not None
    assert result.asset_prefix == "person"
    assert result.class_name == "person"


def test_compute_attributes_main_snaps_sorted_and_unique() -> None:
    tr = _track_with_motion([(0.0, 100, 100), (1.0, 300, 100)])
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=0.0,
        parked_disp_px=10.0,
        color="red",
        t_start_wall=_T0_WALL,
        main_snaps=[3, 1, 2],
    )
    assert result is not None
    assert result.main_snaps == [1, 2, 3]


def test_compute_attributes_main_snap_bboxes_aligned_with_main_snaps() -> None:
    """When the BufferedTrack has per-snap bboxes recorded, the
    TrackRecord must carry a parallel list of bboxes aligned with
    ``main_snaps`` (sorted ascending). Missing per-index bboxes
    become ``None`` so downstream analysis can detect mixed states."""
    tr = _track_with_motion([(0.0, 100, 100), (1.0, 300, 100)])
    tr.snap_fire_bboxes[1] = (50, 80, 150, 180)
    tr.snap_fire_bboxes[3] = (250, 80, 350, 180)
    # snap_index 2 deliberately omitted
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=0.0,
        parked_disp_px=10.0,
        color="red",
        t_start_wall=_T0_WALL,
        main_snaps=[3, 1, 2],
    )
    assert result is not None
    assert result.main_snaps == [1, 2, 3]
    assert result.main_snap_bboxes == [
        [50, 80, 150, 180],  # for snap 1
        None,  # for snap 2 (missing)
        [250, 80, 350, 180],  # for snap 3
    ]


def test_compute_attributes_main_snap_bboxes_none_when_no_snaps() -> None:
    """No snaps -> no bbox list (keeps JSON diff minimal for tracks
    that don't have any 4K assets to begin with)."""
    tr = _track_with_motion([(0.0, 100, 100), (1.0, 300, 100)])
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=0.0,
        parked_disp_px=10.0,
        color="red",
        t_start_wall=_T0_WALL,
    )
    assert result is not None
    assert result.main_snaps == []
    assert result.main_snap_bboxes is None


def test_crop_sample_dataclass_holds_expected_fields() -> None:
    """Light type-shape check so a future field rename triggers the
    test rather than mysteriously breaking downstream consumers."""
    s = CropSample(t=1.0, score=0.9, crop=np.zeros((10, 10, 3), dtype=np.uint8), on_edge=False)
    assert s.t == 1.0
    assert s.score == 0.9
    assert s.crop.shape == (10, 10, 3)
    assert s.on_edge is False


def test_compute_attributes_done_bboxes_parallel_to_main_snaps() -> None:
    """Completion-time bboxes ride a second list parallel to
    ``main_snaps``; indexes whose ``_on_done`` never recorded a bbox
    stay ``None`` so downstream can tell exact hints from windows."""
    tr = _track_with_motion([(0.0, 100, 100), (1.0, 300, 100)])
    tr.snap_fire_bboxes[1] = (50, 80, 150, 180)
    tr.snap_done_bboxes[1] = (90, 80, 190, 180)
    tr.snap_fire_bboxes[3] = (250, 80, 350, 180)
    # snap 3 fired but its completion callback never recorded a bbox.

    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=0.5,
        parked_disp_px=10.0,
        color="red",
        t_start_wall=_T0_WALL,
        main_snaps=[1, 3],
    )

    assert result is not None
    assert result.main_snap_bboxes == [[50, 80, 150, 180], [250, 80, 350, 180]]
    assert result.main_snap_bboxes_done == [[90, 80, 190, 180], None]


# ----------------------------------------------------------------------
# class_suspect kinematics guardrail


def _track_with_box_shape(class_id: int, w: float, h: float) -> BufferedTrack:
    """Moving track (passes the parked filter) whose every bbox is w x h."""
    tr = BufferedTrack(id=7, class_id=class_id)
    for i, cx in enumerate((100.0, 300.0, 500.0)):
        tr.points.append(
            MotionPoint(
                frame_idx=i,
                t=float(i),
                cx=cx,
                cy=200.0,
                x1=cx - w / 2,
                y1=200.0 - h / 2,
                x2=cx + w / 2,
                y2=200.0 + h / 2,
                score=0.9,
            )
        )
    return tr


def _finalize(tr: BufferedTrack) -> TrackRecord:
    result = compute_attributes(
        tr,
        frame_h=_FRAME_H,
        min_duration_s=1.0,
        parked_disp_px=50.0,
        color="red",
        t_start_wall=_T0_WALL,
    )
    assert result is not None
    return result


def test_median_bbox_aspect_is_median_of_ratios() -> None:
    tr = _track_with_box_shape(0, w=50, h=100)  # three samples at 0.5
    tr.points[0].x2 = tr.points[0].x1 + 200  # one wide outlier (2.0)
    assert median_bbox_aspect(tr.points) == pytest.approx(0.5)
    assert median_bbox_aspect([]) == 0.0


def test_person_with_car_shaped_boxes_is_suspect() -> None:
    rec = _finalize(_track_with_box_shape(0, w=160, h=100))  # aspect 1.6
    assert rec.class_name == "person"
    assert rec.class_suspect is True


def test_person_with_person_shaped_boxes_is_not_suspect() -> None:
    rec = _finalize(_track_with_box_shape(0, w=45, h=100))  # aspect 0.45
    assert rec.class_suspect is False


def test_rider_shaped_person_is_not_suspect() -> None:
    # Riders (person-class) max out at aspect ~1.44 in the measured
    # soaks; the 1.5 threshold must leave them alone.
    rec = _finalize(_track_with_box_shape(0, w=144, h=100))
    assert rec.class_suspect is False


def test_wide_car_is_not_suspect() -> None:
    rec = _finalize(_track_with_box_shape(2, w=160, h=100))
    assert rec.class_name == "car"
    assert rec.class_suspect is False
