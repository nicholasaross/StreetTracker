"""Tests for the automatic static-plate filter."""

from __future__ import annotations

from streettracker.analysis.alpr.staticfilter import (
    find_static_spots,
    mark_static_suspects,
)

SUB = (896, 512)
IMG = (4480, 2560)  # exact 5x sub-stream so test arithmetic stays simple


def _rec(tid: int, snap: int, bbox: tuple[int, int, int, int] | None, text: str = "") -> dict:
    return {
        "pipeline": "preferred",
        "track_id": tid,
        "snap_index": snap,
        "det_bbox": list(bbox) if bbox else None,
        "ocr_text": text or None,
        "ocr_conf": 0.95 if text else None,
    }


def _car(x: int) -> tuple[int, int, int, int]:
    """A sub-stream car bbox centred at (x, 230); scales to 5x in IMG."""
    return (x - 40, 200, x + 40, 260)


class TestAffineSeeds:
    def test_static_det_while_car_sweeps_is_static(self) -> None:
        # Same det position on two snaps; the car translated 100 sub px
        # (500 img px), so an attached plate was predicted to move ~500.
        records = [
            _rec(7, 1, (1000, 500, 1035, 522), "AB1234"),
            _rec(7, 2, (1002, 501, 1037, 523), "AB1239"),
        ]
        car_index = {(7, 1): _car(300), (7, 2): _car(400)}
        spots, consistent = find_static_spots(records, car_index, SUB, IMG)
        assert len(spots) == 1
        assert spots[0]["source"] == "within_track"
        assert consistent == set()
        n = mark_static_suspects(records, spots, consistent)
        assert n == 2
        assert all(r.get("static_suspect") for r in records)

    def test_plate_moving_with_car_not_static_and_immune(self) -> None:
        # Det translates with the car: affine-consistent, no spot, and
        # the det keys land in the immune set.
        records = [
            _rec(7, 1, (1400, 1050, 1500, 1080), "AB12CDE"),
            _rec(7, 2, (1900, 1050, 2000, 1080), "AB12CDE"),
        ]
        car_index = {(7, 1): _car(300), (7, 2): _car(400)}
        spots, consistent = find_static_spots(records, car_index, SUB, IMG)
        assert spots == []
        assert (7, 1) in consistent and (7, 2) in consistent

    def test_vanishing_zone_recession_is_ambiguous_not_static(self) -> None:
        # Receding car near the vanishing point: bbox centre moves but
        # the far edge -- where the plate is -- barely does. Predicted
        # plate motion is tiny, so the pair must be treated as unknown,
        # NOT static (this was the failure mode of a naive
        # car-moved/plate-didn't rule).
        records = [
            _rec(7, 1, (2850, 1490, 2920, 1512), "AB12CDE"),
            _rec(7, 2, (2848, 1491, 2916, 1512), "AB12CDE"),
        ]
        # sub boxes: right edge (x2) nearly fixed at the vanishing side,
        # left edge advancing => centre moves ~15 sub px, far edge ~2.
        car_index = {(7, 1): (500, 280, 584, 330), (7, 2): (530, 282, 583, 328)}
        spots, _ = find_static_spots(records, car_index, SUB, IMG)
        assert spots == []

    def test_missing_car_position_is_conservative(self) -> None:
        records = [
            _rec(7, 1, (1000, 500, 1035, 522), "AB1234"),
            _rec(7, 2, (1002, 501, 1037, 523), "AB1239"),
        ]
        spots, _ = find_static_spots(records, {}, SUB, IMG)
        assert spots == []

    def test_missing_sizes_skips_affine(self) -> None:
        records = [
            _rec(7, 1, (1000, 500, 1035, 522), "AB1234"),
            _rec(7, 2, (1002, 501, 1037, 523), "AB1239"),
        ]
        car_index = {(7, 1): _car(300), (7, 2): _car(400)}
        spots, _ = find_static_spots(records, car_index, None, None)
        assert spots == []


class TestCrossTrackSpots:
    def test_many_tracks_same_spot_same_size_is_static(self) -> None:
        # Eight distinct single-snap tracks all "detect" a 33px plate
        # at one fixed position (the parked-car signature).
        records = [
            _rec(t, 1, (2239 + (t % 3), 332, 2272 + (t % 3), 354), f"E4157{t}") for t in range(1, 9)
        ]
        spots, consistent = find_static_spots(records, {}, SUB, IMG)
        assert len(spots) == 1
        assert spots[0]["source"] == "cross_track"
        assert mark_static_suspects(records, spots, consistent) == 8

    def test_inconsistent_widths_not_static(self) -> None:
        # Same position but varying box widths -- fails the width-
        # consistency gate (not one physical object).
        records = [_rec(t, 1, (2200, 330, 2200 + 35 + 18 * t, 352), str(t)) for t in range(1, 9)]
        spots, _ = find_static_spots(records, {}, SUB, IMG)
        assert spots == []

    def test_few_tracks_not_static(self) -> None:
        records = [_rec(t, 1, (2239, 332, 2272, 354), f"E4157{t}") for t in range(1, 5)]
        spots, _ = find_static_spots(records, {}, SUB, IMG)
        assert spots == []


class TestMarking:
    def _spot_records(self) -> list[dict]:
        return [
            _rec(t, 1, (2239 + (t % 3), 332, 2272 + (t % 3), 354), f"E4157{t}") for t in range(1, 9)
        ]

    def test_no_detection_records_never_marked(self) -> None:
        records = self._spot_records() + [_rec(99, 1, None)]
        spots, consistent = find_static_spots(records, {}, SUB, IMG)
        mark_static_suspects(records, spots, consistent)
        assert "static_suspect" not in records[-1]

    def test_far_detection_not_marked(self) -> None:
        records = self._spot_records()
        far = _rec(99, 1, (900, 1500, 1010, 1533), "GL63OWM")
        records.append(far)
        spots, consistent = find_static_spots(records, {}, SUB, IMG)
        mark_static_suspects(records, spots, consistent)
        assert "static_suspect" not in far

    def test_consistent_det_inside_spot_is_immune(self) -> None:
        # A moving car's plate passes through the parked spot's pixels;
        # its det is affine-consistent with its own track and must not
        # be suppressed.
        records = self._spot_records()
        passing = [
            _rec(50, 1, (2240, 333, 2340, 363), "GL63OWM"),
            _rec(50, 2, (2740, 333, 2840, 363), "GL63OWM"),
        ]
        records += passing
        car_index = {(50, 1): _car(448), (50, 2): _car(548)}
        spots, consistent = find_static_spots(records, car_index, SUB, IMG)
        mark_static_suspects(records, spots, consistent)
        assert "static_suspect" not in passing[0]
        assert "static_suspect" not in passing[1]

    def test_no_spots_no_marks(self) -> None:
        records = [_rec(1, 1, (100, 100, 200, 130), "AB12CDE")]
        assert mark_static_suspects(records, [], set()) == 0


class TestRollupExclusion:
    def test_rollup_skips_static_suspects(self) -> None:
        from streettracker.cli.alpr_run import _rollup_by_track

        records = [
            # a genuine moving read
            {
                "pipeline": "preferred",
                "track_id": 5,
                "snap_index": 1,
                "image": "vehicle_5_main_1.jpg",
                "ocr_text": "GL63OWM",
                "ocr_conf": 0.91,
                "det_conf": 0.9,
                "canonical_uk_shape": True,
            },
            # a higher-conf parked-car read on the same track -- would
            # win best-of-N if not excluded
            {
                "pipeline": "preferred",
                "track_id": 5,
                "snap_index": 2,
                "image": "vehicle_5_main_2.jpg",
                "ocr_text": "FD61PVX",
                "ocr_conf": 0.99,
                "det_conf": 0.95,
                "canonical_uk_shape": True,
                "static_suspect": True,
            },
        ]
        rollup = _rollup_by_track(records)
        (track,) = rollup["tracks"]
        assert track["best_preferred"]["ocr_text"] == "GL63OWM"
