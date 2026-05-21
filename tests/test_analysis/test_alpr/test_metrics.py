"""Scoring metrics in ``streettracker.analysis.alpr.metrics``."""

from __future__ import annotations

import pytest

from streettracker.analysis.alpr import metrics

# ---------------------------------------------------------------- Levenshtein


class TestLevenshtein:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("", "", 0),
            ("ABC", "ABC", 0),
            ("ABC", "ABD", 1),     # one substitution
            ("ABC", "AC", 1),      # one deletion
            ("AC", "ABC", 1),      # one insertion
            ("", "ABC", 3),
            ("ABC", "", 3),
            ("KITTEN", "SITTING", 3),  # textbook example
        ],
    )
    def test_basic(self, a: str, b: str, expected: int) -> None:
        assert metrics.levenshtein(a, b) == expected

    def test_wildcard_in_truth_costs_zero(self) -> None:
        # '.' in truth (the b side) matches anything at sub-cost 0
        assert metrics.levenshtein("ABC", "A.C", wildcard=".") == 0

    def test_wildcard_only_on_truth_side(self) -> None:
        # '.' in OCR side (a) is just a regular char — does NOT match truth
        assert metrics.levenshtein("A.C", "ABC", wildcard=".") == 1


# ---------------------------------------------------------------- plate_match


class TestPlateMatch:
    def test_exact(self) -> None:
        assert metrics.plate_match("ABC123", "ABC123", mode="exact") is True

    def test_length_mismatch_is_false(self) -> None:
        assert metrics.plate_match("ABC12", "ABC123") is False

    def test_wildcard_in_truth(self) -> None:
        assert metrics.plate_match("ABC123", "AB..23", mode="exact") is True

    def test_canon_matches_ocr_confusions(self) -> None:
        # OCR returned 'O', truth has '0' → canon matches
        assert metrics.plate_match("OBC123", "0BC123", mode="canon") is True
        # Exact fails for the same pair
        assert metrics.plate_match("OBC123", "0BC123", mode="exact") is False


# ---------------------------------------------------------- aggregate metrics


def _rec(image: str, pipeline: str, *, det: bool = True, ocr: str | None = None,
         track_id: int = 1, snap_index: int = 1, ocr_conf: float | None = None) -> dict:
    """Factory for synthetic ALPR records used across the aggregator tests."""
    return {
        "image": image,
        "image_path": image,
        "track_id": track_id,
        "snap_index": snap_index,
        "class_name": "vehicle",
        "pipeline": pipeline,
        "det_bbox": [10, 20, 100, 60] if det else None,
        "det_conf": 0.9 if det else None,
        "ocr_text": ocr,
        "ocr_raw": ocr,
        "ocr_conf": ocr_conf,
        "crop_path": None,
        "pipeline_ms": 1.0,
        "error": None,
    }


class TestPerPipelineDetectionRate:
    def test_basic(self) -> None:
        records = [
            _rec("a.jpg", "bespoke", det=True),
            _rec("b.jpg", "bespoke", det=False),
            _rec("a.jpg", "preferred", det=True),
            _rec("b.jpg", "preferred", det=True),
        ]
        rates = metrics.per_pipeline_detection_rate(records)
        assert rates["bespoke"] == 0.5
        assert rates["preferred"] == 1.0


class TestPerPipelineReadRate:
    def test_basic(self) -> None:
        records = [
            _rec("a.jpg", "bespoke", ocr="ABC"),
            _rec("b.jpg", "bespoke", ocr=None),
            _rec("a.jpg", "preferred", ocr=""),  # empty string counts as no read
            _rec("b.jpg", "preferred", ocr="XYZ"),
        ]
        rates = metrics.per_pipeline_read_rate(records)
        assert rates["bespoke"] == 0.5
        assert rates["preferred"] == 0.5


class TestCrossPipelineAgreement:
    def test_exact_and_canon(self) -> None:
        records = [
            _rec("a.jpg", "bespoke", ocr="ABC123"),
            _rec("a.jpg", "preferred", ocr="ABC123"),   # exact match
            _rec("b.jpg", "bespoke", ocr="ABCO23"),
            _rec("b.jpg", "preferred", ocr="ABC023"),   # canon match only
            _rec("c.jpg", "bespoke", ocr="XYZ"),
            _rec("c.jpg", "preferred", ocr="QQQ"),      # disagreement
        ]
        agg = metrics.cross_pipeline_agreement(records)
        assert agg["bespoke_vs_preferred_overlap_n"] == 3
        # 1 exact, 1 canon-only, 1 disagreement
        assert agg["bespoke_vs_preferred_exact"] == pytest.approx(1 / 3)
        assert agg["bespoke_vs_preferred_canon"] == pytest.approx(2 / 3)

    def test_skips_pairs_with_missing_read(self) -> None:
        records = [
            _rec("a.jpg", "bespoke", ocr=None),
            _rec("a.jpg", "preferred", ocr="ABC"),
        ]
        agg = metrics.cross_pipeline_agreement(records)
        assert agg["bespoke_vs_preferred_overlap_n"] == 0


class TestLabelBasedAccuracy:
    def test_exact_and_canon_and_mean_ed(self) -> None:
        records = [
            _rec("vehicle_1_main_1.jpg", "bespoke", ocr="ABC123"),
            _rec("vehicle_2_main_1.jpg", "bespoke", ocr="ABCO23"),  # O vs 0
            _rec("vehicle_3_main_1.jpg", "bespoke", ocr="ZZZZ"),    # totally wrong
        ]
        labels = {
            "vehicle_1_main_1.jpg": {"plate": "ABC123"},
            "vehicle_2_main_1.jpg": {"plate": "ABC023"},
            "vehicle_3_main_1.jpg": {"plate": "AAAA"},
        }
        out = metrics.label_based_accuracy(records, labels)
        assert out["bespoke"]["labeled_n"] == 3
        assert out["bespoke"]["exact_accuracy"] == pytest.approx(1 / 3)
        assert out["bespoke"]["canonical_accuracy"] == pytest.approx(2 / 3)
        # ED 0 + 1 + 4 = 5; mean = 5/3
        assert out["bespoke"]["mean_edit_distance"] == pytest.approx(5 / 3)

    def test_excludes_sentinel_labels(self) -> None:
        records = [
            _rec("a.jpg", "bespoke", ocr="ABC"),
        ]
        labels = {"a.jpg": {"plate": metrics.UNREADABLE}}
        out = metrics.label_based_accuracy(records, labels)
        assert out == {}

    def test_no_labels_returns_empty(self) -> None:
        records = [_rec("a.jpg", "bespoke", ocr="ABC")]
        assert metrics.label_based_accuracy(records, {}) == {}


class TestPerTrackBestOfN:
    def test_any_correct_read_passes_track(self) -> None:
        records = [
            _rec("vehicle_1_main_1.jpg", "bespoke", ocr="WRONG", track_id=1),
            _rec("vehicle_1_main_2.jpg", "bespoke", ocr="ABC123", track_id=1),
            _rec("vehicle_2_main_1.jpg", "bespoke", ocr="XYZ", track_id=2),
        ]
        labels = {
            "vehicle_1_main_1.jpg": {"plate": "ABC123"},
            "vehicle_2_main_1.jpg": {"plate": "AAAA"},
        }
        out = metrics.per_track_best_of_n(records, labels)
        # Track 1: ABC123 matched on snap 2 → hit; Track 2: never matched
        assert out["bespoke"]["labeled_tracks"] == 2
        assert out["bespoke"]["best_of_n_accuracy"] == 0.5


class TestCharConfusionTable:
    def test_counts_per_char_mismatches(self) -> None:
        records = [
            _rec("a.jpg", "bespoke", ocr="ABC"),
            _rec("b.jpg", "bespoke", ocr="A8C"),  # B vs 8
        ]
        labels = {
            "a.jpg": {"plate": "ABC"},
            "b.jpg": {"plate": "ABC"},  # truth B at position 2
        }
        out = metrics.char_confusion_table(records, labels)
        # Only one confusion: truth 'B' read as '8'
        confusions = out["bespoke"]
        assert ("B", "8", 1) in confusions
