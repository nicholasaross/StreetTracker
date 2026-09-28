"""Pure-Python helpers in ``streettracker.analysis.alpr.base``."""

from __future__ import annotations

import numpy as np
import pytest

from streettracker.analysis.alpr import base


class TestParseSnapFilename:
    def test_vehicle(self) -> None:
        assert base.parse_snap_filename("vehicle_42_main_3.jpg") == ("vehicle", 42, 3)

    def test_person(self) -> None:
        assert base.parse_snap_filename("person_7_main_1.jpg") == ("person", 7, 1)

    def test_rejects_non_main_snap(self) -> None:
        assert base.parse_snap_filename("vehicle_42_hq.jpg") is None

    def test_rejects_no_extension(self) -> None:
        assert base.parse_snap_filename("vehicle_42_main_3") is None

    def test_rejects_unknown_prefix(self) -> None:
        assert base.parse_snap_filename("car_1_main_1.jpg") is None


class TestNormalizePlateText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("abc-123", "ABC123"),
            ("  AB 12 CD  ", "AB12CD"),
            ("ABC.123", "ABC123"),  # '.' stripped — this is the OCR side
            ("", ""),
            ("!@#$", ""),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert base.normalize_plate_text(raw) == expected


class TestNormalizeLabelText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("AB.123", "AB.123"),  # '.' preserved — wildcards survive
            ("ab 1.cd", "AB1.CD"),
            ("", ""),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert base.normalize_label_text(raw) == expected


class TestCanonicalForScoring:
    def test_collapses_common_confusions(self) -> None:
        # O->0, I->1, S->5, B->8, Z->2
        assert base.canonical_for_scoring("OISZB") == "01528"

    def test_strips_non_alnum_first(self) -> None:
        # 'B' is in the confusion map (→ 8), so we expect 'A801' not 'AB01'.
        assert base.canonical_for_scoring("ab-OI") == "A801"


class TestCanonicalTranslate:
    def test_preserves_wildcard(self) -> None:
        # Used for label text: '.' survives. 'B' is in the confusion map (→ 8).
        assert base.canonical_translate("AB.OI") == "A8.01"


class TestCropWithPadding:
    def test_basic_crop(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = 255
        crop = base.crop_with_padding(img, (40, 40, 60, 60), pad_frac=0.0)
        assert crop.shape == (20, 20, 3)

    def test_padding_expands_crop(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = base.crop_with_padding(img, (40, 40, 60, 60), pad_frac=0.25)
        # 20px bbox * 0.25 = 5px each side -> 30x30
        assert crop.shape == (30, 30, 3)

    def test_padding_clamps_to_image_bounds(self) -> None:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        # bbox at top-left corner; padding would go negative
        crop = base.crop_with_padding(img, (0, 0, 10, 10), pad_frac=0.5)
        # Clamped to image: starts at 0, extends with right/bottom padding (5px)
        assert crop.shape[0] >= 10 and crop.shape[1] >= 10
        assert crop.shape[0] <= 50 and crop.shape[1] <= 50

    def test_zero_size_bbox_is_safe(self) -> None:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        # max(1, 0) safeguards from div by zero
        crop = base.crop_with_padding(img, (10, 10, 10, 10), pad_frac=0.1)
        assert crop.shape[2] == 3  # didn't crash; returned a (possibly empty) 3-ch crop


class TestAtomicWrites:
    def test_atomic_write_text(self, tmp_path) -> None:
        p = tmp_path / "x.json"
        base.atomic_write_text(p, '{"a": 1}')
        assert p.read_text() == '{"a": 1}'
        # No leftover .tmp file
        assert list(tmp_path.iterdir()) == [p]

    def test_atomic_write_bytes(self, tmp_path) -> None:
        p = tmp_path / "x.jpg"
        base.atomic_write_bytes(p, b"\xff\xd8\xff")
        assert p.read_bytes() == b"\xff\xd8\xff"


class TestTimer:
    def test_records_elapsed_ms(self) -> None:
        import time

        with base.Timer() as t:
            time.sleep(0.01)
        # Tolerate scheduler noise; just make sure something nonzero landed.
        assert t.ms > 0
        assert t.ms < 1000  # not 1+ second


class TestPlateResultJson:
    def test_carries_per_character_probabilities(self) -> None:
        result = base.PlateResult(
            image_path="/s/vehicle_1_main_1.jpg",
            image_name="vehicle_1_main_1.jpg",
            track_id=1,
            snap_index=1,
            class_name="vehicle",
            pipeline="preferred",
            detection=base.PlateDetection(bbox=(1, 2, 3, 4), det_confidence=0.9),
            read=base.PlateRead(
                text="AB12CDE", ocr_confidence=0.35, raw_text="AB12CDE",
                char_probs=[0.99, 0.35, 0.99, 0.98, 0.97, 0.99, 0.96],
            ),
            crop_path="/s/alpr_crops/preferred/vehicle_1_main_1.jpg",
            pipeline_ms=1.0,
        )
        d = result.to_json()
        assert d["ocr_conf"] == 0.35
        assert d["ocr_char_probs"] == [0.99, 0.35, 0.99, 0.98, 0.97, 0.99, 0.96]
        assert d["canonical_uk_shape"] is True

    def test_recognizers_without_char_probs_serialise_none(self) -> None:
        read = base.PlateRead(text="ABC", ocr_confidence=0.7, raw_text="ABC")
        assert read.char_probs is None


class TestPlateConfThreshold:
    """The one shared plate-read confidence gate (configs/alpr.json)."""

    def _write(self, path, content) -> None:  # noqa: ANN001
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content))

    def test_default_without_a_config(self) -> None:
        assert base.resolve_plate_conf_threshold() == (0.9, "default")
        assert base.plate_conf_threshold() == 0.9

    def test_config_value_is_used(self, _isolated_plate_conf_config) -> None:  # noqa: ANN001
        self._write(_isolated_plate_conf_config, {"plate_conf_threshold": 0.82})
        value, source = base.resolve_plate_conf_threshold()
        assert value == 0.82
        assert source == str(_isolated_plate_conf_config)

    def test_command_line_override_wins(self, _isolated_plate_conf_config) -> None:  # noqa: ANN001
        self._write(_isolated_plate_conf_config, {"plate_conf_threshold": 0.82})
        assert base.resolve_plate_conf_threshold(0.7) == (0.7, "command line")

    def test_comment_only_file_falls_back_to_default(
        self,
        _isolated_plate_conf_config,  # noqa: ANN001
    ) -> None:
        self._write(_isolated_plate_conf_config, {"_comment": "not set yet"})
        assert base.resolve_plate_conf_threshold() == (0.9, "default")

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ({"plate_conf_treshold": 0.8}, "unknown key"),  # a typo must not fall back to 0.9
            ({"plate_conf_threshold": 1.5}, r"\(0, 1\]"),
            ({"plate_conf_threshold": 0}, r"\(0, 1\]"),
            ({"plate_conf_threshold": "0.8"}, r"\(0, 1\]"),
            ({"plate_conf_threshold": True}, r"\(0, 1\]"),
            ([0.8], "JSON object"),
            ("{not json", "unreadable"),
        ],
    )
    def test_unusable_config_raises_instead_of_falling_back(
        self,
        _isolated_plate_conf_config,  # noqa: ANN001
        content,  # noqa: ANN001
        match: str,
    ) -> None:
        self._write(_isolated_plate_conf_config, content)
        with pytest.raises(ValueError, match=match):
            base.resolve_plate_conf_threshold()

    def test_out_of_range_override_raises(self) -> None:
        with pytest.raises(ValueError, match="command line"):
            base.resolve_plate_conf_threshold(1.2)
