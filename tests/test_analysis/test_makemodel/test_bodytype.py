"""body_type_for: DVSA (make, model) -> coarse body type."""

from __future__ import annotations

import pytest

from streettracker.analysis.makemodel.bodytype import BODY_TYPES, body_type_for


@pytest.mark.parametrize(
    ("make", "model", "expected"),
    [
        # Clean base models across the taxonomy.
        ("FORD", "FIESTA", "hatchback"),
        ("FORD", "MONDEO", "saloon"),
        ("NISSAN", "QASHQAI", "suv"),
        ("VOLKSWAGEN", "TOURAN", "mpv"),
        ("FORD", "TRANSIT", "van"),
        ("FORD", "RANGER", "pickup"),
        ("TOYOTA", "SUPRA", "coupe"),
        # Trim/spec suffixes collapse onto the base model via prefix match.
        ("FORD", "PUMA ST-LINE MHEV", "suv"),
        ("TOYOTA", "COROLLA DESIGN HEV CVT", "hatchback"),
        ("TESLA", "MODEL Y LONG RANGE AWD", "suv"),
        ("MG", "ZS TROPHY HEV AUTO", "suv"),
        # Longest-prefix-first: the more specific model name wins.
        ("FORD", "TRANSIT CONNECT", "van"),
        ("TOYOTA", "YARIS CROSS ICON HEV", "suv"),
        ("CITROEN", "C3 AIRCROSS", "suv"),
        ("CITROEN", "C3", "hatchback"),
        # Case / whitespace normalisation.
        ("ford", "  focus ", "hatchback"),
        ("Mercedes", "C-Class", "saloon"),
    ],
)
def test_known_models(make: str, model: str, expected: str) -> None:
    assert body_type_for(make, model) == expected


def test_mercedes_single_letter_boundary() -> None:
    """Letter-terminated prefixes need a word boundary, so bare 'C'
    (saloon) can't swallow 'CITAN' (van) or 'CLA'."""
    assert body_type_for("MERCEDES-BENZ", "C") == "saloon"
    assert body_type_for("MERCEDES-BENZ", "C 220 D") == "saloon"
    assert body_type_for("MERCEDES-BENZ", "CITAN") == "van"
    assert body_type_for("MERCEDES-BENZ", "CLA") == "saloon"
    assert body_type_for("MERCEDES-BENZ", "SPRINTER 315 CDI") == "van"
    assert body_type_for("MERCEDES-BENZ", "GLA") == "suv"


def test_bmw_numeric_and_digit_boundary() -> None:
    """Digit-terminated codes match plainly so trim letters attach."""
    assert body_type_for("BMW", "116") == "hatchback"
    assert body_type_for("BMW", "116D") == "hatchback"
    assert body_type_for("BMW", "320D M SPORT") == "saloon"
    assert body_type_for("BMW", "520") == "saloon"
    assert body_type_for("BMW", "X3") == "suv"
    assert body_type_for("BMW", "Z4") == "coupe"
    assert body_type_for("BMW", "3 SERIES 320D") == "saloon"


def test_uncovered_and_placeholder_return_empty() -> None:
    assert body_type_for("FORD", "SOME-FUTURE-MODEL") == ""
    assert body_type_for("SAAB", "9-3") == ""  # make not in the table
    assert body_type_for("UNKNOWN", "FIESTA") == ""  # placeholder make
    assert body_type_for(None, None) == ""
    assert body_type_for("FORD", "") == ""


def test_all_mapped_body_types_are_canonical() -> None:
    """Every body type the table can emit is a declared BODY_TYPES class."""
    from streettracker.analysis.makemodel.bodytype import _RAW

    emitted = {bt for by_bt in _RAW.values() for bt in by_bt}
    assert emitted <= set(BODY_TYPES)
