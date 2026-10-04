"""Tests for the UK car-make benchmark (DfT licensed cars, age-matched)."""

from __future__ import annotations

import pytest

from streettracker.web.uk_makes import (
    age_matched_shares,
    load_benchmark,
    make_benchmark,
    uk_shares,
)

BENCH = {
    "latest": {"period": "2026 Q2", "total": 1000, "makes": {"FORD": 100, "TOYOTA": 50}},
    "by_year": {
        "period": "end of 2025",
        "years": {
            "2020": {"total": 100, "makes": {"FORD": 50, "TOYOTA": 50}},
            "2021": {"total": 100, "makes": {"FORD": 10, "TOYOTA": 90}},
        },
    },
}


def test_uk_shares_are_shares_of_all_licensed_cars() -> None:
    assert uk_shares(BENCH) == {"FORD": 0.1, "TOYOTA": 0.05}


def test_age_matching_weights_each_year_by_the_street() -> None:
    shares = age_matched_shares(BENCH, {2020: 1, 2021: 3})
    assert shares["FORD"] == pytest.approx(0.25 * 0.5 + 0.75 * 0.1)
    assert shares["TOYOTA"] == pytest.approx(0.25 * 0.5 + 0.75 * 0.9)


def test_years_outside_the_table_use_the_nearest_year() -> None:
    # 2026 cars (after the table's newest year-end) use 2021; 1990 uses 2020.
    assert age_matched_shares(BENCH, {2026: 1}) == age_matched_shares(BENCH, {2021: 1})
    assert age_matched_shares(BENCH, {1990: 2}) == age_matched_shares(BENCH, {2020: 1})
    assert age_matched_shares(BENCH, {}) == {}


def test_make_benchmark_rows() -> None:
    makes = {"A1": "FORD", "A2": "FORD", "A3": "TOYOTA", "A4": "Saab"}
    years = {"A1": 2020, "A2": 2021, "A3": 2021, "A4": None}
    top = [["FORD", 2], ["TOYOTA", 1], ["Saab", 1]]
    b = make_benchmark(makes, years, top, bench=BENCH)
    assert b is not None and b["n_cars"] == 4 and b["n_with_year"] == 3
    ford, toyota, saab = b["rows"]
    assert ford["share"] == 0.5 and toyota["share"] == 0.25
    # Weights: 2020 one car, 2021 two cars.
    assert ford["uk_share"] == pytest.approx(round(0.5 / 3 + 0.1 * 2 / 3, 4))
    assert ford["uk_share_all"] == 0.1
    assert saab["uk_share"] is None and saab["uk_share_all"] is None  # not in the benchmark


def test_make_benchmark_needs_data() -> None:
    assert make_benchmark({}, {}, [], bench=BENCH) is None
    assert make_benchmark({"A": "FORD"}, {}, [["FORD", 1]], bench={}) is None


def test_shipped_benchmark_matches_dvsa_make_names() -> None:
    bench = load_benchmark()
    assert bench is not None
    assert bench["latest"]["total"] > 30_000_000  # UK licensed cars
    makes = bench["latest"]["makes"]
    # DfT's MERCEDES / SMART are mapped to the DVSA MOT names.
    assert "MERCEDES-BENZ" in makes and "MERCEDES" not in makes
    assert "SMART (MCC)" in makes
    assert all(k in makes for k in ("FORD", "VOLKSWAGEN", "TOYOTA", "VAUXHALL", "MG", "TESLA"))
    assert "2025" in bench["by_year"]["years"]
