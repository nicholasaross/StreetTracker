"""Tests for turning the audit set's labels into rates (review E1.2)."""

from __future__ import annotations

import pytest

from streettracker.analysis.alpr.audit import DIRECTIONS, LIGHT_BANDS
from streettracker.analysis.alpr.audit_score import (
    Rate,
    build_report,
    dvsa_any_wrong,
    format_report,
    labelled_rows,
    on_tracked_car,
    plate_text_equal,
    rescore,
    text_right,
    verified,
    weighted_rate,
)

# ----------------------------------------------------------------------
# One item's answers


@pytest.mark.parametrize(
    ("read", "truth", "expected"),
    [
        ("AB12CDE", "AB12CDE", True),
        ("ab12cde", "AB12CDE", True),
        ("AB12CXE", "AB12C.E", True),  # '.' is a character the labeller couldn't read
        ("AB12CDE", "AB12CDF", False),
        ("AB12CD", "AB12CDE", False),
        ("A012CDE", "AO12CDE", True),  # O and 0 share a glyph on UK plates
        ("AB1ICDE", "AB11CDE", True),  # so do I and 1
        ("AB12CDE", None, None),
        (None, "AB12CDE", False),
    ],
)
def test_plate_text_equal(read: str | None, truth: str | None, expected: bool | None) -> None:
    assert plate_text_equal(read, truth) is expected


def _row(attribution: str, text: str | None, ocr: str = "AB12CDE", **label: object) -> dict:
    return {"ocr_text": ocr, "label": {"attribution": attribution, "text": text, **label}}


def test_on_tracked_car() -> None:
    assert on_tracked_car(_row("tracked", "AB12CDE")) is True
    assert on_tracked_car(_row("parked", "AB12CDE")) is False
    assert on_tracked_car(_row("not_plate", None)) is False
    assert on_tracked_car(_row("unsure", "AB12CDE")) is None


def test_text_right() -> None:
    assert text_right(_row("parked", "AB12CDE")) is True  # right text, wrong car
    assert text_right(_row("tracked", "AB12CDF")) is False
    assert text_right(_row("tracked", None)) is None  # illegible
    assert text_right(_row("not_plate", None)) is False


@pytest.mark.parametrize(
    ("attribution", "text", "expected"),
    [
        ("tracked", "AB12CDE", True),
        ("tracked", "AB12CDF", False),
        ("parked", "AB12CDE", False),
        ("parked", None, False),  # the wrong car settles it, legible or not
        ("unsure", "AB12CDF", False),  # so does the wrong text
        ("unsure", "AB12CDE", None),
        ("tracked", None, None),
    ],
)
def test_verified(attribution: str, text: str | None, expected: bool | None) -> None:
    assert verified(_row(attribution, text)) is expected


def test_dvsa_any_wrong() -> None:
    def row(make: str, colour: str, body: str) -> dict:
        return _row("tracked", "AB12CDE", dvsa={"make": make, "colour": colour, "body": body})

    assert dvsa_any_wrong(row("yes", "yes", "yes")) is False
    assert dvsa_any_wrong(row("yes", "no", "unsure")) is True
    assert dvsa_any_wrong(row("yes", "yes", "unsure")) is None


# ----------------------------------------------------------------------
# Rates


def _rows(spec: dict[str, tuple[float, list[bool | None]]]) -> list[dict]:
    return [
        {"stratum": stratum, "weight": weight, "ok": ok}
        for stratum, (weight, oks) in spec.items()
        for ok in oks
    ]


def _ok(row: dict) -> bool | None:
    return row["ok"]


def test_weighted_rate_weights_each_stratum_by_its_population() -> None:
    # A stands for 100 reads (50 each), B for 300 (150 each).
    rows = _rows({"A": (50.0, [True, False]), "B": (150.0, [True, True])})
    got = weighted_rate(rows, _ok)
    assert (got.n, got.hits) == (4, 3)
    assert got.value == pytest.approx((50 + 150 + 150) / 400)
    assert got.low is not None and got.high is not None
    assert got.low < got.value < got.high


def test_weighted_rate_is_repeatable_for_a_seed() -> None:
    rows = _rows({"A": (10.0, [True, False, True, False, True]), "B": (3.0, [False, True, True])})
    assert weighted_rate(rows, _ok, seed=4) == weighted_rate(rows, _ok, seed=4)


def test_weighted_rate_uses_an_exact_bound_when_every_item_agrees() -> None:
    rows = _rows({"A": (100.0, [True] * 10)})
    got = weighted_rate(rows, _ok)
    assert (got.value, got.high) == (1.0, 1.0)
    assert got.low == pytest.approx(0.05 ** (1 / 10))  # about 1 - 3/n

    none = weighted_rate(_rows({"A": (100.0, [False] * 10)}), _ok)
    assert (none.value, none.low) == (0.0, 0.0)
    assert none.high == pytest.approx(1 - 0.05 ** (1 / 10))


def test_weighted_rate_bound_uses_the_effective_sample_size() -> None:
    # One heavy item and nine light ones: worth far fewer than ten.
    rows = _rows({"A": (1000.0, [True]), "B": (1.0, [True] * 9)})
    got = weighted_rate(rows, _ok)
    assert got.low is not None and got.low < 0.05 ** (1 / 2)


def test_weighted_rate_leaves_out_undecided_and_unselected_items() -> None:
    rows = _rows({"A": (1.0, [True, None, None, False])})
    rows[0]["keep"] = rows[1]["keep"] = rows[3]["keep"] = True
    got = weighted_rate(rows, _ok)
    assert (got.n, got.hits, got.value) == (2, 1, 0.5)
    only_first = weighted_rate(rows, _ok, where=lambda r: r is rows[0])
    assert (only_first.n, only_first.value) == (1, 1.0)


def test_weighted_rate_of_nothing() -> None:
    assert weighted_rate([], _ok) == Rate(0, 0, None, None, None)
    assert weighted_rate(_rows({"A": (1.0, [None])}), _ok).value is None


def test_rate_scaled_and_json() -> None:
    rate = Rate(4, 3, 0.75, 0.5, 0.9).scaled(0.5)
    assert rate.to_json() == {"n": 4, "hits": 3, "value": 0.375, "low": 0.25, "high": 0.45}
    assert Rate(0, 0, None, None, None).scaled(2.0).value is None


def test_labelled_rows_weight_by_the_labelled_count() -> None:
    manifest = {
        "strata": {"A": {"population": 90, "sampled": 3}},
        "items": [{"key": k, "stratum": "A"} for k in ("a", "b", "c")],
    }
    rows = labelled_rows(
        manifest, {"a": {"attribution": "tracked"}, "c": {"attribution": "parked"}}
    )
    assert [(r["key"], r["weight"]) for r in rows] == [("a", 45.0), ("c", 45.0)]
    assert rows[1]["label"] == {"attribution": "parked"}


# ----------------------------------------------------------------------
# The report

RL_DAY = "right to left|day"


def _item(key: str, sample: str, stratum: str, conf: float, gate: bool, **extra: object) -> dict:
    return {
        "key": key,
        "sample": sample,
        "stratum": stratum,
        "session": "s",
        "track_id": int(key[1:]),
        "direction": "right to left",
        "light": "day",
        "ocr_text": "AB12CDE",
        "ocr_conf": conf,
        "gate_pass": gate,
        "n_agree": 1 if gate else 0,
        "support": 3,
        "path": "rank0",
        "track_best": {"ocr_text": "AB12CDE", "ocr_conf": conf, "canonical_uk_shape": True},
        **extra,
    }


PASS = "best|right to left|day|rank0_pass"
FAIL = "best|right to left|day|rank0_fail"
COLOUR = "colour_rejected|left to right"


@pytest.fixture
def manifest() -> dict:
    domains = {
        f"{d}|{light}": {"cars": 0, "cars_with_snaps": 0, "best": 0, "gated": 0}
        for d in DIRECTIONS
        for light in LIGHT_BANDS
    }
    domains[RL_DAY] = {"cars": 200, "cars_with_snaps": 190, "best": 100, "gated": 80}
    dvsa = {"support_band": "5+", "plate": "AB12CDE"}
    items = [
        _item("t1", "best", PASS, 0.95, True),
        _item("t2", "best", PASS, 0.97, True),
        _item("t3", "best", PASS, 0.85, True),
        _item("t4", "best", PASS, 0.99, True),
        _item("t5", "best", PASS, 0.99, True),  # drawn, not labelled yet
        _item("t6", "best", FAIL, 0.60, False),
        _item("t7", "best", FAIL, 0.70, False),
        _item("t8", "colour_rejected", COLOUR, 0.95, True, direction="left to right"),
        _item("t9", "colour_rejected", COLOUR, 0.90, False, direction="left to right"),
        _item("t10", "dvsa", "dvsa|5+", 0.95, True, dvsa=dvsa),
        _item("t11", "dvsa", "dvsa|5+", 0.95, True, dvsa=dvsa),
    ]
    return {
        "items": items,
        "strata": {
            PASS: {"population": 80, "sampled": 5},
            FAIL: {"population": 20, "sampled": 2},
            COLOUR: {"population": 10, "sampled": 2},
            "dvsa|5+": {"population": 50, "sampled": 2},
            "dvsa|1": {"population": 5, "sampled": 0},
        },
        "domains": domains,
    }


def _label(attribution: str, text: str | None, **dvsa: str) -> dict:
    out: dict = {"attribution": attribution, "text": text, "text_first": text, "reconciled": None}
    if dvsa:
        out["dvsa"] = dvsa
    return out


@pytest.fixture
def labels() -> dict:
    yes = {"make": "yes", "colour": "yes", "body": "yes"}
    return {
        "t1": _label("tracked", "AB12CDE"),
        "t2": _label("tracked", "AB12CDE"),
        "t3": _label("tracked", "AB12CDE"),
        "t4": _label("parked", "AB12CDE"),
        "t6": _label("tracked", "AB12CDE"),
        "t7": _label("tracked", "AB12CDF"),
        "t8": _label("other_moving", "AB12CDE"),
        "t9": _label("tracked", "AB12CDE"),
        "t10": _label("tracked", "AB12CDE", **yes),
        "t11": _label("tracked", "AB12CDE", **{**yes, "colour": "no"}),
    }


def test_report_coverage(manifest: dict, labels: dict) -> None:
    coverage = build_report(manifest, labels, n_boot=200)["coverage"]
    assert (coverage["items"], coverage["labelled"]) == (11, 10)
    assert coverage["by_sample"]["best"] == {"items": 7, "labelled": 6}
    assert coverage["unlabelled_strata"] == ["dvsa|1"]


def test_report_read_rates(manifest: dict, labels: dict) -> None:
    reads = build_report(manifest, labels, n_boot=200)["reads"]
    # Gate-passing reads stand for 20 each (80 / 4 labelled), failing ones 10.
    assert reads["overall"]["verified"]["value"] == pytest.approx((3 * 20 + 10) / 100)
    assert reads["overall"]["on_tracked_car"]["value"] == pytest.approx(80 / 100)
    assert reads["overall"]["text_right"]["value"] == pytest.approx(90 / 100)
    assert reads["by_gate"]["pass"]["verified"]["value"] == pytest.approx(0.75)
    assert reads["by_gate"]["fail"]["verified"]["value"] == pytest.approx(0.5)
    assert reads["by_direction"]["left to right"]["verified"]["n"] == 0


def test_report_confidence_table(manifest: dict, labels: dict) -> None:
    table = {
        row["threshold"]: row
        for row in build_report(manifest, labels, n_boot=200)["reads"]["confidence"]
    }
    assert table[0.0]["kept_share"] == 1.0 and table[0.0]["recall"] == 1.0
    # At 0.90: t1, t2 and t4 stay (60 of 100 reads); 60 of the 90 rightly read.
    assert table[0.9]["kept_share"] == pytest.approx(0.6)
    assert table[0.9]["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert table[0.9]["text_right"]["value"] == 1.0
    assert table[0.9]["verified"]["value"] == pytest.approx(40 / 60, abs=1e-4)


def test_report_per_car_rate_is_shape_rate_times_precision(manifest: dict, labels: dict) -> None:
    per_car = build_report(manifest, labels, n_boot=200)["reads"]["per_car"]
    row = per_car[RL_DAY]
    assert (row["shape_rate"], row["gated_rate"]) == (0.5, 0.4)
    assert row["verified_rate"]["value"] == pytest.approx(0.5 * 0.7)
    assert row["verified_gated_rate"]["value"] == pytest.approx(0.4 * 0.75)
    assert per_car["all"]["cars"] == 200
    assert per_car["all"]["verified_rate"]["value"] == pytest.approx(0.35)
    assert per_car["left to right"]["cars"] == 0


def test_report_filters_and_dvsa(manifest: dict, labels: dict) -> None:
    report = build_report(manifest, labels, n_boot=200)
    colour = report["colour_check"]
    assert colour["rightly_rejected"]["value"] == 0.5
    assert colour["good_read_lost"]["value"] == 0.5
    assert colour["good_read_lost_that_passed_the_gate"]["n"] == 1
    assert report["static_filter"]["n"] == 0

    dvsa = report["dvsa"]
    assert dvsa["overall"]["any_wrong"]["value"] == 0.5
    assert dvsa["overall"]["colour_wrong"]["value"] == 0.5
    assert dvsa["overall"]["make_wrong"]["value"] == 0.0
    assert dvsa["overall"]["plate_wrong"]["value"] == 0.0
    assert dvsa["by_support"]["1"]["any_wrong"]["n"] == 0


def test_format_report(manifest: dict, labels: dict) -> None:
    text = format_report(build_report(manifest, labels, n_boot=200))
    assert text.startswith("Audit set: 10 of 11 items labelled")
    assert "dvsa|1" in text  # named as missing from the rates
    assert "right to left, day" in text
    assert " 70.0 %" in text


def test_report_with_nothing_labelled(manifest: dict) -> None:
    report = build_report(manifest, {}, n_boot=50)
    assert report["reads"]["overall"]["verified"]["value"] is None
    assert "Audit set: 0 of 11" in format_report(report)


# ----------------------------------------------------------------------
# The set as a regression check


def test_rescore_compares_the_best_reads_then_and_now() -> None:
    def item(key: str, tid: int, then: str | None) -> dict:
        best = {"ocr_text": then, "canonical_uk_shape": True} if then else None
        return {"key": key, "session": "s", "track_id": tid, "track_best": best}

    manifest = {
        "items": [
            item("a", 1, "AB12CDE"),
            item("b", 2, "XY12ZZ2"),
            item("c", 3, "PA12KED"),
            item("d", 4, "AB12CDE"),
            item("e", 5, "AB12CDE"),
            item("f", 1, "AB12CDE"),  # another snap of track 1
        ]
    }
    labels = {
        "a": {"attribution": "tracked", "text": "AB12CDE"},
        "b": {"attribution": "tracked", "text": "XY12ZZZ"},
        "c": {"attribution": "parked", "text": "PA12KED"},
        "d": {"attribution": "tracked", "text": "AB1.CDE"},  # not fully read: no truth
        "e": {"attribution": "unsure", "text": "AB12CDE"},
        "f": {"attribution": "tracked", "text": "AB12CDE"},
    }
    now = {
        ("s", 1): {"ocr_text": "AB12CDE", "canonical_uk_shape": True},
        ("s", 2): {"ocr_text": "XY12ZZZ", "canonical_uk_shape": True},
        ("s", 3): None,
    }
    got = rescore(manifest, labels, now)
    assert got["known"]["n"] == 2
    assert got["known"]["then"] == {"same": 1, "different": 1}
    assert got["known"]["now"] == {"same": 2}
    assert got["known"]["changed"] == [{"key": "b", "then": "different", "now": "same"}]
    assert got["misattributed"] == {
        "n": 1,
        "then": {"same": 1},
        "now": {"none": 1},
        "changed": [{"key": "c", "then": "same", "now": "none"}],
    }
