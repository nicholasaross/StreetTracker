"""build_showcase() against synthetic multi-session output roots.

Covers cross-session pooling by canonical plate, repeat-``kind``
classification, fuzzy merge of a 1-char OCR diff across sessions, the DVSA
"extras" join (colour/fuel/engine), unread-car exclusion, and ordering.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from streettracker.common.schema import TrackRecord
from streettracker.web.aggregate import (
    _apply_operator_merges,
    _new_pool_rec,
    build_showcase,
    discover_sessions,
)

_FAKE_JPEG = b"\xff\xd8\xff\xe0X\xff\xd9"


def _track(
    tid: int,
    *,
    date: str = "2026-05-26",
    hhmmss: str = "13:00:00",
    direction: str = "left to right",
    color: str = "white",
) -> TrackRecord:
    start = f"{date}T{hhmmss}+01:00"
    unix = 1_700_000_000.0 + tid * 1000
    return TrackRecord(
        track_id=tid,
        class_id=2,
        class_name="car",
        time_start=start,
        time_end=start,
        time_start_unix=unix,
        time_end_unix=unix + 6.0,
        time_start_s=0.0,
        time_end_s=6.0,
        duration_visible=6.0,
        direction=direction,
        speed_px_s=100.0,
        color=color,
        lane="middle",
        avg_confidence=0.9,
        displacement_px=500.0,
        net_displacement_px=480.0,
        num_detections=30,
        asset_prefix="vehicle",
        main_snaps=[1],
    )


def _alpr(*reads: tuple[int, str, float]) -> dict:
    """reads = (track_id, plate, ocr_conf) tuples."""
    return {
        "tracks": [
            {
                "track_id": tid,
                "best_preferred": {
                    "track_id": tid,
                    "snap_index": 1,
                    "image": f"vehicle_{tid}_main_1.jpg",
                    "ocr_text": plate,
                    "ocr_conf": conf,
                    "det_conf": 0.85,
                },
            }
            for tid, plate, conf in reads
        ]
    }


def _dvsa(**by_plate: dict) -> dict:
    return {"labels": by_plate}


def _write_session(
    root: Path,
    name: str,
    tracks: list[TrackRecord],
    alpr: dict,
    dvsa: dict | None = None,
    *,
    crops: bool = True,
) -> Path:
    d = root / name
    d.mkdir()
    (d / f"{name}_data.json").write_text(json.dumps([asdict(t) for t in tracks]))
    (d / f"{name}_alpr_by_track.json").write_text(json.dumps(alpr))
    if dvsa is not None:
        (d / f"{name}_dvsa_labels.json").write_text(json.dumps(dvsa))
    # Snap files on disk: the 4K _main_N always, the _hq/tile crops only when
    # ``crops`` (a --only-main pull ships the main snaps but no small crops).
    for t in tracks:
        for n in t.main_snaps or []:
            (d / f"vehicle_{t.track_id}_main_{n}.jpg").write_bytes(_FAKE_JPEG)
        if crops:
            (d / f"vehicle_{t.track_id}.jpg").write_bytes(_FAKE_JPEG)
            (d / f"vehicle_{t.track_id}_hq.jpg").write_bytes(_FAKE_JPEG)
    return d


def test_single_plated_car_with_dvsa(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1)],
        _alpr((1, "AB12CDE", 0.97)),
        _dvsa(
            AB12CDE={
                "make": "FORD",
                "model": "FOCUS",
                "year": 2018,
                "primary_colour": "Blue",
                "fuel_type": "Petrol",
                "engine_size_cc": 1499,
            }
        ),
    )
    cars = build_showcase(tmp_path)
    assert len(cars) == 1
    c = cars[0]
    assert c.plate == "AB12CDE"
    assert (c.make, c.model, c.year) == ("FORD", "FOCUS", 2018)
    assert c.make_model_source == "dvsa"
    assert (c.primary_colour, c.fuel_type, c.engine_size_cc) == ("Blue", "Petrol", 1499)
    assert c.kind == "one-off"
    assert c.n_visits == 1 and c.n_sessions == 1 and c.n_dates == 1
    # Image URLs point back through the /images/<session>/ route. With all
    # three tiers on disk, thumb = smallest (tile), full = largest (4K main).
    im = c.images[0]
    assert im.thumb == "/images/session_20260526_120000/vehicle_1.jpg"
    assert im.full == "/images/session_20260526_120000/vehicle_1_main_1.jpg"
    assert im.thumb_small is True
    assert c.thumb == "/images/session_20260526_120000/vehicle_1.jpg"


def test_cnn_colour_from_sidecar(tmp_path: Path) -> None:
    """build_showcase pools the confident colour-CNN prediction per car from
    the ``_colour_by_track.json`` sidecar (majority across its tracks)."""
    d = _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1)],
        _alpr((1, "AB12CDE", 0.97)),
        _dvsa(AB12CDE={"make": "FORD", "model": "FOCUS", "primary_colour": "Blue"}),
    )
    (d / "session_20260526_120000_colour_by_track.json").write_text(
        json.dumps({"tracks": [{"track_id": 1, "colour": "silver", "conf": 0.9}]})
    )
    c = build_showcase(tmp_path)[0]
    assert c.cnn_colour == "silver"


def test_cnn_colour_none_without_sidecar(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1)],
        _alpr((1, "AB12CDE", 0.97)),
        _dvsa(AB12CDE={"make": "FORD", "model": "FOCUS", "primary_colour": "Blue"}),
    )
    assert build_showcase(tmp_path)[0].cnn_colour is None


def test_only_main_session_thumb_falls_back_to_4k(tmp_path: Path) -> None:
    # --only-main pull: 4K _main snap on disk, no _hq/tile crop. The thumb
    # must fall back to the (existing) 4K snap rather than a missing crop URL.
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1)],
        _alpr((1, "AB12CDE", 0.97)),
        crops=False,
    )
    c = build_showcase(tmp_path)[0]
    im = c.images[0]
    assert im.thumb == "/images/session_20260526_120000/vehicle_1_main_1.jpg"
    assert im.full == "/images/session_20260526_120000/vehicle_1_main_1.jpg"
    assert im.thumb_small is False
    assert c.thumb == im.thumb  # no smaller option, so the card uses it


def test_card_thumb_prefers_small_crop_across_sightings(tmp_path: Path) -> None:
    # Earliest sighting is --only-main (no crop); a later sighting has a crop.
    # The card thumbnail should prefer the small crop over the 4K snap.
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1, date="2026-05-26")],
        _alpr((1, "AB12CDE", 0.97)),
        crops=False,
    )
    _write_session(
        tmp_path,
        "session_20260528_120000",
        [_track(2, date="2026-05-28")],
        _alpr((2, "AB12CDE", 0.97)),
        crops=True,
    )
    c = build_showcase(tmp_path)[0]
    assert c.thumb == "/images/session_20260528_120000/vehicle_2.jpg"


def test_unread_car_excluded(tmp_path: Path) -> None:
    # Low-conf read -> treated as unread -> not a showcase car (no identity).
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1)],
        _alpr((1, "AB12CDE", 0.40)),
    )
    assert build_showcase(tmp_path) == []


def test_regular_across_different_days(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1, date="2026-05-26", hhmmss="09:00:00")],
        _alpr((1, "AB12CDE", 0.97)),
    )
    _write_session(
        tmp_path,
        "session_20260528_120000",
        [_track(2, date="2026-05-28", hhmmss="17:30:00")],
        _alpr((2, "AB12CDE", 0.98)),
    )
    cars = build_showcase(tmp_path)
    assert len(cars) == 1
    c = cars[0]
    assert c.kind == "different-day"
    assert c.n_visits == 2 and c.n_sessions == 2 and c.n_dates == 2
    # Images sorted by time: the 26th comes before the 28th.
    assert [im.session for im in c.images] == [
        "session_20260526_120000",
        "session_20260528_120000",
    ]


def test_same_day_repeat(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "session_20260526_090000",
        [_track(1, date="2026-05-26", hhmmss="09:00:00")],
        _alpr((1, "AB12CDE", 0.97)),
    )
    _write_session(
        tmp_path,
        "session_20260526_180000",
        [_track(2, date="2026-05-26", hhmmss="18:00:00")],
        _alpr((2, "AB12CDE", 0.97)),
    )
    cars = build_showcase(tmp_path)
    assert len(cars) == 1
    assert cars[0].kind == "same-day"
    assert cars[0].n_dates == 1 and cars[0].n_visits == 2


def test_fuzzy_merge_one_char_diff_across_sessions(tmp_path: Path) -> None:
    # AB12CDE vs AB12CDF: 7-char, 1-char diff -> rapidfuzz ratio 85.7 >= 85.
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1, date="2026-05-26")],
        _alpr((1, "AB12CDE", 0.99)),
    )
    _write_session(
        tmp_path,
        "session_20260527_120000",
        [_track(2, date="2026-05-27")],
        _alpr((2, "AB12CDF", 0.95)),
    )
    cars = build_showcase(tmp_path)
    assert len(cars) == 1, "the 1-char OCR variant should merge into one car"
    c = cars[0]
    assert c.plate == "AB12CDE"  # highest-conf string is canonical
    assert "AB12CDF" in c.plate_variants
    assert c.n_visits == 2


def test_fuzzy_disabled_keeps_variants_separate(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1)],
        _alpr((1, "AB12CDE", 0.99)),
    )
    _write_session(
        tmp_path,
        "session_20260527_120000",
        [_track(2, date="2026-05-27")],
        _alpr((2, "AB12CDF", 0.95)),
    )
    cars = build_showcase(tmp_path, fuzzy_ratio=None)
    assert len(cars) == 2


def test_operator_merge_folds_ocr_splits_into_one_card(tmp_path: Path) -> None:
    # Two OCR misreads of one car (2-char apart, so fuzzy clustering leaves them
    # separate). An operator merge_map folds both into the true plate SK69NJZ.
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1), _track(2, date="2026-05-27")],
        _alpr((1, "SH69NJZ", 0.99), (2, "SH69NJZ", 0.97)),
    )
    _write_session(
        tmp_path,
        "session_20260528_120000",
        [_track(3, date="2026-05-28")],
        _alpr((3, "SK59NJZ", 0.95)),
    )
    # Without the merge: two distinct cards.
    assert len(build_showcase(tmp_path)) == 2

    cars = build_showcase(tmp_path, merge_map={"SH69NJZ": "SK69NJZ", "SK59NJZ": "SK69NJZ"})
    assert len(cars) == 1
    c = cars[0]
    assert c.plate == "SK69NJZ"
    assert c.n_visits == 3  # 2 + 1 sightings pooled
    assert set(c.plate_variants) >= {"SH69NJZ", "SK59NJZ"}


def test_card_make_comes_from_canonical_not_a_variant_dvsa_row(tmp_path: Path) -> None:
    # The canonical plate's DVSA row is a Renault; a merged 1-char misread
    # variant is a real, different VW Golf. The card must show the canonical's
    # make (Renault) -- consistent with its colour -- not the variant's Golf.
    _write_session(
        tmp_path,
        "session_20260101_120000",  # earlier name -> pooled first (the trap)
        [_track(1, date="2026-01-01")],
        _alpr((1, "AA11AAB", 0.99)),
        _dvsa(AA11AAB={"make": "VOLKSWAGEN", "model": "GOLF", "primary_colour": "Silver"}),
    )
    _write_session(
        tmp_path,
        "session_20260201_120000",
        [_track(2, date="2026-02-01"), _track(3, date="2026-02-02"), _track(4, date="2026-02-03")],
        _alpr((2, "AA11AAA", 0.99), (3, "AA11AAA", 0.99), (4, "AA11AAA", 0.99)),
        _dvsa(AA11AAA={"make": "RENAULT", "model": "CAPTUR", "primary_colour": "Blue"}),
    )
    cars = build_showcase(tmp_path)
    assert len(cars) == 1  # merged (frequency picks AA11AAA as canonical)
    c = cars[0]
    assert c.plate == "AA11AAA"
    assert (c.make, c.model) == ("RENAULT", "CAPTUR")
    assert c.primary_colour == "Blue"  # make + colour now agree
    assert c.make_model_source == "dvsa"


def test_operator_merge_set_on_a_variant_still_folds_the_card() -> None:
    # A merge set on any OCR variant of a card (not just its current canonical)
    # must still fold the whole card -- the canonical spelling can shift as read
    # frequencies change, orphaning a merge keyed on the old spelling (the
    # LA68CWT/LA68EWT case).
    pool: dict[str, dict] = {}
    src = pool.setdefault("LA68CWT", _new_pool_rec())
    src["plates"].update({"LA68CWT", "LA68EWT"})  # LA68EWT is a variant here
    src["sessions"].add("s1")
    tgt = pool.setdefault("LA68CWY", _new_pool_rec())
    tgt["plates"].add("LA68CWY")
    tgt["sessions"].add("s2")

    # merge_into was set on the VARIANT LA68EWT, not the canonical LA68CWT.
    _apply_operator_merges(pool, {"LA68EWT": "LA68CWY"})

    assert "LA68CWT" not in pool  # folded away
    assert set(pool) == {"LA68CWY"}
    assert pool["LA68CWY"]["plates"] >= {"LA68CWT", "LA68EWT", "LA68CWY"}


def test_regulars_sorted_first(tmp_path: Path) -> None:
    # A one-off and a different-day regular; the regular must lead.
    _write_session(
        tmp_path,
        "session_20260526_120000",
        [_track(1, date="2026-05-26"), _track(3, date="2026-05-26")],
        _alpr((1, "AB12CDE", 0.97), (3, "ZZ99ZZA", 0.97)),
    )
    _write_session(
        tmp_path,
        "session_20260528_120000",
        [_track(2, date="2026-05-28")],
        _alpr((2, "AB12CDE", 0.97)),
    )
    cars = build_showcase(tmp_path)
    assert cars[0].plate == "AB12CDE" and cars[0].kind == "different-day"
    assert cars[-1].kind == "one-off"


def test_discover_sessions_skips_dirs_without_data(tmp_path: Path) -> None:
    good = _write_session(
        tmp_path, "session_20260526_120000", [_track(1)], _alpr((1, "AB12CDE", 0.97))
    )
    (tmp_path / "session_inprogress").mkdir()  # no _data.json
    (tmp_path / "loose_file.txt").write_text("x")
    found = discover_sessions(tmp_path)
    assert found == [good]
