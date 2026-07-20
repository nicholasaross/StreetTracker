"""Coverage for ``streettracker.analysis.identity``.

The matching logic is pure and is tested with synthetic embeddings, so
the suite needs neither the ``identity`` extra nor a GPU. What matters
most here is the *refusal* behaviour: this is the one module that names
individuals, so "unknown" must win whenever the evidence is weak or
ambiguous. A wrong name is worse than no name.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from streettracker.analysis.identity import (
    UNKNOWN,
    DetectedFace,
    FaceMatch,
    Gallery,
    identify_track,
)


def _vec(*values: float) -> np.ndarray:
    """A unit-norm embedding in as many dims as given."""
    v = np.asarray(values, dtype="float32")
    return v / np.linalg.norm(v)


# Three mutually near-orthogonal identities.
ALICE = _vec(1.0, 0.0, 0.0)
BOB = _vec(0.0, 1.0, 0.0)
CARA = _vec(0.0, 0.0, 1.0)


def _gallery() -> Gallery:
    g = Gallery()
    g.add("alice", ALICE)
    g.add("bob", BOB)
    return g


def _face(embedding: np.ndarray, width: int = 150, snap: int = 1) -> DetectedFace:
    return DetectedFace(embedding=embedding, width_px=width, det_score=0.9, snap_index=snap)


# ----------------------------------------------------------------------
# Gallery.match


def test_match_finds_the_enrolled_person() -> None:
    label, sim, _, _ = _gallery().match(ALICE)
    assert label == "alice"
    assert sim == pytest.approx(1.0)


def test_match_returns_unknown_below_threshold() -> None:
    """An unenrolled person must not be forced onto the nearest label."""
    label, _, _, _ = _gallery().match(CARA)
    assert label == UNKNOWN


def test_match_returns_unknown_when_two_people_are_too_close() -> None:
    """Two enrolled people equidistant from the probe is ambiguous --
    refuse rather than pick the marginally closer one."""
    g = _gallery()
    between = _vec(1.0, 1.0, 0.0)  # equidistant from alice and bob
    label, _, runner, _ = g.match(between, threshold=0.5, margin=0.05)
    assert label == UNKNOWN
    assert runner in ("alice", "bob")


def test_match_accepts_when_the_margin_is_clear() -> None:
    g = _gallery()
    leaning = _vec(1.0, 0.25, 0.0)
    label, _, _, _ = g.match(leaning, threshold=0.5, margin=0.05)
    assert label == "alice"


def test_match_on_an_empty_gallery_is_unknown() -> None:
    label, sim, _, _ = Gallery().match(ALICE)
    assert label == UNKNOWN
    assert sim == 0.0


# ----------------------------------------------------------------------
# identify_track


def test_identify_track_without_imagery_reports_why() -> None:
    """Door tracks recorded before door-snapping have no 4K frames; the
    verdict must say so rather than look like a failed match."""
    m = identify_track([], _gallery(), track_id=7)
    assert m.label == UNKNOWN
    assert m.reason == "no_4k_imagery"


def test_identify_track_reports_an_empty_gallery() -> None:
    m = identify_track([_face(ALICE)], Gallery(), track_id=7)
    assert m.label == UNKNOWN
    assert m.reason == "empty_gallery"


def test_identify_track_takes_the_best_view() -> None:
    """A subject looking away in one snap is still identified from a
    better one -- the strongest match across the track wins."""
    faces = [_face(CARA, snap=1), _face(ALICE, snap=2)]
    m = identify_track(faces, _gallery(), track_id=7)
    assert m.label == "alice"
    assert m.snap_index == 2


def test_identify_track_stays_unknown_for_a_stranger() -> None:
    m = identify_track([_face(CARA)], _gallery(), track_id=7)
    assert m.label == UNKNOWN
    assert m.reason == "below_threshold_or_ambiguous"


# ----------------------------------------------------------------------
# Gallery persistence


def test_gallery_round_trips_through_disk(tmp_path) -> None:
    g = _gallery()
    g.save(tmp_path)
    loaded = Gallery.load(tmp_path)
    assert loaded.labels == ["alice", "bob"]
    label, _, _, _ = loaded.match(ALICE)
    assert label == "alice"


def test_gallery_load_is_forgiving_of_a_missing_or_corrupt_file(tmp_path) -> None:
    assert Gallery.load(tmp_path).labels == []
    (tmp_path / "gallery.json").write_text("{not json", encoding="utf-8")
    assert Gallery.load(tmp_path).labels == []


def test_gallery_save_is_atomic(tmp_path) -> None:
    """A half-written gallery would silently lose enrolments."""
    g = _gallery()
    g.save(tmp_path)
    doc = json.loads((tmp_path / "gallery.json").read_text(encoding="utf-8"))
    assert set(doc) == {"alice", "bob"}
    assert not list(tmp_path.glob("*.tmp"))


def test_face_match_serialises_for_the_session_document() -> None:
    d = FaceMatch(track_id=3, label="alice", similarity=0.8).to_json_dict()
    assert d["track_id"] == 3
    assert d["label"] == "alice"
