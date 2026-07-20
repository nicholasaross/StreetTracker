"""Coverage for ``streettracker.analysis.identity_queue``.

The consent machinery. Two properties matter more than anything else
here and are tested directly:

* a **declined** face is never named -- it lives in its own store, which
  nothing consults when resolving a label;
* **forget** actually erases, including from results already written,
  because consent that can't be withdrawn isn't consent.
"""

from __future__ import annotations

import json

import numpy as np

from streettracker.analysis.identity import Gallery
from streettracker.analysis.identity_queue import (
    DECLINED_FILE,
    PENDING_FILE,
    ClusterStore,
    FaceCluster,
    enrol_cluster,
    forget,
)


def _vec(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype="float32")
    return v / np.linalg.norm(v)


ALICE = _vec(1.0, 0.0, 0.0)
ALICE_AGAIN = _vec(0.97, 0.24, 0.0)  # same person, slightly different view
DRIVER = _vec(0.0, 1.0, 0.0)


def _sighting(track_id: int = 1, time_start: str = "2026-07-20T15:50:00+01:00") -> dict:
    return {"track_id": track_id, "session": "session_x", "time_start": time_start}


# ----------------------------------------------------------------------
# Clustering


def test_repeat_visitor_folds_into_one_cluster() -> None:
    """A driver who calls weekly should be ONE cluster to review, not one
    per visit."""
    store = ClusterStore(filename=PENDING_FILE)
    first = store.add_sighting(DRIVER, _sighting(1))
    second = store.add_sighting(DRIVER, _sighting(2))
    assert first.cluster_id == second.cluster_id
    assert len(store.clusters) == 1
    assert first.n_sightings == 2


def test_different_people_get_separate_clusters() -> None:
    store = ClusterStore(filename=PENDING_FILE)
    store.add_sighting(ALICE, _sighting(1))
    store.add_sighting(DRIVER, _sighting(2))
    assert len(store.clusters) == 2


def test_clustering_tolerates_a_different_view_of_one_person() -> None:
    store = ClusterStore(filename=PENDING_FILE)
    store.add_sighting(ALICE, _sighting(1))
    store.add_sighting(ALICE_AGAIN, _sighting(2))
    assert len(store.clusters) == 1


def test_cluster_tracks_first_and_last_seen() -> None:
    store = ClusterStore(filename=PENDING_FILE)
    store.add_sighting(DRIVER, _sighting(1, "2026-07-01T09:00:00+01:00"))
    store.add_sighting(DRIVER, _sighting(2, "2026-07-15T09:00:00+01:00"))
    cluster = next(iter(store.clusters.values()))
    assert cluster.first_seen.startswith("2026-07-01")
    assert cluster.last_seen.startswith("2026-07-15")


# ----------------------------------------------------------------------
# Decline: suppress the prompt, never name


def test_declined_face_is_recognised_for_suppression() -> None:
    declined = ClusterStore(filename=DECLINED_FILE)
    declined.add_sighting(DRIVER, _sighting(1))
    assert declined.matches(DRIVER)
    assert not declined.matches(ALICE)


def test_declined_store_is_separate_from_the_gallery() -> None:
    """Structural guarantee: a declined person holds no gallery label, so
    no code path can resolve them to a name."""
    gallery = Gallery()
    declined = ClusterStore(filename=DECLINED_FILE)
    declined.add_sighting(DRIVER, _sighting(1))
    assert gallery.labels == []
    label, _, _, _ = gallery.match(DRIVER)
    assert label == "unknown"


def test_declining_then_enrolling_is_a_deliberate_act() -> None:
    """A declined cluster only becomes an identity if someone explicitly
    enrols it -- there is no automatic promotion."""
    declined = ClusterStore(filename=DECLINED_FILE)
    cluster = declined.add_sighting(DRIVER, _sighting(1))
    gallery = Gallery()
    enrol_cluster(cluster, "driver_dave", gallery)
    assert gallery.labels == ["driver_dave"]


# ----------------------------------------------------------------------
# Enrolment


def test_enrol_cluster_adds_every_reference_face(tmp_path) -> None:
    store = ClusterStore(filename=PENDING_FILE)
    store.add_sighting(ALICE, _sighting(1))
    store.add_sighting(ALICE_AGAIN, _sighting(2))
    cluster = next(iter(store.clusters.values()))
    gallery = Gallery()
    n = enrol_cluster(cluster, "alys", gallery)
    assert n == 2
    label, _, _, _ = gallery.match(ALICE)
    assert label == "alys"


def test_stores_round_trip_through_disk(tmp_path) -> None:
    store = ClusterStore(filename=PENDING_FILE)
    store.add_sighting(ALICE, _sighting(1), crop_path="output/s/person_1_main_1.jpg")
    store.save(tmp_path)
    loaded = ClusterStore.load(PENDING_FILE, tmp_path)
    assert len(loaded.clusters) == 1
    cluster = next(iter(loaded.clusters.values()))
    assert cluster.representative_crop == "output/s/person_1_main_1.jpg"
    assert loaded.matches(ALICE)


# ----------------------------------------------------------------------
# Forget -- the reason consent-based enrolment is meaningful


def test_forget_removes_the_person_from_the_gallery(tmp_path) -> None:
    gallery = Gallery()
    gallery.add("alys", ALICE)
    gallery.add("me", DRIVER)
    gallery.save(tmp_path)

    removed = forget("alys", path=tmp_path)
    assert removed["gallery_embeddings"] == 1
    assert Gallery.load(tmp_path).labels == ["me"]


def test_forget_rewrites_results_already_written(tmp_path) -> None:
    """Erasing someone has to reach the output too, or their identity
    survives in every session document."""
    gallery = Gallery()
    gallery.add("alys", ALICE)
    gallery.save(tmp_path)

    session = tmp_path / "session_20260720_144136"
    session.mkdir()
    (session / "session_20260720_144136_identity.json").write_text(
        json.dumps(
            {
                "gallery_labels": ["alys", "me"],
                "summary": {"alys": 1, "unknown": 1},
                "tracks": [
                    {"track_id": 1, "label": "alys", "similarity": 0.9, "runner_up": None},
                    {"track_id": 2, "label": "unknown", "similarity": 0.1, "runner_up": "alys"},
                ],
            }
        ),
        encoding="utf-8",
    )

    removed = forget("alys", path=tmp_path, session_dirs=[session])
    assert removed["identity_rows"] == 1
    assert removed["sessions"] == 1

    doc = json.loads((session / "session_20260720_144136_identity.json").read_text())
    assert [t["label"] for t in doc["tracks"]] == ["unknown", "unknown"]
    assert doc["tracks"][0]["reason"] == "forgotten"
    # The runner-up mention is scrubbed too -- otherwise the name leaks.
    assert doc["tracks"][1]["runner_up"] is None
    assert "alys" not in doc["gallery_labels"]
    assert doc["summary"] == {"unknown": 2}


def test_forget_is_a_no_op_for_an_unknown_label(tmp_path) -> None:
    removed = forget("nobody", path=tmp_path)
    assert removed["gallery_embeddings"] == 0


def test_pop_removes_a_cluster_from_the_queue() -> None:
    store = ClusterStore(filename=PENDING_FILE)
    cluster = store.add_sighting(ALICE, _sighting(1))
    assert store.pop(cluster.cluster_id) is cluster
    assert store.clusters == {}
    assert store.pop("nope") is None


def test_face_cluster_survives_a_json_round_trip() -> None:
    c = FaceCluster(cluster_id="c0001", embeddings=[[1.0, 0.0, 0.0]], sightings=[_sighting()])
    again = FaceCluster.from_json_dict(c.to_json_dict())
    assert again.cluster_id == "c0001"
    assert again.n_sightings == 1
