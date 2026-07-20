"""Consent-gated enrolment: a review queue for unrecognised door faces.

The operator's model (2026-07-20): people who reach the door and aren't
in the gallery get clustered into candidate identities; the operator
then either **enrols** them by name -- having asked, for a visiting
friend -- or **declines** them, as for a delivery driver. Nobody is
enrolled automatically.

Three stores, deliberately separate files:

``gallery.json``   enrolled, consented people. Only these are ever named
                   in output.
``pending.json``   candidate clusters awaiting review. Faces of people
                   who have not consented to anything, so this holds the
                   minimum needed to review: one embedding per sighting,
                   a representative crop path, and counts.
``declined.json``  clusters the operator has rejected.

**The decline list matches but never names.** It exists solely to stop
the same driver reappearing in the queue every week, so
:func:`identify_track`-level output for a declined face stays
``unknown``. Keeping it in its own file (rather than as a gallery label)
makes that structural: nothing reads ``declined.json`` when resolving a
name, so a declined face cannot be promoted into an identity by
accident. Turning one into an identity requires a deliberate enrol.

Retention is the operator's call and is deliberately explicit here
rather than silent: declined entries are kept permanently and pending
clusters do not expire (chosen 2026-07-20). Both stores are therefore
unbounded by design; ``purge_pending`` and ``forget`` are the manual
paths back. Everything here is gitignored and dev-box only -- see
``analysis/identity.py`` for the full handling rules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from streettracker.analysis.identity import DEFAULT_GALLERY_PATH, Gallery, _cosine

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Cosine similarity at which two face embeddings are treated as the same
# person when clustering unknowns. Higher than the gallery match
# threshold (0.45) on purpose: an over-merged cluster would invite the
# operator to enrol two different people under one name, which is the
# worst outcome here. Splitting one person into two clusters is merely
# annoying -- they get reviewed twice.
CLUSTER_THRESHOLD = 0.55

PENDING_FILE = "pending.json"
DECLINED_FILE = "declined.json"


@dataclass(slots=True)
class FaceCluster:
    """A candidate identity: one person's faces, grouped."""

    cluster_id: str
    embeddings: list[list[float]] = field(default_factory=list)
    # Where each sighting came from, for the operator's review.
    sightings: list[dict[str, Any]] = field(default_factory=list)
    representative_crop: str | None = None

    @property
    def n_sightings(self) -> int:
        return len(self.sightings)

    @property
    def first_seen(self) -> str:
        return min((s.get("time_start", "") for s in self.sightings), default="")

    @property
    def last_seen(self) -> str:
        return max((s.get("time_start", "") for s in self.sightings), default="")

    def similarity(self, embedding: np.ndarray) -> float:
        """Best similarity between ``embedding`` and this cluster."""
        if not self.embeddings:
            return 0.0
        return max(_cosine(embedding, vec) for vec in self.embeddings)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> FaceCluster:
        return cls(
            cluster_id=str(d.get("cluster_id", "")),
            embeddings=[[float(v) for v in vec] for vec in d.get("embeddings", [])],
            sightings=list(d.get("sightings", [])),
            representative_crop=d.get("representative_crop"),
        )


@dataclass(slots=True)
class ClusterStore:
    """A named collection of :class:`FaceCluster`, persisted as JSON.

    Backs both the pending queue and the decline list -- same shape,
    different meaning and (critically) different files.
    """

    filename: str
    clusters: dict[str, FaceCluster] = field(default_factory=dict)

    @classmethod
    def load(cls, filename: str, path: Path = DEFAULT_GALLERY_PATH) -> ClusterStore:
        store = cls(filename=filename)
        doc_path = path / filename
        if not doc_path.exists():
            return store
        try:
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return store
        if not isinstance(doc, dict):
            return store
        for cid, raw in doc.items():
            if isinstance(raw, dict):
                store.clusters[str(cid)] = FaceCluster.from_json_dict({**raw, "cluster_id": cid})
        return store

    def save(self, path: Path = DEFAULT_GALLERY_PATH) -> None:
        path.mkdir(parents=True, exist_ok=True)
        doc_path = path / self.filename
        tmp = doc_path.with_suffix(".tmp")
        payload = {cid: c.to_json_dict() for cid, c in self.clusters.items()}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(doc_path)

    def best_match(self, embedding: np.ndarray) -> tuple[FaceCluster | None, float]:
        best: FaceCluster | None = None
        best_sim = 0.0
        for cluster in self.clusters.values():
            sim = cluster.similarity(embedding)
            if sim > best_sim:
                best, best_sim = cluster, sim
        return best, best_sim

    def matches(self, embedding: np.ndarray, *, threshold: float = CLUSTER_THRESHOLD) -> bool:
        """Whether this embedding belongs to any cluster in the store."""
        _, sim = self.best_match(embedding)
        return sim >= threshold

    def add_sighting(
        self,
        embedding: np.ndarray,
        sighting: dict[str, Any],
        *,
        threshold: float = CLUSTER_THRESHOLD,
        crop_path: str | None = None,
    ) -> FaceCluster:
        """Fold one face into the store, merging into an existing cluster
        when it's the same person and starting a new one otherwise."""
        cluster, sim = self.best_match(embedding)
        if cluster is None or sim < threshold:
            cluster = FaceCluster(cluster_id=self._next_id(), representative_crop=crop_path)
            self.clusters[cluster.cluster_id] = cluster
        cluster.embeddings.append([float(v) for v in embedding])
        cluster.sightings.append(sighting)
        if cluster.representative_crop is None:
            cluster.representative_crop = crop_path
        return cluster

    def _next_id(self) -> str:
        n = 1
        while f"c{n:04d}" in self.clusters:
            n += 1
        return f"c{n:04d}"

    def pop(self, cluster_id: str) -> FaceCluster | None:
        return self.clusters.pop(cluster_id, None)


def enrol_cluster(
    cluster: FaceCluster,
    label: str,
    gallery: Gallery,
) -> int:
    """Promote a reviewed cluster into the gallery under ``label``.

    The deliberate act that turns a candidate into an identity. Returns
    how many reference embeddings were added.
    """
    for vec in cluster.embeddings:
        gallery.add(label, vec)
    return len(cluster.embeddings)


def forget(
    label: str,
    *,
    path: Path = DEFAULT_GALLERY_PATH,
    session_dirs: list[Path] | None = None,
) -> dict[str, int]:
    """Erase one enrolled person: gallery entry, and every mention of
    them in per-session identity output.

    This is what makes consent-based enrolment meaningful -- someone who
    asks to be removed has to actually be removable, including from the
    results already written. Returns a per-artefact count of what was
    changed.
    """
    removed = {"gallery_embeddings": 0, "identity_rows": 0, "sessions": 0}

    gallery = Gallery.load(path)
    if label in gallery.embeddings:
        removed["gallery_embeddings"] = len(gallery.embeddings.pop(label))
        gallery.save(path)

    for session_dir in session_dirs or []:
        doc_path = session_dir / f"{session_dir.name}_identity.json"
        if not doc_path.exists():
            continue
        try:
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        changed = 0
        for row in doc.get("tracks", []):
            if row.get("label") == label:
                row["label"] = "unknown"
                row["similarity"] = 0.0
                row["reason"] = "forgotten"
                changed += 1
            if row.get("runner_up") == label:
                row["runner_up"] = None
                row["runner_up_similarity"] = 0.0
        if changed:
            doc["summary"] = _recount(doc.get("tracks", []))
            doc["gallery_labels"] = [x for x in doc.get("gallery_labels", []) if x != label]
            doc_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            removed["identity_rows"] += changed
            removed["sessions"] += 1
    return removed


def _recount(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("label", "unknown"))
        out[key] = out.get(key, 0) + 1
    return out


# ----------------------------------------------------------------------
# CLI: review the queue, enrol or decline, and forget an enrolled person.


def _print_queue(pending: ClusterStore) -> None:
    if not pending.clusters:
        print("[identity] review queue is empty.")
        return
    print(f"[identity] {len(pending.clusters)} cluster(s) awaiting review:\n")
    order = sorted(pending.clusters.values(), key=lambda c: -c.n_sightings)
    for c in order:
        print(f"  {c.cluster_id}  sightings={c.n_sightings:<4d} first={c.first_seen[:19]}")
        print(f"           last={c.last_seen[:19]}")
        if c.representative_crop:
            print(f"           crop: {c.representative_crop}")
    print(
        "\n  enrol:   streettracker identity-review --enrol <cluster-id> --label <name>\n"
        "  decline: streettracker identity-review --decline <cluster-id>\n"
        "\n  Enrol only with that person's consent. Declined faces are kept to stop\n"
        "  them reappearing here, and are never named in output."
    )


def review_main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="streettracker identity-review",
        description="Review unrecognised door faces: enrol with consent, or decline.",
    )
    ap.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    ap.add_argument("--enrol", metavar="CLUSTER_ID", help="promote a cluster into the gallery")
    ap.add_argument("--label", help="person's name, required with --enrol")
    ap.add_argument("--decline", metavar="CLUSTER_ID", help="reject a cluster (never named)")
    ap.add_argument(
        "--purge-queue",
        action="store_true",
        help="delete every unreviewed cluster (the queue does not expire on its own)",
    )
    args = ap.parse_args(argv)

    pending = ClusterStore.load(PENDING_FILE, args.gallery)

    if args.purge_queue:
        n = len(pending.clusters)
        pending.clusters.clear()
        pending.save(args.gallery)
        print(f"[identity] purged {n} unreviewed cluster(s)")
        return 0

    if args.enrol:
        if not args.label:
            print("--enrol needs --label <name>")
            return 2
        cluster = pending.pop(args.enrol)
        if cluster is None:
            print(f"no pending cluster {args.enrol!r}")
            return 1
        gallery = Gallery.load(args.gallery)
        n = enrol_cluster(cluster, args.label, gallery)
        gallery.save(args.gallery)
        pending.save(args.gallery)
        print(f"[identity] enrolled cluster {args.enrol} as {args.label!r} ({n} reference faces)")
        print(f"  gallery now: {', '.join(gallery.labels)}")
        return 0

    if args.decline:
        cluster = pending.pop(args.decline)
        if cluster is None:
            print(f"no pending cluster {args.decline!r}")
            return 1
        declined = ClusterStore.load(DECLINED_FILE, args.gallery)
        declined.clusters[cluster.cluster_id] = cluster
        declined.save(args.gallery)
        pending.save(args.gallery)
        print(f"[identity] declined cluster {args.decline}")
        print("  kept only to suppress future prompts; never named in output.")
        return 0

    _print_queue(pending)
    return 0


def forget_main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="streettracker identity-forget",
        description="Erase an enrolled person from the gallery and all identity output.",
    )
    ap.add_argument("label", help="person to erase")
    ap.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="scan this root's sessions and rewrite their identity output",
    )
    args = ap.parse_args(argv)

    sessions = [p for p in args.output_root.glob("session_*") if p.is_dir()]
    removed = forget(args.label, path=args.gallery, session_dirs=sessions)
    print(f"[identity] forgot {args.label!r}:")
    print(f"  gallery reference faces removed: {removed['gallery_embeddings']}")
    print(f"  identity rows rewritten to unknown: {removed['identity_rows']}")
    print(f"  sessions updated: {removed['sessions']}")
    print("  NOTE: reference crops you enrolled from are NOT deleted -- remove those by hand.")
    return 0
