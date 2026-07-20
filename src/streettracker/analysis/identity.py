"""Per-person identity for door users -- face recognition against an
operator-enrolled gallery.

This is the only part of StreetTracker that identifies *individuals*.
Everything else on the person side is deliberately aggregate: footfall,
walk kinds, door-origin trips ("whose door", not "who"). This module
crosses that line on purpose, at the operator's explicit instruction
(2026-07-20), to answer "which household member took this trip".

Handling rules -- these are load-bearing, not advisory:

* **Dev-box only.** Never install the ``identity`` extra on the Orin.
  The always-on device stays free of biometric processing; it captures
  frames, and identity happens off-device on demand.
* **Never committed, never released.** The gallery, the embeddings and
  every ``*_identity.json`` are gitignored. The StreetTracker repo is
  PUBLIC and the disaster-recovery flow pushes small models and JSON to
  it -- face embeddings must never ride along. They are not recoverable
  from a public backup because they must not be in one.
* **Enrolment is deliberate.** A person enters the gallery only when the
  operator enrols reference crops for them by name. There is no
  automatic clustering of strangers into new identities; unmatched faces
  return ``unknown`` and are counted, not banked.
* **Scope.** The gallery covers the household plus named regulars the
  operator has chosen to enrol. Passers-by on the pavement are not
  enrolled and are not identified.

How it works. ``insightface``'s buffalo_l pack gives an SCRFD face
detector plus an ArcFace embedder; both are pretrained, so this is a
few-shot matcher rather than a model to train -- enrol 5-10 crops per
person and match by cosine similarity. That matters here because the
labelled examples number in the handfuls, which is nowhere near enough
to train anything.

Resolution is the binding constraint. A sub-stream tile/hq crop carries
a ~40-60 px oblique face, below ArcFace's useful floor; a 4K door snap
at 1-3 m gives ~130-200 px. So identity runs on ``*_main_*.jpg`` only,
which is why the runtime now fires door snaps
(``device/snap_planner.consider_door``). Tracks with no 4K snap cannot
be identified and report ``unknown`` with reason ``no_4k_imagery``.

Per track the module takes the best-scoring face across that track's
snaps, so a subject who looks down in one frame can still be identified
from another.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Cosine similarity above which a face is accepted as a gallery match.
# ArcFace on buffalo_l separates same/different identity around 0.35-0.4
# for good crops; 0.45 is deliberately conservative because a false
# "that was your daughter" is worse here than an honest "unknown".
DEFAULT_MATCH_THRESHOLD = 0.45

# A second-best match this close to the best one means the two enrolled
# people are not separable on this crop -- report unknown rather than
# guess between them.
DEFAULT_MARGIN = 0.05

# Below this face width (px in the 4K snap) an embedding is too noisy to
# trust. Measured: door-range faces run 130-200 px, pavement-approach
# faces 70-90 px, sub-stream crops 40-60 px.
MIN_FACE_PX = 60

DEFAULT_GALLERY_PATH = Path("configs/identities")

UNKNOWN = "unknown"


@dataclass(slots=True)
class FaceMatch:
    """One track's identity verdict."""

    track_id: int
    label: str = UNKNOWN
    similarity: float = 0.0
    runner_up: str | None = None
    runner_up_similarity: float = 0.0
    face_px: int = 0
    snap_index: int | None = None
    reason: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Gallery:
    """Enrolled reference embeddings, keyed by person label.

    Persisted as a single JSON file of ``{label: [[float, ...], ...]}``.
    Stored beside the reference crops under ``configs/identities/``,
    which is gitignored in its entirety.
    """

    embeddings: dict[str, list[list[float]]] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return sorted(self.embeddings)

    def add(self, label: str, embedding: Sequence[float] | np.ndarray) -> None:
        self.embeddings.setdefault(label, []).append([float(v) for v in embedding])

    @classmethod
    def load(cls, path: Path = DEFAULT_GALLERY_PATH) -> Gallery:
        doc_path = path / "gallery.json"
        if not doc_path.exists():
            return cls()
        try:
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(doc, dict):
            return cls()
        out: dict[str, list[list[float]]] = {}
        for label, vecs in doc.items():
            if isinstance(vecs, list):
                out[str(label)] = [[float(v) for v in vec] for vec in vecs if isinstance(vec, list)]
        return cls(embeddings=out)

    def save(self, path: Path = DEFAULT_GALLERY_PATH) -> None:
        path.mkdir(parents=True, exist_ok=True)
        doc_path = path / "gallery.json"
        tmp = doc_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.embeddings), encoding="utf-8")
        tmp.replace(doc_path)

    def match(
        self,
        embedding: np.ndarray,
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
    ) -> tuple[str, float, str | None, float]:
        """Best gallery match for one embedding.

        Returns ``(label, similarity, runner_up, runner_up_similarity)``.
        ``label`` is :data:`UNKNOWN` when nothing clears ``threshold``,
        or when the two best labels are within ``margin`` of each other
        (ambiguous -- two enrolled people who look alike on this crop).
        """
        best: list[tuple[float, str]] = []
        for label, vecs in self.embeddings.items():
            if not vecs:
                continue
            sims = [_cosine(embedding, vec) for vec in vecs]
            best.append((max(sims), label))
        if not best:
            return UNKNOWN, 0.0, None, 0.0
        best.sort(reverse=True)
        top_sim, top_label = best[0]
        second_sim, second_label = best[1] if len(best) > 1 else (0.0, None)
        if top_sim < threshold:
            return UNKNOWN, top_sim, second_label, second_sim
        if second_label is not None and (top_sim - second_sim) < margin:
            return UNKNOWN, top_sim, second_label, second_sim
        return top_label, top_sim, second_label, second_sim


def _cosine(a: np.ndarray, b: Sequence[float]) -> float:
    """Cosine similarity. ArcFace embeddings arrive L2-normalised, but a
    gallery can be hand-edited, so normalise defensively."""
    import numpy as np

    v = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(v))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, v) / denom)


# ----------------------------------------------------------------------
# Face extraction. Imports insightface lazily so the module (and its
# tests) stay importable without the `identity` extra installed.


@dataclass(slots=True)
class DetectedFace:
    """One face found in one image."""

    embedding: np.ndarray
    width_px: int
    det_score: float
    snap_index: int | None = None


def build_face_app(det_size: int = 640) -> Any:
    """Load the buffalo_l pack (SCRFD detector + ArcFace embedder).

    Raises a clear error rather than a bare ImportError when the extra
    isn't installed -- this runs on the dev box only, and the failure
    mode otherwise looks like a missing model rather than a missing
    opt-in.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise RuntimeError(
            "per-person identity needs the 'identity' extra: "
            "uv sync --extra identity  (dev box only -- never on the Orin)"
        ) from exc

    app = FaceAnalysis(name="buffalo_l")
    # ctx_id 0 = first CUDA device, -1 = CPU. The 3080 handles this in
    # milliseconds; CPU is ~10x slower but works.
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def faces_in_image(
    app: Any,
    image: np.ndarray,
    *,
    min_face_px: int = MIN_FACE_PX,
    snap_index: int | None = None,
) -> list[DetectedFace]:
    """Every face in one image that clears ``min_face_px``."""
    out: list[DetectedFace] = []
    for face in app.get(image):
        x1, _y1, x2, _y2 = face.bbox
        width = int(x2 - x1)
        if width < min_face_px:
            continue
        out.append(
            DetectedFace(
                embedding=face.normed_embedding,
                width_px=width,
                det_score=float(face.det_score),
                snap_index=snap_index,
            )
        )
    return out


def identify_track(
    faces: Sequence[DetectedFace],
    gallery: Gallery,
    track_id: int,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> FaceMatch:
    """Best identity across all of one track's faces.

    A track usually has several 4K snaps and so several candidate faces;
    the subject may be looking down in one and straight on in another.
    We take the face whose best gallery similarity is highest rather
    than voting, because the failure mode here is a *bad view* (which
    should simply lose) rather than a wrong label appearing repeatedly.
    """
    if not faces:
        return FaceMatch(track_id=track_id, reason="no_4k_imagery")
    if not gallery.embeddings:
        return FaceMatch(track_id=track_id, reason="empty_gallery")

    best: FaceMatch | None = None
    for face in faces:
        label, sim, runner, runner_sim = gallery.match(
            face.embedding, threshold=threshold, margin=margin
        )
        candidate = FaceMatch(
            track_id=track_id,
            label=label,
            similarity=round(sim, 4),
            runner_up=runner,
            runner_up_similarity=round(runner_sim, 4),
            face_px=face.width_px,
            snap_index=face.snap_index,
            reason="" if label != UNKNOWN else "below_threshold_or_ambiguous",
        )
        if best is None or candidate.similarity > best.similarity:
            best = candidate
    assert best is not None
    return best


def enrol_from_crops(
    app: Any,
    label: str,
    crop_paths: Sequence[Path],
    gallery: Gallery,
    *,
    min_face_px: int = MIN_FACE_PX,
) -> tuple[int, list[Path]]:
    """Add reference embeddings for ``label`` from operator-chosen crops.

    Returns ``(n_added, skipped_paths)``. A crop is skipped when no face
    clears ``min_face_px`` -- typically a sub-stream tile, or a shot
    where the subject is looking away. Enrolment is always an explicit
    operator act: nothing here discovers new identities on its own.
    """
    import cv2

    added = 0
    skipped: list[Path] = []
    for path in crop_paths:
        image = cv2.imread(str(path))
        if image is None:
            skipped.append(path)
            continue
        faces = faces_in_image(app, image, min_face_px=min_face_px)
        if not faces:
            skipped.append(path)
            continue
        # Largest face: an enrolment crop should be of one person, and if
        # a bystander is in shot the subject is the near, large one.
        face = max(faces, key=lambda f: f.width_px)
        gallery.add(label, face.embedding)
        added += 1
    return added, skipped


# ----------------------------------------------------------------------
# Session-level driver + CLI.


def identify_session(
    session_dir: Path,
    gallery: Gallery,
    *,
    app: Any | None = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
    min_face_px: int = MIN_FACE_PX,
    door_only: bool = True,
) -> dict[str, Any]:
    """Identify every person track in one session that has 4K imagery.

    ``door_only`` restricts the pass to tracks whose entry or exit falls
    in the door zone -- the household-and-guests population the gallery
    is enrolled for. Turning it off would run face recognition over every
    passer-by on the pavement, which is a much broader act than what this
    was built for, so it is opt-in and off by default.
    """
    import cv2

    from streettracker.analysis.walks import classify_walk_origin, is_own_trip
    from streettracker.common.door_zone import DoorZone

    if app is None:
        app = build_face_app()

    name = session_dir.name
    data_path = session_dir / f"{name}_data.json"
    if data_path.exists():
        records = json.loads(data_path.read_text(encoding="utf-8"))
    else:
        events = session_dir / f"{name}_events.jsonl"
        lines = events.read_text(encoding="utf-8").splitlines() if events.exists() else []
        records = [json.loads(line) for line in lines if line]

    zone = DoorZone.load() if door_only else None
    matches: list[FaceMatch] = []
    n_considered = 0
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("class_suspect"):
            continue
        if zone is not None:
            kind = classify_walk_origin(
                rec.get("entry_point_frac"), rec.get("exit_point_frac"), zone
            )
            if not is_own_trip(kind):
                continue
        n_considered += 1
        track_id = int(rec.get("track_id", -1))
        # `main_snaps` is the runtime's record of which fires actually
        # landed on disk, so it beats globbing for the filenames.
        prefix = rec.get("asset_prefix") or "person"
        faces: list[DetectedFace] = []
        for snap_index in rec.get("main_snaps") or []:
            path = session_dir / f"{prefix}_{track_id}_main_{snap_index}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                continue
            faces.extend(
                faces_in_image(app, image, min_face_px=min_face_px, snap_index=snap_index)
            )
        matches.append(
            identify_track(faces, gallery, track_id, threshold=threshold, margin=margin)
        )

    by_label: dict[str, int] = {}
    for m in matches:
        by_label[m.label] = by_label.get(m.label, 0) + 1
    return {
        "session": name,
        "gallery_labels": gallery.labels,
        "threshold": threshold,
        "margin": margin,
        "door_only": door_only,
        "n_tracks_considered": n_considered,
        "summary": by_label,
        "tracks": [m.to_json_dict() for m in matches],
    }


def _cmd_enrol(args: argparse.Namespace) -> int:
    """`streettracker identity-enrol <label> <crop...>`"""
    gallery = Gallery.load(args.gallery)
    app = build_face_app()
    crops = [Path(p) for p in args.crops]
    added, skipped = enrol_from_crops(app, args.label, crops, gallery, min_face_px=args.min_face_px)
    gallery.save(args.gallery)
    print(f"[identity] enrolled {added} reference face(s) for {args.label!r}")
    if skipped:
        print(f"  skipped {len(skipped)} crop(s) with no usable face (< {args.min_face_px} px):")
        for p in skipped[:10]:
            print(f"    {p.name}")
    print(f"  gallery now holds: {', '.join(gallery.labels) or '(empty)'}")
    print(f"  written to {args.gallery}/gallery.json  (gitignored -- never commit or release)")
    return 0


def _cmd_identify(args: argparse.Namespace) -> int:
    """`streettracker identify <session-dir>`"""
    gallery = Gallery.load(args.gallery)
    if not gallery.embeddings:
        print(f"[identity] gallery at {args.gallery} is empty -- enrol someone first:")
        print("  uv run streettracker identity-enrol <name> <crop.jpg> ...")
        return 1

    session_dir = Path(args.session_dir)
    doc = identify_session(
        session_dir,
        gallery,
        threshold=args.threshold,
        margin=args.margin,
        min_face_px=args.min_face_px,
        door_only=not args.all_people,
    )
    out_path = session_dir / f"{session_dir.name}_identity.json"
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"[identity] wrote {out_path}")
    print(f"  gallery: {', '.join(doc['gallery_labels'])}")
    scope = "all person tracks" if args.all_people else "door-zone tracks only"
    print(f"  scope: {scope}   considered: {doc['n_tracks_considered']}")
    for label, n in sorted(doc["summary"].items(), key=lambda kv: -kv[1]):
        print(f"    {label:<20s} {n}")
    if doc["n_tracks_considered"] and not any(
        k != UNKNOWN for k in doc["summary"]
    ):  # pragma: no cover - operator guidance
        print(
            "  nothing matched. Most likely the tracks predate door snapping,\n"
            "  so they carry no 4K imagery -- check the 'no_4k_imagery' reason."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="streettracker identify",
        description="Per-person identity for door users (dev box only; never on the Orin).",
    )
    ap.add_argument("session_dir", help="session directory to identify")
    ap.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    ap.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--min-face-px", type=int, default=MIN_FACE_PX)
    ap.add_argument(
        "--all-people",
        action="store_true",
        help="identify EVERY person track, not just door-zone ones. This runs face "
        "recognition over passers-by; off by default on purpose.",
    )
    return _cmd_identify(ap.parse_args(argv))


def enrol_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="streettracker identity-enrol",
        description="Enrol reference faces for one person (dev box only).",
    )
    ap.add_argument("label", help="person's name/label, e.g. 'alys'")
    ap.add_argument("crops", nargs="+", help="reference image paths (4K snaps work best)")
    ap.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    ap.add_argument("--min-face-px", type=int, default=MIN_FACE_PX)
    return _cmd_enrol(ap.parse_args(argv))
