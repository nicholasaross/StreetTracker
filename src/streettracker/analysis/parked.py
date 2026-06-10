"""Stationary parked-plate ("beacon") detection over per-image ALPR reads.

A car parked inside the camera's view leaves a distinctive signature in a
session's per-image ALPR output (``<session>_alpr.json``): the SAME plate
text detected at (nearly) the SAME image position across MANY DIFFERENT
moving tracks. The plate detector, fed each passing car's (possibly stale)
pre-crop, keeps finding the crisp static plate of the parked car instead —
so the parked car's identity is attributed to every passer-by.

Real example (session_20260601_173815): a Dacia Duster parked in view for
~44 minutes was read 100+ times across ~70 distinct passing tracks, all
within ±60 px of one spot — making it the showcase's #1 "regular" with 55
phantom visits, while its genuine arrival/departure went unrecorded.

This module detects those beacons and converts them from an aliasing bug
into first-class data:

* ``detect_parked`` clusters same-plate reads by image position. A cluster
  read by >= ``min_tracks`` distinct MOVING host tracks (record
  ``speed_px_s >= moving_speed_min``) is a beacon. Its reads on moving
  hosts go into the suppression set; reads on slow/stationary hosts (the
  parked car's own track, if any) are kept, so the car retains its own
  identity.
* Each beacon yields a :class:`ParkedEpisode` — the parked car's plate,
  position, and observed parked window (bounded by the host tracks'
  timestamps). The arrival/departure information the aliasing was hiding.
* ``best_unsuppressed_read`` re-anchors an affected track to its own best
  remaining read (the passing car's real plate, when one was read).

Position clustering is resolution-independent: the merge radius scales
with the median detected plate width (a parked plate wobbles by well under
one plate width; a moving car's consecutive reads travel many).

Contrast that distinguishes beacon from regular driver, measured on this
install: the parked Duster's reads sat at sigma ~(58, 16) px, while
FD61PVX (a resident's car that genuinely drives through daily) scatters
±200-380 px per session. The defaults below sit comfortably between.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from streettracker.analysis.dvsa import is_canonical_uk_plate

# A position cluster needs at least this many distinct MOVING host tracks
# to count as a beacon. BotSORT occasionally splits one stationary car
# into a handful of its own tracks; requiring 4+ moving hosts keeps a
# queued/idling car's self-reads from tripping the detector.
PARKED_MIN_TRACKS = 4

# Hosts at or above this ``speed_px_s`` (inference-frame px/s) count as
# "moving" — i.e. passing traffic that cannot legitimately own a static
# plate. Genuine passes on this scene run ~45-90 px/s; the parked Duster's
# own departure track averaged ~1 px/s.
PARKED_MOVING_SPEED_MIN = 15.0

# Reads below this OCR confidence are ignored for clustering (they're too
# noisy to place) — but anchors all sit at >= 0.9, so anything that could
# have minted a phantom visit is well inside the clustered set.
PARKED_MIN_READ_CONF = 0.5

# Cluster merge radius: max(eps_min_px, eps_plate_widths * median plate
# width among that text's reads).
PARKED_EPS_MIN_PX = 80.0
PARKED_EPS_PLATE_WIDTHS = 1.5


@dataclass(slots=True)
class ParkedEpisode:
    """One parked-car stint: a plate observed static across many tracks."""

    plate: str  # normalised plate text of the beacon reads
    center: tuple[float, float]  # mean read centre, snap pixel coords
    n_reads: int  # beacon reads folded into this episode
    n_tracks: int  # distinct host tracks those reads landed on
    first_seen: str  # ISO time_start of the earliest host track
    last_seen: str  # ISO time_start of the latest host track
    duration_minutes: float  # last_seen - first_seen
    track_ids: list[int]  # moving hosts whose reads were suppressed
    canonical_shape: bool  # plate text matches a UK plate shape

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["center"] = [round(self.center[0], 1), round(self.center[1], 1)]
        return d


@dataclass(slots=True)
class ParkedDetection:
    """Detector output: episodes + the read-identity suppression set."""

    episodes: list[ParkedEpisode] = field(default_factory=list)
    # (track_id, snap_index) of every read attributed to a moving host from
    # a beacon position. Anchor selection must not trust these reads.
    suppressed: set[tuple[int, int]] = field(default_factory=set)
    # All candidate per-image reads keyed by track_id (suppressed or not),
    # so callers can re-anchor an affected track without re-reading the
    # per-image JSON.
    reads_by_track: dict[int, list[dict[str, Any]]] = field(default_factory=dict)


def load_alpr_entries(session_dir: Path) -> list[dict[str, Any]]:
    """Per-image ALPR entries (``<session>_alpr.json``), or ``[]`` when the
    file is absent/unreadable (sessions where ``alpr-run`` never ran)."""
    path = session_dir / f"{session_dir.name}_alpr.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def normalize_plate(text: Any) -> str:
    return (text or "").strip().upper().replace(" ", "")


def _read_key(entry: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return (int(entry["track_id"]), int(entry["snap_index"]))
    except (KeyError, TypeError, ValueError):
        return None


def _records_by_tid(records: Any) -> dict[int, dict[str, Any]]:
    """Accept ``data.json``-shaped list[dict] or a prebuilt tid->record map."""
    if isinstance(records, dict):
        return {int(k): v for k, v in records.items()}
    out: dict[int, dict[str, Any]] = {}
    for r in records or []:
        tid = r.get("track_id")
        if tid is not None:
            out[int(tid)] = r
    return out


def detect_parked(
    entries: list[dict[str, Any]],
    records: Any,
    *,
    min_tracks: int = PARKED_MIN_TRACKS,
    moving_speed_min: float = PARKED_MOVING_SPEED_MIN,
    min_read_conf: float = PARKED_MIN_READ_CONF,
    eps_min_px: float = PARKED_EPS_MIN_PX,
    eps_plate_widths: float = PARKED_EPS_PLATE_WIDTHS,
) -> ParkedDetection:
    """Find parked-plate beacons in a session's per-image ALPR reads.

    ``entries`` is the parsed ``<session>_alpr.json`` list;``records`` the
    parsed ``<session>_data.json`` (list or tid-keyed dict) supplying each
    host track's ``speed_px_s`` / ``time_start``. Returns the episodes plus
    the suppression set of ``(track_id, snap_index)`` read identities.
    """
    recs = _records_by_tid(records)
    detection = ParkedDetection()

    # Candidate reads: preferred-pipeline, positioned, confident enough to
    # have been (or to become) an anchor. Group by normalised text.
    by_text: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        if e.get("pipeline") not in (None, "preferred"):
            continue
        key = _read_key(e)
        if key is None:
            continue
        detection.reads_by_track.setdefault(key[0], []).append(e)
        text = normalize_plate(e.get("ocr_text"))
        bbox = e.get("det_bbox")
        if not text or not bbox or (e.get("ocr_conf") or 0.0) < min_read_conf:
            continue
        by_text.setdefault(text, []).append(e)

    for text, reads in by_text.items():
        if len({_read_key(e) for e in reads}) < min_tracks:
            continue  # cheap prefilter: not enough reads to involve min_tracks hosts
        widths = [e["det_bbox"][2] - e["det_bbox"][0] for e in reads]
        eps = max(eps_min_px, eps_plate_widths * statistics.median(widths))

        # Greedy centroid clustering, highest-conf reads seed first so the
        # crisp parked reads define the cluster centre.
        clusters: list[dict[str, Any]] = []  # {cx, cy, n, reads}
        for e in sorted(reads, key=lambda r: -(r.get("ocr_conf") or 0.0)):
            b = e["det_bbox"]
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            home = None
            for c in clusters:
                if (c["cx"] - cx) ** 2 + (c["cy"] - cy) ** 2 <= eps * eps:
                    home = c
                    break
            if home is None:
                clusters.append({"cx": cx, "cy": cy, "n": 1, "reads": [e]})
            else:
                # Running-mean centroid keeps the cluster anchored.
                home["cx"] += (cx - home["cx"]) / (home["n"] + 1)
                home["cy"] += (cy - home["cy"]) / (home["n"] + 1)
                home["n"] += 1
                home["reads"].append(e)

        for c in clusters:
            tids = {int(e["track_id"]) for e in c["reads"]}
            # Self-host exclusion: a host whose track ALSO read this same
            # text OUTSIDE the cluster is the plate's own car driving by
            # (a regular read at a consistent road position on every pass
            # clusters across its own tracks; e.g. VK74TPO, a genuine 4-
            # visit regular, false-beaconed without this). Foreign passing
            # traffic never reads the beacon text anywhere else.
            cluster_ids = {id(e) for e in c["reads"]}
            self_hosts = {
                int(e["track_id"])
                for e in reads
                if id(e) not in cluster_ids and int(e["track_id"]) in tids
            }
            counted = tids - self_hosts
            moving = {
                t
                for t in counted
                if (recs.get(t, {}).get("speed_px_s") or 0.0) >= moving_speed_min
            }
            if len(moving) < min_tracks:
                continue
            # Beacon. Suppress reads on foreign moving hosts; the parked
            # car's own tracks (slow, or self-hosting) keep their reads.
            suppressed_keys = [
                k
                for e in c["reads"]
                if int(e["track_id"]) in moving and (k := _read_key(e)) is not None
            ]
            detection.suppressed.update(suppressed_keys)
            starts = sorted(
                str(recs[t]["time_start"])
                for t in counted
                if t in recs and recs[t].get("time_start")
            )
            unix = sorted(
                float(recs[t]["time_start_unix"])
                for t in counted
                if t in recs and recs[t].get("time_start_unix") is not None
            )
            duration_min = (unix[-1] - unix[0]) / 60.0 if len(unix) >= 2 else 0.0
            detection.episodes.append(
                ParkedEpisode(
                    plate=text,
                    center=(c["cx"], c["cy"]),
                    n_reads=len(c["reads"]),
                    n_tracks=len(counted),
                    first_seen=starts[0] if starts else "",
                    last_seen=starts[-1] if starts else "",
                    duration_minutes=round(duration_min, 1),
                    track_ids=sorted(moving),
                    canonical_shape=is_canonical_uk_plate(text),
                )
            )

    detection.episodes.sort(key=lambda ep: (-ep.n_reads, ep.plate))
    return detection


def merge_episodes(
    episodes: list[ParkedEpisode],
    canonical: dict[str, str],
    *,
    eps_px: float = 200.0,
) -> list[ParkedEpisode]:
    """Fold episodes that are OCR variants of one physical parked car.

    The same parked plate is often read under 2+ spellings (the Duster:
    ``LX19PXR`` x100 + ``LX11PXR`` x5 at the same spot). ``canonical`` is
    the plate->canonical map from fuzzy clustering; episodes sharing a
    canonical plate whose centres sit within ``eps_px`` merge into one,
    keyed by the canonical spelling.
    """
    groups: dict[str, list[ParkedEpisode]] = {}
    for ep in episodes:
        groups.setdefault(canonical.get(ep.plate, ep.plate), []).append(ep)

    out: list[ParkedEpisode] = []
    for plate, eps_group in groups.items():
        merged: list[ParkedEpisode] = []
        for ep in sorted(eps_group, key=lambda e: -e.n_reads):
            home = None
            for m in merged:
                dx, dy = m.center[0] - ep.center[0], m.center[1] - ep.center[1]
                if dx * dx + dy * dy <= eps_px * eps_px:
                    home = m
                    break
            if home is None:
                merged.append(
                    ParkedEpisode(
                        plate=plate,
                        center=ep.center,
                        n_reads=ep.n_reads,
                        n_tracks=ep.n_tracks,
                        first_seen=ep.first_seen,
                        last_seen=ep.last_seen,
                        duration_minutes=ep.duration_minutes,
                        track_ids=list(ep.track_ids),
                        canonical_shape=is_canonical_uk_plate(plate),
                    )
                )
            else:
                home.n_reads += ep.n_reads
                home.n_tracks += ep.n_tracks
                home.track_ids = sorted(set(home.track_ids) | set(ep.track_ids))
                if ep.first_seen and (not home.first_seen or ep.first_seen < home.first_seen):
                    home.first_seen = ep.first_seen
                if ep.last_seen and ep.last_seen > home.last_seen:
                    home.last_seen = ep.last_seen
                home.duration_minutes = max(home.duration_minutes, ep.duration_minutes)
        out.extend(merged)
    out.sort(key=lambda ep: (-ep.n_reads, ep.plate))
    return out


def best_unsuppressed_read(
    reads: list[dict[str, Any]],
    suppressed: set[tuple[int, int]],
    *,
    conf_threshold: float,
    canonical_only: bool = True,
) -> dict[str, Any] | None:
    """Re-anchor one track: its highest-conf per-image read that is not
    beacon-suppressed (and clears the same conf + canonical-shape gates the
    rollup anchor had to). ``None`` when nothing genuine remains — the
    track is then treated as unread."""
    best: dict[str, Any] | None = None
    for e in reads:
        conf = e.get("ocr_conf") or 0.0
        if conf < conf_threshold:
            continue
        key = _read_key(e)
        if key is None or key in suppressed:
            continue
        text = normalize_plate(e.get("ocr_text"))
        if not text:
            continue
        if canonical_only and not is_canonical_uk_plate(text):
            continue
        if best is None or conf > (best.get("ocr_conf") or 0.0):
            best = e
    if best is None:
        return None
    return {
        "track_id": best.get("track_id"),
        "snap_index": best.get("snap_index"),
        "image": best.get("image"),
        "ocr_text": best.get("ocr_text"),
        "ocr_conf": best.get("ocr_conf"),
        "det_conf": best.get("det_conf"),
    }
