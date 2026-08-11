"""Cross-session showcase aggregation.

Pools the *plated* per-session vehicles produced by
:func:`streettracker.analysis.vehicles.build_vehicles` into one
:class:`ShowcaseCar` per physical car, keyed by canonical UK plate. This is
the read model the website renders.

Why not reuse :func:`streettracker.analysis.vehicles.build_cross_session`
directly? That function already pools by canonical plate and classifies the
repeat ``kind`` (different-day / same-day / one-off), but it discards the
per-visit image detail (it keeps only ``(time, direction)``). The gallery
needs the snap filenames per sighting, so we re-do the pooling here, reusing
the same building blocks:

* ``build_vehicles`` -- per-session, DVSA-joined, fuzzy-clustered, canonical
  -only plated vehicles (its ``Vehicle.visits`` carry ``track_id`` +
  ``best_image``).
* ``_cluster_plates_by_similarity`` -- the cross-session canonical mapping so
  a 1-character OCR diff doesn't split one car across two sessions.

On top of that we join the *richer* DVSA fields (``primary_colour`` /
``fuel_type`` / ``engine_size_cc``) straight from each session's
``<session>_dvsa_labels.json`` -- they're harvested but not surfaced by the
``Vehicle`` dataclass.

The on-disk ``<session>_vehicles.json`` is deliberately ignored: several were
written before ``dvsa-label`` ran and carry zero make/model. We always rebuild
live.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from streettracker.analysis.vehicles import (
    FUZZY_RATIO_DEFAULT,
    Vehicle,
    _cluster_plates_by_similarity,
    build_vehicles,
    make_distinct_vehicle_checker,
)


@dataclass(slots=True)
class ShowcaseImage:
    """One sighting's images, as **existence-checked**, server-relative URLs
    (``/images/<session>/<file>``; the server route maps them back to the
    session output dir).

    ``thumb`` is the smallest crop present on disk (tile -> hq -> the 4K main
    snap); ``full`` is the largest (4K main -> hq -> tile). ``thumb_small`` is
    True when ``thumb`` is a real small crop (tile/hq) rather than a downscaled
    4K snap -- the gallery prefers a ``thumb_small`` sighting for the card so a
    car that only appears in ``--only-main`` sessions (which ship the 4K snaps
    but no ``_hq``/tile crops) doesn't force a heavy 4K thumbnail when another
    sighting has a small one. Only files that exist are referenced, so the UI
    never shows a broken image."""

    session: str
    track_id: int
    time: str
    direction: str
    color: str
    thumb: str | None
    full: str | None
    thumb_small: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ShowcaseCar:
    """One physical car, pooled across every session it appears in.

    ``kind`` mirrors :class:`streettracker.analysis.vehicles.CrossVehicle`:
    ``different-day`` (a regular -- seen on >=2 calendar dates),
    ``same-day`` (>=2 sightings on one date), ``one-off`` (single sighting).
    """

    plate: str
    make: str | None
    model: str | None
    year: int | None
    make_model_source: str | None
    primary_colour: str | None
    fuel_type: str | None
    engine_size_cc: int | None
    n_visits: int
    n_sessions: int
    n_dates: int
    kind: str
    first_seen: str
    last_seen: str
    # Single best thumbnail for the gallery card: the smallest crop available
    # across all sightings (prefers a real small crop over a 4K snap). None
    # only if the car has no readable snap on disk.
    thumb: str | None = None
    directions: dict[str, int] = field(default_factory=dict)
    colors: dict[str, int] = field(default_factory=dict)
    # Confidence-weighted majority colour from the colour CNN (reads the 4K
    # snap; ``colour_source="cnn"``), pooled across this car's tracks. Far more
    # reliable than the low-res HSV ``colors`` histogram — the template prefers
    # it. None where no session carries a ``_colour_by_track.json`` for the car.
    cnn_colour: str | None = None
    plate_variants: list[str] = field(default_factory=list)
    # Parked-car stints from the stationary-beacon detector (one dict per
    # episode, see ``analysis.parked.ParkedEpisode.to_json_dict``):
    # windows where this car sat in view while its static plate was read
    # across passing tracks. Those reads are already excluded from
    # ``n_visits``/``images``; the episode documents the parked window.
    parked_episodes: list[dict[str, Any]] = field(default_factory=list)
    images: list[ShowcaseImage] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["images"] = [
            im.to_json_dict() if isinstance(im, ShowcaseImage) else im for im in d["images"]
        ]
        return d


_KIND_RANK = {"different-day": 0, "same-day": 1, "one-off": 2}


def discover_sessions(output_root: Path) -> list[Path]:
    """Return session dirs under ``output_root`` that carry a ``_data.json``.

    A session needs its per-track ``data.json`` for ``build_vehicles`` to read;
    in-progress sessions (snaps but no finalized data) are skipped. Sorted by
    name (chronological, since names are ``session_YYYYMMDD_HHMMSS``)."""
    if not output_root.is_dir():
        return []
    return sorted(
        d for d in output_root.iterdir() if d.is_dir() and (d / f"{d.name}_data.json").exists()
    )


def _load_dvsa_extras(session_dir: Path) -> dict[str, dict[str, Any]]:
    """Return ``{plate -> label-row}`` from a session's DVSA harvest, or ``{}``.

    The rows carry ``primary_colour`` / ``fuel_type`` / ``engine_size_cc`` in
    addition to make/model/year -- fields the ``Vehicle`` dataclass drops."""
    path = session_dir / f"{session_dir.name}_dvsa_labels.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    labels = payload.get("labels")
    return labels if isinstance(labels, dict) else {}


def _load_colour_by_track(session_dir: Path) -> dict[int, str]:
    """Return ``{track_id -> CNN colour}`` from a session's
    ``_colour_by_track.json`` (confident tracks only), or ``{}``."""
    path = session_dir / f"{session_dir.name}_colour_by_track.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, str] = {}
    for t in payload.get("tracks", []):
        col, tid = t.get("colour"), t.get("track_id")
        if col and tid is not None:
            out[int(tid)] = str(col)
    return out


def resolve_image_urls(
    output_root: Path,
    session: str,
    track_id: int,
    *,
    prefix: str = "vehicle",
    best_image: str | None = None,
) -> tuple[str | None, str | None, bool]:
    """Existence-checked ``(thumb, full, thumb_small)`` for one track's snaps.

    Sessions pulled with ``--only-main`` ship the 4K ``_main_N.jpg`` snaps but
    not the ``_hq``/tile crops, so we stat each candidate and only reference
    files that are actually on disk: ``thumb`` is the smallest present
    (tile -> hq -> the 4K ``best_image``), ``full`` the largest. ``thumb_small``
    is True when ``thumb`` is a real small crop (not a downscaled 4K snap).
    Shared with :mod:`streettracker.web.stats` (fastest-car thumbnails)."""
    base = output_root / session

    def _url(name: str | None) -> str | None:
        return f"/images/{session}/{name}" if name and (base / name).is_file() else None

    tile = _url(f"{prefix}_{track_id}.jpg")
    hq = _url(f"{prefix}_{track_id}_hq.jpg")
    best = _url(best_image)
    small = tile or hq
    return small or best, best or hq or tile, small is not None


def _image_urls(output_root: Path, session: str, visit: Any) -> ShowcaseImage:
    """Build a :class:`ShowcaseImage` for one sighting (gallery use)."""
    thumb, full, small = resolve_image_urls(
        output_root,
        session,
        visit.track_id,
        prefix=visit.asset_prefix or "vehicle",
        best_image=visit.best_image,  # 4K snap that produced the best ALPR read
    )
    return ShowcaseImage(
        session=session,
        track_id=visit.track_id,
        time=visit.time_start,
        direction=visit.direction,
        color=visit.color,
        thumb=thumb,
        full=full,
        thumb_small=small,
    )


def build_showcase(
    output_root: Path,
    *,
    fuzzy_ratio: int | None = FUZZY_RATIO_DEFAULT,
) -> list[ShowcaseCar]:
    """Build the cross-session showcase: one :class:`ShowcaseCar` per plate.

    Only plated (identified) cars are emitted -- unread tracks have no stable
    cross-session identity to key user metadata on. Cars are returned
    regulars-first (different-day, then same-day, then one-off; ties by visit
    count desc, then plate) so the gallery's natural order leads with the
    recurring vehicles.
    """
    sessions = discover_sessions(output_root)

    # (session_name, Vehicle) for every plated vehicle, plus the max OCR
    # confidence per raw plate string (seeds the cross-session clustering) and
    # the richer DVSA rows keyed by raw plate.
    plated: list[tuple[str, str, Vehicle]] = []
    conf_by_plate: dict[str, float] = {}
    extras_by_plate: dict[str, dict[str, Any]] = {}
    colours_by_plate: dict[str, Counter] = {}
    cnn_by_session: dict[str, dict[int, str]] = {}
    for d in sessions:
        cnn_by_session[d.name] = _load_colour_by_track(d)
        for v in build_vehicles(d, fuzzy_ratio=fuzzy_ratio):
            if v.plate is None:
                continue
            plated.append((d.name, v.plate, v))
            conf_by_plate[v.plate] = max(conf_by_plate.get(v.plate, 0.0), v.plate_conf or 0.0)
            colours_by_plate.setdefault(v.plate, Counter()).update(v.colors)
        for plate, row in _load_dvsa_extras(d).items():
            if isinstance(row, dict):
                extras_by_plate.setdefault(plate, row)

    # Cross-session canonical map: collapse OCR variants of one physical plate
    # seen in different sessions (same fuzzy logic build_cross_session uses).
    # The DVSA-distinct veto keeps provably different registered vehicles
    # (e.g. a red UP and a white GOLF one character apart) on separate cards.
    if fuzzy_ratio is not None:
        canonical = _cluster_plates_by_similarity(
            list(conf_by_plate.items()),
            ratio=fuzzy_ratio,
            is_distinct=make_distinct_vehicle_checker(extras_by_plate, colours_by_plate),
        )
    else:
        canonical = {p: p for p in conf_by_plate}

    pool: dict[str, dict[str, Any]] = {}
    for sname, vplate, v in plated:
        key = canonical.get(vplate, vplate)
        rec = pool.setdefault(
            key,
            {
                "make": None,
                "model": None,
                "year": None,
                "make_model_source": None,
                "sessions": set(),
                "dates": set(),
                "plates": set(),
                "directions": Counter(),
                "colors": Counter(),
                "cnn_colors": Counter(),
                "parked": [],
                "images": [],
            },
        )
        rec["sessions"].add(sname)
        # Track every raw plate string folded in (canonical + this vehicle's
        # plate + its within-session fuzzy variants) so the DVSA extras join
        # can match on any of them.
        rec["plates"].add(vplate)
        for pv, _c in v.plate_variants:
            rec["plates"].add(pv)
        if v.make and rec["make"] is None:
            rec["make"], rec["model"], rec["year"], rec["make_model_source"] = (
                v.make,
                v.model,
                v.year,
                v.make_model_source,
            )
        rec["directions"].update(v.directions)
        rec["colors"].update(v.colors)
        rec["parked"].extend(v.parked_episodes)
        sess_cnn = cnn_by_session.get(sname, {})
        for visit in v.visits:
            rec["dates"].add(visit.time_start[:10])
            rec["images"].append(_image_urls(output_root, sname, visit))
            cc = sess_cnn.get(visit.track_id)
            if cc:
                rec["cnn_colors"][cc] += 1

    out: list[ShowcaseCar] = []
    for plate, rec in pool.items():
        images = sorted(rec["images"], key=lambda im: im.time)
        # Card thumbnail: prefer a sighting with a real small crop; otherwise
        # fall back to the earliest sighting's (possibly 4K) thumb.
        card_img = next((im for im in images if im.thumb_small), images[0] if images else None)
        n_visits = len(images)
        n_dates = len(rec["dates"])
        kind = "different-day" if n_dates >= 2 else "same-day" if n_visits >= 2 else "one-off"
        # DVSA extras: first matching raw plate among the folded-in strings.
        extras: dict[str, Any] = {}
        for cand in (plate, *sorted(rec["plates"])):
            if cand in extras_by_plate:
                extras = extras_by_plate[cand]
                break
        variants = sorted(p for p in rec["plates"] if p != plate)
        out.append(
            ShowcaseCar(
                plate=plate,
                make=rec["make"],
                model=rec["model"],
                year=rec["year"],
                make_model_source=rec["make_model_source"],
                primary_colour=extras.get("primary_colour") or None,
                fuel_type=extras.get("fuel_type") or None,
                engine_size_cc=extras.get("engine_size_cc"),
                n_visits=n_visits,
                n_sessions=len(rec["sessions"]),
                n_dates=n_dates,
                kind=kind,
                first_seen=images[0].time if images else "",
                last_seen=images[-1].time if images else "",
                thumb=card_img.thumb if card_img else None,
                directions=dict(rec["directions"]),
                colors=dict(rec["colors"]),
                cnn_colour=(rec["cnn_colors"].most_common(1)[0][0] if rec["cnn_colors"] else None),
                plate_variants=variants,
                parked_episodes=sorted(rec["parked"], key=lambda e: e.get("first_seen") or ""),
                images=images,
            )
        )

    out.sort(key=lambda c: (_KIND_RANK[c.kind], -c.n_visits, c.plate))
    return out
