"""Per-person-track activity enrichment.

Folds a session's person tracks (``data.json``) into
``{session}_people.json``: each person track gets an activity ``kind``
(``walker`` / ``jogger`` / ``cyclist``) plus companion pairings — dog
tracks set the ``dog_walker`` flag, bicycle tracks produce the
``cyclist`` kind.

Track records carry no trajectory, so pairing is temporal +
directional: a companion track (dog / bicycle) is attributed to the
person track with the largest time overlap among those overlapping at
least ``--min-overlap-s`` seconds in the same recorded direction. Each
companion pairs with at most one person; a person can accumulate
several companions (two dogs, one walker).

Speed: ``speed_px_s`` is inference-frame pixels. With a calibration
(``--m-per-px``, ``--road-length-m``, or ``configs/showcase.json`` —
same resolution scheme as the showcase) speeds convert to m/s and the
walker/jogger split happens at ``--jogger-min-ms`` (default 2.5 m/s:
the valley between the walking mass and the jogging bump in the
five-soak person-coverage measurement of 2026-07-07,
``.claude/person_coverage.py``). Bicycle-paired tracks are classified
``cyclist`` FIRST — riders detect as person, and the soak's >4 m/s
person-speed tail was riders, so a speed-only jogger split would
mislabel them. Uncalibrated sessions classify every non-cyclist as a
walker (surfaced in the output's ``params``).

Output schema::

    {
      "session": "session_20260628_073521",
      "params": {"m_per_px": 0.0421, "jogger_min_m_s": 2.5,
                 "min_overlap_s": 3.0},
      "summary": {"n_person_tracks": 4211, "walkers": ..., "joggers": ...,
                  "cyclists": ..., "dog_walkers": ...,
                  "n_dog_tracks": ..., "n_dogs_paired": ...,
                  "n_bicycle_tracks": ..., "n_bicycles_paired": ...},
      "people": [
        {"track_id": 42, "time_start": "...", "time_end": "...",
         "direction": "left to right", "duration_visible": 12.8,
         "speed_px_s": 37.1, "speed_m_s": 1.56, "num_detections": 61,
         "kind": "walker", "dog_walker": true,
         "dog_track_ids": [43], "bicycle_track_ids": []},
        ...
      ]
    }

One person track ~= one pass, with the same BotSORT-split caveat as
cars. Dog and bicycle tracks only exist where the deployment's
``inference.vehicle_classes`` includes COCO classes 16 / 1.

Run via the CLI::

    streettracker people output/session_20260628_073521
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Walker/jogger boundary, m/s. Empirical: the pooled five-soak person
# speed histogram (13,350 tracks) shows the walking mass ending ~2.0-2.5
# and a distinct jogging bump at 2.5-4.0.
JOGGER_MIN_M_S_DEFAULT = 2.5

# A companion (dog/bicycle) must co-occur with a person for at least
# this long to pair. Median person dwell on this scene is ~11 s.
MIN_OVERLAP_S_DEFAULT = 3.0

# Speed classification needs this many detections — same guard as the
# stats page's fastest board, against 2-3 frame BotSORT glitch tracks
# whose displacement/duration spikes.
MIN_DETECTIONS_FOR_SPEED = 6

# Traced-road travel-axis length in the 896x512 inference frame (px).
# Mirrors web/stats.py DEFAULT_ROAD_AXIS_PX (kept local: analysis/ does
# not import from web/). Measured from .claude/triggers_proposal.json.
DEFAULT_ROAD_AXIS_PX = 801.0


@dataclass(slots=True)
class PersonTrack:
    """One person track's enrichment row (``people`` array element)."""

    track_id: int
    time_start: str
    time_end: str
    time_start_unix: float
    time_end_unix: float
    direction: str
    duration_visible: float
    speed_px_s: float
    speed_m_s: float | None  # None when uncalibrated
    num_detections: int
    kind: str  # "walker" | "jogger" | "cyclist"
    dog_walker: bool
    dog_track_ids: list[int] = field(default_factory=list)
    bicycle_track_ids: list[int] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _overlap_s(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Seconds of wall-clock overlap between two track records."""
    start = max(float(a["time_start_unix"]), float(b["time_start_unix"]))
    end = min(float(a["time_end_unix"]), float(b["time_end_unix"]))
    return end - start


def pair_companions(
    persons: list[dict[str, Any]],
    companions: list[dict[str, Any]],
    *,
    min_overlap_s: float = MIN_OVERLAP_S_DEFAULT,
) -> dict[int, list[int]]:
    """Attribute companion tracks (dogs / bicycles) to person tracks.

    Each companion pairs with the person track it overlaps longest,
    among persons overlapping >= ``min_overlap_s`` in the same recorded
    direction. Returns ``{person_track_id: [companion_track_id, ...]}``
    (only persons that gained a companion appear).
    """
    paired: dict[int, list[int]] = {}
    for comp in companions:
        best_person: int | None = None
        best_overlap = min_overlap_s
        for person in persons:
            if person.get("direction") != comp.get("direction"):
                continue
            ov = _overlap_s(person, comp)
            if ov >= best_overlap:
                best_overlap = ov
                best_person = person["track_id"]
        if best_person is not None:
            paired.setdefault(best_person, []).append(comp["track_id"])
    return paired


def classify_kind(
    *,
    speed_m_s: float | None,
    num_detections: int,
    has_bicycle: bool,
    jogger_min_m_s: float = JOGGER_MIN_M_S_DEFAULT,
) -> str:
    """Activity kind for one person track. Cyclist wins over speed —
    riders detect as person and would otherwise land in the jogger
    bucket. Jogger needs a calibrated speed and enough detections to
    trust it; everything else is a walker."""
    if has_bicycle:
        return "cyclist"
    if (
        speed_m_s is not None
        and num_detections >= MIN_DETECTIONS_FOR_SPEED
        and speed_m_s >= jogger_min_m_s
    ):
        return "jogger"
    return "walker"


def build_people(
    records: list[dict[str, Any]],
    *,
    m_per_px: float | None = None,
    jogger_min_m_s: float = JOGGER_MIN_M_S_DEFAULT,
    min_overlap_s: float = MIN_OVERLAP_S_DEFAULT,
) -> tuple[list[PersonTrack], dict[str, Any]]:
    """Fold a session's raw track records into person enrichment rows
    plus the summary block. ``records`` is the parsed ``data.json``
    array; non-person/dog/bicycle records are ignored."""
    persons = [r for r in records if r.get("class_name") == "person"]
    dogs = [r for r in records if r.get("class_name") == "dog"]
    bikes = [r for r in records if r.get("class_name") == "bicycle"]

    dog_by_person = pair_companions(persons, dogs, min_overlap_s=min_overlap_s)
    bike_by_person = pair_companions(persons, bikes, min_overlap_s=min_overlap_s)

    out: list[PersonTrack] = []
    for r in persons:
        tid = r["track_id"]
        speed_px_s = float(r.get("speed_px_s") or 0.0)
        speed_m_s = round(speed_px_s * m_per_px, 2) if m_per_px is not None else None
        num_detections = int(r.get("num_detections") or 0)
        dog_ids = sorted(dog_by_person.get(tid, []))
        bike_ids = sorted(bike_by_person.get(tid, []))
        out.append(
            PersonTrack(
                track_id=tid,
                time_start=r.get("time_start") or "",
                time_end=r.get("time_end") or "",
                time_start_unix=float(r.get("time_start_unix") or 0.0),
                time_end_unix=float(r.get("time_end_unix") or 0.0),
                direction=r.get("direction") or "",
                duration_visible=float(r.get("duration_visible") or 0.0),
                speed_px_s=speed_px_s,
                speed_m_s=speed_m_s,
                num_detections=num_detections,
                kind=classify_kind(
                    speed_m_s=speed_m_s,
                    num_detections=num_detections,
                    has_bicycle=bool(bike_ids),
                    jogger_min_m_s=jogger_min_m_s,
                ),
                dog_walker=bool(dog_ids),
                dog_track_ids=dog_ids,
                bicycle_track_ids=bike_ids,
            )
        )

    summary = {
        "n_person_tracks": len(out),
        "walkers": sum(1 for p in out if p.kind == "walker"),
        "joggers": sum(1 for p in out if p.kind == "jogger"),
        "cyclists": sum(1 for p in out if p.kind == "cyclist"),
        "dog_walkers": sum(1 for p in out if p.dog_walker),
        "n_dog_tracks": len(dogs),
        "n_dogs_paired": sum(len(v) for v in dog_by_person.values()),
        "n_bicycle_tracks": len(bikes),
        "n_bicycles_paired": sum(len(v) for v in bike_by_person.values()),
    }
    return out, summary


def resolve_m_per_px(
    m_per_px: float | None,
    road_length_m: float | None,
    config_path: Path = Path("configs/showcase.json"),
) -> float | None:
    """Calibration resolution, mirroring the showcase's scheme: explicit
    ``--m-per-px`` wins, then ``--road-length-m`` / road-axis pixels,
    then ``configs/showcase.json`` (either key), else uncalibrated."""
    if m_per_px is not None:
        return m_per_px
    if road_length_m is not None:
        return road_length_m / DEFAULT_ROAD_AXIS_PX
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if cfg.get("m_per_px") is not None:
            return float(cfg["m_per_px"])
        if cfg.get("road_length_m") is not None:
            return float(cfg["road_length_m"]) / DEFAULT_ROAD_AXIS_PX
    return None


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="streettracker people",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_dir", type=Path)
    ap.add_argument(
        "--jogger-min-ms",
        type=float,
        default=JOGGER_MIN_M_S_DEFAULT,
        help=f"walker/jogger boundary in m/s (default {JOGGER_MIN_M_S_DEFAULT})",
    )
    ap.add_argument(
        "--min-overlap-s",
        type=float,
        default=MIN_OVERLAP_S_DEFAULT,
        help="minimum person/companion time overlap to pair, seconds "
        f"(default {MIN_OVERLAP_S_DEFAULT})",
    )
    ap.add_argument(
        "--m-per-px",
        type=float,
        default=None,
        help="speed calibration; overrides --road-length-m and configs/showcase.json",
    )
    ap.add_argument(
        "--road-length-m",
        type=float,
        default=None,
        help="visible road length in metres (converted via the traced "
        f"road axis, {DEFAULT_ROAD_AXIS_PX:.0f} px)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default <session>/<session>_people.json)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2
    session = session_dir.name
    data_path = session_dir / f"{session}_data.json"
    if not data_path.exists():
        print(f"missing {data_path}", file=sys.stderr)
        return 2
    records = json.loads(data_path.read_text(encoding="utf-8"))

    m_per_px = resolve_m_per_px(args.m_per_px, args.road_length_m)
    people, summary = build_people(
        records,
        m_per_px=m_per_px,
        jogger_min_m_s=args.jogger_min_ms,
        min_overlap_s=args.min_overlap_s,
    )

    out_path = args.out or (session_dir / f"{session}_people.json")
    out_path.write_text(
        json.dumps(
            {
                "session": session,
                "params": {
                    "m_per_px": m_per_px,
                    "jogger_min_m_s": args.jogger_min_ms,
                    "min_overlap_s": args.min_overlap_s,
                },
                "summary": summary,
                "people": [p.to_json_dict() for p in people],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[people] wrote {out_path}  ({summary['n_person_tracks']} person tracks)")
    print(
        f"  walkers: {summary['walkers']}   joggers: {summary['joggers']}   "
        f"cyclists: {summary['cyclists']}   dog walkers: {summary['dog_walkers']}"
    )
    if summary["n_dog_tracks"]:
        loose = summary["n_dog_tracks"] - summary["n_dogs_paired"]
        print(
            f"  dog tracks: {summary['n_dog_tracks']}  "
            f"(paired {summary['n_dogs_paired']}, unpaired {loose})"
        )
    if m_per_px is None:
        print("  (uncalibrated -- no jogger split; set configs/showcase.json or --road-length-m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
