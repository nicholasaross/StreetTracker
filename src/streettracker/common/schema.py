"""Track record + session metadata dataclasses.

These are the canonical schemas — used by the device runtime when writing
session output, and by analysis tools when reading it. Field set ported
verbatim from NanoTracker's `compute_attributes()` return dict so existing
sessions remain parseable.

Records are JSON-serialised via `to_json_dict()` / `from_json_dict()` to
keep the wire format stable across Python version bumps. We don't pickle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ----------------------------------------------------------------------
# Track record (one per finalized track)
# ----------------------------------------------------------------------

@dataclass(slots=True)
class TrackRecord:
    """One finalized track. Written as a single JSONL line and folded into
    the session's `_data.json` array at finalize time.

    Fields:
        identity:  track_id, class_id, class_name, asset_prefix
        time:      time_start[/end][/_unix][/_s], duration_visible
        motion:    direction, speed_px_s, displacement_px, net_displacement_px, lane
        detection: avg_confidence, num_detections
        attrs:     color, make/model/year (off-device DVSA/CNN enrichment)
        assets:    main_snaps (list of int N values whose _main_N.jpg landed)
                   main_snap_bboxes (parallel list of sub-stream bboxes at fire time)
    """

    track_id: int
    class_id: int
    class_name: str
    time_start: str           # ISO-local with tz offset
    time_end: str
    time_start_unix: float
    time_end_unix: float
    time_start_s: float       # seconds since session start
    time_end_s: float
    duration_visible: float
    direction: str            # "left to right" | "right to left"
    speed_px_s: float
    color: str                # see common.color.vote_color()
    lane: str                 # "top" | "middle" | "bottom"
    avg_confidence: float
    displacement_px: float
    net_displacement_px: float
    num_detections: int
    asset_prefix: str = "vehicle"   # "vehicle" | "person"
    main_snaps: list[int] = field(default_factory=list)
    # Parallel to ``main_snaps``: the BotSORT-tracked sub-stream bbox at
    # the moment that snap was fired, as ``[x1, y1, x2, y2]`` in
    # sub-stream pixel coords (matches the resolution in
    # ``SessionMeta.frame_size``). ``None`` per element if the runtime
    # didn't capture a bbox for that snap; outer ``None`` for older
    # sessions written before the field existed. Analysis-side ALPR
    # uses these to pre-crop the 4K snap to the *tracked* vehicle
    # (vs the largest in frame); eliminates the parked-car aliasing
    # surfaced by the 2026-05-25 soak re-run. See
    # :func:`streettracker.device.runtime._fire_snap`.
    main_snap_bboxes: list[list[int] | None] | None = None
    # Parallel to ``main_snaps``: the tracked bbox re-captured when the
    # snap's HTTP task COMPLETED (the 4K image is exposed near response
    # time, ~0.7-1.3 s after the fire decision -- Step 16 measured 96 %
    # of R→L read failures as the car exiting its fire-time bbox in
    # that gap). Where present this is the car's true position in the
    # saved image; offline hint resolution prefers it over
    # ``main_snap_bboxes``. ``None`` for sessions written before the
    # field existed.
    main_snap_bboxes_done: list[list[int] | None] | None = None
    # Make/model/year enrichment, filled OFF-DEVICE by a post-process
    # pass (never the live runtime). ``make_model_source`` is "dvsa"
    # when set from the DVSA MOT harvest (``streettracker dvsa-apply``)
    # and "cnn" once the CompCars classifier lands. All None on a
    # freshly-finalised track and on sessions written before the fields.
    make: str | None = None
    model: str | None = None
    year: int | None = None
    make_model_source: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> TrackRecord:
        # Tolerate extra fields (forward compatibility with future analysers
        # that add columns). Drop unknown keys silently.
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ----------------------------------------------------------------------
# Session metadata
# ----------------------------------------------------------------------

@dataclass(slots=True)
class IRPeriod:
    """A stretch of frames in which IR/night mode was active and inference
    was paused. Persisted so analysis can tell 'no traffic this hour' apart
    from 'we were asleep'."""

    start: str          # ISO local
    end: str
    duration_s: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionMeta:
    session_label: str
    session_start_unix: float
    frames_processed: int = 0
    pipe_fps: float = 0.0
    avg_infer_ms: float = 0.0
    ir_periods: list[IRPeriod] = field(default_factory=list)
    # Reolink snapshot stats: counters + fire-latency percentiles + the
    # frame-count where the blur gate suppressed a fire. ``None`` when
    # the session ran with the snapshotter disabled (``batch`` mode,
    # or ``snapshot.enabled=False`` in config). Otherwise a flat dict --
    # see :meth:`ReolinkSnapshotter.latency_summary` and
    # ``streettracker.device.runtime.build_session_meta`` for the
    # exact keys. Surfaced here so a multi-day soak's ``_meta.json``
    # carries the data needed to tune trigger placement post-cutover
    # without parsing the journal.
    snap_stats: dict[str, Any] | None = None
    # Sub-stream frame size ``[width, height]`` (typically `[896, 512]`
    # for the Reolink sub-stream we drive inference from). Captured at
    # session start. ``None`` for older sessions written before the
    # field existed. Analysis tools use this together with
    # :attr:`TrackRecord.main_snap_bboxes` to scale per-snap bboxes
    # from the sub-stream coord system into the 4K snap coords (each
    # snap's actual pixel size is read from the JPEG at load time, so
    # different mainstream resolutions are handled transparently).
    frame_size: list[int] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict() recurses dataclasses; ir_periods is already in dict form.
        return d

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> SessionMeta:
        ir = [IRPeriod(**p) for p in d.get("ir_periods", [])]
        return cls(
            session_label=d["session_label"],
            session_start_unix=d["session_start_unix"],
            frames_processed=d.get("frames_processed", 0),
            pipe_fps=d.get("pipe_fps", 0.0),
            avg_infer_ms=d.get("avg_infer_ms", 0.0),
            ir_periods=ir,
            snap_stats=d.get("snap_stats"),
            frame_size=d.get("frame_size"),
        )


# ----------------------------------------------------------------------
# Asset-prefix helper (matches NanoTracker's _class_asset_prefix)
# ----------------------------------------------------------------------

_PERSON_CLASS_ID = 0  # COCO class 0 = person; everything else maps to vehicle.


def asset_prefix_for_class(class_id: int) -> str:
    return "person" if class_id == _PERSON_CLASS_ID else "vehicle"
