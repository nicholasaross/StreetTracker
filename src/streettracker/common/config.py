"""Strict JSON config loader for StreetTracker.

Loads ``configs/camera.json`` (per-install, gitignored) into a tree of
frozen ``@dataclass`` objects. Schema mirrors NanoTracker's
``camera_config.json`` because that is what is already deployed on the
Orin -- a fresh deploy needs no migration step, only an scp of the
existing file plus this loader's strict validation.

Properties of the loader
~~~~~~~~~~~~~~~~~~~~~~~~

* **Frozen output.** ``StreetTrackerConfig`` and every nested config
  object is ``@dataclass(frozen=True, slots=True)``. Runtime code can
  treat the loaded config as immutable for the lifetime of the session.
* **Strict on unknown keys.** Any key not declared on the matching
  dataclass raises ``ConfigError`` rather than being silently dropped.
  This catches typos like ``cofidence_threshold`` that would otherwise
  use the default and look like working config.
* **Comment-friendly.** Keys whose name starts with an underscore
  (``_comment``, ``_note``, etc.) are stripped before validation,
  matching the convention in ``configs/camera.example.json``.
* **Nano-compat carve-out.** The legacy ``nano`` block from
  NanoTracker's config schema is accepted and exposed on the loaded
  object (so existing deployed configs load unchanged) but its fields
  are device-decoder hints that StreetTracker does not consume.
* **Type-coerced where unambiguous.** JSON ``list`` -> ``tuple``,
  ``str`` -> ``Path`` for known path fields. No string -> int magic.
* **Descriptive errors.** Error messages include the JSON path of the
  offending node (e.g. ``snapshot.snap_gate.trigger_directions``).
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised on invalid config layout, unknown keys, or type mismatches.

    Always carries the JSON path of the offending node in the message so
    operators can find the line in their config file quickly.
    """


# ----------------------------------------------------------------------
# Camera + streams


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Reolink network address + auth.

    ``model`` is free-form metadata (e.g. ``"Reolink RLC-1224A"``); the
    runtime ignores it but it's useful to keep in the file for the
    operator and for the snapshot HTTP user-agent.
    """

    ip: str
    username: str
    password: str
    model: str = ""


@dataclass(frozen=True, slots=True)
class PortsConfig:
    """Reolink service ports. Defaults match standard Reolink units."""

    http: int = 80
    rtsp: int = 554


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """One Reolink stream entry. The live deploy registers two
    ("main" 4K H.265 and "sub" 896x512 H.264); the runtime picks one
    via ``NanoCompatConfig.preferred_stream`` (legacy compat) or by
    explicit CLI override."""

    name: str
    quality: str  # "main" | "sub"
    codec: str  # "h264" | "h265"
    path: str  # RTSP URL path, e.g. "/h264Preview_01_sub"


# ----------------------------------------------------------------------
# NanoCompat -- the legacy nano-specific block carried in the deployed
# camera_config.json. Accepted by the loader, not consumed by the
# StreetTracker runtime (Orin uses different decoder defaults).


@dataclass(frozen=True, slots=True)
class NanoCompatConfig:
    """Legacy NanoTracker decoder hints. Honored only for stream-name
    selection on the Orin; the actual decoder choice lives in the
    ``sources/`` modules and uses NVDEC unconditionally."""

    preferred_stream: str = "sub"
    rtsp_transport: str = "tcp"
    latency_ms: int = 200
    decoder: str = "nvv4l2decoder"


# ----------------------------------------------------------------------
# Inference + tracking


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """YOLO inference knobs. ``engine_path`` is resolved relative to
    the repo root if not absolute."""

    engine_path: Path
    input_size: int = 640
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    vehicle_classes: tuple[int, ...] = (2,)  # COCO car-only by default


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """BotSORT-side tuning. ``iou_match_threshold`` and ``max_misses``
    are kept for parity with NanoTracker's hand-rolled tracker config
    but BotSORT (via Ultralytics) reads its own parameters from a yaml;
    these values are exposed so the operator can still hand-tune from
    one place. ``by_class`` is an optional dict of class-name overrides
    -- left untyped here because the runtime has not yet decided how to
    apply per-class minima."""

    iou_match_threshold: float = 0.30
    max_misses: int = 15
    min_track_duration_s: float = 1.0
    parked_displacement_px: float = 50.0
    by_class: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)


# ----------------------------------------------------------------------
# Output + dashboard


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Where session artifacts land + HTML regen cadence."""

    dir: Path = Path("output")
    save_thumbnails: bool = True
    session_label_prefix: str = "session"
    html_idle_seconds: float = 10.0
    html_refresh_seconds: int = 15


@dataclass(frozen=True, slots=True)
class HttpConfig:
    """Built-in dashboard. ``enabled=False`` skips the aiohttp server."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080


# ----------------------------------------------------------------------
# Snap planner config


@dataclass(frozen=True, slots=True)
class SnapGateSpec:
    """Operator-traced road polygon + axis triggers, loaded from JSON.

    Mirrors the ``RoadGateConfig`` dataclass in
    ``device.snap_planner`` but is frozen (config-time) instead of
    mutable (runtime). Conversion happens in ``cli/run.py`` when the
    SnapPlanner is wired up.
    """

    polygon_frac: tuple[tuple[float, float], ...]
    trigger_t_prime: tuple[float, ...]
    t_usable_frac: tuple[float, float] = (0.0, 1.0)
    trigger_directions: tuple[str, ...] | None = None
    # Pipeline-mode (continuous in-band) firing. Both default to the
    # disabled values so a pre-pipeline camera.json loads unchanged.
    # When ``pipeline_interval_ms > 0`` the planner fires an additional
    # snap every N ms a track spends inside the usable t-band, on top
    # of the existing trigger-crossing fires.
    pipeline_interval_ms: int = 0
    pipeline_max_per_track: int = 10


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    """Reolink 4K snapshot tuning. Most knobs feed ``SnapPlannerConfig``;
    ``max_concurrent``, ``timeout_s``, ``keep_days``,
    ``cleanup_interval_s`` feed the snapshotter HTTP client (PR 3b)."""

    enabled: bool = True
    area_threshold_frac: float = 0.05
    max_snaps_per_track: int = 3
    min_sharpness: float = 0.0

    # Legacy peak/decay knobs (still consulted in non-road-gate mode).
    refire_growth_factor: float = 1.5
    decay_ratio: float = 0.92
    plateau_frames: int = 20
    max_wait_frames: int = 90
    exit_margin_frac: float = 0.04
    post_fire_cooldown_frames: int = 30

    # HTTP client (snapshotter, PR 3b).
    timeout_s: float = 5.0
    max_concurrent: int = 2

    # Periodic cleanup of stale *_main_*.jpg files.
    keep_days: int = 7
    cleanup_interval_s: float = 3600.0

    # Road-gate mode (the live deployment uses this; legacy modes only
    # fire when this is None).
    snap_gate: SnapGateSpec | None = None


# ----------------------------------------------------------------------
# Top-level


@dataclass(frozen=True, slots=True)
class StreetTrackerConfig:
    """Root config object. Load via :meth:`from_json_file` or
    :meth:`from_dict`."""

    camera: CameraConfig
    ports: PortsConfig
    streams: tuple[StreamConfig, ...]
    inference: InferenceConfig
    tracking: TrackingConfig
    output: OutputConfig
    http: HttpConfig
    snapshot: SnapshotConfig
    nano: NanoCompatConfig = dataclasses.field(default_factory=NanoCompatConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreetTrackerConfig:
        return _load(cls, data, "$")

    @classmethod
    def from_json_file(cls, path: str | Path) -> StreetTrackerConfig:
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def stream_by_name(self, name: str) -> StreamConfig:
        """Helper for the runtime: look up a stream entry by ``name``.

        Raises ``ConfigError`` (not ``KeyError``) so the error chain
        stays consistent with load-time validation messages.
        """
        for s in self.streams:
            if s.name == name:
                return s
        names = ", ".join(s.name for s in self.streams)
        raise ConfigError(f"stream {name!r} not declared in config (available: {names})")

    def stream_by_name_or_quality(self, key: str) -> StreamConfig:
        """Lookup by ``name`` first, then fall back to matching ``quality``.

        Migrated NanoTracker configs commonly carry descriptive stream
        names (``"RTSP sub (H.264)"``) while ``nano.preferred_stream``
        is the legacy short token (``"sub"``). Strict name lookup
        would fail; this fallback keeps such configs working. If the
        key matches multiple streams by quality, the first declared
        one wins -- the operator should rename their streams to
        unique values if that's ambiguous.
        """
        try:
            return self.stream_by_name(key)
        except ConfigError:
            pass
        for s in self.streams:
            if s.quality == key:
                return s
        names = ", ".join(f"{s.name!r}(quality={s.quality!r})" for s in self.streams)
        raise ConfigError(f"no stream matches {key!r} by name or quality (available: {names})")

    def validate_paths(self) -> None:
        """Cross-field validation that requires filesystem access.

        Kept separate from :meth:`from_json_file` so unit tests can
        load the example config (which references engine files that
        only exist on the Orin) without filesystem checks. The CLI
        entry points (``cli/run.py``, ``cli/batch.py``) call this
        right after load so config errors surface at startup with a
        clear JSON path -- *not* deep inside Ultralytics with a
        generic ``FileNotFoundError`` traceback after a 5-second
        engine-load attempt.

        Currently checks:

        * ``inference.engine_path`` resolves to an existing file. The
          path is interpreted relative to CWD (matching how
          Ultralytics' ``check_file`` resolves it).

        Raises ``ConfigError`` with the failed JSON path and the
        absolute resolved path on first failure.
        """
        engine = self.inference.engine_path
        resolved = engine if engine.is_absolute() else Path.cwd() / engine
        if not resolved.is_file():
            raise ConfigError(
                f"$.inference.engine_path: file {str(engine)!r} does not "
                f"exist (resolved to {resolved!s}). Build it with "
                f"`streettracker export-engine <weights.pt>` on this device "
                f"-- TRT engines are not portable across GPU architectures."
            )


# ----------------------------------------------------------------------
# Loader plumbing.
#
# Recursive descent over the @dataclass tree. Each node calls _load
# with the field type; _load dispatches on the type to either drop into
# a nested dataclass, coerce a list to a tuple, etc.


def _strip_comments(data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose names start with an underscore. Operators put
    documentation comments in the JSON file under ``_comment`` keys;
    those are not part of the schema."""
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _load(cls: Any, data: Any, path: str) -> Any:
    """Recursive type-driven loader.

    ``path`` is the JSON path of the current node, used purely for
    error messages.
    """
    # Optional / Union (e.g. ``SnapGateSpec | None``).
    origin = typing.get_origin(cls)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(cls) if a is not type(None)]
        if data is None:
            if type(None) in typing.get_args(cls):
                return None
            raise ConfigError(f"{path}: expected {cls}, got null")
        if len(args) != 1:
            raise ConfigError(f"{path}: complex unions not supported in config schema ({cls})")
        return _load(args[0], data, path)

    # Dataclass -- recurse over declared fields.
    if dataclasses.is_dataclass(cls) and isinstance(cls, type):
        if not isinstance(data, dict):
            raise ConfigError(
                f"{path}: expected object for {cls.__name__}, got {type(data).__name__}"
            )
        return _load_dataclass(cls, _strip_comments(data), path)

    # Tuple of fixed type (e.g. ``tuple[float, ...]``).
    if origin is tuple:
        if not isinstance(data, list):
            raise ConfigError(f"{path}: expected list for tuple field, got {type(data).__name__}")
        # Use a distinct name from the list-typed ``args`` bound in
        # the Union branch above; mypy otherwise complains about the
        # re-assignment narrowing list[Any] -> tuple[Any, ...].
        tuple_args = typing.get_args(cls)
        # ``tuple[X, ...]`` -- homogenous, variable-length
        if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
            return tuple(_load(tuple_args[0], v, f"{path}[{i}]") for i, v in enumerate(data))
        # ``tuple[X, Y]`` -- fixed-length, mixed types
        if len(data) != len(tuple_args):
            raise ConfigError(
                f"{path}: expected exactly {len(tuple_args)} elements, got {len(data)}"
            )
        return tuple(
            _load(t, v, f"{path}[{i}]")
            for i, (t, v) in enumerate(zip(tuple_args, data, strict=True))
        )

    # Dict (left untyped for now -- e.g. tracking.by_class).
    if origin is dict or cls is dict:
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected object, got {type(data).__name__}")
        return data

    # Path: JSON string -> pathlib.Path
    if cls is Path:
        if not isinstance(data, str):
            raise ConfigError(f"{path}: expected string for Path, got {type(data).__name__}")
        return Path(data)

    # Primitives.
    if cls is bool:
        # Check bool BEFORE int (bool is a subclass of int in Python).
        if not isinstance(data, bool):
            raise ConfigError(f"{path}: expected bool, got {type(data).__name__}")
        return data
    if cls is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise ConfigError(f"{path}: expected int, got {type(data).__name__}")
        return data
    if cls is float:
        # Accept int as float (common JSON convention -- 60 instead of 60.0).
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ConfigError(f"{path}: expected number, got {type(data).__name__}")
        return float(data)
    if cls is str:
        if not isinstance(data, str):
            raise ConfigError(f"{path}: expected string, got {type(data).__name__}")
        return data

    # Any / object -- pass through (rare; only fields explicitly typed Any).
    if cls is Any:
        return data

    raise ConfigError(f"{path}: unsupported config field type {cls!r}")


def _load_dataclass(cls: type, data: dict[str, Any], path: str) -> Any:
    """Pop known fields off the dict, recursing into each. Anything
    left over is an unknown key -- raise."""
    # ``from __future__ import annotations`` stores field types as
    # strings on the dataclass; resolve them through the module
    # globals once per call. typing.get_type_hints does the eval.
    try:
        resolved = typing.get_type_hints(cls)
    except Exception as exc:  # pragma: no cover - schema bug, not config bug
        raise ConfigError(
            f"{path}: could not resolve type hints for {cls.__name__}: {exc}"
        ) from exc
    kwargs: dict[str, Any] = {}
    remaining = dict(data)
    for f in fields(cls):
        field_type = resolved.get(f.name, f.type)
        if f.name in remaining:
            value = remaining.pop(f.name)
            kwargs[f.name] = _load(field_type, value, f"{path}.{f.name}")
        else:
            # Field missing -- only OK if there's a default.
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f"{path}.{f.name}: required field missing from config")
            # Default applies on construction; nothing to do here.
    if remaining:
        unknown = ", ".join(sorted(remaining))
        raise ConfigError(f"{path}: unknown key(s) for {cls.__name__}: {unknown}")
    return cls(**kwargs)
