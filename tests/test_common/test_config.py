"""Coverage for ``streettracker.common.config``.

The loader is the single point of contact between operator-written JSON
and every downstream subsystem (snap planner, snapshotter, runtime
loop). A regression here -- silent default for a typoed key, accepted
string where an int was expected -- would manifest as a hard-to-debug
runtime divergence days later, so the tests exercise every error path
explicitly.

Where possible, tests use the *real* ``configs/camera.example.json``
checked into the repo as the happy-path fixture. That keeps the schema
in the loader and the schema in the example file from drifting apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streettracker.common.config import (
    CameraConfig,
    ConfigError,
    StreamConfig,
    StreetTrackerConfig,
    TrackingConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "configs" / "camera.example.json"


# ----------------------------------------------------------------------
# Happy path: the example file in the repo loads.


def test_example_config_loads_cleanly() -> None:
    """The checked-in ``configs/camera.example.json`` must always parse.

    If a new field is added to the dataclass schema without updating
    the example, this test fails -- which is the desired guard."""
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    assert isinstance(cfg, StreetTrackerConfig)
    assert cfg.camera.model == "Reolink RLC-1224A"
    assert cfg.camera.username == "admin"
    assert cfg.ports.http == 80
    assert cfg.ports.rtsp == 554
    assert len(cfg.streams) == 2
    assert cfg.inference.input_size == 640
    assert cfg.inference.vehicle_classes == (2,)
    assert cfg.snapshot.enabled is True
    assert cfg.snapshot.snap_gate is not None
    assert len(cfg.snapshot.snap_gate.trigger_t_prime) == 6
    assert cfg.snapshot.snap_gate.trigger_directions == (
        "forward",
        "forward",
        "forward",
        "reverse",
        "reverse",
        "reverse",
    )


def test_stream_by_name_helper() -> None:
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    sub = cfg.stream_by_name("sub")
    assert isinstance(sub, StreamConfig)
    assert sub.codec == "h264"
    assert sub.path == "/h264Preview_01_sub"


def test_stream_by_name_unknown_raises_config_error_not_keyerror() -> None:
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    with pytest.raises(ConfigError, match="stream 'nonexistent' not declared"):
        cfg.stream_by_name("nonexistent")


# ----------------------------------------------------------------------
# stream_by_name_or_quality -- fallback for migrated configs whose
# stream names don't match the legacy preferred_stream token.


def test_stream_by_name_or_quality_hits_name_first() -> None:
    """When the key matches a stream name exactly, that wins regardless
    of any quality match. Keeps deterministic behaviour for configs
    that follow the example's name == quality convention."""
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    s = cfg.stream_by_name_or_quality("sub")
    assert s.name == "sub"
    assert s.quality == "sub"


def test_stream_by_name_or_quality_falls_back_to_quality() -> None:
    """Reproduces the cutover failure: streams have descriptive names
    but preferred_stream is the legacy short token. Strict-name lookup
    would fail; quality fallback recovers."""
    data = _full_minimal()
    data["streams"] = [
        {
            "name": "RTSP main (H.265)",
            "quality": "main",
            "codec": "h265",
            "path": "/h264Preview_01_main",
        },
        {
            "name": "RTSP sub (H.264)",
            "quality": "sub",
            "codec": "h264",
            "path": "/h264Preview_01_sub",
        },
    ]
    cfg = StreetTrackerConfig.from_dict(data)
    s = cfg.stream_by_name_or_quality("sub")  # legacy token; no name matches
    assert s.name == "RTSP sub (H.264)"
    assert s.quality == "sub"


def test_stream_by_name_or_quality_raises_when_neither_matches() -> None:
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    with pytest.raises(ConfigError, match="no stream matches 'nope' by name or quality"):
        cfg.stream_by_name_or_quality("nope")


def test_stream_by_name_or_quality_quality_match_is_first_declared() -> None:
    """When multiple streams share the same quality (unusual but
    possible), the first declared wins. This is documented; ambiguous
    configs should be cleaned up rather than relying on order."""
    data = _full_minimal()
    data["streams"] = [
        {"name": "alpha", "quality": "sub", "codec": "h264", "path": "/a"},
        {"name": "beta", "quality": "sub", "codec": "h264", "path": "/b"},
    ]
    cfg = StreetTrackerConfig.from_dict(data)
    s = cfg.stream_by_name_or_quality("sub")
    assert s.name == "alpha"


# ----------------------------------------------------------------------
# validate_paths -- engine_path filesystem check.


def test_validate_paths_passes_when_engine_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: engine file exists at the resolved relative location."""
    data = _full_minimal()
    data["inference"]["engine_path"] = "fake.engine"
    cfg = StreetTrackerConfig.from_dict(data)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fake.engine").write_bytes(b"FAKE_TRT_ENGINE_BYTES")
    cfg.validate_paths()  # should not raise


def test_validate_paths_raises_with_json_path_when_engine_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces the second cutover failure: engine_path points at a
    file that wasn't built. The error message must include the JSON
    path AND the resolved absolute path so the operator can fix the
    config without spelunking Python tracebacks."""
    data = _full_minimal()
    data["inference"]["engine_path"] = "yolov8n_fp16.engine"
    cfg = StreetTrackerConfig.from_dict(data)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError) as exc_info:
        cfg.validate_paths()
    msg = str(exc_info.value)
    assert "$.inference.engine_path" in msg
    assert "yolov8n_fp16.engine" in msg
    assert "does not exist" in msg
    # The resolved absolute path should appear so the operator can
    # confirm CWD assumptions match their expectations.
    assert str(tmp_path) in msg
    # And the message should hint at the on-device build step.
    assert "export-engine" in msg


def test_validate_paths_accepts_absolute_engine_path(tmp_path: Path) -> None:
    """Absolute paths bypass the CWD-relative resolution -- common for
    operators who keep engines outside the repo dir."""
    data = _full_minimal()
    engine = tmp_path / "abs.engine"
    engine.write_bytes(b"FAKE")
    data["inference"]["engine_path"] = str(engine)
    cfg = StreetTrackerConfig.from_dict(data)
    cfg.validate_paths()  # no raise


# ----------------------------------------------------------------------
# Comment-stripping.


def test_underscore_prefixed_keys_are_stripped_silently() -> None:
    """Operators document JSON inline via ``_comment`` keys. The loader
    must accept any number of these in any block without complaint."""
    data = _full_minimal()
    data["_comment"] = "top-level note"
    data["_another_comment"] = ["multi", "line", "list"]
    data["camera"]["_doc"] = {"nested": "dict is fine too"}
    data["snapshot"]["_doc"] = "comments inside any block"
    result = StreetTrackerConfig.from_dict(data)
    assert result.camera.ip == "192.168.1.72"  # config still loads


# ----------------------------------------------------------------------
# Unknown-key strictness.


def test_unknown_top_level_key_raises() -> None:
    data = _full_minimal()
    data["snapsho"] = {}  # typo of "snapshot"
    with pytest.raises(ConfigError, match=r"unknown key.*snapsho"):
        StreetTrackerConfig.from_dict(data)


def test_unknown_nested_key_raises_with_json_path() -> None:
    data = _full_minimal()
    data["snapshot"]["min_sharpnss"] = 100.0  # typo of min_sharpness
    with pytest.raises(ConfigError) as excinfo:
        StreetTrackerConfig.from_dict(data)
    msg = str(excinfo.value)
    assert "$.snapshot" in msg
    assert "min_sharpnss" in msg


def test_unknown_deeply_nested_key_raises_with_json_path() -> None:
    """The path in the error message must walk into ``snap_gate``."""
    data = _full_minimal()
    data["snapshot"]["snap_gate"] = {
        "polygon_frac": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        "trigger_t_prime": [0.5],
        "trigger_direction": ["forward"],  # typo (missing trailing 's')
    }
    with pytest.raises(ConfigError) as excinfo:
        StreetTrackerConfig.from_dict(data)
    msg = str(excinfo.value)
    assert "snapshot.snap_gate" in msg
    assert "trigger_direction" in msg


# ----------------------------------------------------------------------
# Type strictness.


def test_int_field_rejects_string() -> None:
    data = _full_minimal()
    data["ports"]["http"] = "80"  # string, not int
    with pytest.raises(ConfigError, match="expected int"):
        StreetTrackerConfig.from_dict(data)


def test_bool_field_rejects_int_zero() -> None:
    """bool is a subclass of int in Python -- we must reject the 0/1 form."""
    data = _full_minimal()
    data["snapshot"]["enabled"] = 0
    with pytest.raises(ConfigError, match="expected bool"):
        StreetTrackerConfig.from_dict(data)


def test_int_field_rejects_bool() -> None:
    """Mirror of the above: ints must not accept True/False either."""
    data = _full_minimal()
    data["ports"]["http"] = True
    with pytest.raises(ConfigError, match="expected int"):
        StreetTrackerConfig.from_dict(data)


def test_float_field_accepts_int_value() -> None:
    """JSON commonly writes ``5`` where a float was expected. Accept."""
    data = _full_minimal()
    data["snapshot"]["timeout_s"] = 5  # no decimal
    cfg = StreetTrackerConfig.from_dict(data)
    assert isinstance(cfg.snapshot.timeout_s, float)
    assert cfg.snapshot.timeout_s == 5.0


def test_string_field_rejects_list() -> None:
    data = _full_minimal()
    data["camera"]["ip"] = ["192", "168", "1", "72"]
    with pytest.raises(ConfigError, match="expected string"):
        StreetTrackerConfig.from_dict(data)


def test_dataclass_field_rejects_array() -> None:
    """A field declared as ``CameraConfig`` must reject an array."""
    data = _full_minimal()
    data["camera"] = ["192.168.1.72", "admin", "pw"]
    with pytest.raises(ConfigError, match="expected object for CameraConfig"):
        StreetTrackerConfig.from_dict(data)


# ----------------------------------------------------------------------
# Required vs optional fields.


def test_missing_required_field_raises() -> None:
    data = _full_minimal()
    del data["camera"]["password"]
    with pytest.raises(ConfigError, match=r"\$.camera.password.*required"):
        StreetTrackerConfig.from_dict(data)


def test_missing_optional_field_uses_default() -> None:
    """``camera.model`` defaults to empty string; leaving it out is OK."""
    data = _full_minimal()
    del data["camera"]["model"]
    cfg = StreetTrackerConfig.from_dict(data)
    assert cfg.camera.model == ""


def test_entire_nano_block_optional_via_default_factory() -> None:
    """The ``nano`` compat block is optional -- new clean configs may
    omit it entirely. Defaults to a NanoCompatConfig with all defaults."""
    data = _full_minimal()
    del data["nano"]
    cfg = StreetTrackerConfig.from_dict(data)
    assert cfg.nano.preferred_stream == "sub"
    assert cfg.nano.decoder == "nvv4l2decoder"


def test_snap_gate_optional_in_snapshot() -> None:
    """A snapshot block with no ``snap_gate`` key falls back to legacy
    peak/decay mode (``None`` means road-gate is not configured)."""
    data = _full_minimal()
    assert "snap_gate" not in data["snapshot"]  # sanity
    cfg = StreetTrackerConfig.from_dict(data)
    assert cfg.snapshot.snap_gate is None


def test_snap_gate_null_in_snapshot() -> None:
    """Explicit JSON null for the optional gate is also acceptable."""
    data = _full_minimal()
    data["snapshot"]["snap_gate"] = None
    cfg = StreetTrackerConfig.from_dict(data)
    assert cfg.snapshot.snap_gate is None


# ----------------------------------------------------------------------
# Tuple coercion.


def test_lists_are_coerced_to_tuples_in_loaded_config() -> None:
    """``vehicle_classes`` is declared as ``tuple[int, ...]`` -- JSON
    sends a list. Make sure the loaded value is a tuple (i.e., hashable
    and immutable, matching the frozen dataclass posture)."""
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    assert isinstance(cfg.inference.vehicle_classes, tuple)
    assert isinstance(cfg.streams, tuple)
    assert isinstance(cfg.snapshot.snap_gate.polygon_frac, tuple)
    assert isinstance(cfg.snapshot.snap_gate.polygon_frac[0], tuple)


def test_nested_tuple_of_pairs() -> None:
    """``polygon_frac: tuple[tuple[float, float], ...]`` -- two levels."""
    data = _full_minimal()
    data["snapshot"]["snap_gate"] = {
        "polygon_frac": [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]],
        "trigger_t_prime": [0.3, 0.7],
    }
    cfg = StreetTrackerConfig.from_dict(data)
    sg = cfg.snapshot.snap_gate
    assert sg is not None
    assert sg.polygon_frac == ((0.0, 0.0), (1.0, 0.5), (0.5, 1.0))
    assert sg.trigger_t_prime == (0.3, 0.7)
    assert sg.t_usable_frac == (0.0, 1.0)  # default
    assert sg.trigger_directions is None


def test_snap_gate_pipeline_fields_default_to_disabled() -> None:
    """A snap_gate block without pipeline fields must default to the
    disabled values, preserving pre-pipeline camera.json compatibility."""
    data = _full_minimal()
    data["snapshot"]["snap_gate"] = {
        "polygon_frac": [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]],
        "trigger_t_prime": [0.5],
    }
    cfg = StreetTrackerConfig.from_dict(data)
    sg = cfg.snapshot.snap_gate
    assert sg is not None
    assert sg.pipeline_interval_ms == 0
    assert sg.pipeline_max_per_track == 10


def test_snap_gate_pipeline_fields_round_trip() -> None:
    """When pipeline_interval_ms and pipeline_max_per_track are set in
    JSON they must reach the loaded config unchanged."""
    data = _full_minimal()
    data["snapshot"]["snap_gate"] = {
        "polygon_frac": [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]],
        "trigger_t_prime": [0.5],
        "pipeline_interval_ms": 400,
        "pipeline_max_per_track": 8,
    }
    cfg = StreetTrackerConfig.from_dict(data)
    sg = cfg.snapshot.snap_gate
    assert sg is not None
    assert sg.pipeline_interval_ms == 400
    assert sg.pipeline_max_per_track == 8


def test_fixed_length_tuple_validates_arity() -> None:
    """``t_usable_frac: tuple[float, float]`` (fixed 2-tuple)."""
    data = _full_minimal()
    data["snapshot"]["snap_gate"] = {
        "polygon_frac": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        "trigger_t_prime": [0.5],
        "t_usable_frac": [0.1, 0.5, 0.9],  # too many
    }
    with pytest.raises(ConfigError, match="expected exactly 2 elements"):
        StreetTrackerConfig.from_dict(data)


# ----------------------------------------------------------------------
# Path coercion.


def test_engine_path_is_loaded_as_pathlib_path() -> None:
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    assert isinstance(cfg.inference.engine_path, Path)
    assert isinstance(cfg.output.dir, Path)


def test_engine_path_rejects_non_string() -> None:
    data = _full_minimal()
    data["inference"]["engine_path"] = 12345
    with pytest.raises(ConfigError, match="expected string for Path"):
        StreetTrackerConfig.from_dict(data)


# ----------------------------------------------------------------------
# Immutability.


def test_loaded_config_is_frozen() -> None:
    """Try mutating the loaded config -- frozen dataclass should refuse."""
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    with pytest.raises(dataclasses_FrozenInstanceError):
        cfg.camera.ip = "10.0.0.1"  # type: ignore[misc]


# ----------------------------------------------------------------------
# JSON-file path edge cases.


def test_from_json_file_with_str_path(tmp_path: Path) -> None:
    """``from_json_file`` must accept ``str`` paths, not just ``Path``."""
    out = tmp_path / "camera.json"
    out.write_text(json.dumps(_full_minimal()))
    cfg = StreetTrackerConfig.from_json_file(str(out))
    assert cfg.camera.ip == "192.168.1.72"


def test_from_json_file_with_invalid_json_propagates_json_decode_error(
    tmp_path: Path,
) -> None:
    """JSON syntax errors surface as ``json.JSONDecodeError``, not as
    ``ConfigError``. That preserves the line / column info from the
    JSON parser."""
    out = tmp_path / "camera.json"
    out.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        StreetTrackerConfig.from_json_file(out)


# ----------------------------------------------------------------------
# Helpers.


def _build_minimal_camera() -> CameraConfig:
    return CameraConfig(ip="192.168.1.72", username="admin", password="p")


def _full_minimal() -> dict:
    """Smallest dict that satisfies every required field on the
    schema. Tests mutate copies of this to provoke specific errors."""
    return {
        "camera": {
            "model": "Test",
            "ip": "192.168.1.72",
            "username": "admin",
            "password": "secret",
        },
        "ports": {"http": 80, "rtsp": 554},
        "streams": [
            {"name": "sub", "quality": "sub", "codec": "h264", "path": "/h264Preview_01_sub"},
        ],
        "nano": {
            "preferred_stream": "sub",
            "rtsp_transport": "tcp",
            "latency_ms": 200,
            "decoder": "nvv4l2decoder",
        },
        "inference": {
            "engine_path": "yolov8m.engine",
            "input_size": 640,
            "conf_threshold": 0.30,
            "iou_threshold": 0.45,
            "vehicle_classes": [2],
        },
        "tracking": {
            "iou_match_threshold": 0.30,
            "max_misses": 15,
            "min_track_duration_s": 1.0,
            "parked_displacement_px": 50.0,
            "by_class": {},
        },
        "output": {
            "dir": "output",
            "save_thumbnails": True,
            "session_label_prefix": "session",
            "html_idle_seconds": 10.0,
            "html_refresh_seconds": 15,
        },
        "http": {"enabled": True, "host": "0.0.0.0", "port": 8080},
        "snapshot": {
            "enabled": True,
            "area_threshold_frac": 0.05,
            "max_snaps_per_track": 3,
            "min_sharpness": 0.0,
            "post_fire_cooldown_frames": 30,
            "max_concurrent": 2,
            "timeout_s": 5.0,
            "keep_days": 7,
            "cleanup_interval_s": 3600.0,
        },
    }


# Local alias so a single ``import dataclasses`` doesn't pollute the
# top-level namespace just for one rare assertion.
from dataclasses import FrozenInstanceError as dataclasses_FrozenInstanceError  # noqa: E402


def test_snap_gate_accepts_pipeline_t_usable_by_direction() -> None:
    """Schema-additive: the per-direction pipeline band key loads. The
    strict loader passes dict values through raw (lists stay lists);
    range/key validation happens at planner construction."""
    data = _full_minimal()
    data["snapshot"]["snap_gate"] = {
        "polygon_frac": [[0.1, 0.4], [0.9, 0.4], [0.9, 0.6], [0.1, 0.6]],
        "trigger_t_prime": [0.5],
        "pipeline_t_usable_by_direction": {
            "forward": [0.10, 0.20],
            "reverse": [0.25, 0.60],
        },
    }
    cfg = StreetTrackerConfig.from_dict(data)
    spec = cfg.snapshot.snap_gate
    assert spec is not None
    assert spec.pipeline_t_usable_by_direction == {
        "forward": [0.10, 0.20],
        "reverse": [0.25, 0.60],
    }


def test_snap_gate_per_direction_band_defaults_to_none() -> None:
    """Configs written before the field existed keep loading (the live
    camera.json must parse unchanged on the deploy's restart-old-config
    step)."""
    cfg = StreetTrackerConfig.from_json_file(EXAMPLE_PATH)
    spec = cfg.snapshot.snap_gate
    assert spec is not None
    assert spec.pipeline_t_usable_by_direction is None


# ----------------------------------------------------------------------
# Per-class finalize minima (TrackingConfig.by_class).
#
# The key was accepted by the loader but silently ignored by the runtime
# until 2026-07-20, so operator-configured person thresholds never took
# effect and person tracks were judged at the car-scale defaults.


def _tracking(**kw: object) -> TrackingConfig:
    return TrackingConfig(**kw)  # type: ignore[arg-type]


def test_minima_for_falls_back_to_top_level_without_override() -> None:
    cfg = _tracking(min_track_duration_s=1.0, parked_displacement_px=50.0)
    assert cfg.minima_for(2, "car") == (1.0, 50.0)


def test_minima_for_applies_class_id_override() -> None:
    """The live camera.json keys by COCO id string ("0" = person)."""
    cfg = _tracking(
        min_track_duration_s=1.0,
        parked_displacement_px=50.0,
        by_class={"0": {"min_track_duration_s": 0.5, "parked_displacement_px": 30}},
    )
    assert cfg.minima_for(0, "person") == (0.5, 30.0)
    assert cfg.minima_for(2, "car") == (1.0, 50.0)


def test_minima_for_applies_class_name_override() -> None:
    cfg = _tracking(
        min_track_duration_s=1.0,
        parked_displacement_px=50.0,
        by_class={"person": {"parked_displacement_px": 20}},
    )
    # Name matches; the unspecified key keeps the top-level value.
    assert cfg.minima_for(0, "person") == (1.0, 20.0)


def test_minima_for_prefers_class_id_over_name() -> None:
    cfg = _tracking(
        by_class={
            "0": {"parked_displacement_px": 30},
            "person": {"parked_displacement_px": 99},
        },
    )
    assert cfg.minima_for(0, "person")[1] == 30.0


def test_minima_for_survives_a_malformed_override() -> None:
    """A bad value must not take the runtime down mid-session."""
    cfg = _tracking(
        min_track_duration_s=1.0,
        parked_displacement_px=50.0,
        by_class={"0": {"parked_displacement_px": "thirty"}},
    )
    assert cfg.minima_for(0, "person") == (1.0, 50.0)
