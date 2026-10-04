"""The shared plate gate: which plate reads count as a car's identity.

Two modes, chosen in ``configs/alpr.json`` (``plate_gate``):

- ``"conf"`` (the default when the file doesn't say): a read counts when its
  min-character OCR confidence is at least ``plate_conf_threshold`` (0.90).
- ``"combined"`` (review E1.4, 2026-10-04): a read counts when its confidence
  is at least ``supported_conf_threshold`` (0.80) **and** it is corroborated:
  another snap of the same track read the same string (``n_agree`` > 0 on the
  rollup's best read) or at least ``min_support`` tracks anywhere have this
  plate as their best read. Measured over 54k best reads it labels 15.5 %
  more tracks than ``conf >= 0.90`` with a lower not-on-register rate
  (1.4 % vs 2.1 %), because a read that one other observation agrees with is
  far more trustworthy than a confident lone read.

Plate colour (``plate_colour``) is applied upstream: reads whose plate colour
contradicts the track's direction are ``colour_suspect`` and never become a
track's best read, so every gate sees colour-consistent reads only.

Used by dvsa-label (which tracks get labelled), vehicles + the showcase (plate
identity) and the stats page's fastest-car plates.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from streettracker.analysis.alpr import base

GATE_MODES = ("conf", "combined")
DEFAULT_SUPPORTED_CONF = 0.8
DEFAULT_MIN_SUPPORT = 2
_KEYS = ("plate_conf_threshold", "plate_gate", "supported_conf_threshold", "min_support")


@dataclass(frozen=True, slots=True)
class PlateGate:
    mode: str = "conf"
    conf_threshold: float = base.DEFAULT_PLATE_CONF_THRESHOLD
    supported_conf: float = DEFAULT_SUPPORTED_CONF
    min_support: int = DEFAULT_MIN_SUPPORT

    @property
    def min_conf(self) -> float:
        """The lowest confidence any read can pass at (for pre-filters)."""
        return self.supported_conf if self.mode == "combined" else self.conf_threshold

    @property
    def needs_support(self) -> bool:
        return self.mode == "combined"

    def passes(self, conf: float | None, *, n_agree: int = 0, support: int = 0) -> bool:
        """Whether a read clears the gate. ``n_agree``: other snaps of the
        same track that read the same string; ``support``: tracks anywhere
        whose best read is this plate (including this one)."""
        c = float(conf or 0.0)
        if self.mode == "combined":
            return c >= self.supported_conf and (n_agree > 0 or support >= self.min_support)
        return c >= self.conf_threshold

    def describe(self) -> str:
        if self.mode == "combined":
            return (
                f"combined: conf >= {self.supported_conf:g} and (a snap agrees or "
                f">= {self.min_support} tracks read the plate)"
            )
        return f"conf >= {self.conf_threshold:g}"

    def to_json(self) -> dict[str, Any]:
        if self.mode == "combined":
            return {
                "mode": self.mode,
                "supported_conf_threshold": self.supported_conf,
                "min_support": self.min_support,
            }
        return {"mode": self.mode, "plate_conf_threshold": self.conf_threshold}


def resolve_plate_gate(override_conf: float | None = None) -> tuple[PlateGate, str]:
    """The plate gate and where it came from.

    ``override_conf`` (a ``--conf-threshold`` flag) forces the plain
    ``conf`` gate at that value. Otherwise ``configs/alpr.json``
    (:data:`base.PLATE_CONF_CONFIG`, read at call time) decides; no file means
    the default ``conf`` gate. A file that can't be used raises
    ``ValueError`` rather than silently undoing a calibration.
    """
    if override_conf is not None:
        return (
            PlateGate(conf_threshold=base._valid_threshold(override_conf, "command line")),
            "command line",
        )
    path = base.PLATE_CONF_CONFIG
    if not path.is_file():
        return PlateGate(), "default"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable ({exc})") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a JSON object")
    unknown = sorted(k for k in cfg if k not in _KEYS and not k.startswith("_"))
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}; expected some of {list(_KEYS)}")
    mode = cfg.get("plate_gate", "conf")
    if mode not in GATE_MODES:
        raise ValueError(f"{path}: plate_gate must be one of {list(GATE_MODES)}, got {mode!r}")
    conf = (
        base._valid_threshold(cfg["plate_conf_threshold"], str(path))
        if "plate_conf_threshold" in cfg
        else base.DEFAULT_PLATE_CONF_THRESHOLD
    )
    supported = (
        base._valid_threshold(cfg["supported_conf_threshold"], str(path))
        if "supported_conf_threshold" in cfg
        else DEFAULT_SUPPORTED_CONF
    )
    min_support = cfg.get("min_support", DEFAULT_MIN_SUPPORT)
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError(f"{path}: min_support must be a whole number >= 1, got {min_support!r}")
    return PlateGate(mode, conf, supported, min_support), str(path)


def plate_gate(override_conf: float | None = None) -> PlateGate:
    """:func:`resolve_plate_gate` without the source."""
    return resolve_plate_gate(override_conf)[0]


# (output_root, ((file, mtime_ns), ...)) -> support counts.
_SUPPORT_CACHE: dict[tuple[str, tuple[tuple[str, int], ...]], Counter[str]] = {}


def load_plate_support(output_root: Path) -> Counter[str]:
    """``plate -> number of tracks`` whose UK-shaped best read is that plate,
    over every ``session_*/<session>_alpr_by_track.json`` under
    ``output_root``. Cached until any rollup file changes."""
    from streettracker.analysis.dvsa import is_canonical_uk_plate

    files = sorted(output_root.glob("session_*/session_*_alpr_by_track.json"))
    key_files: list[tuple[str, int]] = []
    for f in files:
        try:
            key_files.append((str(f), f.stat().st_mtime_ns))
        except OSError:
            continue
    key = (str(output_root.resolve()), tuple(key_files))
    hit = _SUPPORT_CACHE.get(key)
    if hit is not None:
        return hit
    support: Counter[str] = Counter()
    for name, _m in key_files:
        try:
            tracks = json.loads(Path(name).read_text(encoding="utf-8")).get("tracks", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        for t in tracks:
            best = t.get("best_preferred") if isinstance(t, dict) else None
            if not isinstance(best, dict):
                continue
            plate = str(best.get("ocr_text") or "").replace(" ", "").upper()
            if plate and is_canonical_uk_plate(plate):
                support[plate] += 1
    _SUPPORT_CACHE.clear()  # one output root in practice; keep only the newest
    _SUPPORT_CACHE[key] = support
    return support


def read_passes(
    gate: PlateGate,
    read: Mapping[str, Any],
    support: Mapping[str, int] | None,
    *,
    n_agree: int | None = None,
) -> bool:
    """Gate one read dict (a rollup best read or a per-image read).
    ``n_agree`` defaults to the read's own ``n_agree`` field (rollups since
    2026-10-04; absent = 0)."""
    plate = str(read.get("ocr_text") or "").replace(" ", "").upper()
    agree = int(read.get("n_agree") or 0) if n_agree is None else n_agree
    return gate.passes(
        read.get("ocr_conf"),
        n_agree=agree,
        support=(support or {}).get(plate, 0),
    )
