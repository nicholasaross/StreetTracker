"""Local introspection for the control-panel radiator.

Everything here is read-only and answers the operator's decision questions from
on-disk state alone:

* **Sessions** — what's been pulled, how big, and how far through the
  enrichment chain (ALPR -> DVSA -> vehicles -> make/model) each one is.
* **Corpus** — the size + make distribution of the latest UK make-classifier
  crop set under ``runs/``.
* **Model** — metadata of the production classifier
  (``analysis/makemodel/models/makemodel_b0.pt``): backbone, input size,
  validation make@1, and the corpus it was trained on.
* **Training runs** — best val make@1 of each ``runs/uk_make_*`` candidate.
* **Recommendations** — *should I retrain? should I promote?* — each a verdict
  plus the evidence numbers behind it.

Reads are cached by file mtime so the background rescan is cheap even with many
sessions: a closed session's ``data.json`` is parsed exactly once. The module
is deliberately torch-light — model metadata comes from a tiny sidecar JSON
written at promotion time, falling back to a one-off (cached) checkpoint read
only when no sidecar exists yet.
"""

from __future__ import annotations

import contextlib
import functools
import json
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from streettracker.control.orin import parse_session_start

# ----------------------------------------------------------------------
# mtime-keyed read cache
# ----------------------------------------------------------------------

_CACHE: dict[tuple[str, str, int], Any] = {}
_CACHE_CAP = 4096

# Where the cache is persisted, once the server tells us the output root.
# ``None`` keeps the cache purely in-process, which is what the tests and any
# library caller get by default.
_CACHE_FILE = "control_introspect_cache.json"
_CACHE_PATH: Path | None = None
_CACHE_DIRTY = False
_CACHE_LOCK = threading.Lock()


def _by_mtime(path: Path, tag: str, loader: Callable[[], Any]) -> Any:
    """Memoise ``loader()`` against ``path``'s mtime under ``tag``.

    A closed session's files never change, so this turns a per-poll re-parse
    into a one-time cost; a live/being-pulled session re-reads only when its
    mtime advances. Returns ``None`` if the path has vanished.
    """
    global _CACHE_DIRTY
    try:
        mt = path.stat().st_mtime_ns
    except OSError:
        return None
    key = (str(path), tag, mt)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        known = key in _CACHE
    if not known:
        hit = loader()
        with _CACHE_LOCK:
            if len(_CACHE) > _CACHE_CAP:
                _CACHE.clear()
            _CACHE[key] = hit
            _CACHE_DIRTY = True
    return hit


# ----------------------------------------------------------------------
# On-disk persistence
#
# The cache above is per-process, so every panel restart re-globbed every
# session directory and re-parsed every data.json from scratch. That is
# O(files under output/) -- 400k files across 30 sessions by 2026-07-20 --
# and it is paid at the worst possible moment, because you restart the panel
# right after pulling a large session, when the OS file cache is cold and a
# pull or enrich is still hammering the same disk.
#
# Staleness is impossible by construction: the mtime is part of the key, so a
# changed file simply misses and re-loads. Entries whose file has vanished or
# moved on are dropped when saving, which bounds growth without a TTL.


def _is_jsonable(value: Any) -> bool:
    """Whether a cached value survives a JSON round trip.

    Most loaders return ints and plain dicts. A few (corpus, model checkpoint,
    training runs) return dataclasses, which don't -- and don't need to: there
    are a handful of them and they're cheap to rebuild, unlike the per-session
    globs and parses this exists for. Rather than maintain a list of which tags
    are safe, just try.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def load_cache(output_root: Path) -> int:
    """Point the cache at ``output_root`` and load any previous contents.

    Returns the number of entries restored. Never raises: a corrupt or absent
    cache file simply starts empty -- it is an optimisation, not state.
    """
    global _CACHE_PATH, _CACHE_DIRTY
    _CACHE_PATH = output_root / _CACHE_FILE
    try:
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(raw, list):
        return 0

    restored = 0
    with _CACHE_LOCK:
        for row in raw:
            if not isinstance(row, dict):
                continue
            try:
                key = (str(row["path"]), str(row["tag"]), int(row["mtime_ns"]))
            except (KeyError, TypeError, ValueError):
                continue
            _CACHE[key] = row.get("value")
            restored += 1
        _CACHE_DIRTY = False
    return restored


def save_cache() -> int:
    """Persist the cache if it changed. Returns the number of entries written.

    Drops entries whose file no longer has the cached mtime, so a session that
    was deleted or re-pulled doesn't accumulate dead keys forever.
    """
    global _CACHE_DIRTY
    if _CACHE_PATH is None or not _CACHE_DIRTY:
        return 0

    with _CACHE_LOCK:
        items = list(_CACHE.items())

    rows = []
    for (path_s, tag, mt), value in items:
        if not _is_jsonable(value):
            continue
        try:
            if Path(path_s).stat().st_mtime_ns != mt:
                continue  # superseded; the live key will be written instead
        except OSError:
            continue  # gone
        rows.append({"path": path_s, "tag": tag, "mtime_ns": mt, "value": value})

    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(_CACHE_PATH)
    except OSError:
        return 0  # a cache we can't write is not worth failing a poll over

    _CACHE_DIRTY = False
    return len(rows)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------


@dataclass(slots=True)
class SessionInfo:
    label: str
    start_unix: float | None
    end_unix: float | None
    duration_s: float | None
    n_tracks: int
    n_cars: int
    n_persons: int
    n_vehicle_snaps: int  # vehicle_*_main_*.jpg count on disk
    n_person_snaps: int  # person_*_main_*.jpg count on disk
    # Enrichment progress: which post-process passes have run for this session.
    has_alpr: bool
    # Crop path that produced the ALPR results, from the provenance stamp
    # in <session>_static_plates.json ("fullframe" since 2026-07-28).
    # None = no stamp: either ALPR never ran, or it ran with the pre-
    # fullframe pipeline and needs a re-run to recover R->L reads.
    alpr_crop_mode: str | None
    n_dvsa_labels: int  # 0 if dvsa-label hasn't run
    has_vehicles: bool
    has_makemodel: bool
    has_people: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_session_dirs(output_root: Path) -> list[Path]:
    """All ``session_*`` directories under ``output_root``, newest first.

    Session labels are timestamped, so the reverse lexical order is
    newest-to-oldest.
    """
    if not output_root.is_dir():
        return []
    return sorted((p for p in output_root.glob("session_*") if p.is_dir()), reverse=True)


def _count_snaps(session_dir: Path, prefix: str) -> int:
    # Cache by the directory's own mtime: it advances when snaps land (e.g.
    # during a pull), so a still-growing session recounts and a closed one
    # counts once. ``prefix`` is "vehicle" or "person".
    def _do() -> int:
        return sum(1 for _ in session_dir.glob(f"{prefix}_*_main_*.jpg"))

    return _by_mtime(session_dir, f"snapcount_{prefix}", _do) or 0


def _track_counts(session_dir: Path, label: str) -> dict[str, Any]:
    """Track totals + time span, parsed once per session (raw JSON, no
    dataclass construction — cheaper and forward-compatible)."""
    data_path = session_dir / f"{label}_data.json"
    events_path = session_dir / f"{label}_events.jsonl"

    def _from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
        starts = [r["time_start_unix"] for r in records if r.get("time_start_unix") is not None]
        ends = [r["time_end_unix"] for r in records if r.get("time_end_unix") is not None]
        return {
            "n_tracks": len(records),
            "n_cars": sum(1 for r in records if r.get("class_name") == "car"),
            "n_persons": sum(1 for r in records if r.get("class_name") == "person"),
            "start_unix": min(starts) if starts else None,
            "end_unix": max(ends) if ends else None,
        }

    if data_path.is_file():

        def _do() -> dict[str, Any]:
            raw = _read_json(data_path)
            return _from_records(raw if isinstance(raw, list) else [])

        return _by_mtime(data_path, "trackcounts", _do) or _empty_counts()

    if events_path.is_file():

        def _do_ev() -> dict[str, Any]:
            recs: list[dict[str, Any]] = []
            try:
                for raw_line in events_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if line:
                        with contextlib.suppress(json.JSONDecodeError):
                            recs.append(json.loads(line))
            except OSError:
                pass
            return _from_records(recs)

        return _by_mtime(events_path, "trackcounts", _do_ev) or _empty_counts()

    return _empty_counts()


def _empty_counts() -> dict[str, Any]:
    return {"n_tracks": 0, "n_cars": 0, "n_persons": 0, "start_unix": None, "end_unix": None}


def _dvsa_label_count(session_dir: Path, label: str) -> int:
    path = session_dir / f"{label}_dvsa_labels.json"
    if not path.is_file():
        return 0

    def _do() -> int:
        data = _read_json(path)
        labels = data.get("labels") if isinstance(data, dict) else None
        return len(labels) if isinstance(labels, dict) else 0

    return _by_mtime(path, "dvsacount", _do) or 0


def _alpr_crop_mode(session_dir: Path, label: str) -> str | None:
    """Crop-path provenance from ``<session>_static_plates.json``.

    The sidecar (written by alpr-run since the static filter shipped)
    carries a ``crop_mode`` stamp since 2026-07-28. Returns the stamp
    string, or ``None`` when the sidecar or stamp is absent -- i.e. the
    session's ALPR predates the fullframe crop path.
    """
    path = session_dir / f"{label}_static_plates.json"
    if not path.is_file():
        return None

    def _do() -> str | None:
        data = _read_json(path)
        mode = data.get("crop_mode") if isinstance(data, dict) else None
        return mode if isinstance(mode, str) else None

    return _by_mtime(path, "cropmode", _do)


def session_info(session_dir: Path) -> SessionInfo:
    """Build a :class:`SessionInfo` for one ``session_*`` directory."""
    label = session_dir.name
    counts = _track_counts(session_dir, label)
    start = counts["start_unix"]
    end = counts["end_unix"]
    # Prefer the meta's recorded session start (the true t0) over the first
    # finalised track; fall back to the label timestamp.
    meta_path = session_dir / f"{label}_meta.json"
    if meta_path.is_file():
        meta = _by_mtime(meta_path, "metastart", lambda: _read_json(meta_path))
        if isinstance(meta, dict) and meta.get("session_start_unix") is not None:
            start = float(meta["session_start_unix"])
    if start is None:
        start = parse_session_start(label)

    duration = (end - start) if (start is not None and end is not None and end >= start) else None

    return SessionInfo(
        label=label,
        start_unix=start,
        end_unix=end,
        duration_s=duration,
        n_tracks=counts["n_tracks"],
        n_cars=counts["n_cars"],
        n_persons=counts["n_persons"],
        n_vehicle_snaps=_count_snaps(session_dir, "vehicle"),
        n_person_snaps=_count_snaps(session_dir, "person"),
        has_alpr=(session_dir / f"{label}_alpr_by_track.json").is_file(),
        alpr_crop_mode=_alpr_crop_mode(session_dir, label),
        n_dvsa_labels=_dvsa_label_count(session_dir, label),
        has_vehicles=(session_dir / f"{label}_vehicles.json").is_file(),
        has_makemodel=(session_dir / f"{label}_makemodel_by_track.json").is_file(),
        has_people=(session_dir / f"{label}_people.json").is_file(),
    )


def list_local_sessions(output_root: Path) -> list[SessionInfo]:
    return [session_info(d) for d in discover_session_dirs(output_root)]


# ----------------------------------------------------------------------
# Training corpus (runs/uk_crops_*)
# ----------------------------------------------------------------------


@dataclass(slots=True)
class CorpusInfo:
    name: str
    path: str
    mtime: float
    n_crops: int
    n_makes: int
    n_cars: int
    per_make_cars: dict[str, int] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _corpus_from_manifest(d: Path) -> CorpusInfo | None:
    manifest = _read_json(d / "manifest.json")
    if not isinstance(manifest, dict):
        return None
    samples = manifest.get("samples") or []
    makes = manifest.get("makes") or []
    cars_by_make: dict[str, set[str]] = defaultdict(set)
    cars: set[str] = set()
    for s in samples:
        if not isinstance(s, dict):
            continue
        mk, car = s.get("make"), s.get("car")
        if mk is not None and car is not None:
            cars_by_make[mk].add(car)
            cars.add(car)
    return CorpusInfo(
        name=d.name,
        path=str(d),
        mtime=d.stat().st_mtime,
        n_crops=len(samples),
        n_makes=len(makes) if makes else len(cars_by_make),
        n_cars=len(cars),
        per_make_cars={m: len(c) for m, c in sorted(cars_by_make.items())},
    )


def latest_corpus(runs_dir: Path) -> CorpusInfo | None:
    """The most recently built ``runs/uk_crops_*`` that carries a manifest."""
    if not runs_dir.is_dir():
        return None
    candidates = [
        p for p in runs_dir.glob("uk_crops_*") if p.is_dir() and (p / "manifest.json").is_file()
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return _by_mtime(newest / "manifest.json", "corpus", lambda: _corpus_from_manifest(newest))


# ----------------------------------------------------------------------
# Production model + training runs
# ----------------------------------------------------------------------


@dataclass(slots=True)
class ModelInfo:
    path: str
    mtime: float
    source: str  # "sidecar" | "checkpoint" | "unavailable"
    arch: str | None = None
    input_size: int | None = None
    n_makes: int | None = None
    val_make_top1: float | None = None
    promoted_at: str | None = None
    trained_corpus: dict[str, Any] | None = None  # {n_cars, n_makes, n_crops, name}

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def production_model_path() -> Path:
    """Resolve ``makemodel_b0.pt`` from the installed package location."""
    import streettracker.analysis.makemodel as mm

    return Path(mm.__file__).resolve().parent / "models" / "makemodel_b0.pt"


def model_sidecar_path(model_path: Path) -> Path:
    """Sidecar metadata path for a model: ``makemodel_b0.pt`` ->
    ``makemodel_b0.meta.json`` (written by the panel at promotion time)."""
    return model_path.with_suffix(".meta.json")


def _model_from_checkpoint(model_path: Path) -> ModelInfo:
    """One-off torch read of a checkpoint's metadata (no sidecar yet).

    Reads only the small non-tensor keys; memory-maps the file so the big
    state-dict tensors aren't materialised. Any failure (torch missing,
    unreadable file) degrades to ``source="unavailable"`` rather than breaking
    the radiator.
    """
    try:
        import torch

        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
        except (TypeError, RuntimeError):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
        head_sizes = ckpt.get("head_sizes", {}) if isinstance(ckpt, dict) else {}
        n_makes = head_sizes.get("make")
        if n_makes is None and isinstance(meta.get("make_names"), list):
            n_makes = len(meta["make_names"])
        return ModelInfo(
            path=str(model_path),
            mtime=model_path.stat().st_mtime,
            source="checkpoint",
            arch=ckpt.get("arch") if isinstance(ckpt, dict) else None,
            input_size=meta.get("input_size"),
            n_makes=n_makes,
            val_make_top1=meta.get("val_make_top1"),
        )
    except Exception:
        # The radiator must survive any torch import / IO / format error.
        return ModelInfo(
            path=str(model_path), mtime=model_path.stat().st_mtime, source="unavailable"
        )


def model_info(model_path: Path | None = None, *, allow_torch: bool = True) -> ModelInfo | None:
    """Production-model metadata, sidecar-first then a cached checkpoint read.

    ``allow_torch=False`` skips the checkpoint fallback (used on the eager
    startup path so the first request isn't blocked on a torch import); the
    background rescan fills it in.
    """
    path = model_path or production_model_path()
    if not path.is_file():
        return None

    sidecar = model_sidecar_path(path)
    if sidecar.is_file():
        data = _by_mtime(sidecar, "modelsidecar", lambda: _read_json(sidecar))
        if isinstance(data, dict):
            return ModelInfo(
                path=str(path),
                mtime=path.stat().st_mtime,
                source="sidecar",
                arch=data.get("arch"),
                input_size=data.get("input_size"),
                n_makes=data.get("n_makes"),
                val_make_top1=data.get("val_make_top1"),
                promoted_at=data.get("promoted_at"),
                trained_corpus=data.get("trained_corpus"),
            )

    if not allow_torch:
        return ModelInfo(path=str(path), mtime=path.stat().st_mtime, source="unavailable")
    return _by_mtime(path, "modelckpt", lambda: _model_from_checkpoint(path))


@dataclass(slots=True)
class TrainingRunInfo:
    name: str
    path: str
    mtime: float
    best_epoch: int | None
    best_val_make_top1: float | None
    n_makes: int | None
    n_epochs: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_from_history(d: Path) -> TrainingRunInfo | None:
    summary = _read_json(d / "history.json")
    if not isinstance(summary, dict):
        return None
    makes = summary.get("makes")
    return TrainingRunInfo(
        name=d.name,
        path=str(d),
        mtime=(d / "history.json").stat().st_mtime,
        best_epoch=summary.get("best_epoch"),
        best_val_make_top1=summary.get("best_val_make_top1"),
        n_makes=len(makes) if isinstance(makes, list) else None,
        n_epochs=len(summary.get("history") or []),
    )


def training_runs(runs_dir: Path) -> list[TrainingRunInfo]:
    """Summaries of every ``runs/uk_make_*`` with a ``history.json``, newest
    first."""
    if not runs_dir.is_dir():
        return []
    out: list[TrainingRunInfo] = []
    for d in runs_dir.glob("uk_make_*"):
        if not d.is_dir() or not (d / "history.json").is_file():
            continue
        info = _by_mtime(d / "history.json", "run", functools.partial(_run_from_history, d))
        if info is not None:
            out.append(info)
    out.sort(key=lambda r: r.mtime, reverse=True)
    return out


# ----------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------


@dataclass(slots=True)
class Recommendation:
    kind: str  # "retrain" | "promote"
    verdict: str  # "recommend" | "hold" | "unknown"
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


# Defaults for "is corpus growth worth a retrain?" — any one tripping is enough.
RETRAIN_CARS_PCT = 0.15
RETRAIN_MAKES_DELTA = 3
RETRAIN_CROPS_DELTA = 5000
# Minimum val make@1 gain for a candidate to be worth promoting.
PROMOTE_MARGIN = 0.005


def retrain_recommendation(
    corpus: CorpusInfo | None,
    model: ModelInfo | None,
    *,
    cars_pct: float = RETRAIN_CARS_PCT,
    makes_delta: int = RETRAIN_MAKES_DELTA,
    crops_delta: int = RETRAIN_CROPS_DELTA,
) -> Recommendation:
    """Should the corpus be rebuilt + the CNN retrained?

    Compares the latest corpus against the corpus the production model was
    trained on (recorded in the model sidecar at promotion). With no recorded
    baseline — e.g. the current manually-promoted model — the verdict is
    ``unknown`` but the current corpus size is still surfaced.
    """
    if corpus is None:
        return Recommendation(
            "retrain", "unknown", "No UK crop corpus found under runs/ (build one first).", {}
        )
    current = {"n_cars": corpus.n_cars, "n_makes": corpus.n_makes, "n_crops": corpus.n_crops}
    trained = model.trained_corpus if model else None
    if not isinstance(trained, dict) or trained.get("n_cars") is None:
        return Recommendation(
            "retrain",
            "unknown",
            "Production model has no recorded training corpus — promote a model "
            "through the panel to start tracking growth. Current corpus: "
            f"{corpus.n_cars} cars / {corpus.n_makes} makes / {corpus.n_crops} crops.",
            {"current": current},
        )

    base_cars = max(int(trained.get("n_cars") or 0), 1)
    d_cars = corpus.n_cars - int(trained.get("n_cars") or 0)
    pct = d_cars / base_cars
    d_makes = corpus.n_makes - int(trained.get("n_makes") or 0)
    d_crops = corpus.n_crops - int(trained.get("n_crops") or 0)

    tripped = []
    if pct >= cars_pct:
        tripped.append(f"+{d_cars} cars ({pct:.0%} ≥ {cars_pct:.0%})")
    if d_makes >= makes_delta:
        tripped.append(f"+{d_makes} makes (≥ {makes_delta})")
    if d_crops >= crops_delta:
        tripped.append(f"+{d_crops} crops (≥ {crops_delta})")

    evidence = {
        "trained": trained,
        "current": current,
        "deltas": {"cars": d_cars, "cars_pct": round(pct, 4), "makes": d_makes, "crops": d_crops},
        "thresholds": {"cars_pct": cars_pct, "makes": makes_delta, "crops": crops_delta},
    }
    if tripped:
        return Recommendation(
            "retrain", "recommend", "Corpus grew: " + "; ".join(tripped) + ".", evidence
        )
    return Recommendation(
        "retrain",
        "hold",
        f"Corpus growth below threshold (+{d_cars} cars, +{d_makes} makes, +{d_crops} crops).",
        evidence,
    )


def promote_recommendation(
    model: ModelInfo | None,
    runs: list[TrainingRunInfo],
    *,
    margin: float = PROMOTE_MARGIN,
) -> Recommendation:
    """Should the best training run replace the production model?"""
    scored = [r for r in runs if r.best_val_make_top1 is not None]
    if not scored:
        return Recommendation("promote", "unknown", "No scored training runs under runs/.", {})
    best = max(scored, key=lambda r: r.best_val_make_top1 or 0.0)
    cand_acc = best.best_val_make_top1 or 0.0
    prod_acc = model.val_make_top1 if model else None
    candidate = {"name": best.name, "val_make_top1": cand_acc, "n_makes": best.n_makes}

    if prod_acc is None:
        return Recommendation(
            "promote",
            "unknown",
            f"Best candidate {best.name} scores make@1 {cand_acc:.3f}, but the production "
            "model's accuracy is unknown (no sidecar) — can't compare.",
            {"candidate": candidate, "production": None},
        )

    delta = cand_acc - prod_acc
    evidence = {
        "candidate": candidate,
        "production": {"val_make_top1": prod_acc},
        "delta": round(delta, 4),
        "margin": margin,
    }
    if delta >= margin:
        return Recommendation(
            "promote",
            "recommend",
            f"{best.name} make@1 {cand_acc:.3f} beats production {prod_acc:.3f} "
            f"(+{delta:.3f} ≥ {margin:.3f}).",
            evidence,
        )
    return Recommendation(
        "promote",
        "hold",
        f"Best candidate {best.name} ({cand_acc:.3f}) does not beat production "
        f"{prod_acc:.3f} by ≥ {margin:.3f}.",
        evidence,
    )


# ----------------------------------------------------------------------
# Composite snapshot for the API / cache
# ----------------------------------------------------------------------


def local_snapshot(
    output_root: Path,
    runs_dir: Path,
    model_path: Path | None = None,
    *,
    allow_torch: bool = True,
) -> dict[str, Any]:
    """All local introspection in one dict, ready to cache + serialise.

    Composes the session list, latest corpus, production-model metadata,
    training-run summaries, and the two recommendations.
    """
    sessions = list_local_sessions(output_root)
    corpus = latest_corpus(runs_dir)
    model = model_info(model_path, allow_torch=allow_torch)
    runs = training_runs(runs_dir)
    return {
        "sessions": [s.to_json_dict() for s in sessions],
        "corpus": corpus.to_json_dict() if corpus else None,
        "model": model.to_json_dict() if model else None,
        "runs": [r.to_json_dict() for r in runs],
        "recommendations": {
            "retrain": retrain_recommendation(corpus, model).to_json_dict(),
            "promote": promote_recommendation(model, runs).to_json_dict(),
        },
        "totals": _session_totals(sessions),
    }


def _session_totals(sessions: list[SessionInfo]) -> dict[str, int]:
    return {
        "n_sessions": len(sessions),
        "n_tracks": sum(s.n_tracks for s in sessions),
        "n_cars": sum(s.n_cars for s in sessions),
        "n_persons": sum(s.n_persons for s in sessions),
        "n_vehicle_snaps": sum(s.n_vehicle_snaps for s in sessions),
        "n_person_snaps": sum(s.n_person_snaps for s in sessions),
        "n_enriched": sum(1 for s in sessions if s.has_makemodel),
    }
