"""Head-to-head comparison of a candidate vs production CNN on shared held-out cars.

Works for all three heads (``--target make | colour | body_type``); the text
below uses make, the original case.

A training run's val make@1 is only comparable with production's when both
were scored on the same cars, with the same makes, from the same kind of
crops. None of that held once the corpus moved to plate-anchored crops
(2026-09-24): production's recorded 0.451 was scored on stale-hint crops
(~2/3 of which missed the car), a different val split and 45 makes. This
module scores both models on ONE held-out set instead:

* **Cars** -- the candidate corpus's by-car val split (the exact cars the
  candidate never trained on), minus every car in production's training
  corpus (so production isn't scored on cars it learnt). With
  ``--eval-session`` they instead come from the DVSA labels of the named
  sessions, minus every car in production's AND the candidate's training
  corpora: fresh cars neither model has seen. Use it when the two corpora
  share most of their cars, which leaves the val-split mode only a handful
  (2026-10-01: 67 cars, because 93 % of the rebuilt corpus's cars were
  already in production's). ``--include-trained-cars`` keeps the cars the
  models trained on: new PASSES of known cars, from sessions recorded after
  every model's training data. On a street of regulars (88 % of one day's
  plated cars were already in the corpora) that is most of what the
  classifier sees, and it gives far more cars than the fresh-only set; it
  rewards recognising known cars, so read it beside the fresh-car result.
* **Tracks** -- each held-out car's DVSA-labelled tracks (capped per car so
  a few regulars can't dominate), and ALL of each track's 4K snaps, as the
  ``makemodel`` command would classify them.
* **Crops** -- each model with its own crop settings
  (:func:`vehicle_locator.resolve_crop_settings`): production "as deployed",
  plus production on clean fullframe crops, plus the candidate.

The headline metric is per-track make@1 (a track = one pass; a track with no
prediction counts as wrong), with the candidate-minus-production delta and a
95 % bootstrap CI resampled BY CAR (tracks of one car aren't independent).
Per-car accuracy and a shared-makes-only view are reported alongside.

For colour, a second "grouped" score also counts near-misses inside one
colour family as right (``vehicles._colour_group``: white/silver/grey ->
light, ...), the metric the colour head was first judged on.

Caveat: held-out cars are plated (DVSA-labelled) cars, which skew nearer /
daylit / sharper than the unplated majority the classifier mostly serves.
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from streettracker.analysis.alpr.base import atomic_write_text

_SAMPLE_NAME_RE = re.compile(
    r"^(?P<car>.+)_(?P<session>session_\d{8}_\d{6})_(?P<tid>\d+)_(?P<n>\d+)\.jpg$"
)
DEFAULT_MAX_TRACKS_PER_CAR = 3
_BOOTSTRAP_ITERS = 2000
_LEGACY_TRAIN_PAD = 0.1


@dataclasses.dataclass(slots=True)
class HeldOutCar:
    car: str
    label: str  # the DVSA-derived truth for the target (make / colour / body type)
    tracks: list[tuple[str, int]]  # (session name, track id)


TARGETS = ("make", "colour", "body_type")
_NOUN = {"make": "make", "colour": "colour", "body_type": "body_type"}


@dataclasses.dataclass(slots=True)
class Contender:
    """One model + its crop settings, behind a uniform (label, conf) predict."""

    crop_mode: str
    pad_frac: float
    classes: set[str]
    predict: Any  # Callable[[np.ndarray, box | None], tuple[str, float] | None]


def load_contender(
    target: str,
    checkpoint: Path,
    *,
    device: str,
    crop_mode: str = "auto",
    pad_frac: float | None = None,
) -> Contender:
    """Load ``checkpoint`` as the ``target`` head's classifier (each head's
    own inference class, so crop mode + pad follow its checkpoint)."""
    if target == "make":
        from streettracker.analysis.makemodel.infer import MakeModelClassifier

        mm = MakeModelClassifier(checkpoint, device=device, crop_mode=crop_mode, pad_frac=pad_frac)
        if not mm.make_only:
            raise ValueError(f"{checkpoint} is not a UK make-only model")

        def predict_make(image: Any, box: Any) -> tuple[str, float] | None:
            cands = mm.classify(image, box)
            return (cands[0].make, cands[0].conf) if cands else None

        return Contender(mm.crop_mode, mm.pad_frac, set(mm.make_names), predict_make)
    if target == "colour":
        from streettracker.analysis.makemodel.colour_infer import ColourClassifier

        cc = ColourClassifier(checkpoint, device=device, crop_mode=crop_mode, pad_frac=pad_frac)
        return Contender(cc.crop_mode, cc.pad_frac, set(cc.colours), cc.classify)
    if target == "body_type":
        from streettracker.analysis.makemodel.bodytype_infer import BodyTypeClassifier

        bc = BodyTypeClassifier(checkpoint, device=device, crop_mode=crop_mode, pad_frac=pad_frac)
        return Contender(bc.crop_mode, bc.pad_frac, set(bc.body_types), bc.classify)
    raise ValueError(f"target must be one of {TARGETS}, got {target!r}")


def default_production(target: str) -> Path:
    """The production checkpoint each head's inference command uses."""
    if target == "colour":
        from streettracker.analysis.makemodel.colour_infer import DEFAULT_MODEL
    elif target == "body_type":
        from streettracker.analysis.makemodel.bodytype_infer import DEFAULT_MODEL
    else:
        from streettracker.analysis.makemodel.infer import DEFAULT_MODEL
    return DEFAULT_MODEL


def _colour_group(colour: str) -> str:
    from streettracker.analysis.vehicles import _colour_group as group

    return group(colour) or ""


def _manifest(corpus_dir: Path) -> dict[str, Any]:
    return json.loads((corpus_dir / "manifest.json").read_text())


def corpus_cars(corpus_dir: Path) -> set[str]:
    """Every car (plate) in a corpus manifest."""
    return {s["car"] for s in _manifest(corpus_dir)["samples"]}


def _sample_tracks(
    label_of: dict[str, str],
    tracks_of: dict[str, set[tuple[str, int]]],
    *,
    seed: int,
    max_tracks_per_car: int,
    max_cars: int,
) -> list[HeldOutCar]:
    """Cap each car's tracks and optionally the number of cars, seeded so
    every model compared on the same cars gets the same tracks."""
    rng = random.Random(f"compare:{seed}")
    out = []
    for car in sorted(label_of):
        tracks = sorted(tracks_of[car])
        if max_tracks_per_car and len(tracks) > max_tracks_per_car:
            tracks = sorted(rng.sample(tracks, max_tracks_per_car))
        out.append(HeldOutCar(car, label_of[car], tracks))
    if max_cars and len(out) > max_cars:
        out = sorted(rng.sample(out, max_cars), key=lambda c: c.car)
    return out


def held_out_cars(
    corpus_dir: Path,
    *,
    exclude_cars: set[str] | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
    max_tracks_per_car: int = DEFAULT_MAX_TRACKS_PER_CAR,
    max_cars: int = 0,
    target: str = "make",
) -> list[HeldOutCar]:
    """The candidate corpus's val cars for ``target`` (the same by-car split,
    stratified by the same label, as the trainer used), minus
    ``exclude_cars``, each with up to ``max_tracks_per_car`` tracks. Cars
    with no label for the target (e.g. an uncovered body-type model) are
    dropped, exactly as the trainer drops them."""
    from streettracker.analysis.makemodel.uk_dataset import split_val_cars

    samples = [s for s in _manifest(corpus_dir)["samples"] if s.get(target)]
    val = split_val_cars(samples, label_field=target, val_frac=val_frac, seed=seed)
    exclude = exclude_cars or set()
    label_of: dict[str, str] = {}
    tracks_of: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for s in samples:
        car = s["car"]
        if car not in val or car in exclude:
            continue
        m = _SAMPLE_NAME_RE.match(Path(s["path"]).name)
        if not m:
            continue
        label_of[car] = s[target]
        tracks_of[car].add((m["session"], int(m["tid"])))
    return _sample_tracks(
        label_of, tracks_of, seed=seed, max_tracks_per_car=max_tracks_per_car, max_cars=max_cars
    )


def target_label(row: dict[str, Any], target: str) -> str:
    """A DVSA label row's truth for ``target``, derived exactly as
    ``makemodel-build-uk`` labels a corpus crop ("" when it has none)."""
    from streettracker.analysis.makemodel.bodytype import body_type_for, normalize_make
    from streettracker.analysis.makemodel.colour import colour_class_for

    if target == "make":
        return normalize_make(row.get("make"))
    if target == "colour":
        return colour_class_for(row.get("primary_colour"))
    if target == "body_type":
        return body_type_for(row.get("make"), row.get("model"))
    raise ValueError(f"target must be one of {TARGETS}, got {target!r}")


def session_cars(
    output_root: Path,
    sessions: list[str],
    *,
    exclude_cars: set[str] | None = None,
    target: str = "make",
    seed: int = 0,
    max_tracks_per_car: int = DEFAULT_MAX_TRACKS_PER_CAR,
    max_cars: int = 0,
) -> list[HeldOutCar]:
    """Held-out cars from ``sessions``' DVSA labels instead of a corpus val
    split: every labelled car there (its tracks in those sessions only)
    except ``exclude_cars``, the cars any compared model trained on. A car
    seen under a misread plate in training escapes the exclusion; the plate
    gate makes that rare."""
    exclude = exclude_cars or set()
    label_of: dict[str, str] = {}
    tracks_of: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for sess in sessions:
        path = output_root / sess / f"{sess}_dvsa_labels.json"
        try:
            labels = json.loads(path.read_text(encoding="utf-8")).get("labels") or {}
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: no readable DVSA labels ({exc})") from exc
        for plate, row in labels.items():
            if plate in exclude or not isinstance(row, dict):
                continue
            label = target_label(row, target)
            tids = row.get("track_ids") or []
            if not label or not tids:
                continue
            label_of.setdefault(plate, label)
            tracks_of[plate].update((sess, int(t)) for t in tids)
    return _sample_tracks(
        label_of, tracks_of, seed=seed, max_tracks_per_car=max_tracks_per_car, max_cars=max_cars
    )


def _vote(preds: list[tuple[str, float]]) -> str | None:
    """Confidence-weighted top-1 vote (the ``makemodel`` per-track rule)."""
    weight: dict[str, float] = defaultdict(float)
    for make, conf in preds:
        weight[make] += conf
    return max(weight, key=lambda k: weight[k]) if weight else None


def score_rows(
    cars: list[HeldOutCar],
    preds: dict[str, dict[tuple[str, int], list[tuple[str, float]]]],
    known_classes: dict[str, set[str]],
    group: Any = None,
) -> dict[str, dict[str, Any]]:
    """Per-contender accuracy from per-track snap predictions.

    ``preds[name][(session, tid)]`` holds that contender's (label, conf) per
    classified snap. Returns per-contender metrics plus per-car correctness
    vectors (``_track_hits``, and ``_track_hits_grouped`` when ``group``
    maps a label to its family) used for the paired bootstrap.
    """
    shared = set.intersection(*known_classes.values()) if known_classes else set()
    out: dict[str, dict[str, Any]] = {}
    for name, by_track in preds.items():
        n_tr = n_tr_ok = n_tr_pred = n_tr_gok = 0
        n_sh = n_sh_ok = 0
        n_car_ok = n_car_sh = n_car_sh_ok = n_car_gok = 0
        track_hits: dict[str, list[int]] = {}
        track_hits_g: dict[str, list[int]] = {}
        for c in cars:
            hits, hits_g = [], []
            car_preds: list[tuple[str, float]] = []
            for key in c.tracks:
                p = by_track.get(key, [])
                car_preds.extend(p)
                guess = _vote(p)
                ok = int(guess == c.label)
                hits.append(ok)
                n_tr += 1
                n_tr_ok += ok
                n_tr_pred += guess is not None
                if group is not None:
                    gok = int(guess is not None and group(guess) == group(c.label) != "")
                    hits_g.append(gok)
                    n_tr_gok += gok
                if c.label in shared:
                    n_sh += 1
                    n_sh_ok += ok
            track_hits[c.car] = hits
            track_hits_g[c.car] = hits_g
            car_guess = _vote(car_preds)
            car_ok = int(car_guess == c.label)
            n_car_ok += car_ok
            if group is not None:
                n_car_gok += int(car_guess is not None and group(car_guess) == group(c.label) != "")
            if c.label in shared:
                n_car_sh += 1
                n_car_sh_ok += car_ok
        row: dict[str, Any] = {
            "n_tracks": n_tr,
            "track_acc": round(n_tr_ok / n_tr, 4) if n_tr else None,
            "track_acc_shared_classes": round(n_sh_ok / n_sh, 4) if n_sh else None,
            "car_acc": round(n_car_ok / len(cars), 4) if cars else None,
            "car_acc_shared_classes": round(n_car_sh_ok / n_car_sh, 4) if n_car_sh else None,
            "coverage": round(n_tr_pred / n_tr, 4) if n_tr else None,
            "_track_hits": track_hits,
        }
        if group is not None:
            row["track_acc_grouped"] = round(n_tr_gok / n_tr, 4) if n_tr else None
            row["car_acc_grouped"] = round(n_car_gok / len(cars), 4) if cars else None
            row["_track_hits_grouped"] = track_hits_g
        out[name] = row
    return out


def paired_delta(
    hits_a: dict[str, list[int]],
    hits_b: dict[str, list[int]],
    *,
    iters: int = _BOOTSTRAP_ITERS,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Per-track accuracy of ``a`` minus ``b`` with a 95 % bootstrap CI,
    resampling cars (a car's tracks move together)."""
    cars = sorted(set(hits_a) & set(hits_b))
    if not cars:
        return 0.0, 0.0, 0.0

    def delta(sample: list[str]) -> float:
        a = sum(sum(hits_a[c]) for c in sample)
        b = sum(sum(hits_b[c]) for c in sample)
        n = sum(len(hits_a[c]) for c in sample)
        return (a - b) / n if n else 0.0

    point = delta(cars)
    rng = random.Random(seed)
    boots = sorted(delta([rng.choice(cars) for _ in cars]) for _ in range(iters))
    return point, boots[int(0.025 * iters)], boots[int(0.975 * iters) - 1]


def _file_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime": st.st_mtime}


def _production_corpus(production: Path, runs_dir: Path) -> Path | None:
    """Production's training corpus dir, from the promotion sidecar."""
    sidecar = production.with_suffix(".meta.json")
    try:
        name = (json.loads(sidecar.read_text()).get("trained_corpus") or {}).get("name")
    except (OSError, json.JSONDecodeError):
        return None
    if not name:
        return None
    d = runs_dir / name
    return d if (d / "manifest.json").is_file() else None


def excluded_cars(
    corpus_dir: Path,
    exclude_corpora: list[Path] | None,
    *,
    eval_sessions: list[str] | None = None,
    include_trained_cars: bool = False,
) -> set[str]:
    """The cars dropped from the held-out set: every car in
    ``exclude_corpora`` (production's training corpus by default), plus the
    candidate's own corpus when scoring ``eval_sessions`` (the val-split mode
    gets that from the split itself). ``include_trained_cars`` (sessions
    only) drops none: new passes of known cars are scored too."""
    if include_trained_cars:
        if not eval_sessions:
            raise ValueError("include_trained_cars needs eval_sessions")
        return set()
    exclude: set[str] = set()
    for d in exclude_corpora or []:
        exclude |= corpus_cars(d)
    if eval_sessions:
        exclude |= corpus_cars(corpus_dir)
    return exclude


def compare(
    corpus_dir: Path,
    candidate: Path,
    production: Path,
    *,
    output_root: Path = Path("output"),
    exclude_corpora: list[Path] | None = None,
    road_polygon_frac: list[tuple[float, float]] | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
    max_tracks_per_car: int = DEFAULT_MAX_TRACKS_PER_CAR,
    max_cars: int = 0,
    device: str = "cpu",
    vehicle_detector: Any = None,
    target: str = "make",
    eval_sessions: list[str] | None = None,
    include_trained_cars: bool = False,
) -> dict[str, Any]:
    """Run the head-to-head and return the report dict (see module doc)."""
    import cv2  # type: ignore[import-untyped]

    from streettracker.analysis.snap_assets import discover_vehicle_snaps
    from streettracker.analysis.vehicle_locator import SnapVehicleLocator, VehicleBoxCache

    exclude = excluded_cars(
        corpus_dir,
        exclude_corpora,
        eval_sessions=eval_sessions,
        include_trained_cars=include_trained_cars,
    )
    if eval_sessions:
        cars = session_cars(
            output_root,
            eval_sessions,
            exclude_cars=exclude,
            target=target,
            seed=seed,
            max_tracks_per_car=max_tracks_per_car,
            max_cars=max_cars,
        )
    else:
        cars = held_out_cars(
            corpus_dir,
            exclude_cars=exclude,
            val_frac=val_frac,
            seed=seed,
            max_tracks_per_car=max_tracks_per_car,
            max_cars=max_cars,
            target=target,
        )

    prod_auto = load_contender(target, production, device=device)
    contenders: dict[str, Contender] = {"production": prod_auto}
    if prod_auto.crop_mode != "fullframe":
        # Production on clean crops at its TRAINING pad: every legacy corpus
        # was built at 0.1 (recomputed crops match the saved ones, 2026-09-24
        # audit); only the make head's inference default was the looser 0.25.
        contenders["production_clean_crops"] = load_contender(
            target, production, device=device, crop_mode="fullframe", pad_frac=_LEGACY_TRAIN_PAD
        )
    contenders["candidate"] = load_contender(target, candidate, device=device)

    # Work list grouped by session: each snap decoded once, located once per
    # crop mode, then classified by every contender.
    wanted: dict[str, set[int]] = defaultdict(set)
    for c in cars:
        for sess, tid in c.tracks:
            wanted[sess].add(tid)
    work: list[tuple[str, Path, int, int]] = []
    for sess in sorted(wanted):
        for path, tid, n, _cls in discover_vehicle_snaps(output_root / sess):
            if tid in wanted[sess]:
                work.append((sess, path, tid, n))
    total = len(work)
    print(
        f"[makemodel-compare] {len(cars)} held-out cars, "
        f"{sum(len(c.tracks) for c in cars)} tracks, {total} snaps; "
        + (
            "kept cars the models trained on (scoring their new passes)"
            if include_trained_cars
            else f"excluded {len(exclude)} cars from training corpora"
        )
        + (f"; eval sessions: {', '.join(eval_sessions)}" if eval_sessions else ""),
        flush=True,
    )

    preds: dict[str, dict[tuple[str, int], list[tuple[str, float]]]] = {
        name: defaultdict(list) for name in contenders
    }
    modes = sorted({clf.crop_mode for clf in contenders.values()})
    cur_sess: str | None = None
    locs: dict[str, SnapVehicleLocator] = {}
    t0 = time.time()
    for i, (sess, path, tid, n) in enumerate(work, 1):
        if sess != cur_sess:  # work is session-ordered: one set of locators at a time
            for loc in locs.values():
                loc.close()
            sd = output_root / sess
            cache = VehicleBoxCache(sd, detector=vehicle_detector) if "fullframe" in modes else None
            locs = {
                mode: SnapVehicleLocator(
                    sd,
                    mode,
                    road_polygon_frac=road_polygon_frac if mode == "fullframe" else None,
                    cache=cache if mode == "fullframe" else None,
                )
                for mode in modes
            }
            cur_sess = sess
        image = cv2.imread(str(path))
        if image is not None:
            boxes = {mode: loc.locate(path, tid, n, image)[0] for mode, loc in locs.items()}
            for name, clf in contenders.items():
                pred = clf.predict(image, boxes[clf.crop_mode])
                if pred:
                    preds[name][(sess, tid)].append((pred[0], float(pred[1])))
        if i % 200 == 0 or i == total:
            print(f"[batch] {i}/{total} done ({time.time() - t0:.0f}s)", flush=True)
    for loc in locs.values():
        loc.close()

    known = {name: clf.classes for name, clf in contenders.items()}
    group = _colour_group if target == "colour" else None
    scores = score_rows(cars, preds, known, group=group)
    point, lo, hi = paired_delta(
        scores["candidate"]["_track_hits"], scores["production"]["_track_hits"], seed=seed
    )
    grouped_delta = None
    if group is not None:
        g_pt, g_lo, g_hi = paired_delta(
            scores["candidate"]["_track_hits_grouped"],
            scores["production"]["_track_hits_grouped"],
            seed=seed,
        )
        grouped_delta = {
            "metric": "track_acc_grouped",
            "candidate_minus_production": round(g_pt, 4),
            "ci95": [round(g_lo, 4), round(g_hi, 4)],
        }
    rows = []
    for name, clf in contenders.items():
        row = {k: v for k, v in scores[name].items() if not k.startswith("_")}
        row.update(name=name, crop_mode=clf.crop_mode, pad_frac=clf.pad_frac)
        rows.append(row)
    # Does production itself read better on clean crops? (a switch that needs
    # no retrain -- only an inference --crop-mode change)
    clean_delta = None
    if "production_clean_crops" in scores:
        c_pt, c_lo, c_hi = paired_delta(
            scores["production_clean_crops"]["_track_hits"],
            scores["production"]["_track_hits"],
            seed=seed,
        )
        clean_delta = {
            "metric": "track_acc",
            "clean_minus_deployed": round(c_pt, 4),
            "ci95": [round(c_lo, 4), round(c_hi, 4)],
        }

    def n_classes(name: str) -> dict[str, int]:
        n = len(known[name])
        return {"n_classes": n, **({"n_makes": n} if target == "make" else {})}

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "corpus": str(corpus_dir),
        "candidate": {**_file_fingerprint(candidate), **n_classes("candidate")},
        "production": {**_file_fingerprint(production), **n_classes("production")},
        "excluded_corpora": [] if include_trained_cars else [str(d) for d in exclude_corpora or []],
        "eval_sessions": list(eval_sessions or []),
        "include_trained_cars": include_trained_cars,
        "n_excluded_cars": len(exclude),
        "n_cars": len(cars),
        "n_tracks": sum(len(c.tracks) for c in cars),
        "n_snaps": total,
        "max_tracks_per_car": max_tracks_per_car,
        "rows": rows,
        "delta": {
            "metric": "track_acc",
            "candidate_minus_production": round(point, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        },
        "grouped_delta": grouped_delta,
        "production_clean_crop_delta": clean_delta,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"[makemodel-compare] {report['n_cars']} cars / {report['n_tracks']} tracks / "
        f"{report['n_snaps']} snaps"
    )

    def pct(v: float | None) -> str:
        return f"{100 * v:.1f}%" if v is not None else "-"

    noun = _NOUN.get(report.get("target", "make"), "make")
    grouped = report.get("grouped_delta") is not None
    head = f"  {'model':<24}{'crops':<15}{'track@1':>9}{'shared':>9}{'car@1':>8}{'cover':>8}"
    print(head + (f"{'grouped':>9}" if grouped else ""))
    for r in report["rows"]:
        crops = f"{r['crop_mode']}@{r['pad_frac']}"
        print(
            f"  {r['name']:<24}{crops:<15}"
            f"{pct(r['track_acc']):>9}{pct(r['track_acc_shared_classes']):>9}"
            f"{pct(r['car_acc']):>8}{pct(r['coverage']):>8}"
            + (f"{pct(r.get('track_acc_grouped')):>9}" if grouped else "")
        )
    d = report["delta"]
    lo, hi = d["ci95"]
    print(
        f"[makemodel-compare] candidate - production per-track {noun}@1: "
        f"{100 * d['candidate_minus_production']:+.1f} pp (95% CI {100 * lo:+.1f}..{100 * hi:+.1f})"
    )
    g = report.get("grouped_delta")
    if g:
        glo, ghi = g["ci95"]
        print(
            f"[makemodel-compare] candidate - production per-track grouped {noun}: "
            f"{100 * g['candidate_minus_production']:+.1f} pp "
            f"(95% CI {100 * glo:+.1f}..{100 * ghi:+.1f})"
        )
    c = report.get("production_clean_crop_delta")
    if c:
        clo, chi = c["ci95"]
        print(
            f"[makemodel-compare] production on clean crops - as deployed: "
            f"{100 * c['clean_minus_deployed']:+.1f} pp (95% CI {100 * clo:+.1f}..{100 * chi:+.1f})"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``streettracker makemodel-compare <corpus_dir> --candidate <best.pt>``."""
    import argparse

    import torch

    from streettracker.analysis.alpr.fullframe import load_road_polygon

    ap = argparse.ArgumentParser(prog="streettracker makemodel-compare")
    ap.add_argument("corpus_dir", type=Path, help="the candidate's training corpus")
    ap.add_argument("--candidate", type=Path, required=True, help="candidate best.pt")
    ap.add_argument(
        "--target",
        choices=TARGETS,
        default="make",
        help="which head: make (default), colour or body_type",
    )
    ap.add_argument(
        "--production",
        type=Path,
        default=None,
        help="production checkpoint (default: the target's installed model)",
    )
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument(
        "--exclude-corpus",
        type=Path,
        nargs="*",
        default=None,
        help="corpora whose cars are dropped from the held-out set (default: production's "
        "training corpus, from its .meta.json sidecar)",
    )
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument(
        "--eval-session",
        nargs="+",
        default=None,
        metavar="SESSION",
        help="score on these sessions' DVSA-labelled cars (minus every car in production's and "
        "the candidate's training corpora) instead of the candidate corpus's val split",
    )
    ap.add_argument(
        "--include-trained-cars",
        action="store_true",
        help="with --eval-session: keep cars the models trained on, scoring their new passes "
        "(sessions recorded after all training data) alongside fresh cars",
    )
    ap.add_argument("--road-polygon", type=Path, default=Path(".claude/triggers_proposal.json"))
    ap.add_argument("--max-tracks-per-car", type=int, default=DEFAULT_MAX_TRACKS_PER_CAR)
    ap.add_argument("--max-cars", type=int, default=0, help="random subset of cars (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: <candidate dir>/compare.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    if args.production is None:
        args.production = default_production(args.target)

    for p in (args.corpus_dir / "manifest.json", args.candidate, args.production):
        if not p.exists():
            print(f"[makemodel-compare] not found: {p}")
            return 1
    if args.include_trained_cars and not args.eval_session:
        print("[makemodel-compare] --include-trained-cars needs --eval-session")
        return 2
    exclude = args.exclude_corpus
    if exclude is None and not args.include_trained_cars:
        prod_corpus = _production_corpus(args.production, args.runs_dir)
        if prod_corpus is None:
            print(
                "[makemodel-compare] production's training corpus is unknown (no sidecar or "
                "corpus dir) -- pass --exclude-corpus, or production may be scored on cars "
                "it trained on"
            )
            return 1
        exclude = [prod_corpus]
    for sess in args.eval_session or []:
        if not (args.output_root / sess / f"{sess}_dvsa_labels.json").is_file():
            print(f"[makemodel-compare] {sess}: no DVSA labels under {args.output_root}")
            return 1

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    report = compare(
        args.corpus_dir,
        args.candidate,
        args.production,
        output_root=args.output_root,
        exclude_corpora=exclude,
        road_polygon_frac=load_road_polygon(args.road_polygon, log_prefix="[makemodel-compare]"),
        val_frac=args.val_frac,
        seed=args.seed,
        max_tracks_per_car=args.max_tracks_per_car,
        max_cars=args.max_cars,
        device=device,
        target=args.target,
        eval_sessions=args.eval_session,
        include_trained_cars=args.include_trained_cars,
    )
    if not report["n_cars"]:
        print("[makemodel-compare] no held-out cars left after exclusions")
        return 1
    out = args.out or args.candidate.parent / "compare.json"
    if out.is_file():  # keep the report this one replaces
        out.replace(out.with_name(f"{out.stem}.prev{out.suffix}"))
    atomic_write_text(out, json.dumps(report, indent=2))
    _print_report(report)
    print(f"[makemodel-compare] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
