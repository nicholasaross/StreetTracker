"""Head-to-head make-classifier comparison on shared held-out cars.

A training run's val make@1 is only comparable with production's when both
were scored on the same cars, with the same makes, from the same kind of
crops. None of that held once the corpus moved to plate-anchored crops
(2026-09-24): production's recorded 0.451 was scored on stale-hint crops
(~2/3 of which missed the car), a different val split and 45 makes. This
module scores both models on ONE held-out set instead:

* **Cars** -- the candidate corpus's by-car val split (the exact cars the
  candidate never trained on), minus every car in production's training
  corpus (so production isn't scored on cars it learnt).
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
    make: str
    tracks: list[tuple[str, int]]  # (session name, track id)


def _manifest(corpus_dir: Path) -> dict[str, Any]:
    return json.loads((corpus_dir / "manifest.json").read_text())


def corpus_cars(corpus_dir: Path) -> set[str]:
    """Every car (plate) in a corpus manifest."""
    return {s["car"] for s in _manifest(corpus_dir)["samples"]}


def held_out_cars(
    corpus_dir: Path,
    *,
    exclude_cars: set[str] | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
    max_tracks_per_car: int = DEFAULT_MAX_TRACKS_PER_CAR,
    max_cars: int = 0,
) -> list[HeldOutCar]:
    """The candidate corpus's val cars (same split as the trainer), minus
    ``exclude_cars``, each with up to ``max_tracks_per_car`` tracks."""
    from streettracker.analysis.makemodel.uk_dataset import split_val_cars

    samples = _manifest(corpus_dir)["samples"]
    val = split_val_cars(samples, label_field="make", val_frac=val_frac, seed=seed)
    exclude = exclude_cars or set()
    make_of: dict[str, str] = {}
    tracks_of: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for s in samples:
        car = s["car"]
        if car not in val or car in exclude:
            continue
        m = _SAMPLE_NAME_RE.match(Path(s["path"]).name)
        if not m:
            continue
        make_of[car] = s["make"]
        tracks_of[car].add((m["session"], int(m["tid"])))

    rng = random.Random(f"compare:{seed}")
    out = []
    for car in sorted(make_of):
        tracks = sorted(tracks_of[car])
        if max_tracks_per_car and len(tracks) > max_tracks_per_car:
            tracks = sorted(rng.sample(tracks, max_tracks_per_car))
        out.append(HeldOutCar(car, make_of[car], tracks))
    if max_cars and len(out) > max_cars:
        out = sorted(rng.sample(out, max_cars), key=lambda c: c.car)
    return out


def _vote(preds: list[tuple[str, float]]) -> str | None:
    """Confidence-weighted top-1 vote (the ``makemodel`` per-track rule)."""
    weight: dict[str, float] = defaultdict(float)
    for make, conf in preds:
        weight[make] += conf
    return max(weight, key=lambda k: weight[k]) if weight else None


def score_rows(
    cars: list[HeldOutCar],
    preds: dict[str, dict[tuple[str, int], list[tuple[str, float]]]],
    known_makes: dict[str, set[str]],
) -> dict[str, dict[str, Any]]:
    """Per-contender accuracy from per-track snap predictions.

    ``preds[name][(session, tid)]`` holds that contender's (make, conf) per
    classified snap. Returns per-contender metrics plus per-car correctness
    vectors (``_track_hits``) used for the paired bootstrap.
    """
    shared = set.intersection(*known_makes.values()) if known_makes else set()
    out: dict[str, dict[str, Any]] = {}
    for name, by_track in preds.items():
        n_tr = n_tr_ok = n_tr_pred = 0
        n_sh = n_sh_ok = 0
        n_car_ok = n_car_sh = n_car_sh_ok = 0
        track_hits: dict[str, list[int]] = {}
        for c in cars:
            hits = []
            car_preds: list[tuple[str, float]] = []
            for key in c.tracks:
                p = by_track.get(key, [])
                car_preds.extend(p)
                guess = _vote(p)
                ok = int(guess == c.make)
                hits.append(ok)
                n_tr += 1
                n_tr_ok += ok
                n_tr_pred += guess is not None
                if c.make in shared:
                    n_sh += 1
                    n_sh_ok += ok
            track_hits[c.car] = hits
            car_ok = int(_vote(car_preds) == c.make)
            n_car_ok += car_ok
            if c.make in shared:
                n_car_sh += 1
                n_car_sh_ok += car_ok
        out[name] = {
            "n_tracks": n_tr,
            "track_acc": round(n_tr_ok / n_tr, 4) if n_tr else None,
            "track_acc_shared_makes": round(n_sh_ok / n_sh, 4) if n_sh else None,
            "car_acc": round(n_car_ok / len(cars), 4) if cars else None,
            "car_acc_shared_makes": round(n_car_sh_ok / n_car_sh, 4) if n_car_sh else None,
            "coverage": round(n_tr_pred / n_tr, 4) if n_tr else None,
            "_track_hits": track_hits,
        }
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
) -> dict[str, Any]:
    """Run the head-to-head and return the report dict (see module doc)."""
    import cv2  # type: ignore[import-untyped]

    from streettracker.analysis.makemodel.infer import MakeModelClassifier
    from streettracker.analysis.snap_assets import discover_vehicle_snaps
    from streettracker.analysis.vehicle_locator import SnapVehicleLocator, VehicleBoxCache

    exclude: set[str] = set()
    for d in exclude_corpora or []:
        exclude |= corpus_cars(d)
    cars = held_out_cars(
        corpus_dir,
        exclude_cars=exclude,
        val_frac=val_frac,
        seed=seed,
        max_tracks_per_car=max_tracks_per_car,
        max_cars=max_cars,
    )

    prod_auto = MakeModelClassifier(production, device=device)
    contenders: dict[str, MakeModelClassifier] = {"production": prod_auto}
    if prod_auto.crop_mode != "fullframe":
        # Production on clean crops at its TRAINING pad: every legacy corpus
        # was built at 0.1 (recomputed crops match the saved ones, 2026-09-24
        # audit); only its inference default was the looser 0.25.
        contenders["production_clean_crops"] = MakeModelClassifier(
            production, device=device, crop_mode="fullframe", pad_frac=_LEGACY_TRAIN_PAD
        )
    contenders["candidate"] = MakeModelClassifier(candidate, device=device)
    for name, clf in contenders.items():
        if not clf.make_only:
            raise ValueError(f"{name} checkpoint is not a UK make-only model")

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
        f"excluded {len(exclude)} production-training cars",
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
                cands = clf.classify(image, boxes[clf.crop_mode])
                if cands:
                    preds[name][(sess, tid)].append((cands[0].make, cands[0].conf))
        if i % 200 == 0 or i == total:
            print(f"[batch] {i}/{total} done ({time.time() - t0:.0f}s)", flush=True)
    for loc in locs.values():
        loc.close()

    known = {name: set(clf.make_names) for name, clf in contenders.items()}
    scores = score_rows(cars, preds, known)
    point, lo, hi = paired_delta(
        scores["candidate"]["_track_hits"], scores["production"]["_track_hits"], seed=seed
    )
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
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": str(corpus_dir),
        "candidate": {**_file_fingerprint(candidate), "n_makes": len(known["candidate"])},
        "production": {**_file_fingerprint(production), "n_makes": len(known["production"])},
        "excluded_corpora": [str(d) for d in exclude_corpora or []],
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
        "production_clean_crop_delta": clean_delta,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"[makemodel-compare] {report['n_cars']} cars / {report['n_tracks']} tracks / "
        f"{report['n_snaps']} snaps"
    )

    def pct(v: float | None) -> str:
        return f"{100 * v:.1f}%" if v is not None else "-"

    print(f"  {'model':<24}{'crops':<15}{'track@1':>9}{'shared':>9}{'car@1':>8}{'cover':>8}")
    for r in report["rows"]:
        crops = f"{r['crop_mode']}@{r['pad_frac']}"
        print(
            f"  {r['name']:<24}{crops:<15}"
            f"{pct(r['track_acc']):>9}{pct(r['track_acc_shared_makes']):>9}"
            f"{pct(r['car_acc']):>8}{pct(r['coverage']):>8}"
        )
    d = report["delta"]
    lo, hi = d["ci95"]
    print(
        f"[makemodel-compare] candidate - production per-track make@1: "
        f"{100 * d['candidate_minus_production']:+.1f} pp (95% CI {100 * lo:+.1f}..{100 * hi:+.1f})"
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
    from streettracker.analysis.makemodel.infer import DEFAULT_MODEL

    ap = argparse.ArgumentParser(prog="streettracker makemodel-compare")
    ap.add_argument("corpus_dir", type=Path, help="the candidate's training corpus")
    ap.add_argument("--candidate", type=Path, required=True, help="candidate best.pt")
    ap.add_argument("--production", type=Path, default=DEFAULT_MODEL)
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
    ap.add_argument("--road-polygon", type=Path, default=Path(".claude/triggers_proposal.json"))
    ap.add_argument("--max-tracks-per-car", type=int, default=DEFAULT_MAX_TRACKS_PER_CAR)
    ap.add_argument("--max-cars", type=int, default=0, help="random subset of cars (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: <candidate dir>/compare.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    for p in (args.corpus_dir / "manifest.json", args.candidate, args.production):
        if not p.exists():
            print(f"[makemodel-compare] not found: {p}")
            return 1
    exclude = args.exclude_corpus
    if exclude is None:
        prod_corpus = _production_corpus(args.production, args.runs_dir)
        if prod_corpus is None:
            print(
                "[makemodel-compare] production's training corpus is unknown (no sidecar or "
                "corpus dir) -- pass --exclude-corpus, or production may be scored on cars "
                "it trained on"
            )
            return 1
        exclude = [prod_corpus]

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
    )
    if not report["n_cars"]:
        print("[makemodel-compare] no held-out cars left after exclusions")
        return 1
    out = args.out or args.candidate.parent / "compare.json"
    atomic_write_text(out, json.dumps(report, indent=2))
    _print_report(report)
    print(f"[makemodel-compare] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
