"""VLM vs CNN make-classification bake-off.

Phase A — labelled val, apples-to-apples: reproduces the exact by-car
val split of a crop corpus (same seed/logic as training), takes each
val car's largest crop, and scores three predictors on the same images:
the B6 candidate, the B5 production checkpoint, and a local VLM via
Ollama (constrained to the corpus make list).

Phase B — the OOD slice: unlabelled (no-DVSA) car tracks from a recent
session, cropped with the same VehicleCropper. No ground truth exists
here, so we compare prediction DISTRIBUTIONS: the 2026-06-08 audit
showed the CNN collapses (VW ~28 % at ~0.94 confidence) on exactly this
slice; a healthy predictor should show a plausible make mix instead.

    uv run python .claude/vlm_bakeoff.py --vlm qwen3-vl:8b [--limit 5]

Writes per-item predictions to .claude/vlm_bakeoff_results.json.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434/api/chat"


def vlm_predict(model: str, image_path: Path, makes: list[str]) -> tuple[str, str]:
    """One constrained make classification. Returns (parsed_make|UNPARSED, raw)."""
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    prompt = (
        "This is a vehicle photographed from behind or at an oblique angle by a "
        "UK street camera. Identify the vehicle's manufacturer (make). Respond "
        "with exactly one make from this list and nothing else: "
        + ", ".join(makes)
        + ". If unsure, give your best guess from the list."
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(OLLAMA, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = json.loads(resp.read())["message"]["content"].strip()
    up = " " + re.sub(r"[^A-Z0-9\- ]", " ", raw.upper()) + " "
    hits = [m for m in makes if f" {m} " in up or up.strip() == m]
    if not hits:  # tolerate answers embedding the make without clean boundaries
        hits = [m for m in makes if m in up]
    return (max(hits, key=len) if hits else "UNPARSED"), raw


def cnn_batch(ckpt_path: str, images: list[Path]) -> list[str]:
    """Predict make for each image with one CNN checkpoint, then free VRAM."""
    import torch
    from PIL import Image

    from streettracker.analysis.makemodel.dataset import build_eval_transform
    from streettracker.analysis.makemodel.model import load_checkpoint

    model, meta = load_checkpoint(ckpt_path)
    names = list(meta["make_names"])
    tf = build_eval_transform(int(meta["input_size"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = torch.stack(
                [tf(Image.open(p).convert("RGB")) for p in images[i : i + 16]]
            ).to(dev)
            idx = model(batch)["make"].argmax(dim=1).tolist()
            out.extend(names[j] for j in idx)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def build_ood_crops(session_dir: Path, n: int, out_dir: Path, size: int) -> list[dict]:
    """Crop one largest-bbox snap for each of n unlabelled car tracks."""
    import cv2

    from streettracker.analysis.makemodel.cropper import VehicleCropper
    from streettracker.analysis.snap_assets import (
        discover_vehicle_snaps,
        load_bbox_index,
        resolve_bbox_hint,
    )

    name = session_dir.name
    data = json.loads((session_dir / f"{name}_data.json").read_text(encoding="utf-8"))
    unlabelled = {
        r["track_id"] for r in data if r.get("class_name") == "car" and not r.get("make")
    }
    bbox_index, sub_size = load_bbox_index(session_dir)
    best: dict[int, tuple[int, tuple, str]] = {}
    for path, tid, snap_index, _cls in discover_vehicle_snaps(session_dir):
        if tid not in unlabelled:
            continue
        hint = resolve_bbox_hint(path, tid, snap_index, bbox_index, sub_size)
        if hint is None:
            continue
        area = max(0, hint[2] - hint[0]) * max(0, hint[3] - hint[1])
        if tid not in best or area > best[tid][0]:
            best[tid] = (area, hint, str(path))
    tids = sorted(best)
    random.Random(0).shuffle(tids)
    cropper = VehicleCropper(output_size=size)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tid in tids:
        if len(rows) >= n:
            break
        _area, hint, src = best[tid]
        img = cv2.imread(src)
        if img is None:
            continue
        crop = cropper.crop(img, hint)
        if crop is None:
            continue
        p = out_dir / f"ood_{tid}.jpg"
        cv2.imwrite(str(p), crop)
        rows.append({"track_id": tid, "path": str(p)})
    return rows


def acc(pred: list[str], truth: list[str]) -> float:
    return sum(p == t for p, t in zip(pred, truth)) / len(truth) if truth else 0.0


def dist(preds: list[str], top: int = 6) -> str:
    c = Counter(preds)
    n = len(preds)
    return "  ".join(f"{m} {100 * k / n:.0f}%" for m, k in c.most_common(top))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="qwen3-vl:8b")
    ap.add_argument("--corpus", default="runs/uk_crops_0707_576")
    ap.add_argument("--b6", default="runs/uk_make_0707_b6/best.pt")
    ap.add_argument(
        "--b5", default="src/streettracker/analysis/makemodel/models/makemodel_b0.pt"
    )
    ap.add_argument("--ood-session", default="output/session_20260703_103110")
    ap.add_argument("--ood-n", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None, help="cap val cars (smoke)")
    ap.add_argument(
        "--b5-train-corpus",
        default="runs/uk_crops_0627_512",
        help="corpus the b5 checkpoint trained on; its TRAIN cars are excluded "
        "from b5 scoring (cross-generation leakage: cars recur across corpora)",
    )
    args = ap.parse_args(argv)

    from streettracker.analysis.makemodel.uk_dataset import UKMakeDataset

    corpus = Path(args.corpus)
    ds = UKMakeDataset(corpus, "val", seed=0)
    makes = list(ds.make_names)
    first_per_car: dict[str, tuple[str, str]] = {}
    for s in ds._samples:  # manifest order = largest bbox first per car
        if s.car not in first_per_car:
            first_per_car[s.car] = (s.path, makes[s.make_index])
    cars = sorted(first_per_car)
    random.Random(0).shuffle(cars)
    if args.limit:
        cars = cars[: args.limit]
    val_paths = [corpus / first_per_car[c][0] for c in cars]
    truth = [first_per_car[c][1] for c in cars]
    print(f"[bakeoff] val: {len(cars)} cars ({len(makes)} makes)", flush=True)

    # Cars B5 trained on (its own corpus generation) — leaked for b5 scoring.
    b5_train_cars: set[str] = set()
    if Path(args.b5_train_corpus, "manifest.json").exists():
        old_train = UKMakeDataset(args.b5_train_corpus, "train", seed=0)
        b5_train_cars = {s.car for s in old_train._samples}
    clean_idx = [i for i, c in enumerate(cars) if c not in b5_train_cars]
    print(
        f"[bakeoff] b5-clean subset: {len(clean_idx)}/{len(cars)} val cars "
        f"unseen by the b5 checkpoint",
        flush=True,
    )

    ood_dir = Path(".claude/vlm_bakeoff_ood")
    ood = build_ood_crops(Path(args.ood_session), args.ood_n if not args.limit else 3, ood_dir, 576)
    ood_paths = [Path(r["path"]) for r in ood]
    print(f"[bakeoff] ood: {len(ood)} unlabelled tracks cropped", flush=True)

    results: dict = {"makes": makes, "val_cars": cars, "truth": truth, "ood": ood}
    for tag, ckpt in (("b6", args.b6), ("b5", args.b5)):
        preds = cnn_batch(ckpt, val_paths + ood_paths)
        results[f"{tag}_val"] = preds[: len(val_paths)]
        results[f"{tag}_ood"] = preds[len(val_paths) :]
        print(f"[bakeoff] {tag} val make@1 = {acc(results[f'{tag}_val'], truth):.3f}", flush=True)

    vlm_val, vlm_ood, raws = [], [], []
    for i, p in enumerate(val_paths + ood_paths):
        pred, raw = vlm_predict(args.vlm, p, makes)
        (vlm_val if i < len(val_paths) else vlm_ood).append(pred)
        raws.append(raw)
        if (i + 1) % 25 == 0:
            print(f"[bakeoff] vlm {i + 1}/{len(val_paths) + len(ood_paths)}", flush=True)
    results["vlm_val"], results["vlm_ood"], results["vlm_raw"] = vlm_val, vlm_ood, raws

    results["b5_train_cars_in_val"] = len(cars) - len(clean_idx)
    Path(".claude/vlm_bakeoff_results.json").write_text(json.dumps(results, indent=1))

    def sub(preds: list[str]) -> tuple[list[str], list[str]]:
        return [preds[i] for i in clean_idx], [truth[i] for i in clean_idx]

    print("\n=== BAKE-OFF: labelled val (per-car, largest crop) ===")
    print(f"  full sample (n={len(cars)}):")
    print(f"    VLM {args.vlm:<16} make@1 = {acc(vlm_val, truth):.3f}  "
          f"(unparsed {vlm_val.count('UNPARSED')})")
    print(f"    CNN B6 (production)   make@1 = {acc(results['b6_val'], truth):.3f}")
    print(f"    [B5 excluded here: {len(cars) - len(clean_idx)} of these cars "
          f"were in its training set]")
    print(f"  b5-clean subset (n={len(clean_idx)}, unseen by ALL three):")
    for tag, preds in (("VLM", vlm_val), ("B6 ", results["b6_val"]), ("B5 ", results["b5_val"])):
        p, t = sub(preds)
        print(f"    {tag} make@1 = {acc(p, t):.3f}")
    print("\n=== OOD slice (unlabelled tracks — distributions, no ground truth) ===")
    print(f"  VLM: {dist(vlm_ood)}")
    print(f"  B6 : {dist(results['b6_ood'])}")
    print(f"  B5 : {dist(results['b5_ood'])}")
    agree = sum(a == b for a, b in zip(vlm_ood, results["b6_ood"]))
    print(f"  VLM/B6 agreement on OOD: {100 * agree / len(ood):.0f}%" if ood else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
