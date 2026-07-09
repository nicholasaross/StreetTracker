"""Body-type model verdict: per-class + per-car recall on its own val split.

Reproduces the by-car val split (seed 0) the training scored 0.694 on and
reports (a) per-crop per-class recall/precision + top confusions, and (b)
the per-CAR majority-vote accuracy -- the real use, since a car yields
several crops and one body-type tag.

    uv run python .claude/bodytype_eval.py [runs/uk_body_0708_b0/best.pt]
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

from streettracker.analysis.makemodel.dataset import build_eval_transform
from streettracker.analysis.makemodel.model import load_checkpoint
from streettracker.analysis.makemodel.uk_dataset import UKMakeDataset

CORPUS = "runs/uk_crops_0708_bt_384"


def main(argv: list[str]) -> int:
    ckpt = argv[0] if argv else "runs/uk_body_0708_b0/best.pt"
    model, meta = load_checkpoint(ckpt)
    names = list(meta["body_type_names"])
    tf = build_eval_transform(int(meta["input_size"]))
    root = Path(CORPUS)
    val = UKMakeDataset(CORPUS, "val", label_field="body_type", class_names=tuple(names), seed=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    n = len(names)
    conf = [[0] * n for _ in range(n)]
    car_votes: dict[str, Counter] = defaultdict(Counter)
    car_true: dict[str, int] = {}
    samples = val._samples
    with torch.no_grad():
        for i in range(0, len(samples), 64):
            chunk = samples[i : i + 64]
            batch = torch.stack(
                [tf(Image.open(root / s.path).convert("RGB")) for s in chunk]
            ).to(dev)
            preds = model(batch)["body_type"].argmax(1).tolist()
            for s, p in zip(chunk, preds):
                conf[s.make_index][p] += 1
                car_votes[s.car][p] += 1
                car_true[s.car] = s.make_index

    support = [sum(conf[t]) for t in range(n)]
    col = [sum(conf[t][p] for t in range(n)) for p in range(n)]
    tot = sum(support)
    print(f"per-crop val: {tot} crops  top1={sum(conf[i][i] for i in range(n)) / tot:.3f}\n")
    print(f"{'class':<11}{'support':>8}{'recall':>8}{'prec':>8}   top confusion")
    for i in range(n):
        rec = conf[i][i] / support[i] if support[i] else 0
        prec = conf[i][i] / col[i] if col[i] else 0
        wrong = sorted(((conf[i][j], names[j]) for j in range(n) if j != i), reverse=True)
        tc = f"{wrong[0][1]} {100 * wrong[0][0] / support[i]:.0f}%" if support[i] and wrong[0][0] else "-"
        print(f"{names[i]:<11}{support[i]:>8}{rec:>8.2f}{prec:>8.2f}   {tc}")

    car_ok = sum(car_votes[c].most_common(1)[0][0] == car_true[c] for c in car_true)
    big = {"hatchback", "suv"}
    big_ids = [i for i, nm in enumerate(names) if nm in big]
    car_big = [c for c in car_true if car_true[c] in big_ids]
    car_big_ok = sum(car_votes[c].most_common(1)[0][0] == car_true[c] for c in car_big)
    print(f"\nper-CAR majority vote: {len(car_true)} cars  top1={car_ok / len(car_true):.3f}")
    print(
        f"  hatchback+suv cars only ({len(car_big)}, {100 * len(car_big) / len(car_true):.0f}% "
        f"of cars): {car_big_ok / len(car_big):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
