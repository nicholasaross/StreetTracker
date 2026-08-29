"""Validate the behavioural vehicle classifier against the real corpus.

The classifier is purely behavioural (an owner/name tag is NOT a resident
prior -- it means "known car", which includes visiting family). This cross-tabs
the operator-tagged cars against what behaviour calls them and prints the
overall bucket + certainty distribution and coverage.

    uv run python .claude/classify_eval.py [output]

Local-only: reads plate/owner PII from the metadata store. Do not paste output
into any shared artifact.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # the reasons carry a → arrow

from streettracker.web.aggregate import build_showcase
from streettracker.web.classify import BUCKET_LABEL
from streettracker.web.metadata import MetadataStore

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")

store = MetadataStore(root / "showcase_metadata.json").load()
owner_plates = {
    p
    for p, e in store.items()
    if isinstance(e, dict) and ((e.get("name") or "").strip() or (e.get("owner") or "").strip())
}

cars = build_showcase(root)
print(f"{len(cars)} plated cars across {root}\n")

buckets = Counter(c.classification for c in cars)
certs = Counter(c.classification_certainty for c in cars)
print("Bucket distribution:")
for b, n in buckets.most_common():
    print(f"  {BUCKET_LABEL.get(b, b):18} {n:5}  ({100 * n / len(cars):4.1f}%)")
print("\nCertainty:")
for c, n in certs.most_common():
    print(f"  {c:8} {n:5}")
classified = sum(n for b, n in buckets.items() if b != "unclassified")
above_low = sum(
    1 for c in cars if c.classification != "unclassified" and c.classification_certainty != "low"
)
print(
    f"\nCoverage: {classified}/{len(cars)} classified ({100 * classified / len(cars):.1f}%), "
    f"{above_low} above 'low' ({100 * above_low / len(cars):.1f}%)"
)

# Ground-truth residents: what did blind behaviour call the tagged owners?
by_plate = {c.plate: c for c in cars}
print(f"\nOperator-tagged cars ({len(owner_plates)} tagged) — behavioural call:")
gt = Counter()
seen = 0
for p in sorted(owner_plates):
    c = by_plate.get(p)
    if c is None:
        print(f"  {p:10} — not in showcase (unread this corpus)")
        continue
    seen += 1
    gt[c.classification] += 1
    owner = (store[p].get("owner") or store[p].get("name") or "").strip()
    print(
        f"  {p:10} {owner:16} -> {c.classification:12} ({c.classification_certainty}) "
        f"| {c.classification_reason}"
    )
print(f"\n  tagged-owner buckets: {dict(gt)}  ({gt.get('resident', 0)}/{seen} called resident)")

# A few example justifications per bucket.
print("\nSample reasons per bucket:")
for b in ("resident", "ymca_staff", "visitor", "brief"):
    ex = [c for c in cars if c.classification == b][:3]
    if ex:
        print(f"  [{b}]")
        for c in ex:
            print(f"    {c.plate:10} {c.classification_reason}")
