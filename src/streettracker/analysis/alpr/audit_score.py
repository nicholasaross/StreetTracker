"""What the audit set's labels say (review E1.2).

:mod:`streettracker.analysis.alpr.audit` draws a stratified sample of plate
reads; a person then says, for each, whether the boxed plate is on the
tracked car and what it really reads. This module turns those answers into
rates for every read, not just the sampled ones.

**Weights.** An item stands for ``population / labelled`` reads of its
stratum, so a rate over several strata is a weighted mean (the Hajek ratio
estimator). The labelled count is used, not the sampled one: the labelling
order is random, so a part-labelled set is still a random sample of each
stratum. A stratum with nothing labelled yet is simply missing from the
estimate, and :func:`build_report` lists those.

**Intervals.** 95 %, from a bootstrap that resamples the labelled items
within each stratum. When every counted item agrees (all right or all wrong)
the bootstrap has no spread to find, so the interval is then the exact
one-sided bound for that many items (the "rule of three"), using the
effective sample size of the weights.

**Can't tell.** An item the labeller could not decide (which car is tracked,
or an illegible plate) is left out of the rate it can't inform, not counted
as wrong.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from streettracker.analysis.alpr.audit import (
    DIRECTIONS,
    DVSA_FIELDS,
    LIGHT_BANDS,
    SAMPLES,
    SUPPORT_BANDS,
)

Row = dict[str, Any]
Outcome = Callable[[Row], "bool | None"]
Where = Callable[[Row], bool]

CONFIDENCE_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
PATH_GROUPS = ("rank0", "rank1", "fallback")

# UK plates use one glyph for O and 0 and one for I and 1, so a person
# reading a picture cannot tell them apart and neither spelling is wrong.
_SAME_GLYPH = str.maketrans({"O": "0", "I": "1"})
_WILDCARD = "."


# ----------------------------------------------------------------------
# One item's answers


def plate_text_equal(read: str | None, truth: str | None) -> bool | None:
    """Whether a read matches the plate a person read off the picture.

    ``None`` when there is no truth to compare with. ``.`` in the truth is a
    character the labeller could not make out and matches anything; O/0 and
    I/1 count as the same character.
    """
    if truth is None:
        return None
    got = (read or "").upper().translate(_SAME_GLYPH)
    want = truth.upper().translate(_SAME_GLYPH)
    return len(got) == len(want) and all(
        w in (_WILDCARD, g) for g, w in zip(got, want, strict=True)
    )


def on_tracked_car(row: Row) -> bool | None:
    """Is the boxed plate on the tracked car? ``None`` if the labeller
    couldn't tell which car is tracked."""
    attribution = row["label"]["attribution"]
    return None if attribution == "unsure" else attribution == "tracked"


def off_tracked_car(row: Row) -> bool | None:
    on = on_tracked_car(row)
    return None if on is None else not on


def text_right(row: Row) -> bool | None:
    """Did the pipeline read the boxed plate's characters correctly,
    whichever car it is on? Text read off something that isn't a plate is
    wrong; an illegible plate gives ``None``."""
    label = row["label"]
    if label["attribution"] == "not_plate":
        return False
    return plate_text_equal(row.get("ocr_text"), label.get("text"))


def verified(row: Row) -> bool | None:
    """The read is the tracked car's plate, correctly read: the only kind
    of read that should count."""
    tracked, text = on_tracked_car(row), text_right(row)
    if tracked is False or text is False:
        return False
    if tracked is None or text is None:
        return None
    return True


def _dvsa_wrong(field: str) -> Outcome:
    def outcome(row: Row) -> bool | None:
        answer = row["label"].get("dvsa", {}).get(field)
        return {"yes": False, "no": True}.get(answer)

    return outcome


def dvsa_any_wrong(row: Row) -> bool | None:
    """The register's record doesn't fit the car in some respect. ``None``
    when nothing is wrong but something couldn't be judged."""
    answers = [row["label"].get("dvsa", {}).get(f) for f in DVSA_FIELDS]
    if "no" in answers:
        return True
    return False if all(a == "yes" for a in answers) else None


# ----------------------------------------------------------------------
# Rates


@dataclass(frozen=True, slots=True)
class Rate:
    """A weighted rate over ``n`` decided items, ``hits`` of them counted,
    with a 95 % interval. All ``None`` when ``n`` is 0."""

    n: int
    hits: int
    value: float | None
    low: float | None
    high: float | None

    def scaled(self, factor: float) -> Rate:
        """The rate times a known constant (a precision times the share of
        cars it applies to)."""
        if self.value is None or self.low is None or self.high is None:
            return self
        return Rate(self.n, self.hits, self.value * factor, self.low * factor, self.high * factor)

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "hits": self.hits,
            "value": _round(self.value),
            "low": _round(self.low),
            "high": _round(self.high),
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def labelled_rows(manifest: Mapping[str, Any], labels: Mapping[str, Any]) -> list[Row]:
    """The labelled items, each with its ``label`` and the ``weight`` it
    carries: how many reads of its stratum it stands for."""
    rows = [
        {**item, "label": labels[item["key"]]}
        for item in manifest["items"]
        if item["key"] in labels
    ]
    labelled = Counter(r["stratum"] for r in rows)
    for r in rows:
        r["weight"] = manifest["strata"][r["stratum"]]["population"] / labelled[r["stratum"]]
    return rows


def _one_sided_bound(n_effective: float) -> float:
    """The largest rate still consistent (at 95 %) with seeing none in
    ``n_effective`` items: ``1 - 0.05 ** (1 / n)``, about ``3 / n``."""
    return 1.0 - 0.05 ** (1.0 / max(n_effective, 1.0))


def weighted_rate(
    rows: Sequence[Row],
    outcome: Outcome,
    *,
    where: Where | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Rate:
    """The share of reads for which ``outcome`` holds, among those ``where``
    admits and ``outcome`` can decide. ``rows`` must be every labelled row of
    the strata involved (see :func:`labelled_rows`): the bootstrap resamples
    whole strata, then applies ``where``."""
    by_stratum: dict[str, list[tuple[float, float, float]]] = {}
    for r in rows:
        decided = outcome(r) if where is None or where(r) else None
        by_stratum.setdefault(r["stratum"], []).append(
            (float(r["weight"]), 0.0 if decided is None else 1.0, 1.0 if decided else 0.0)
        )
    arrays = [np.array(by_stratum[s]) for s in sorted(by_stratum)]
    counted = [a[a[:, 1] > 0] for a in arrays]
    n = sum(len(a) for a in counted)
    hits = int(sum(a[:, 2].sum() for a in counted))
    if n == 0:
        return Rate(0, 0, None, None, None)
    weights = np.concatenate([a[:, 0] for a in counted])
    value = float(sum((a[:, 0] * a[:, 2]).sum() for a in counted) / weights.sum())
    if hits in (0, n):
        bound = _one_sided_bound(float(weights.sum() ** 2 / (weights**2).sum()))
        return Rate(n, hits, value, 0.0 if hits == 0 else 1.0 - bound, bound if hits == 0 else 1.0)

    rng = np.random.default_rng(seed)
    top = np.zeros(n_boot)
    bottom = np.zeros(n_boot)
    for a in arrays:
        picks = rng.integers(0, len(a), size=(n_boot, len(a)))
        top += (a[:, 0] * a[:, 2])[picks].sum(axis=1)
        bottom += (a[:, 0] * a[:, 1])[picks].sum(axis=1)
    usable = bottom > 0
    low, high = np.percentile(top[usable] / bottom[usable], [2.5, 97.5])
    return Rate(n, hits, value, float(low), float(high))


def _weighted_share(rows: Iterable[Row], inside: Where) -> float | None:
    total = inside_total = 0.0
    for r in rows:
        total += r["weight"]
        if inside(r):
            inside_total += r["weight"]
    return inside_total / total if total else None


# ----------------------------------------------------------------------
# The report


def _three(rows: Sequence[Row], rate: Callable[..., Rate], where: Where | None = None) -> Row:
    return {
        "on_tracked_car": rate(rows, on_tracked_car, where=where).to_json(),
        "text_right": rate(rows, text_right, where=where).to_json(),
        "verified": rate(rows, verified, where=where).to_json(),
    }


def _path_group(row: Row) -> str:
    return row["path"] if row["path"] in ("rank0", "rank1") else "fallback"


def _corroboration(row: Row) -> str:
    if row["n_agree"] > 0:
        return "snap_agrees"
    return "tracks_only" if row["support"] >= 2 else "neither"


def _confidence_table(rows: Sequence[Row], rate: Callable[..., Rate]) -> list[Row]:
    """Precision and recall of the read text against a confidence floor."""
    right = [r for r in rows if text_right(r)]
    table = []
    for threshold in CONFIDENCE_THRESHOLDS:

        def kept(r: Row, floor: float = threshold) -> bool:
            return float(r.get("ocr_conf") or 0.0) >= floor

        table.append(
            {
                "threshold": threshold,
                "kept_share": _round(_weighted_share(rows, kept)),
                "text_right": rate(rows, text_right, where=kept).to_json(),
                "verified": rate(rows, verified, where=kept).to_json(),
                "recall": _round(_weighted_share(right, kept)),
            }
        )
    return table


def _per_car(
    rows: Sequence[Row], domains: Mapping[str, Mapping[str, int]], rate: Callable[..., Rate]
) -> dict[str, Row]:
    """The verified read rate per car: the share of car tracks whose best
    read is their own plate, correctly read. It is the share with a
    UK-shaped best read (counted exactly, over every track) times the audited
    precision of those reads."""
    groups: dict[str, list[str]] = {"all": list(domains)}
    for direction in DIRECTIONS:
        groups[direction] = [k for k in domains if k.startswith(f"{direction}|")]
    for light in LIGHT_BANDS:
        groups[light] = [k for k in domains if k.endswith(f"|{light}")]
    groups.update({k: [k] for k in domains})

    out: dict[str, Row] = {}
    for name, keys in groups.items():
        total = {
            f: sum(domains[k][f] for k in keys)
            for f in ("cars", "cars_with_snaps", "best", "gated")
        }
        cars = total["cars"]

        def inside(r: Row, keys: frozenset[str] = frozenset(keys)) -> bool:
            return f"{r['direction']}|{r['light']}" in keys

        def inside_gated(r: Row, inside: Where = inside) -> bool:
            return inside(r) and bool(r["gate_pass"])

        shape = total["best"] / cars if cars else 0.0
        gated = total["gated"] / cars if cars else 0.0
        out[name] = {
            **total,
            "shape_rate": _round(shape),
            "gated_rate": _round(gated),
            "verified_rate": rate(rows, verified, where=inside).scaled(shape).to_json(),
            "verified_gated_rate": rate(rows, verified, where=inside_gated).scaled(gated).to_json(),
        }
    return out


def _by(
    rows: Sequence[Row],
    rate: Callable[..., Rate],
    key: Callable[[Row], str],
    values: Iterable[str],
) -> dict[str, Row]:
    out: dict[str, Row] = {}
    for value in values:

        def matches(r: Row, value: str = value) -> bool:
            return key(r) == value

        out[value] = _three(rows, rate, matches)
    return out


def _reads_report(
    rows: Sequence[Row], domains: Mapping[str, Mapping[str, int]], rate: Callable[..., Rate]
) -> Row:
    return {
        "n": len(rows),
        "overall": _three(rows, rate),
        "by_direction": _by(rows, rate, lambda r: r["direction"], DIRECTIONS),
        "by_light": _by(rows, rate, lambda r: r["light"], LIGHT_BANDS),
        "by_path": _by(rows, rate, _path_group, PATH_GROUPS),
        "by_gate": _by(
            rows, rate, lambda r: "pass" if r["gate_pass"] else "fail", ("pass", "fail")
        ),
        "by_corroboration": _by(
            rows, rate, _corroboration, ("snap_agrees", "tracks_only", "neither")
        ),
        "confidence": _confidence_table(rows, rate),
        "per_car": _per_car(rows, domains, rate),
    }


def _filter_report(rows: Sequence[Row], rate: Callable[..., Rate]) -> Row:
    """A filter's rejected reads: how many were rightly set aside (the plate
    is not on the tracked car) and how many were good reads lost."""

    def would_pass(r: Row) -> bool:
        return bool(r["gate_pass"])

    out: Row = {
        "n": len(rows),
        "rightly_rejected": rate(rows, off_tracked_car).to_json(),
        "good_read_lost": rate(rows, verified).to_json(),
        "good_read_lost_that_passed_the_gate": rate(rows, verified, where=would_pass).to_json(),
        "by_direction": {},
    }
    for direction in DIRECTIONS:

        def matches(r: Row, direction: str = direction) -> bool:
            return bool(r["direction"] == direction)

        out["by_direction"][direction] = {
            "rightly_rejected": rate(rows, off_tracked_car, where=matches).to_json(),
            "good_read_lost": rate(rows, verified, where=matches).to_json(),
        }
    return out


def _dvsa_report(rows: Sequence[Row], rate: Callable[..., Rate]) -> Row:
    """How often a DVSA label describes a different car, by how many tracks
    read the plate."""

    def unverified(r: Row) -> bool | None:
        ok = verified(r)
        return None if ok is None else not ok

    def block(where: Where | None) -> Row:
        out = {
            "plate_wrong": rate(rows, unverified, where=where).to_json(),
            "any_wrong": rate(rows, dvsa_any_wrong, where=where).to_json(),
        }
        for f in DVSA_FIELDS:
            out[f"{f}_wrong"] = rate(rows, _dvsa_wrong(f), where=where).to_json()
        return out

    report: Row = {"n": len(rows), "overall": block(None), "by_support": {}}
    for band in SUPPORT_BANDS:

        def matches(r: Row, band: str = band) -> bool:
            return bool(r["dvsa"]["support_band"] == band)

        report["by_support"][band] = block(matches)
    return report


def _coverage(manifest: Mapping[str, Any], rows: Sequence[Row]) -> Row:
    labelled = Counter(r["stratum"] for r in rows)
    by_sample: dict[str, dict[str, int]] = {s: {"items": 0, "labelled": 0} for s in SAMPLES}
    for item in manifest["items"]:
        by_sample[item["sample"]]["items"] += 1
    for r in rows:
        by_sample[r["sample"]]["labelled"] += 1
    return {
        "items": len(manifest["items"]),
        "labelled": len(rows),
        "by_sample": by_sample,
        # Strata the estimates can't see yet: reads exist, none is labelled.
        "unlabelled_strata": sorted(
            s for s, info in manifest["strata"].items() if info["population"] and not labelled[s]
        ),
    }


def build_report(
    manifest: Mapping[str, Any],
    labels: Mapping[str, Any],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Everything the labels so far say, as JSON-ready data."""
    rows = labelled_rows(manifest, labels)
    rate = partial(weighted_rate, n_boot=n_boot, seed=seed)
    of = {s: [r for r in rows if r["sample"] == s] for s in SAMPLES}
    return {
        "coverage": _coverage(manifest, rows),
        "reads": _reads_report(of["best"], manifest["domains"], rate),
        "colour_check": _filter_report(of["colour_rejected"], rate),
        "static_filter": _filter_report(of["static_rejected"], rate),
        "dvsa": _dvsa_report(of["dvsa"], rate),
    }


# ----------------------------------------------------------------------
# The report as text


def _pct(rate: Mapping[str, Any] | None) -> str:
    if not rate or rate.get("value") is None:
        return "    -"
    return (
        f"{100 * rate['value']:5.1f} % ({100 * rate['low']:.1f}-{100 * rate['high']:.1f}, "
        f"n={rate['n']})"
    )


def _three_lines(title: str, groups: Mapping[str, Row]) -> list[str]:
    lines = [f"  {title:<22} {'verified':<28} {'on tracked car':<28} text right"]
    for name, block in groups.items():
        lines.append(
            f"    {name:<20} {_pct(block['verified']):<28} "
            f"{_pct(block['on_tracked_car']):<28} {_pct(block['text_right'])}"
        )
    return lines


def _read_lines(reads: Row) -> list[str]:
    lines = [f"Best reads ({reads['n']} labelled)"]
    lines += _three_lines("all", {"every best read": reads["overall"]})
    for title in ("by_direction", "by_light", "by_path", "by_gate", "by_corroboration"):
        lines += _three_lines(title.replace("_", " "), reads[title])
    lines += ["", "  Read text against a confidence floor"]
    lines.append(f"    {'conf >=':<8} {'kept':>6} {'recall':>7}  {'text right':<28} verified")
    for row in reads["confidence"]:
        kept = "-" if row["kept_share"] is None else f"{100 * row['kept_share']:.0f} %"
        recall = "-" if row["recall"] is None else f"{100 * row['recall']:.0f} %"
        lines.append(
            f"    {row['threshold']:<8.2f} {kept:>6} {recall:>7}  "
            f"{_pct(row['text_right']):<28} {_pct(row['verified'])}"
        )
    lines += ["", "  Verified read rate per car track (shape rate x audited precision)"]
    lines.append(
        f"    {'':<24} {'cars':>7} {'UK-shaped':>10} {'gated':>7}  "
        f"{'verified, any read':<28} verified, gated"
    )
    for name, row in reads["per_car"].items():
        lines.append(
            f"    {name.replace('|', ', '):<24} {row['cars']:>7} "
            f"{100 * row['shape_rate']:>8.1f} % {100 * row['gated_rate']:>5.1f} %  "
            f"{_pct(row['verified_rate']):<28} {_pct(row['verified_gated_rate'])}"
        )
    return lines


def _filter_lines(title: str, block: Row) -> list[str]:
    lines = [
        f"{title} ({block['n']} labelled)",
        f"  rightly rejected (plate not on the tracked car)  {_pct(block['rightly_rejected'])}",
        f"  a good read lost                                 {_pct(block['good_read_lost'])}",
        "  a good read lost that passed the gate            "
        + _pct(block["good_read_lost_that_passed_the_gate"]),
    ]
    for direction, row in block["by_direction"].items():
        lines.append(f"    {direction:<16} rightly rejected {_pct(row['rightly_rejected'])}")
    return lines


def _dvsa_lines(dvsa: Row) -> list[str]:
    lines = [f"DVSA labels ({dvsa['n']} labelled cars)"]
    fields = ("any_wrong", "plate_wrong", *(f"{f}_wrong" for f in DVSA_FIELDS))
    for name, block in (("all", dvsa["overall"]), *dvsa["by_support"].items()):
        label = "all cars" if name == "all" else f"read by {name} track(s)"
        lines.append(f"  {label}")
        lines += [f"    {f.replace('_', ' '):<14} {_pct(block[f])}" for f in fields]
    return lines


def format_report(report: Mapping[str, Any]) -> str:
    """:func:`build_report`'s result as the text the script prints."""
    coverage = report["coverage"]
    lines = [f"Audit set: {coverage['labelled']} of {coverage['items']} items labelled"]
    for sample, counts in coverage["by_sample"].items():
        lines.append(f"  {sample:<16} {counts['labelled']:>4} / {counts['items']}")
    if coverage["unlabelled_strata"]:
        lines.append(
            f"  {len(coverage['unlabelled_strata'])} strata have no labelled item yet, so "
            f"their reads are missing from the rates below:"
        )
        lines += [f"    {s}" for s in coverage["unlabelled_strata"]]
    lines += ["", "Rates are weighted to every read; brackets are 95 % intervals.", ""]
    lines += _read_lines(report["reads"])
    lines += [
        "",
        *_filter_lines(
            "Plate-colour check: would-be best reads it rejected", report["colour_check"]
        ),
    ]
    lines += [
        "",
        *_filter_lines(
            "Static-plate filter: would-be best reads it rejected", report["static_filter"]
        ),
    ]
    lines += ["", *_dvsa_lines(report["dvsa"])]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# The set as a regression check


def _state(best: Mapping[str, Any] | None, plate: str) -> str:
    """How a track's best read stands against a plate: ``"same"``,
    ``"different"`` or ``"none"`` (no UK-shaped best read)."""
    if not best or not best.get("canonical_uk_shape") or not best.get("ocr_text"):
        return "none"
    return "same" if plate_text_equal(str(best["ocr_text"]), plate) else "different"


def rescore(
    manifest: Mapping[str, Any],
    labels: Mapping[str, Any],
    current_best: Mapping[tuple[str, int], Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Compare the audited tracks' best reads now with when the set was
    drawn. Counts over the labelled tracks, not weighted: this is a
    before-and-after on a fixed set, for judging a change to the pipeline.

    ``known`` are tracks whose own plate the labeller read in full, so the
    best read is right (``same``), wrong (``different``) or absent.
    ``misattributed`` are tracks whose audited read was another car's plate;
    ``same`` means the track still carries that plate.
    """
    out: dict[str, Any] = {}
    seen: set[tuple[str, tuple[str, int]]] = set()
    for item in manifest["items"]:
        label = labels.get(item["key"])
        text = (label or {}).get("text")
        if not label or not text:
            continue
        attribution = label["attribution"]
        if attribution == "tracked" and _WILDCARD not in text:
            kind = "known"
        elif attribution in ("other_moving", "parked"):
            kind = "misattributed"
        else:
            continue
        track = (item["session"], int(item["track_id"]))
        if (kind, track) in seen:
            continue
        seen.add((kind, track))
        block = out.setdefault(kind, {"n": 0, "then": Counter(), "now": Counter(), "changed": []})
        then, now = _state(item.get("track_best"), text), _state(current_best.get(track), text)
        block["n"] += 1
        block["then"][then] += 1
        block["now"][now] += 1
        if then != now:
            block["changed"].append({"key": item["key"], "then": then, "now": now})
    for block in out.values():
        block["then"], block["now"] = dict(block["then"]), dict(block["now"])
    return out
