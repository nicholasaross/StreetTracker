"""UK car-make benchmark for the /stats "Top makes" chart.

``data/uk_car_makes.json`` (built by ``.claude/build_uk_make_benchmark.py`` from
DfT vehicle licensing statistics, Open Government Licence v3.0) holds licensed
cars in the UK by make: for the latest quarter, and for each year of first
registration. The chart compares the street's identified cars with the UK
**age-matched** to the street: each registration year's UK make shares,
weighted by how many identified cars come from that year. A street whose cars
are newer or older than the national fleet then isn't read as a brand
preference. The plain share of all licensed cars is kept alongside.

There is no regional (e.g. South East) equivalent: DfT publishes no data that
combines make with geography.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data" / "uk_car_makes.json"


@lru_cache(maxsize=1)
def load_benchmark(path: Path = DATA) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _key(make: str) -> str:
    return make.strip().upper()


def uk_shares(bench: Mapping[str, Any]) -> dict[str, float]:
    """Share of all UK licensed cars by make (latest quarter)."""
    latest = bench.get("latest") or {}
    total = float(latest.get("total") or 0)
    if total <= 0:
        return {}
    return {m: n / total for m, n in (latest.get("makes") or {}).items()}


def age_matched_shares(
    bench: Mapping[str, Any], year_counts: Mapping[int, int]
) -> dict[str, float]:
    """UK make shares weighted by the registration years in ``year_counts``.

    Years after the table's newest (cars registered since that year-end) use the
    newest year's shares; years before its oldest use the oldest's.
    """
    years = (bench.get("by_year") or {}).get("years") or {}
    table = {int(y): v for y, v in years.items() if str(y).isdigit()}
    n_total = sum(n for n in year_counts.values() if n > 0)
    if not table or n_total <= 0:
        return {}
    first, last = min(table), max(table)
    shares: dict[str, float] = {}
    for year, n in year_counts.items():
        if n <= 0:
            continue
        row = table.get(min(max(int(year), first), last)) or {}
        total = float(row.get("total") or 0)
        if total <= 0:
            continue
        weight = n / n_total
        for make, count in (row.get("makes") or {}).items():
            shares[make] = shares.get(make, 0.0) + weight * count / total
    return shares


def make_benchmark(
    makes_by_plate: Mapping[str, str],
    years_by_plate: Mapping[str, int | None],
    top: list[list[Any]],
    bench: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Rows for the chart: each top make's share of the street's identified
    cars, its age-matched UK share and its share of all UK licensed cars.
    ``None`` when the benchmark file is missing."""
    bench = bench if bench is not None else load_benchmark()
    if not bench or not makes_by_plate:
        return None
    n_cars = len(makes_by_plate)
    year_counts: dict[int, int] = {}
    for plate in makes_by_plate:
        y = years_by_plate.get(plate)
        if isinstance(y, int) and y > 1900:
            year_counts[y] = year_counts.get(y, 0) + 1
    matched = age_matched_shares(bench, year_counts)
    plain = uk_shares(bench)
    rows = []
    for make, n in top:
        k = _key(str(make))
        rows.append(
            {
                "make": make,
                "n": n,
                "share": round(n / n_cars, 4),
                "uk_share": round(matched[k], 4) if k in matched else None,
                "uk_share_all": round(plain[k], 4) if k in plain else None,
            }
        )
    return {
        "rows": rows,
        "n_cars": n_cars,
        "n_with_year": sum(year_counts.values()),
        "latest_period": (bench.get("latest") or {}).get("period"),
        "by_year_period": (bench.get("by_year") or {}).get("period"),
        "source": bench.get("source"),
        "licence": bench.get("licence"),
    }
