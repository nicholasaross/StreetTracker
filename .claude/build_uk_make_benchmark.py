"""Build the UK car-make benchmark shipped with the showcase's /stats page.

Reads three DfT vehicle licensing statistics data files (United Kingdom,
Open Government Licence v3.0; download from
https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-files):

- ``df_VEH0120_UK.csv``: vehicles at the end of each quarter by body type,
  make and model (the latest quarter gives today's licensed cars by make);
- ``df_VEH0124_AM.csv`` + ``df_VEH0124_NZ.csv``: vehicles at the end of each
  year by make, model and year of first registration.

Writes ``src/streettracker/web/data/uk_car_makes.json``: licensed cars by make
for the latest quarter, and licensed cars by make for each year of first
registration (latest year-end), keeping makes that ever reach 0.05 % of a
year's cars (the rest stay in each year's total). DfT make names are mapped to
the DVSA MOT names the showcase uses (MERCEDES -> MERCEDES-BENZ, SMART ->
SMART (MCC)). The /stats page re-weights the per-year shares by the
registration years of the cars actually identified on the street
(``web/uk_makes.py``), so a street with newer or older cars than average isn't
read as a brand preference.

    uv run python .claude/build_uk_make_benchmark.py <dir with the three CSVs>
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src/streettracker/web/data/uk_car_makes.json"
# DfT make -> DVSA MOT make, where they differ for cars seen on this street.
ALIASES = {"MERCEDES": "MERCEDES-BENZ", "SMART": "SMART (MCC)"}
MIN_SHARE = 0.0005  # keep a make in a year if it reaches 0.05 % of that year's cars
FIRST_YEAR = 1960  # earlier first-registration years fold into this one


def _rows(path: Path) -> csv.DictReader:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:  # VEH0124 is Windows-1252
        text = raw.decode("cp1252")
    return csv.DictReader(text.splitlines())


def _num(v: str | None) -> float:
    if v in (None, "", "[z]", "[x]"):
        return 0.0
    if v == "[c]":  # confidential: 1-4 vehicles
        return 2.5
    return float(v)


def main(src: Path) -> int:
    q_files = sorted(src.glob("df_VEH0120_UK.csv"))
    y_files = sorted(src.glob("df_VEH0124_*.csv"))
    if not q_files or len(y_files) < 2:
        raise SystemExit(f"{src}: need df_VEH0120_UK.csv and df_VEH0124_AM/NZ.csv")

    quarter = ""
    latest: Counter[str] = Counter()
    for r in _rows(q_files[0]):
        if not quarter:
            quarter = next(k for k in r if k[:4].isdigit())  # newest column comes first
        if r["BodyType"] == "Cars" and r["LicenceStatus"] == "Licensed":
            latest[ALIASES.get(r["Make"], r["Make"])] += _num(r[quarter])

    year_end = ""
    by_year: dict[int, Counter[str]] = defaultdict(Counter)
    for f in y_files:
        for r in _rows(f):
            if not year_end:
                year_end = next(k for k in r if k.isdigit())
            if r["BodyType"] != "Cars" or r["LicenceStatus"] != "Licensed":
                continue
            y = r["YearFirstUsed"]
            if not y.isdigit():  # imports etc.: no first-registration year
                continue
            by_year[max(int(y), FIRST_YEAR)][ALIASES.get(r["Make"], r["Make"])] += _num(r[year_end])

    keep = {
        m
        for counts in by_year.values()
        for m, n in counts.items()
        if n >= MIN_SHARE * sum(counts.values())
    }
    out = {
        "source": (
            "Department for Transport, vehicle licensing statistics: df_VEH0120_UK "
            f"(licensed cars, {quarter}) and df_VEH0124 (licensed cars at the end of "
            f"{year_end} by year of first registration). United Kingdom."
        ),
        "source_url": "https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-files",
        "licence": "Open Government Licence v3.0",
        "built": date.today().isoformat(),
        "aliases": ALIASES,
        "latest": {
            "period": quarter,
            "total": round(sum(latest.values())),
            "makes": {m: round(n) for m, n in latest.most_common() if m in keep},
        },
        "by_year": {
            "period": f"end of {year_end}",
            "years": {
                str(y): {
                    "total": round(sum(c.values())),
                    "makes": {m: round(n) for m, n in c.most_common() if m in keep and n >= 1},
                }
                for y, c in sorted(by_year.items())
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {len(keep)} makes, "
        f"{len(by_year)} registration years, latest {quarter} {out['latest']['total']:,} cars"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")))
