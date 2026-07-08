"""DVSA model name -> coarse body type.

The make classifier answers a hard question (which of ~45 marques, from a
rear/oblique street view) at ~45 % make@1. Body type is an *easier*,
coarser question read from the vehicle's silhouette -- hatchback vs SUV
vs van -- expected to be far more reliable, and useful for the unreadable
majority where DVSA gives nothing: "silver SUV at 08:14" beats a make
guess that's wrong half the time.

DVSA returns make + model but **no body type**, so we derive it from the
model name via a curated lookup. Two facts shape the design:

* DVSA model strings carry trim/spec suffixes ("PUMA ST-LINE MHEV",
  "COROLLA DESIGN HEV CVT", "MODEL Y LONG RANGE AWD"), so we match by
  **model-name prefix**, longest-first, collapsing ~1,400 raw strings
  onto a few hundred base models.
* A prefix that ends in a **letter** needs a word boundary after it
  (space / hyphen / end) so "C" doesn't eat "CITAN"; a prefix ending in
  a **digit** matches plainly so "320" catches "320D".

**Approximation, by design.** One model name can span several body styles
(a Golf is a hatch or an estate; a 3-Series a saloon, estate or coupe);
each model maps to its *predominant* UK silhouette. That is acceptable
for a coarse target and is why body type is a separate, lossy signal, not
a replacement for make. Models we don't cover return ``""`` (dropped,
like a missing make) -- honest partial coverage beats guessing.
"""

from __future__ import annotations

import re

# --- DVSA make -> class label (lives here so uk_dataset can import body_type_for
# without a cycle; re-exported from uk_dataset for existing callers) ----------

# DVSA sometimes returns two spellings for one marque (e.g. both "MERCEDES"
# and "MERCEDES-BENZ"), which would otherwise split a make across two classes
# and dilute its training signal. Fold known synonyms onto one canonical label.
_MAKE_SYNONYMS = {
    "MERCEDES": "MERCEDES-BENZ",
}

# DVSA also emits placeholder "makes" for vehicles it can't attribute.
# These are not marques and must never become training classes — the
# literal "UNKNOWN" crossed min_cars_per_make=5 and trained as a real
# class in the 0707 corpus (caught in the 2026-07-08 VLM bake-off, where
# the VLM used it as an escape hatch for 12% of OOD answers).
_MAKE_PLACEHOLDERS = frozenset({"UNKNOWN", "NOT KNOWN", "NONE"})


def normalize_make(make: str | None) -> str:
    """DVSA make -> class label. DVSA names are already canonical UK
    strings ("FORD", "MERCEDES-BENZ", "LAND ROVER"); upper/strip, collapse
    internal whitespace, then fold known synonyms (see ``_MAKE_SYNONYMS``)
    for stable folder/label names. Placeholder values ("UNKNOWN", ...)
    normalise to ``""`` so callers drop them like a missing make."""
    if not make:
        return ""
    name = " ".join(make.upper().split())
    if name in _MAKE_PLACEHOLDERS:
        return ""
    return _MAKE_SYNONYMS.get(name, name)


# Canonical body-type classes (rear/oblique-distinguishable silhouettes).
BODY_TYPES: tuple[str, ...] = (
    "hatchback",
    "saloon",
    "estate",
    "suv",
    "mpv",
    "van",
    "pickup",
    "coupe",
)

# make -> {body_type: [model-name prefixes]}. Authored per marque for the
# models seen on this scene (top ~150 (make,model) pairs cover the bulk)
# plus common UK models generally. Longest-prefix-first matching means
# e.g. "TRANSIT CONNECT" (van) is tried before "TRANSIT", and
# "YARIS CROSS" (suv) before "YARIS".
_RAW: dict[str, dict[str, list[str]]] = {
    "FORD": {
        "hatchback": ["FIESTA", "FOCUS", "KA", "FUSION", "ESCORT"],
        "saloon": ["MONDEO"],
        "suv": ["PUMA", "KUGA", "ECOSPORT", "EDGE", "EXPLORER"],
        "mpv": ["GRAND C-MAX", "C-MAX", "S-MAX", "B-MAX", "GALAXY", "TOURNEO"],
        "van": ["TRANSIT CONNECT", "TRANSIT CUSTOM", "TRANSIT COURIER", "TRANSIT"],
        "pickup": ["RANGER"],
        "coupe": ["MUSTANG"],
    },
    "VOLKSWAGEN": {
        "hatchback": ["GOLF", "POLO", "UP", "BEETLE", "ID.3", "ID3", "FOX", "LUPO"],
        "saloon": ["PASSAT", "ARTEON", "JETTA", "CC"],
        "suv": ["TIGUAN", "T-ROC", "T-CROSS", "TOUAREG", "ID.4", "ID4", "ID.5", "ID5", "TAIGO"],
        "mpv": ["TOURAN", "SHARAN", "CARAVELLE", "MULTIVAN"],
        "van": ["TRANSPORTER", "CADDY", "CRAFTER"],
        "coupe": ["SCIROCCO"],
    },
    "VAUXHALL": {
        "hatchback": ["CORSA", "ASTRA", "ADAM", "VIVA", "AGILA"],
        "saloon": ["INSIGNIA", "VECTRA"],
        "suv": ["MOKKA", "CROSSLAND", "GRANDLAND", "ANTARA"],
        "mpv": ["ZAFIRA", "MERIVA", "COMBO LIFE"],
        "van": ["VIVARO", "COMBO", "MOVANO"],
    },
    "NISSAN": {
        "hatchback": ["MICRA", "LEAF", "PULSAR"],
        "saloon": ["ALTIMA"],
        "suv": ["QASHQAI", "JUKE", "X-TRAIL", "MURANO", "ARIYA"],
        "mpv": ["NOTE", "NV200 COMBI"],
        "van": ["NV200", "PRIMASTAR", "INTERSTAR", "NV400"],
        "pickup": ["NAVARA"],
    },
    "TOYOTA": {
        "hatchback": ["YARIS", "AYGO", "AURIS", "PRIUS", "COROLLA", "IQ"],
        "saloon": ["AVENSIS", "CAMRY"],
        "suv": [
            "YARIS CROSS",
            "C-HR",
            "RAV4",
            "RAV-4",
            "LAND CRUISER",
            "HIGHLANDER",
            "URBAN CRUISER",
        ],
        "mpv": ["VERSO", "PROACE VERSO", "PREVIA", "ESTIMA"],
        "van": ["PROACE CITY", "PROACE"],
        "pickup": ["HILUX"],
        "coupe": ["GT86", "SUPRA"],
    },
    "HONDA": {
        "hatchback": ["JAZZ", "CIVIC", "E"],
        "saloon": ["ACCORD", "INSIGHT"],
        "suv": ["CR-V", "HR-V", "ZR-V"],
        "mpv": ["JADE", "FR-V"],
    },
    "KIA": {
        "hatchback": ["PICANTO", "RIO", "CEED", "PRO CEED"],
        "saloon": ["OPTIMA", "STINGER"],
        "suv": ["SPORTAGE", "SORENTO", "NIRO", "STONIC", "SOUL", "XCEED", "SELTOS", "EV6"],
        "mpv": ["VENGA", "CARENS", "CARNIVAL"],
    },
    "HYUNDAI": {
        "hatchback": ["I10", "I20", "I30", "IONIQ"],
        "saloon": ["I40"],
        "suv": ["TUCSON", "SANTA FE", "KONA", "IX35", "BAYON", "NEXO"],
        "mpv": ["IX20"],
        "van": ["I800", "ILOAD"],
    },
    "AUDI": {
        "hatchback": ["A1", "A2", "A3", "S3", "RS3"],
        "saloon": ["A4", "A6", "A7", "A8", "S4", "S6", "RS4", "RS6"],
        "suv": ["Q2", "Q3", "Q4", "Q5", "Q7", "Q8", "SQ5", "E-TRON"],
        "coupe": ["A5", "S5", "TT", "R8"],
    },
    "MERCEDES-BENZ": {
        "hatchback": ["A-CLASS", "A"],
        "saloon": ["C-CLASS", "C", "E-CLASS", "E", "S-CLASS", "S", "CLA", "CLS"],
        "suv": [
            "GLA",
            "GLB",
            "GLC",
            "GLE",
            "GLS",
            "GLK",
            "GL",
            "G-CLASS",
            "G",
            "ML",
            "EQA",
            "EQB",
            "EQC",
        ],
        "mpv": ["B-CLASS", "B", "V-CLASS", "V", "VIANO"],
        "van": ["SPRINTER", "VITO", "CITAN"],
        "coupe": ["SL", "SLK", "SLC", "CLK", "AMG GT"],
    },
    "BMW": {
        "hatchback": ["1 SERIES", "2 SERIES ACTIVE TOURER", "2 SERIES GRAN TOURER", "I3"],
        "saloon": ["3 SERIES", "5 SERIES", "7 SERIES", "I4", "M3", "M5"],
        "suv": ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "IX1", "IX3", "IX"],
        "coupe": ["2 SERIES", "4 SERIES", "6 SERIES", "8 SERIES", "Z4", "I8", "M2", "M4"],
    },
    "MAZDA": {
        "hatchback": ["2", "3"],
        "saloon": ["6"],
        "suv": ["CX-3", "CX-30", "CX-5", "CX-60"],
        "coupe": ["MX-5"],
    },
    "SKODA": {
        "hatchback": ["FABIA", "SCALA", "CITIGO", "RAPID", "OCTAVIA"],
        "saloon": ["SUPERB"],
        "suv": ["KODIAQ", "KAROQ", "KAMIQ", "YETI", "ENYAQ"],
        "mpv": ["ROOMSTER"],
    },
    "SEAT": {
        "hatchback": ["IBIZA", "LEON", "MII", "TOLEDO"],
        "suv": ["ARONA", "ATECA", "TARRACO"],
        "mpv": ["ALHAMBRA", "ALTEA"],
    },
    "RENAULT": {
        "hatchback": ["CLIO", "MEGANE", "TWINGO", "ZOE"],
        "saloon": ["LAGUNA"],
        "suv": ["CAPTUR", "KADJAR", "KOLEOS", "ARKANA"],
        "mpv": ["GRAND SCENIC", "SCENIC"],
        "van": ["KANGOO", "TRAFIC", "MASTER"],
    },
    "PEUGEOT": {
        "hatchback": ["107", "108", "206", "207", "208", "306", "307", "308", "RCZ"],
        "saloon": ["508"],
        "suv": ["2008", "3008", "5008", "4007", "4008"],
        "van": ["PARTNER", "EXPERT", "BOXER"],
        "mpv": ["RIFTER", "TRAVELLER"],
    },
    "CITROEN": {
        "hatchback": ["C1", "C2", "C3", "C4", "DS3"],
        "saloon": ["C5"],
        "suv": ["C3 AIRCROSS", "C4 CACTUS", "C5 AIRCROSS"],
        "mpv": ["C3 PICASSO", "C4 PICASSO", "GRAND C4", "BERLINGO MULTISPACE"],
        "van": ["BERLINGO", "DISPATCH", "RELAY"],
    },
    "FIAT": {
        "hatchback": ["500", "PANDA", "PUNTO", "TIPO"],
        "suv": ["500X"],
        "mpv": ["500L", "QUBO", "MULTIPLA"],
        "van": ["DOBLO", "DUCATO"],
    },
    "MINI": {
        "hatchback": ["COOPER", "ONE", "MINI", "CONVERTIBLE", "ELECTRIC", "CLUBMAN"],
        "suv": ["COUNTRYMAN"],
        "coupe": ["PACEMAN"],
    },
    "LAND ROVER": {
        "suv": [
            "RANGE ROVER EVOQUE",
            "RANGE ROVER SPORT",
            "RANGE ROVER VELAR",
            "RANGE ROVER",
            "DISCOVERY SPORT",
            "DISCOVERY",
            "DEFENDER",
            "FREELANDER",
        ],
    },
    "JAGUAR": {
        "saloon": ["XE", "XF", "XJ", "X-TYPE", "S-TYPE"],
        "suv": ["F-PACE", "E-PACE", "I-PACE"],
        "coupe": ["F-TYPE"],
    },
    "VOLVO": {
        "hatchback": ["V40", "C30"],
        "saloon": ["S40", "S60", "S80", "S90", "V50", "V60", "V70", "V90"],
        "suv": ["XC40", "XC60", "XC90"],
        "coupe": ["C70"],
    },
    "TESLA": {
        "saloon": ["MODEL 3", "MODEL S"],
        "suv": ["MODEL Y", "MODEL X"],
    },
    "MG": {
        "hatchback": ["3", "4", "ZR"],
        "saloon": ["5", "6"],
        "suv": ["ZS", "HS", "GS", "MARVEL"],
    },
    "DACIA": {
        "hatchback": ["SANDERO", "SPRING"],
        "saloon": ["LOGAN"],
        "suv": ["DUSTER"],
        "mpv": ["JOGGER"],
    },
    "SUZUKI": {
        "hatchback": ["SWIFT", "BALENO", "CELERIO", "ALTO", "SPLASH"],
        "suv": ["VITARA", "SX4", "S-CROSS", "IGNIS", "JIMNY"],
    },
    "MITSUBISHI": {
        "hatchback": ["MIRAGE", "COLT"],
        "suv": ["OUTLANDER", "ASX", "SHOGUN", "ECLIPSE CROSS"],
        "pickup": ["L200"],
    },
    "PORSCHE": {
        "saloon": ["PANAMERA", "TAYCAN"],
        "suv": ["CAYENNE", "MACAN"],
        "coupe": ["911", "CAYMAN", "BOXSTER", "718"],
    },
    "LEXUS": {
        "hatchback": ["CT"],
        "saloon": ["IS", "ES", "GS", "LS"],
        "suv": ["NX", "RX", "UX", "RZ"],
        "coupe": ["RC", "LC"],
    },
    "CUPRA": {
        "hatchback": ["BORN", "LEON"],
        "suv": ["FORMENTOR", "ATECA", "TERRAMAR"],
    },
    "DS": {
        "hatchback": ["DS3", "DS4"],
        "saloon": ["DS9"],
        "suv": ["DS3 CROSSBACK", "DS7"],
    },
    "SMART (MCC)": {
        "hatchback": ["FORTWO", "FORFOUR"],
    },
}


def _flatten(raw: dict[str, dict[str, list[str]]]) -> dict[str, list[tuple[str, str]]]:
    """make -> [(prefix, body_type), ...] sorted longest-prefix-first so the
    most specific model name wins (TRANSIT CONNECT before TRANSIT)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for make, by_bt in raw.items():
        pairs = [(p, bt) for bt, prefixes in by_bt.items() for p in prefixes]
        pairs.sort(key=lambda pb: -len(pb[0]))
        out[make] = pairs
    return out


_BY_MAKE = _flatten(_RAW)
_BMW_SERIES = {
    1: "hatchback",
    2: "coupe",
    3: "saloon",
    4: "coupe",
    5: "saloon",
    6: "coupe",
    7: "saloon",
    8: "coupe",
}
_BMW_CODE = re.compile(r"^M?([1-8])[0-9]{2}")


def _prefix_match(model: str, prefix: str) -> bool:
    """``model`` begins with ``prefix`` at a model-name boundary.

    A digit-terminated prefix ("320", "A1") matches plainly so trim
    letters attach ("320D"); a letter-terminated prefix ("C", "TRANSIT")
    needs the next char to be a boundary (space / hyphen / end) so it
    can't swallow a longer unrelated model ("CITAN", "TRANSPORTER")."""
    if model == prefix:
        return True
    if not model.startswith(prefix):
        return False
    if prefix[-1].isdigit():
        return True
    return model[len(prefix)] in " -"


def body_type_for(make: str | None, model: str | None) -> str:
    """Coarse body type for a DVSA (make, model), or ``""`` if uncovered.

    Placeholder makes ("UNKNOWN") and unmapped models return ``""`` so
    callers can drop them exactly like a missing label."""
    mk = normalize_make(make)
    if not mk:
        return ""
    md = " ".join((model or "").upper().split())
    if not md:
        return ""
    for prefix, bt in _BY_MAKE.get(mk, ()):
        if _prefix_match(md, prefix):
            return bt
    if mk == "BMW":
        m = _BMW_CODE.match(md)
        if m:
            return _BMW_SERIES[int(m.group(1))]
    return ""
