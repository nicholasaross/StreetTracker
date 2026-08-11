"""DVSA ``primary_colour`` -> coarse colour class (classifier training label).

The live colour label (``common.color.vote_color``) is a hand-tuned HSV vote
on a *low-res sub-stream* crop with an 8-colour palette. Against DVSA ground
truth it scores ~40 % (grouped) — light-coloured cars (white/silver/grey, the
plurality) drift to black/blue, and orange/brown/purple can't be emitted at
all. This module supplies the training label for a learned colour classifier
that instead reads the full-resolution 4K snap (the same "resolution is the
lever" fix that lifted the make classifier).

DVSA registers a wide, capitalised, owner-entered vocabulary (Grey, Black,
Blue, White, Silver, Red, Green, Orange, Beige, Brown, Purple, Yellow, Gold,
Bronze, Pink, and case variants like "BLACK"). :func:`colour_class_for` folds
that onto a compact set of camera-distinguishable classes:

* **white / silver / grey kept SEPARATE.** The whole point of a 4K classifier
  is to test whether it can tell these apart, which the HSV voter cannot. The
  by-car val split decides empirically whether to fold them; until then they
  stay distinct.
* Rare or off-palette registrations fold onto the nearest class (gold ->
  yellow, bronze -> brown, beige -> silver, cream -> white, navy ->
  blue, maroon/burgundy/pink -> red, turquoise -> blue, violet -> purple).
* Genuinely uninformative values (multi-colour, unknown, placeholders) return
  ``""`` so callers drop them exactly like a missing make (the
  :func:`streettracker.analysis.makemodel.bodytype.body_type_for` convention).

Thin classes are handled downstream: ``extract_crops`` counts cars per class
and the trainer's by-car split needs >=2 cars to contribute a val car, so a
class with a single registered car simply won't be learnable — no special
casing here, the same as makes.
"""

from __future__ import annotations

# Canonical colour classes (camera-distinguishable at 4K). Order is the stable
# class order used when a corpus is missing an explicit list.
COLOUR_CLASSES: tuple[str, ...] = (
    "white",
    "silver",
    "grey",
    "black",
    "blue",
    "red",
    "green",
    "yellow",
    "orange",
    "brown",
    "purple",
)

# DVSA colour string (lowercased) -> class. Identity entries make the accepted
# vocabulary explicit; the rest fold synonyms / near-neighbours. Anything not
# here (multi-colour, "not applicable", typos) -> "" (dropped).
_COLOUR_SYNONYMS: dict[str, str] = {
    # direct
    "white": "white",
    "silver": "silver",
    "grey": "grey",
    "gray": "grey",  # US spelling, just in case
    "black": "black",
    "blue": "blue",
    "red": "red",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "brown": "brown",
    "purple": "purple",
    # folds onto the nearest kept class
    "gold": "yellow",
    "bronze": "brown",
    "beige": "silver",
    "cream": "white",
    "ivory": "white",
    "navy": "blue",
    "turquoise": "blue",
    "maroon": "red",
    "burgundy": "red",
    "pink": "red",
    "violet": "purple",
}

# Registrations that carry no usable colour signal.
_COLOUR_PLACEHOLDERS = frozenset(
    {"", "unknown", "not known", "not applicable", "none", "multi-colour", "multicolour"}
)


def colour_class_for(primary_colour: str | None) -> str:
    """DVSA ``primary_colour`` -> a :data:`COLOUR_CLASSES` label, or ``""``.

    Lowercases/strips, then maps via :data:`_COLOUR_SYNONYMS`. Placeholder
    or off-vocabulary values return ``""`` so callers drop them like a
    missing label (mirrors ``body_type_for`` / ``normalize_make``)."""
    name = " ".join(str(primary_colour or "").lower().split())
    if name in _COLOUR_PLACEHOLDERS:
        return ""
    return _COLOUR_SYNONYMS.get(name, "")
