"""colour_class_for: DVSA primary_colour -> coarse colour class."""

from __future__ import annotations

import pytest

from streettracker.analysis.makemodel.colour import (
    _COLOUR_SYNONYMS,
    COLOUR_CLASSES,
    colour_class_for,
)


@pytest.mark.parametrize(
    ("dvsa", "expected"),
    [
        # DVSA's capitalised vocabulary -> lowercase class.
        ("White", "white"),
        ("Silver", "silver"),
        ("Grey", "grey"),
        ("Black", "black"),
        ("Blue", "blue"),
        ("Red", "red"),
        ("Green", "green"),
        ("Yellow", "yellow"),
        ("Orange", "orange"),
        ("Brown", "brown"),
        ("Purple", "purple"),
        # Case / whitespace variants seen in the harvest ("BLACK", "SILVER").
        ("BLACK", "black"),
        ("  silver ", "silver"),
        ("GRAY", "grey"),  # US spelling
        # Near-neighbour folds onto a kept class.
        ("Gold", "yellow"),
        ("Bronze", "brown"),
        ("Beige", "silver"),
        ("Cream", "white"),
        ("Navy", "blue"),
        ("Turquoise", "blue"),
        ("Maroon", "red"),
        ("Burgundy", "red"),
        ("Pink", "red"),
    ],
)
def test_known_colours(dvsa: str, expected: str) -> None:
    assert colour_class_for(dvsa) == expected


def test_placeholder_and_unknown_return_empty() -> None:
    """Uninformative registrations drop like a missing make ("")."""
    assert colour_class_for(None) == ""
    assert colour_class_for("") == ""
    assert colour_class_for("Unknown") == ""
    assert colour_class_for("Not known") == ""
    assert colour_class_for("Multi-colour") == ""
    assert colour_class_for("some-future-colour") == ""


def test_all_synonym_targets_are_canonical() -> None:
    """Every class the map can emit is a declared COLOUR_CLASSES member."""
    assert set(_COLOUR_SYNONYMS.values()) <= set(COLOUR_CLASSES)


def test_white_silver_grey_stay_distinct() -> None:
    """The 4K classifier's whole point is telling these apart -- they must
    NOT collapse the way vehicles._colour_group folds them to 'light'."""
    assert len({colour_class_for(c) for c in ("White", "Silver", "Grey")}) == 3
