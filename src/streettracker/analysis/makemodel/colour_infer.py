"""Run the learned colour classifier over a closed session's 4K snaps.

The off-device batch counterpart to ``makemodel-train-uk --target colour``.
Sibling to :mod:`infer` (make) and :mod:`bodytype_infer` (body type): the crop,
the preprocessing and the command itself are shared
(:mod:`~streettracker.analysis.makemodel.session_infer`); this module is the
``colour`` head, emitting white/silver/grey/black/blue/red/green/... .

It exists because the live HSV vote (``common.color.vote_color``) reads a
*low-res sub-stream* crop with an 8-colour palette and scores ~40 % against
DVSA (light cars drift to black/blue; orange/brown/purple can't be emitted).
Reading the full-resolution 4K snap with a trained head is the fix -- the same
"resolution is the lever" move that lifted the make classifier. Predictions
carry ``colour_source="cnn"``, distinct from the HSV ``color`` field, which is
left untouched.

Reads the session's ``*_main_*.jpg`` snaps + per-snap BotSORT bboxes, crops
each to the tracked vehicle, classifies, and writes::

    <session>_colour.json            # per-image top-1 (colour, conf)
    <session>_colour_by_track.json   # per-track confidence-weighted best

CLI::

    streettracker colour <session_dir>
        [--model PATH]            # default: analysis/makemodel/models/colour_b0.pt
        [--conf-threshold 0.5] [--pad-frac 0.1] [--input-size N] [--limit N] [--cpu]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streettracker.analysis.makemodel.session_infer import (
    TopOneClassifier,
    TopOneInference,
    aggregate_top_one,
)

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "colour_b0.pt"


class ColourClassifier(TopOneClassifier):
    """Loaded colour checkpoint + cropper + preprocessing."""

    head = "colour"
    label = "colour"
    # Colour is read from the vehicle's paint, so crop tight (like body type,
    # unlike the make model's looser 0.25) to keep road/sky/adjacent cars out
    # of the vote. MUST match the training corpus crop (makemodel-build-uk
    # defaults to 0.1 for the colour/body-type corpora) -- a looser pad drags
    # background in and biases the prediction.
    DEFAULT_PAD_FRAC = 0.1

    @property
    def colours(self) -> tuple[str, ...]:
        """The head's class names, in the checkpoint's order."""
        return self.class_names


def aggregate_by_track(
    per_image_rows: list[dict[str, Any]], conf_threshold: float
) -> list[dict[str, Any]]:
    """Confidence-weighted vote of each snap's top-1 into a per-track best.

    Sums per-colour top-1 confidence across a track's snaps and takes the
    argmax. ``colour`` is emitted only when the winner's best single read
    clears ``conf_threshold`` (else ``None``); ``conf`` is kept regardless."""
    return aggregate_top_one(
        per_image_rows, conf_threshold, field="colour", source_field="colour_source"
    )


class _ColourInference(TopOneInference[ColourClassifier]):
    tag = "colour"
    classifier = ColourClassifier
    source_field = "colour_source"
    train_command = "makemodel-train-uk ... --target colour"
    pad_help = (
        "crop context pad (default: the checkpoint's training pad, else 0.1); "
        "MUST match the training corpus — a looser "
        "pad drags background into the vote (see class docstring)"
    )

    def default_model(self) -> Path:
        return DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``streettracker colour <session_dir>``."""
    return _ColourInference().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
