"""Run the body-type classifier over a closed session's 4K snaps.

The off-device batch counterpart to ``makemodel-train-uk --target
body_type``. Sibling to :mod:`infer` (the make classifier): the crop, the
preprocessing and the command itself are shared
(:mod:`~streettracker.analysis.makemodel.session_infer`); this module is the
``body_type`` head, emitting a coarse silhouette class {hatchback, saloon,
suv, mpv, van, pickup, coupe}. Its value is the *unreadable majority*: a
plated car already has a reliable body type from its DVSA model (see
:func:`streettracker.analysis.makemodel.bodytype.body_type_for`), so the
model earns its keep on the cars with no plate.

Reads the session's ``*_main_*.jpg`` snaps + per-snap BotSORT bboxes,
crops each to the tracked vehicle, classifies, and writes::

    <session>_bodytype.json           # per-image top-1 (class, conf)
    <session>_bodytype_by_track.json  # per-track confidence-weighted best

**Verdict (2026-07-09):** ~0.69 per-crop / 0.67 per-car overall, but the
signal is concentrated -- hatchback + suv (77 % of cars) read ~0.75,
the minority classes are weak. Consumers should treat it as a coarse,
best-effort estimate (the ``conf`` field + a threshold let them gate).

CLI::

    streettracker bodytype <session_dir>
        [--model PATH]            # default: analysis/makemodel/models/bodytype_b0.pt
        [--conf-threshold 0.5] [--pad-frac 0.25] [--input-size N] [--limit N] [--cpu]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streettracker.analysis.makemodel.session_infer import (
    TopOneClassifier,
    TopOneInference,
    aggregate_top_one,
)

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "bodytype_b0.pt"


class BodyTypeClassifier(TopOneClassifier):
    """Loaded body-type checkpoint + cropper + preprocessing."""

    head = "body_type"
    label = "body-type"
    # Body type is read from the silhouette, so the crop framing must match
    # training: makemodel-build-uk crops at pad_frac=0.1, and a looser
    # inference pad (the make model's 0.25) shrinks the car in-frame and
    # skews a *shape* classifier hard toward boxy/van predictions -- on
    # session_20260703 it dropped CNN-vs-DVSA body-type agreement from 79 %
    # (pad 0.1) to 42 % (pad 0.25), inflating "van" from 15 % to 33 %.
    DEFAULT_PAD_FRAC = 0.1

    @property
    def body_types(self) -> tuple[str, ...]:
        """The head's class names, in the checkpoint's order."""
        return self.class_names


def aggregate_by_track(
    per_image_rows: list[dict[str, Any]], conf_threshold: float
) -> list[dict[str, Any]]:
    """Confidence-weighted vote of each snap's top-1 into a per-track best.

    Sums per-body-type top-1 confidence across a track's snaps and takes
    the argmax. ``body_type`` is emitted only when the winner's best single
    read clears ``conf_threshold`` (else ``None``); ``conf`` is kept
    regardless for debugging."""
    return aggregate_top_one(per_image_rows, conf_threshold, field="body_type")


class _BodyTypeInference(TopOneInference[BodyTypeClassifier]):
    tag = "bodytype"
    classifier = BodyTypeClassifier
    train_command = "makemodel-train-uk ... --target body_type"
    pad_help = (
        "crop context pad (default: the checkpoint's training pad, else 0.1); "
        "MUST match the training corpus — a looser "
        "pad skews the shape classifier toward van (see class docstring)"
    )

    def default_model(self) -> Path:
        return DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``streettracker bodytype <session_dir>``."""
    return _BodyTypeInference().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
