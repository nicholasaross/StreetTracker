"""Where each of a session's files lives.

A session is one directory, ``<output root>/<label>/``, and every file the
tracker or an enrichment command writes about it is named
``<label>_<suffix>`` inside that directory. :class:`SessionFiles` spells
those names once, so readers and writers can't drift apart on a suffix.

The property list doubles as an index of the per-session artefacts; the
format of each is described where it is written (and in
``docs/session_files.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SessionFiles:
    """Paths of one session's files. Nothing here touches the disk."""

    session_dir: Path

    @property
    def label(self) -> str:
        """The session label, e.g. ``session_20261004_095219``."""
        return self.session_dir.name

    def _file(self, suffix: str) -> Path:
        return self.session_dir / f"{self.session_dir.name}{suffix}"

    # --- written by the tracker (device runtime) ---------------------------

    @property
    def events(self) -> Path:
        """One finalised track per line, appended live (crash-safe)."""
        return self._file("_events.jsonl")

    @property
    def ir_events(self) -> Path:
        """IR-mode start/end transitions, appended live."""
        return self._file("_ir_events.jsonl")

    @property
    def trajectories(self) -> Path:
        """One thinned bbox trajectory per kept track (since 2026-10-04)."""
        return self._file("_trajectories.jsonl")

    @property
    def data(self) -> Path:
        """Every track record as one array; written at a clean shutdown."""
        return self._file("_data.json")

    @property
    def meta(self) -> Path:
        """Session metadata, IR periods and snap statistics."""
        return self._file("_meta.json")

    @property
    def hourly(self) -> Path:
        """Per-hour rollup."""
        return self._file("_hourly.json")

    @property
    def summary_html(self) -> Path:
        """The session's dashboard page."""
        return self._file("_summary.html")

    # --- plate reading (alpr-run and friends) ------------------------------

    @property
    def alpr(self) -> Path:
        """Per-image plate reads."""
        return self._file("_alpr.json")

    @property
    def alpr_by_track(self) -> Path:
        """Per-track rollup of the plate reads (best read per track)."""
        return self._file("_alpr_by_track.json")

    @property
    def static_plates(self) -> Path:
        """Static-plate spots plus the stamps recording how ALPR was run."""
        return self._file("_static_plates.json")

    @property
    def alpr_labels(self) -> Path:
        """Hand-entered plate labels (alpr-label)."""
        return self._file("_alpr_labels.json")

    @property
    def alpr_scores(self) -> Path:
        """Pipeline scores against the labels (alpr-score)."""
        return self._file("_alpr_scores.json")

    @property
    def alpr_report_html(self) -> Path:
        """Pipeline comparison page (alpr-report)."""
        return self._file("_alpr_report.html")

    # --- vehicle enrichment ------------------------------------------------

    @property
    def dvsa_labels(self) -> Path:
        """DVSA make/model/year per plate (dvsa-label)."""
        return self._file("_dvsa_labels.json")

    @property
    def vehicles(self) -> Path:
        """Per-vehicle aggregation (vehicles)."""
        return self._file("_vehicles.json")

    @property
    def vehicle_boxes(self) -> Path:
        """Cache of full-frame vehicle boxes per 4K snap."""
        return self._file("_vehicle_boxes.json")

    @property
    def stationary_vehicles(self) -> Path:
        """Vehicles found stopped in the road, which the CNN commands
        refuse as a passing car's crop."""
        return self._file("_stationary_vehicles.json")

    @property
    def makemodel(self) -> Path:
        """Per-image make predictions (makemodel)."""
        return self._file("_makemodel.json")

    @property
    def makemodel_by_track(self) -> Path:
        """Per-track make prediction."""
        return self._file("_makemodel_by_track.json")

    @property
    def bodytype(self) -> Path:
        """Per-image body-type predictions (bodytype)."""
        return self._file("_bodytype.json")

    @property
    def bodytype_by_track(self) -> Path:
        """Per-track body type."""
        return self._file("_bodytype_by_track.json")

    @property
    def colour(self) -> Path:
        """Per-image colour predictions (colour)."""
        return self._file("_colour.json")

    @property
    def colour_by_track(self) -> Path:
        """Per-track colour."""
        return self._file("_colour_by_track.json")

    # --- people enrichment -------------------------------------------------

    @property
    def people(self) -> Path:
        """Per-person-track activity (people)."""
        return self._file("_people.json")
