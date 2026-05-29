"""Vehicle make/model + vehicle-type classification (off-device).

Phase 1 of the make/model plan: read a closed session's 4K snaps +
``main_snap_bboxes``, crop each vehicle into a classifier-friendly
square, and emit per-track attributes (vehicle type now, full
make/model in Phase 2). The DVSA labelled corpus from Phase 0.3 is
the regression test for both heads.

Layout mirrors :mod:`streettracker.analysis.alpr`:

- ``cropper`` -- bbox-hint vehicle crop for classifier input.
  Distinct from :mod:`streettracker.analysis.alpr.precrop`'s
  ``PreCropDetector`` which crops *tight* for the plate detector;
  this one crops *with context* (~25 % pad, square aspect) for the
  make/model classifier.
- Future: ``classifier``, ``training/``, ``cli``.
"""
