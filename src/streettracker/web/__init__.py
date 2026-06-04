"""Vehicle showcase website.

A small local ``aiohttp`` site that browses the *enriched* cars across all
session outputs on the dev box: plate-read (ANPR) cars joined to DVSA MOT
make/model/year/colour, with the regularly-appearing ones featured, plus a
per-car metadata editor ("this is my car", "this is Shaun's car").

Distinct from :mod:`streettracker.device.dashboard` (which serves one live
session's static summary on the Orin): this is an off-device, cross-session,
read-enriched gallery with write-back user metadata. It reuses the
plate-anchored aggregator in :mod:`streettracker.analysis.vehicles`.

Entry point: ``streettracker showcase`` -> :func:`streettracker.web.server.main`.
"""
