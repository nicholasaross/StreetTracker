# Session files

What the tracker and each enrichment command write for a session, and what the fields mean.
[`CLAUDE.md`](../CLAUDE.md#session-files) keeps a one-line summary per file; this is the full
reference. Paths come from `SessionFiles` (`src/streettracker/common/session_files.py`) and
record fields from `src/streettracker/common/schema.py`.

Moved from `CLAUDE.md` on 2026-11-11 with the text unchanged, so it still carries dated notes
on when each field arrived.

## Output schema (preserved from NanoTracker)

Per finalized track:

| File                         | Quality         | Use                    |
| ---------------------------- | --------------- | ---------------------- |
| `{prefix}_{id}.jpg`          | q=85, ~80px     | dashboard tile         |
| `{prefix}_{id}_hq.jpg`       | q=95, ~250px    | quick color/silhouette |
| `{prefix}_{id}_main_{N}.jpg` | 4K Reolink HTTP | ALPR / make-model      |

`{prefix}` is `vehicle` or `person`. `N` is `1..max_snaps_per_track`.
The Orin prunes only the 4K `_main_` snaps (7 days); tiles and HQ crops stay.
The vehicle HQ crop (sharpest, largest sub-stream view) is the one picture
sure to show the tracked car, so `pull --only-main` fetches it since
2026-10-04 and the `/stats` fastest-cars board shows it; `pull --crops-only`
backfills older sessions without touching their JSON.

Session files:

- `{session}_events.jsonl` — appended line-per-track (crash-safe)
- `{session}_trajectories.jsonl` — one line per kept track:
  `{"track_id", "t0_unix", "points": [[dt, x1, y1, x2, y2], ...]}`, every
  3rd sub-stream bbox plus the last (≤ ~200 rows). With
  `TrackRecord.main_snap_fire_unix` / `main_snap_done_unix` it lets analysis
  fit when each 4K snap was exposed (review E1.5(c); sessions recorded from
  the 2026-10 deploy on; `common.output.read_trajectories_jsonl`)
- `{session}_data.json` — array of records, written at session end
- `{session}_meta.json` — session-level metadata + IR periods + snap_stats.
  `_data.json` / `_meta.json` / `_hourly.json` are written only at a clean
  shutdown; after a power cut or crash, `pull` rebuilds them from
  `_events.jsonl` + the meta embedded in `_summary.html` (or run
  `recover-session`), and the meta then carries a `recovered` block
  (2026-10-23). The runtime replaces each of the three in one step and
  writes `_data.json` last, so a session that has a `_data.json` has all
  three whole (code of 2026-11-13; an earlier runtime wrote them in place,
  data first)
- `{session}_hourly.json` — per-hour rollup
- `{session}_summary.html` + `index.html` — dashboard + auto-redirect
- `{session}_alpr.json` + `{session}_alpr_by_track.json` — per-image
  - per-track ALPR rollup (after running `alpr-run`). `ocr_conf` is the
    probability of the weakest decoded character, with the per-character
    probabilities in `ocr_char_probs`, when `_static_plates.json` carries
    `"ocr_conf": "min_char"`; without that stamp it is the pre-2026-09-28
    always-~1.0 value (run `alpr-rescore`). Since 2026-10-04 each read with a
    saved crop carries `plate_colour` (yellow/white/mono/unsure) and
    `colour_suspect: true` when the colour contradicts the track's direction
    (R→L shows the white front plate, L→R the yellow rear); suspects are left
    out of the rollup like `static_suspect`, and each best read carries
    `n_agree` (other snaps reading the same string). Stamp
    `"plate_colour": "hsv_v1"` in `_static_plates.json` (`alpr-run`, or
    `alpr-colour` for older sessions). Since 2026-10-09 a read carries
    `bottom_retry: true` when the fullframe crop path found it on its last
    resort: a vehicle that overlaps the hint and whose bottom-centre, but
    not its box centre, is on the road (tall vans and SUVs). The static
    filter checks these reads but doesn't learn spots from them
    (`.claude/onroad_bottom_centre_check.py`).
- `{session}_vehicles.json` — per-vehicle plate-anchored aggregation
  (after running `vehicles`); carries DVSA `make`/`model`/`year` once
  `dvsa-label` has harvested for the session
- `{session}_dvsa_labels.json` — DVSA MOT `make`/`model`/`year` per
  plate (after `dvsa-label`); `dvsa-apply` folds it onto `data.json` +
  `events.jsonl` per-track records. `conf_threshold` records the plate gate
  the track attributions reflect (absent on harvests before 2026-09-28), and
  `plate_gate` the full gate (mode + settings) since 2026-10-04
- `cross_session_repeats.json` — repeat vehicles pooled across a cohort
  of sessions (after `vehicles --across`; written to the output root)
- `{session}_makemodel.json` + `{session}_makemodel_by_track.json` —
  per-image top-k + per-track confidence-weighted CNN make/model
  predictions (after `makemodel`; `make_model_source="cnn"`)
- `{session}_bodytype.json` + `{session}_bodytype_by_track.json` —
  per-image + per-track confidence-weighted coarse body type (after
  `bodytype`; classes hatchback/saloon/suv/mpv/van/pickup/coupe). The
  stats page's "Vehicle body types" mix prefers the DVSA-model-derived
  class for plated cars and falls back to this CNN for the unplated
  majority. **CRITICAL: `bodytype` inference must use `--pad-frac 0.1`
  to match the training corpus** (the CLI default; the make model's
  0.25 loosens the crop and skews a _shape_ classifier toward "van" —
  CNN-vs-DVSA agreement dropped 79 %→42 % at 0.25 on session_20260703).
- `{session}_colour.json` + `{session}_colour_by_track.json` —
  per-image + per-track confidence-weighted CNN vehicle colour (after
  `colour`; `colour_source="cnn"`; classes white/silver/grey/black/blue/
  red/green/yellow/orange/brown/purple). **Replaces the rubbish HSV
  vote** (`common.color.vote_color`, the `color` field): the HSV path
  reads a low-res sub-stream crop with an 8-colour palette and scores
  only ~40 % grouped / 32 % fine per-track vs DVSA `primary_colour`
  (light cars drift to black/blue; can't emit silver/orange/brown/
  purple). The colour head reads the 4K snap (same "resolution is the
  lever" fix as make) → **87 % grouped / 74 % fine per-track** on the
  same held-out val cars (`.claude/color_accuracy.py`, shipped
  2026-08-11, b0@384 `models/colour_b0.pt`). **⮕ 2026-09-27: that 87 %
  scored confident passes only** — the harness skips passes whose colour
  is blank (best read < 0.5 conf, 22 % of all passes); the strict
  head-to-head (every pass counts) put that model at 44.8 % exact / 53.3 %
  grouped. It was replaced on 2026-09-27 by the clean-crop retrain (see
  [Make/model classification](history/makemodel.md#makemodel-classification)). Same `--pad-frac 0.1`
  discipline as bodytype. The HSV `color` field is left untouched
  (offline sidecar, no Orin/schema change). **Wired in 2026-08-11:** the
  `/stats` colour mix prefers DVSA `primary_colour` → this CNN → "unknown"
  (HSV was the last resort until 2026-10-04, review E0.8); the
  showcase car page shows the CNN `cnn_colour`; the control-panel
  **enrich** playbook runs `colour` after `bodytype`; and the sessions
  table shows a **Colour** enrichment badge (`has_colour` =
  `_colour_by_track.json` present). All existing sessions backfilled.
- `{session}_vehicle_boxes.json` — cache of full-frame YOLOv8m@1920
  vehicle boxes (car/moto/bus/truck) per 4K snap, written by
  `makemodel-build-uk --crop-mode plate` and the `makemodel`/`bodytype`/
  `colour` commands in fullframe crop mode, so one YOLO pass per session
  is shared by all of them. Disposable — rebuilt if deleted or if the
  detector params change (see `analysis/vehicle_locator.py`).
- `{session}_stationary_vehicles.json` — written by `makemodel`,
  `bodytype` and `colour` in fullframe crop mode (each rewrites it; the
  content is the same). The vehicles stopped in the road that the commands
  refused as a passing car's crop (`analysis/stationary.py`, 2026-11-14):
  - `vehicles`: one entry per vehicle and stretch of time, with `box` (4K
    px), `first` / `last` (UTC, the first and last snap whose crop it would
    have taken), `n_crops` and `n_tracks`.
  - `n_crops_refused`, and `n_left_without_crop` for the snaps where no
    other candidate was left.
  - `rule`: the thresholds in force.

  A session with no such file was last inferred before the guard, and its
  per-track make, colour and body type can still hold a stopped vehicle's
  values. An entry starts one track late and ends one early: the first and
  last cars to pass a stopped vehicle are not covered.
- `{session}_people.json` — per-person-track activity enrichment
  (after `people`): kind walker/jogger/cyclist + `dog_walker` flag via
  temporal+direction pairing with dog/bicycle tracks. Jogger split
  uses the showcase speed calibration (default boundary 2.5 m/s);
  cyclist beats jogger so riders (who detect as person) don't land in
  the jogger bucket. Dog pairing needs COCO class 16 in the live
  `vehicle_classes`. `class_suspect` tracks (kinematics guardrail) are
  excluded (`n_suspect_excluded`). BotSORT-split fragments merge into
  walks (`walk_id` per track; summary `walks` / `n_split_merged`; rule:
  same direction, start within [-2 s, +3 s] of the walk's end, colours
  not known-different — merges ~7.6 % of person tracks on the measured
  soaks). Since 2026-07-19 also **out-and-back round trips** (walk +
  later opposite-direction walk, same kind/dog-flag, colours compatible,
  gap 3 min–2 h; both rows share `round_trip_id`; summary
  `round_trips` + `round_trips_chance` + `away_minutes_median`).
  **~75-80 % of raw pairs are coincidence on this pavement** — the
  time-shift control (`chance_round_trips`) measures the floor and the
  stats page reports only the excess as "likely" trips. The `/stats`
  People section also mines **recurring walk patterns** (dog walks /
  joggers / cyclists recurring in a 40-min time-of-day window on ≥3
  dates, ≤1.6 walks/day so rush-hour waves don't qualify; ranked by
  regularity) — aggregate street statistics; walks are never matched
  across days (privacy line unchanged). Since 2026-07-19 also
  **door-origin ("my walks")** — the _only_ person-side individual
  feature, and deliberately non-biometric: `TrackRecord` now carries
  `entry_point_frac`/`exit_point_frac` (first/last detection centroid,
  fractional; captured by `compute_attributes` when `frame_w` is
  passed), and `analysis/walks.py` (`DoorZone` + `classify_walk_origin`)
  flags a walk as `originated`/`returned`/`round_trip` when its
  entry/exit falls in an operator-traced door polygon
  (`configs/door_zone.json`, gitignored + `.example` tracked). `people`
  tags each row's `door_origin` + counts `summary.own_trips` when a zone
  is configured. This identifies the _household's_ trips by where they
  enter/leave frame — NOT who takes them yet, that's the next step. **Forward-only**: needs the post-2026-07-19 runtime
  (schema-additive, Orin redeploy), so existing sessions currently read `unknown`;
  door zone can be assumed not to have changed, so should be able to reprocess.

JSON record fields: see `common/schema.py` (`TrackRecord`, `SessionMeta`).
