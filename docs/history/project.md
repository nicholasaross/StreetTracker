# Project history: checkpoints, deploys, people side, migration

> Moved from `CLAUDE.md` on 2026-11-11. The text is unchanged apart from link targets and
> heading levels. It is a dated record, written as the work happened: later entries correct
> earlier ones, and passages marked SUPERSEDED, CORRECTION or UPDATE say where. "Above" and
> "below" refer to the old file's layout. For what is true now, read "Current state" in
> [`CLAUDE.md`](../../CLAUDE.md).

## ⮕ Start here — 2026-06-13 checkpoint

Everything from the last sprint is **merged (#57–#64) and on `main`** (clean
tree). The detailed history lives in the sections below; this block is the
"resume from cold" summary.

The handoff that followed this paragraph in the old file is in
[plate_gate.md](plate_gate.md).

### Live on the Orin (as recorded 2026-11-10)

**Live on the Orin** (#63 runtime bundle deployed 2026-06-13; code updated to
`5f4ceb4` on 2026-10-04 for E1.5(c), see step 6.2 above; `6707789` on
2026-10-26 for the startup wait for the boot's first NTP sync (PR #136) and
`YOLO_AUTOINSTALL=false` (#137); **now `0b817b7`**, pulled on 2026-11-10 and
started by a panel roll at 17:18 GMT; service active, NRestarts=0). Its one
behaviour change over `6707789` is the summary-page stall fix (PR #145, see
[Device runtime notes](../../CLAUDE.md#device-runtime-notes-orin-nano-8gb-super)); it also
carries the shared file helpers (#144, imports only on the device) and
formatting / type annotations (#143). `session_20261110_171836` is the first
session recorded with the fix. Rollback: `git checkout 6707789` + restart.
Still live from earlier deploys:

- **per-direction pipeline bands** `pipeline_t_usable_by_direction =
{forward(R→L):[0.10,0.20], reverse(L→R):[0.30,0.60]}` (reverse lower edge
  nudged 0.25→0.30 on 2026-06-19, **validated 2026-06-22** on the ~72h
  `session_20260619_111111` soak — L→R per-car 73-82 % → **92.3 %**, net
  per-car ~48 % → **57.9 %**). **SUSTAINED-RATE CORRECTION (2026-07-09):
  that 57.9 % was ONE favorable soak — three later completion-bbox
  sessions (0628/0703/0707) cluster at net **~40-45 %** (L→R ~73-80 %,
  R→L ~9-11 %), and 0628 predates the 07-07 dog-class deploy so it's
  session variance, not snap-contention. The reverse-band conclusion is
  unchanged (L→R = win, R→L HW-only); only the headline number was
  over-anchored on one good soak — quote ~40-45 % net, not 57.9 %.**
  **Post-deploy verdict
  (2026-06-19, two ~70h soaks) falsified the R→L half** of the 2026-06-11
  re-analysis: L→R is the win (per-car ~73-82 %), R→L is camera-geometry-capped
  at ~15-20 % at _every_ position (the "R→L reads far" finding was a
  motion-window-hint artifact that completion-time bboxes corrected; honest net
  per-car ~48 %, not the inflated 63.9 %). R→L now needs hardware, not tuning.
  Detail in the [snap-gate config layout](../../CLAUDE.md#snap-gate-config-layout) section.
  **⮕ SUPERSEDED 2026-07-28 — R→L SOLVED IN SOFTWARE (read this before
  quoting anything above):** every "R→L camera-geometry-capped / HW-only /
  needs hardware" claim in this bullet is **falsified**. The cap was
  contaminated crops (recorded done-bboxes sit a median ~510 4K px off the
  car + ~34-53 % of R→L "reads" were static parked/background plates), NOT
  geometry. The full-frame trajectory-matched crop path (`alpr-run
--crop-mode fullframe`, the default since 2026-07-28, PR #94/#95)
  recovered R→L to **per-car parity in software** — production **R→L 69.7 %
  / L→R 66.8 %, net 68.3 %** (was ~9-17 % R→L / ~45 % net). No 2nd camera.
  The remaining variance is now time-of-day (day ~74.7 % vs dusk/dark
  ~45.4 %) → the residual capture lever is **night capture, characterised
  2026-08-12 as MOTION BLUR, not underexposure** (night plates are still
  detected but glyph-smeared → fix is a faster night shutter + IR fill, NOT
  brightening). Not geometry. See Next-steps item 5.
- **`vehicle_classes = [0,1,2,3,5,7]`** (person+car+bike+moto+bus+truck; was
  `[0,2]`).
- **completion-time bbox capture** (`TrackRecord.main_snap_bboxes_done`) —
  sessions recorded _after_ the deploy carry the bbox at snap-landing time, and
  `alpr-run` prefers it (exact crops, no extrapolation).
- rollback: on-device `configs/camera.json.bak.20260613T075913Z`, git `523c820`.

**State:** `session_20260608_125639` pulled (56 GB) + enriched (alpr + dvsa =
1,174 new labels, 59 parked beacons suppressed). Showcase live on **:8090**
(~2,700 cars).

### People side

**People side (shipped 2026-07-07):** `streettracker people` writes
`{session}_people.json` — walker/jogger/cyclist kinds (jogger ≥2.5 m/s,
the empirical valley; cyclist via bicycle-track pairing so riders don't
pollute joggers) + `dog_walker` via temporal+direction pairing (PR #70).
Panel-integrated: allowed job kind (net lane), 6th enrich-playbook step,
sessions-table badge (PR #71). All 23 local sessions backfilled (~28k
classified: 76 % walkers / 23 % joggers / 1 % cyclists; pre-06-13 jogger
counts include unfiltered riders, ~+4pp — no bicycle class then).
**Dog class 16 live on the Orin** since 2026-07-07 08:11
(`vehicle_classes=[0,1,2,3,5,7,16]`, rollback
`camera.json.bak.20260707T065646Z`); 3 dog tracks in the first 5 h, so
YOLOv8m sees dogs here. Coverage soak (`.claude/person_coverage.py`,
5 soaks / 13,359 person tracks): 66.7 % of person tracks get a 4K snap
(cars 94.4 %) — gap is short-dwell/BotSORT-splits, NOT band geometry;
no person-specific snap gate needed. Both ancestor repos archived
2026-07-07 (migration closed).

### The 2026-07-07 next-steps list

**Next steps (updated 2026-07-07), priority order** (older list; the
2026-09-28 handoff above takes priority):

1. **Merge PR #72** (stats-page people kinds; CI green, awaiting operator
   merge) → restart the showcase on :8090 to pick it up.
2. **Dog-walk first light — tokenless operator loop.** Let the live
   session soak a few days → panel **roll** → **enrich** (now writes
   `_people.json`) → first real `dog_walkers` count on `/stats`.
3. **Person-count hardening:** (a) class-flip guardrail — cross-check
   `class_name` against `speed_px_s` + bbox aspect ratio at finalize
   (kills the parked-car-labelled-person corruption, hardens footfall);
   (b) same-session walk dedup — BotSORT splits inflate walk counts;
   within-session appearance matching only (reuse `vote_color` infra),
   deliberately NO cross-session person re-id (privacy line).
4. Make classifier (B6, the VLM bake-off, body type): moved to
   [makemodel.md](makemodel.md#b6-the-vlm-bake-off-and-body-type-2026-07-08-to-2026-09-25).
5. R→L solved in software, and night capture: moved to
   [anpr_tuning.md](anpr_tuning.md#rl-solved-in-software-night-capture-2026-07-28-to-2026-10-05).
6. **JP7:** wheel gate re-checked 2026-07-07 — still no jp7 index.
   Re-run `scripts/check_jp7_wheel.ps1` before any flash; stay on JP6.
7. **Parked by choice — people P0 (retention policy).** Deliberately
   deprioritised (operator call, 2026-07-07). Person 4K imagery
   accumulates indefinitely on the dev box (Orin prunes at 7 d, dev box
   never); records-forever/sweep-images-after-N-days is the sketched
   design. Revisit when ready — it stays last until the operator says
   otherwise.

**Ops (learned the hard way):** launch long jobs **detached (WMI/VBS, outside
the Claude helper tree)** AND **verify completion by artifact mtimes, never a
log tail** — a reaped batch's frozen log is indistinguishable from a working
one (cost ~2 days, 2026-06-11→13). **The dev box idle-sleeps** (~Kernel-Power
42; killed the 06-13 overnight build) — unattended long jobs need a
`SetThreadExecutionState` wake-lock (pattern in `.claude/rebuild_0614.ps1`) or
run them daytime while active. Showcase restart: `uv run streettracker
showcase --output-root output --port 8090` (detached). Orin prunes 4K snaps
after **7 days** — pull within the week. `pull --skip-existing` is resumable +
large-session-safe since #64.

## Migration status

Clean-slate replacement for VehicleTracker + NanoTracker. **Migration
complete** — both ancestor repos archived on GitHub 2026-07-07.

| Phase | Scope                                                                   | Status                                                                                                                 |
| ----- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 0     | repo init + pyproject + CI + configs                                    | done                                                                                                                   |
| 1     | `common/`: schema, color, output, hourly, summary                       | done                                                                                                                   |
| 2     | `inference/` (Ultralytics) + `sources/` (RTSP, file)                    | done                                                                                                                   |
| 3     | `device/`: live runtime, snapshotter, dashboard, IR                     | done — PRs #7/#8/#10/#11/#12/#13                                                                                       |
| 4a    | `analysis/`: recolor + debug-color                                      | done                                                                                                                   |
| 4b    | `analysis/alpr/` wholesale port                                         | done                                                                                                                   |
| 5     | CLI: `pull`, `export-engine`, `setup_orin.sh`, systemd                  | done                                                                                                                   |
| 6     | (opt) Nano archive role                                                 | not started                                                                                                            |
| 7     | cutover: enable systemd on Orin + decommission Nano + archive old repos | **done** — Orin live since 2026-05-22; `VehicleTracker` + `NanoTracker` archived 2026-07-07 with superseded-by banners |

## Cutover status

Phase 7 cutover happened **2026-05-22**. Orin `streettracker.service`
active since 15:42 BST; old Jetson Nano `nano_tracker.py` SIGTERM'd
cleanly after 986,619 frames over 27.6h. **Repo archival DONE
2026-07-07** — both repos archived with a "Superseded by StreetTracker
as of 2026-07-07; read-only" banner prepended to each CLAUDE.md.

Two config-mismatch bugs surfaced + hardened in PR #15:

1. **Stream-name lookup is forgiving.** `cli/run.py` falls back to
   matching `stream.quality` when `nano.preferred_stream` doesn't match
   any `stream.name`. Matters for configs migrated from NanoTracker.
2. **`engine_path` validated at config load.** Missing engine file
   surfaces as `ConfigError: $.inference.engine_path: file ... does not
exist (resolved to <abs>)` at startup instead of a 5s-later
   Ultralytics traceback.

A fresh deploy where you scp the live `configs/camera.json` back into
place Just Works without manual edits even if engine target or stream
names differ from the example.

## Summary-page stall (found and fixed 2026-11-10)

From the old "Device runtime notes" list.

- **Summary-page stall: found, fixed and deployed 2026-11-10 (live from
  `session_20261110_171836`; confirmed on that session 2026-11-13).**
  The runtime rebuilt the summary page inline in `process_frame`, 11.76 s
  after each kept track ended (10 s idle + the 16 frames a track takes to
  expire), converting every record the session held. On the Orin that took
  **0.5 s + 0.24 ms per record** (0.75 s at 1,000 records, 2.4 s at 8,000),
  and no frame was processed meanwhile. Measured on the seven sessions with
  trajectory logs (`.claude/loop_stall_check.py`, 46,376 tracks): **5.5 % of
  kept tracks end at a rebuild**, about 2.2 % are one pass recorded as two
  tracks, and 4K snaps in flight at a rebuild get a late completion time. So
  some "BotSORT splits" were self-inflicted, and data late in a long session
  is worse than early. The fix keeps the page's rows ready
  (`common.summary.SummaryRows`, one row serialised per kept track at
  finalise) and writes the files from a worker thread
  (`runtime.start_html_write`); the on-loop cost of a rebuild at 8,000 records
  drops from ~1.4 s to ~3 ms on the dev box. Timed on the Orin itself before
  the restart, on the then-live session's 6,838 records: the old page
  generation took 1,254 ms, the new render 4.9 ms (plus a 14 ms file write in
  the worker thread), with identical output. **Still to do:** once
  `session_20261110_171836` has been rolled and pulled, run
  `uv run --no-sync python .claude/loop_stall_check.py output session_20261110_171836`.
  Tracks ending at a rebuild instant should be down to the chance level (a
  pre-fix session shows about 13 times chance, e.g. 479 against 36), with no
  pile of ends at the instant. Sessions recorded before the deploy keep their
  splits.
- **Confirmed 2026-11-13** on `session_20261110_171836`, the first session
  recorded with the fix, with the same script. Tracks ending at a rebuild
  instant: 29 against 21 expected by chance (pre-fix: about 13 times chance),
  an excess of 8 = 0.2 % of kept tracks (was 5.5 %). 29 of 2,309 rebuilds
  coincided with a track end (1.3 %); about one pass was recorded as two
  tracks (0.0 %, was about 2.2 %). No track showed a logged-point gap at a
  rebuild, so the stall-length table is empty. 4K snaps fired in the 0.7 s
  before a rebuild landed in a median 0.54 s (n = 183) against 0.55 s for all
  others (n = 11,579).

## Rest of the Orin runtime work from the 2026-11-10 review (2026-11-13)

Three changes to `device/runtime.py`, the remainder of the review's Tranche 3.
On `main`; not deployed when written (the Orin stayed on `0b817b7`).

- **One sharpness measurement per track per frame.** `process_frame` walked
  the active tracks three times (trigger, pipeline, door) and measured the
  Laplacian sharpness of the same box on the same frame in both the trigger
  and the pipeline pass. It now measures each track once
  (`_probe_active_tracks`) and the three passes share the result. The passes
  were kept whole and in their old order on purpose: the snapshotter refuses
  a submit once `max_concurrent` snaps are in flight, so the order of submits
  across tracks decides which fire is dropped, and a track-by-track loop
  would have let one car's pipeline fire take the slot another car's trigger
  crossing needed. Checked by loading the old and new `runtime.py` side by
  side and driving both through 600 randomised scenes with the real
  `SnapPlanner` (road-gate and right-half modes, pipeline on and off, a door
  zone on half of them, a snapshotter that drops at a cap of 1, 2 or 4):
  7,601 fire attempts, 1,899 drops and 2,745 blur skips, with every submit,
  drop, counter and record identical. `process_frame` was split into
  `_update_ir` and `_plan_snaps` and no longer carries `# noqa: C901`.
- **The shutdown drain runs on the event loop.** `_drain_active` used to run
  in a worker thread under `asyncio.wait_for`. A finalise-time rescue fire
  there would have called `asyncio.create_task` off the loop and raised, and
  the exception would have left `run_session`'s `finally` block before
  `write_session_outputs`, so the session would have ended with no
  `_data.json`. It never happened live (the road gate never rescues, and
  `batch` has no snapshotter). A thread that outlived its timeout would also
  have gone on appending records while the outputs were written. The drain
  now runs on the loop, yields between tracks, stops at `shutdown_grace_s`
  (logging how many tracks it did not reach), and logs a track that fails to
  finalise instead of abandoning the rest. `IRDetector.open_period_start_unix`
  replaces the runtime's reach into the detector's private field.
- **End-of-session JSON is atomic, `_data.json` last.** `save_data_json`,
  `save_meta_json` and the hourly rollup go through
  `common.fileio.atomic_write_json`, in the order hourly, meta, data (it was
  data, meta, hourly, each written in place). `pull` decides a session needs
  rebuilding by a missing `_data.json`, so a shutdown killed part-way
  (systemd's stop timeout) is now always rebuilt; before, a kill after
  `_data.json` left a session that looked finalised with no meta or hourly
  file. File contents are byte-for-byte what they were.
