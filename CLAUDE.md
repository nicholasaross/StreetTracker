# CLAUDE.md <img src="docs/assets/st26trk.svg" alt="ST26 TRK" height="40" align="right"/>

Guidance for Claude Code when working in StreetTracker.

**System map (2026-10-26): [`docs/architecture/`](docs/architecture/README.md)**, published
as a private artifact. It is a one-page architecture of the hardware, the live pipeline,
enrichment, training, storage and sites, and gives every component a discussion ID (`RT-6`,
`EN-5` …). Update and republish it when a component changes; the README says how.

## How this file is organised

This file holds what every session needs: the current state, the standing rules, the commands and
the deploy order. Dated results and procedures for particular jobs live in `docs/`. Read the one
that matches the task before starting.

| Before you… | Read |
| --- | --- |
| change the snap gate, move a trigger or re-sketch the road | [`docs/snap_gate.md`](docs/snap_gate.md) |
| redeploy or reflash the Orin, restore the dev box or plan the JetPack 7 move | [`docs/deployment.md`](docs/deployment.md) |
| read, write or extend a per-session file | [`docs/session_files.md`](docs/session_files.md) |
| change the showcase site or the control panel | [`docs/sites.md`](docs/sites.md) |
| quote a plate-read rate, or revisit a capture or crop setting | [`docs/history/anpr_tuning.md`](docs/history/anpr_tuning.md) |
| retrain, compare or promote a classifier | [`docs/history/makemodel.md`](docs/history/makemodel.md) |
| change the plate gate, or pick up the data-integrity plan | [`docs/data_integrity_review.md`](docs/data_integrity_review.md) and [`docs/history/plate_gate.md`](docs/history/plate_gate.md) |
| need earlier project history (migration, cutover, people side, old plans) | [`docs/history/project.md`](docs/history/project.md) |

**Keep this file short.** When a result lands, edit "Current state" so it says what is true now,
and append the dated account (numbers, method, what it replaced) to the matching file in
`docs/history/`. Don't add a correction under an old claim here: replace the claim.

## Current state (2026-11-11)

### Live on the Orin

- **Code `0b817b7`**, pulled 2026-11-10 and started by a panel roll at 17:18 GMT. Rollback:
  `git checkout 6707789` + restart. JetPack 6, Python 3.10.
- **`session_20261110_171836`** is the first session recorded with the summary-page stall fix
  (see [Device runtime notes](#device-runtime-notes-orin-nano-8gb-super)). Its check is still to
  run (Open work, item 1).
- **Config.** `vehicle_classes = [0,1,2,3,5,7,16]` (person, bicycle, car, motorcycle, bus, truck,
  dog). The snap gate is the road polygon with six triggers, plus pipeline mode (400 ms, 15 per
  track) with per-direction bands. The live block is under
  [Snap-gate config layout](#snap-gate-config-layout).
- **Camera.** Forced colour (`dayNight: Color`) since 2026-09-28. In IR mode the frames are
  monochrome and the runtime skips inference, so the tracker is blind (it was for 41.7 % of
  September). Don't return the camera to automatic day/night, or add an IR illuminator, without
  an IR-mode inference option (review R16).
- **Retention.** The Orin deletes 4K snaps after 7 days and keeps tiles, HQ crops and JSON. Pull a
  session within the week.
- **Fields by recording date.** A session recorded before a deploy lacks that deploy's fields;
  treat a missing field as unknown. Bicycle, motorcycle, bus and truck tracks and
  `main_snap_bboxes_done` date from 2026-06-13; dog tracks and `class_suspect` from 2026-07-07;
  `entry_point_frac` / `exit_point_frac` from 2026-07-19; `main_snap_fire_unix` /
  `main_snap_done_unix` and `_trajectories.jsonl` from 2026-10-04.

### Plate reading

- **Command.** The panel's enrich playbook runs `alpr-run <session> --pipeline preferred
  --pre-crop --ghost-mask .claude/ghost_mask.json`. The bespoke pipeline adds no useful signal.
- **Crops.** `alpr-run` defaults to `--crop-mode fullframe`: it detects vehicles in the whole 4K
  frame and picks the one that matches the track's trajectory. `--crop-mode hint` is the legacy
  path, which cropped around a box the car had already left.
- **Read rate.** Per car, on canonical UK plates with fullframe crops (2026-07-28): R→L 69.7 %,
  L→R 66.8 %, net 68.3 %. Direction is not the limit and no second camera is needed. Every
  earlier "R→L is capped by camera geometry" claim was a crop artefact.
- **Low light is the limit.** Day (06-18 h) reads about 75 % per car; dusk and dark about 45 %.
  The cause is motion blur (plates are found but smeared), plus R→L front-plate detection falling
  to 17-41 % at 07-08 h and 18-19 h under headlight glare. As the days shorten this covers both
  rush hours: gated reads at 17:00 were 54.5 % in July and 5.2 % on 1 October. The lever is a
  faster shutter scheduled for 07-09 h and 17-19 h. It is the operator's decision (`SetIsp` needs
  the camera's admin account). Brightening or a longer exposure makes it worse, and a shutter cap
  all day cost about 23 pp in daylight.
- **Night counts are sound.** In forced colour, weekday dark-hour car counts match summer
  daylight: nights are quiet, not missed.
- **Confidence.** `ocr_conf` is the probability of the weakest decoded character, on sessions
  stamped `"ocr_conf": "min_char"` in `_static_plates.json`. Re-score an unstamped session with
  `alpr-rescore`.
- **Plate gate.** One shared gate, `configs/alpr.json` (gitignored; template
  `configs/alpr.example.json`), read by `dvsa-label`, `vehicles`, the showcase and `/stats`
  through `analysis/alpr/gate.py` (`PlateGate`). It is set to `"plate_gate": "combined"`: a read
  counts at minimum-character confidence ≥ 0.80 when another snap of the track agrees, or when at
  least two tracks anywhere read the same plate. Reads whose plate colour contradicts the
  direction (`colour_suspect`; R→L shows the white front plate, L→R the yellow rear) and static
  parked plates (`static_suspect`) never become a track's best read. A malformed file is an
  error, never a silent fallback. After any change to it, run the panel's **platecheck**
  playbook; plates that newly pass cost DVSA lookups. 41.8 % of car journeys carry a DVSA label
  (2026-10-04).
- **Unverified.** Only the plate-colour rule checks that a plate belongs to the tracked car; the
  DVSA register can't see that. Until the E1.2 audit set exists
  ([`docs/data_integrity_review.md`](docs/data_integrity_review.md)), read rates are
  canonical-shape rates, not verified reads.
- **Tried and rejected** (don't repeat without new evidence): multi-frame character voting
  (−43 pp against best-of-N; `alpr/consensus.py` is kept as infrastructure), a daytime shutter
  cap (Anti-Smearing, 2026-05-31), and a fuzzy plate ratio below 85 (it merges different cars
  across regional prefixes).

### Classifiers

All three production heads were trained on `runs/uk_crops_0929_576` (4,347 cars / 76,033 crops /
62 makes; plate-anchored crops, pad 0.1; labels from DVSA lookups of gated plate reads). Per-track
accuracy on the unseen sessions `session_20260924_104939` + `session_20260930_211033`:

| Head | Run | Promoted | Fresh cars | All cars, new passes |
| --- | --- | --- | --- | --- |
| make (EfficientNet-B6 @528) | `uk_make_0929_b6` | 2026-10-01 | 85.0 % (117 cars) | 95.1 % (710) |
| colour (B0 @384) | `uk_colour_1002_b0` | 2026-10-03 | 83.3 % (118) | 93.5 % (711) |
| body type (B0 @384) | `uk_body_1002_b0` | 2026-10-03 | 89.1 % (109) | 95.9 % (674) |

Weights are in `src/streettracker/analysis/makemodel/models/` (`makemodel_b0.pt`, `colour_b0.pt`,
`bodytype_b0.pt`, each with a `.meta.json` sidecar; the dated backups beside them are untracked).
The make model is too big to commit: it is GitHub Release `models-2026-10-26`, and each promotion
publishes `makemodel-<run>`.

- **These scores are on plated cars.** `/stats` mostly uses the heads on unplated cars, where the
  crop comes from the trajectory rule and accuracy is untested (review E2.6).
- **Sources.** `make_model_source` and `colour_source` say where a value came from: `"dvsa"` is
  the register (truth for a correctly read plate), `"cnn"` a prediction. `/stats` prefers DVSA,
  then the CNN, then "unknown". The HSV `color` field is not a colour source (about 40 % grouped
  accuracy): leave it alone and never fall back to it.
- **Crops must match training.** Inference with `--crop-mode auto` follows the checkpoint; these
  models infer `fullframe` at pad 0.1. `bodytype` and `colour` need `--pad-frac 0.1` (the CLI
  default): 0.25 skews the shape classifier toward "van".
- **Judge a candidate only head-to-head.** `makemodel-compare <corpus> --candidate <best.pt>
  --eval-session <unseen sessions>` scores the candidate and production on the same cars with a
  by-car bootstrap interval. Add `--include-trained-cars` for the larger known-car set and
  `--target colour|body_type` for the other heads. Never compare raw validation scores, or scores
  across corpora or crop families: that mistake produced most of the superseded numbers in the
  history. About 88 % of this street's cars are regulars, so a week yields roughly 100 fresh cars.
- **Training.** B6 @528 trains at batch 6 on the 10 GB RTX 3080 (batch 8 spills into system
  memory and runs about 5× slower); the build-train playbook defaults to it. A per-car crop cap
  doesn't help.
- **What moved accuracy:** clean crops (the clean-crop make retrain gained 22.5 pp per track),
  clean labels, input resolution and more cars.
  **What didn't:** CompCars pre-training (about 1 % on UK cars) and a local VLM (Qwen3-VL 8B lost
  clearly, and it starves panel jobs of memory: never run it beside them).
- **The next retrain** waits on cluster-aware exclusion in `makemodel-compare` (11 % of the
  "fresh" test cars were misreads of regulars) and on fresh cars building up (review E2.5).

### People side

- `streettracker people` writes `_people.json`: walker / jogger / cyclist (jogger ≥ 2.5 m/s; a
  person paired with a bicycle track is a cyclist), a `dog_walker` flag, BotSORT-split fragments
  merged into walks, out-and-back round trips, and door-origin trips when
  `configs/door_zone.json` (gitignored) exists. Field notes are in
  [`docs/session_files.md`](docs/session_files.md).
- About 75-80 % of raw round-trip pairs are coincidence on this pavement. Report only the excess
  over the time-shift control.
- Most "joggers" are near-pavement walkers who move fast in pixel terms (Phase 0, E0.7), until
  ground-plane speed exists (E3.2).
- A "person" whose median box is at least 1.5× wider than it is tall is flagged `class_suspect`
  and left out of people and car counts.
- **Privacy line.** Walks are never matched across days, and this repo does no cross-session
  person re-identification. The door zone identifies the household's trips by where they enter
  and leave the frame, not by who takes them.
- **Parked by the operator (2026-07-07):** a retention policy for person 4K imagery on the dev
  box. It stays last until they say otherwise.

### Open work, in order

1. **Confirm the stall fix.** Once `session_20261110_171836` has been rolled and pulled, run
   `uv run --no-sync python .claude/loop_stall_check.py output session_20261110_171836`. Tracks
   ending at a rebuild instant should be at the chance level (a pre-fix session shows about 13
   times chance).
2. **Data-integrity plan** (review §4, revised 2026-10-04; its first four steps are done). Next
   is the E1.2 audit set, stratified toward flagged reads, then calibrating the plate-colour
   check and the combined gate against it. After that: E2.6 (plate-blind test first), E2.1, E2.3
   and E2.7; the E3.1 hand count, then the night A/B (E2.8); and the retrain (E2.5) last.
3. **Low-light shutter schedule.** Waiting on the operator's decision (see Plate reading).
4. **Code-quality roadmap** (2026-11-10 review). Done: showcase rebuild speed, CI checks, the
   shared file helpers, the stall fix, this file's split and the rest of the Orin runtime work
   (2026-11-13: one sharpness measurement per track, the shutdown drain on the event loop, atomic
   end-of-session JSON). That last part is on `main` but not on the Orin; it goes out with the
   next code deploy and needs no config change. The rest are unapproved proposals; ask before
   starting one.
5. **JetPack 7.** Blocked until `scripts/check_jp7_wheel.ps1` exits 0. Stay on JetPack 6; the
   plan is in [`docs/deployment.md`](docs/deployment.md#jetpack-7-upgrade-plan).

### Working rules

- **Cloud sessions have no `output/`.** They can do code work only. Anything that reads session
  data runs on the dev box, and the operator pastes the output back.
- **Long jobs on the dev box.** The box idle-sleeps, and background children of a Claude session
  can be reaped. Run long jobs through the control panel (it holds a wake-lock for a whole
  playbook) or detached with a wake-lock, and confirm completion from artefact modification
  times, never from a log tail: a dead job's frozen log looks like a working one.
- **Services.** The showcase (:8090) and the control panel (:8095) run as the scheduled tasks
  `StreetTrackerShowcase` and `StreetTrackerControlPanel`. Check the task before launching a
  second copy. New data needs `POST /api/refresh` on the showcase; a code change needs a task
  restart, and restarting the panel kills its running jobs.
- **Never run `alpr-run` on the Orin.** The live tracker uses all of it; pull to the dev box.
- **Size the test set before trusting a rate.** A make@1 figure needs more than about 200
  held-out cars.

## Project overview

StreetTracker unifies VehicleTracker (dev-box, file input) and NanoTracker
(original Jetson Nano, live RTSP) onto one Python 3.10 + Ultralytics +
TensorRT stack, targeting Jetson Orin Nano 8GB Super as the primary device.

```
RTSP H.264/H.265   ┐
   or              ├─▶ FrameSource ─▶ Ultralytics YOLO(.engine).track()
MP4 (NVDEC on Orin)┘                  (BotSORT integrated)
                                          │
                                          ▼
                          attribute compute (direction/speed/color/lane)
                                          │
                                          ▼
                          per-track finalize ─▶ EventLog (jsonl, fsync)
                                          │
                                          ▼
                          idle ▶ regenerate summary HTML + hourly rollup
                                 ▶ on demand ▶ Reolink 4K HTTP snapshot
```

### Compatibility rules

- **Python 3.10 today, 3.12 the JP7 target.** `.python-version` is pinned
  to `3.10`. Bumping past 3.10 broke the Orin on 2026-05-25 (no `cp311`
  torch wheel on `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`).
  Always verify wheel availability before changing the pin. The systemd
  unit passes `--no-sync` so transient `uv` flaps can't reinstall PyPI
  torch on top of the Jetson wheel — but a manual `uv sync` after `git
pull` _will_ rebuild at the new pin and is fatal if torch won't
  resolve. The 3.10 → 3.12 move bundles with the JP7 flash.
- **No Python 3.6 hacks.** Use `@dataclass(slots=True)` and PEP-604
  unions (`X | None`); the sys.path reorder / NamedTuple-for-dataclass /
  `# type:` comments are gone.
- **TRT engines are not portable** across GPU architectures. Always
  build engines ON the target device.

## Architecture

```
src/streettracker/
├── common/                 # shared across runtime + analysis
│   ├── schema.py           # TrackRecord, SessionMeta @dataclass
│   ├── color.py            # COLOR_RANGES + vote_color()
│   ├── config.py           # frozen-dataclass strict JSON loader
│   ├── summary.py          # HTML dashboard generation
│   ├── hourly.py           # build_hourly_rollup()
│   ├── output.py           # EventLog, save_json, JSONL readers
│   ├── fileio.py           # atomic_write_json/text/bytes, read_json, iter_jsonl
│   └── session_files.py    # SessionFiles(session_dir): every per-session file path
├── inference/              # YOLO + BotSORT via Ultralytics
├── sources/                # RTSP (FFmpeg), file (NVDEC on Orin)
├── device/                 # Orin-only: live runtime, snapshotter, dashboard, IR
├── analysis/               # off-device: ALPR, recolor, vehicles, makemodel
└── cli/                    # `streettracker` entry + subcommands
```

Single import root: `from streettracker.common.schema import TrackRecord`.

**Session files (2026-11-10).** Write any file that outlives the command with
`common.fileio.atomic_write_json` / `atomic_write_text` (temp file, then
rename; on Windows the rename retries while the showcase or panel has the
target open), not a plain `write_text`. Build per-session paths with
`SessionFiles(session_dir).<name>` (`common/session_files.py`) instead of an
`f"{label}_<suffix>"` literal; a new per-session file gets a property there
and a line in `tests/test_common/test_session_files.py`. The device's summary
page and end-of-session JSON use the same helpers; only its image saves (4K
snaps, thumbnails) and the append-only logs write in place.

### Device runtime notes (Orin Nano 8GB Super)

- JP6 ships Ubuntu 22.04 + Py 3.10; JP7 ships Ubuntu 24.04 + Py 3.12
  natively. uv handles both via `.python-version`.
- Ultralytics' built-in TRT path (`YOLO('best.engine')`) replaces
  NanoTracker's hand-rolled YOLOv8 decode + numpy NMS + bespoke IoU
  tracker.
- Live RTSP from Reolink: use FFmpeg backend (`cv2.CAP_FFMPEG` +
  `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`). GStreamer stalls
  on Reolink keyframes.
- MP4 input on Orin uses GStreamer + `nvv4l2decoder` (NVDEC); only the
  live-RTSP case is broken with cv2-GStreamer.
- **No time through a power cut.** The Orin's RTC comes back unset and
  systemd-timesyncd restores a stale saved time (3 h slow on 2026-10-23), so
  `streettracker run` waits up to 300 s for the boot's first NTP sync
  (timesyncd's `/run/systemd/timesync/synchronized`, else `timedatectl`)
  before naming the session; `--clock-sync-timeout-s` (0 = off). A restart
  on a synced device doesn't wait. Fitting an RTC backup battery would avoid
  the wait.
- **The summary page is written off the event loop** (since `0b817b7`, 2026-11-10). The runtime
  keeps the page's rows ready (`common.summary.SummaryRows`, one row serialised per kept track at
  finalise) and writes the files from a worker thread (`runtime.start_html_write`). Before the
  fix each rebuild stopped frame processing for 0.5 s + 0.24 ms per record held: 5.5 % of kept
  tracks ended at a rebuild and about 2.2 % were one pass recorded as two tracks. Sessions
  recorded before `session_20261110_171836` keep those splits, more of them late in a long
  session. Don't put per-session work back on the loop.
  Measurements: [`docs/history/project.md`](docs/history/project.md).
- **Snap decisions run as three whole passes** over the active tracks: every track's trigger,
  then every track's pipeline fire, then the door zone (`runtime._plan_snaps`). Each track's box
  and sharpness are measured once per frame and shared. Keep the passes whole: the snapshotter
  drops a submit once `max_concurrent` snaps are in flight, so going track by track would let one
  car's pipeline fire take the slot another car's trigger needed.
- **Shutdown.** The drain that finalises the active tracks runs on the event loop
  (`runtime._drain_active`), because a finalise can submit a snapshot. The three end-of-session
  JSON files are each replaced in one step, `_data.json` last, so a session with a `_data.json`
  has all three whole and one without is rebuilt by `pull`.
- **Track class and file prefix.** A track's class is the `argmax` of confidence-weighted votes
  (`BufferedTrack.class_votes`), not the last detection. Its file prefix is locked at finalise
  and snaps saved under another prefix are renamed, so file names always agree with
  `record.asset_prefix`.

### Phase 3 architectural decisions (locked in)

- **Asyncio throughout.** Entry is `asyncio.run(run_session(config))`.
  Blocking sources (`cv2.VideoCapture.read()`) go to a background thread
  pushing into an `asyncio.Queue`. Blocking inference goes to
  `loop.run_in_executor`. HTTP is `aiohttp`.
- **Frozen dataclasses + strict JSON loader.** `common/config.py` rejects
  unknown keys with a JSON-path error (catches typos and forces
  schema-additive config changes through the proper deploy order — see
  [Schema-additive config changes](#schema-additive-config-changes)).
- **Graceful shutdown via `loop.add_signal_handler(SIGTERM, ...)`,**
  bounded by a 30s timeout so systemd won't SIGKILL mid-write. Stop the
  source first, drain active tracks through finalize, write
  summary HTML + data.json + meta.json + hourly.json, then `loop.stop()`.
- **Heavy unit tests + manual Orin smoke.** No recorded-frames
  integration test (would balloon the repo + CI skips torch /
  ultralytics). End-to-end validation by ssh against the live Reolink.
- **Config loading is forgiving on stream names and strict on the engine.** `cli/run.py` falls
  back to matching `stream.quality` when `nano.preferred_stream` matches no `stream.name`, and a
  missing `inference.engine_path` fails at load with a JSON-path `ConfigError`.

## Session files

Full field notes are in [`docs/session_files.md`](docs/session_files.md). Record fields are
defined in `common/schema.py` (`TrackRecord`, `SessionMeta`) and paths in `SessionFiles`.

Per finalised track (`{prefix}` is `vehicle` or `person`): `{prefix}_{id}.jpg` (dashboard tile),
`{prefix}_{id}_hq.jpg` (the sharpest, largest sub-stream view, and the one picture sure to show
the tracked object) and `{prefix}_{id}_main_{N}.jpg` (4K snaps, `N` from 1; a snap lands about
0.7 s after it fires, so the car may have moved on).

| Session file | Written by | Holds |
| --- | --- | --- |
| `_events.jsonl` | runtime, per track | one record per kept track; survives a crash |
| `_trajectories.jsonl` | runtime | thinned box path per kept track (from 2026-10-04) |
| `_data.json`, `_meta.json`, `_hourly.json` | runtime, at a clean shutdown | records; session metadata with IR periods and `snap_stats`; hourly rollup. After a power cut, `pull` (or `recover-session`) rebuilds them and the meta carries a `recovered` block |
| `_summary.html`, `index.html` | runtime | the Orin dashboard page |
| `_alpr.json`, `_alpr_by_track.json`, `_static_plates.json` | `alpr-run`; updated by `alpr-rescore` and `alpr-colour` | per-image reads; each track's best read (suspects left out, `n_agree`); the stamps that say which fixes the session has |
| `_dvsa_labels.json` | `dvsa-label` | register make / model / year / colour per plate, and the gate it used. Lookups are paid for and cached here |
| `_vehicles.json` | `vehicles` | per-vehicle aggregation. It can be stale: the showcase rebuilds it live |
| `_makemodel*.json`, `_bodytype*.json`, `_colour*.json` | the CNN commands | per-image and `_by_track` predictions |
| `_vehicle_boxes.json` | any fullframe-crop command | cache of 4K vehicle boxes; disposable |
| `_people.json` | `people` | activity per person track |

The output root also holds `cross_session_repeats.json` (`vehicles --across`),
`showcase_metadata.json` (the showcase's only writable state) and `control_jobs.json` /
`control_playbooks.json` (panel history).

`dvsa-apply` writes make and model onto car records only, so harvest rows for vans, buses or
tracks with no record never reach `_data.json`. That is expected.

## Common tasks

```bash
uv run pytest                                            # tests
uv run ruff check src/ tests/                            # lint
uv run ruff format src/ tests/                           # format
uv run mypy src/                                         # type-check
uv run streettracker batch sample.mp4                    # batch dev-box
uv run streettracker export-engine yolov8m.pt            # build TRT on device
uv run streettracker alpr-run output/<session>           # offline ALPR
uv run streettracker alpr-rescore output/<session>       # recompute plate-read confidence from saved crops (2026-09-28 fix); --dry-run to preview
uv run streettracker alpr-colour output/<session>        # plate-colour check on saved crops (E1.3): drop other cars' plates from the rollup; --dry-run to preview
uv run streettracker dvsa-label output/<session>         # DVSA make/model harvest (needs configs/dvsa.json; plate gate from configs/alpr.json: conf >= 0.9 by default, or the combined gate)
uv run streettracker dvsa-apply output/<session>         # fold DVSA make/model onto per-track records
uv run streettracker vehicles output/<session>           # per-vehicle aggregation (+ DVSA make/model)
uv run streettracker vehicles output/<a> --across output/<b> ...  # cross-session repeat vehicles
uv run streettracker people output/<session>             # person activity enrichment (dog walkers / joggers / cyclists)
uv run streettracker showcase --output-root output       # local website: enriched + recurring cars (http://127.0.0.1:8090/)
uv run streettracker makemodel output/<session>          # CNN make/model inference -> _makemodel.json
uv run streettracker bodytype output/<session>           # CNN body-type inference -> _bodytype_by_track.json (pad_frac 0.1)
uv run streettracker colour output/<session>             # CNN vehicle-colour inference -> _colour_by_track.json (pad_frac 0.1); replaces the HSV colour vote
# Mine the Orin -> grow the UK make-classifier corpus (run pull from PowerShell):
uv run streettracker pull --session <S> --only-main      # pull a session's 4K snaps + vehicle HQ crops + JSON from the Orin
uv run streettracker pull --session <S> --crops-only     # fetch just the missing vehicle HQ crops (never the JSON, so safe on an enriched session)
uv run streettracker recover-session output/<session>    # rebuild data/meta/hourly JSON for a session the device never finalised (pull does this itself)
uv run streettracker makemodel-build-uk runs/uk_crops --output-size 512  # DVSA-labelled UK make crops @512 (auto-discovers sessions); --crop-mode plate (default) = plate-anchored crops, needs alpr-run output
uv run streettracker model-release                       # publish the production make model as a public GitHub Release (promote's last step; idempotent; refuses checkpoints holding plate-shaped strings or paths; --dry-run)
uv run streettracker makemodel-compare runs/uk_crops --candidate runs/uk_make_X/best.pt  # head-to-head vs production on shared held-out cars -> runs/uk_make_X/compare.json (the report it replaces is kept as compare.prev.json); --target colour|body_type for the other heads (colour also reports a grouped score); --eval-session <S>... scores fresh sessions' cars instead of the corpus val split, minus every car in production's and the candidate's corpora (+ --include-trained-cars to keep those cars and score their new passes)
uv run streettracker makemodel-train-uk runs/uk_crops --input-size 512   # train the UK make classifier (default backbone efficientnet_b0; production used --backbone efficientnet_b6 --input-size 528 --batch-size 6; --target colour|body_type for the other heads). Judge the run with makemodel-compare, not its validation score
```

**Windows + Git Bash gotcha for `streettracker pull`**: MSYS rewrites
POSIX-looking arguments (`/home/...`) into Windows paths before Python
sees them. Either run from PowerShell / cmd, or prefix with
`MSYS_NO_PATHCONV=1` and pass `--key` as a Windows-style path.

## Tests and CI

Tests at HEAD (2026-11-10): **about 1,300 passing on the dev box, on Python 3.10.**
CI runs them in two jobs: `test` (no torch; the 8 torch-gated classifier modules
skip there) and `test-torch` (those 8 modules, 92 tests, on CPU wheels). CI also
enforces `ruff check`, `ruff format --check`, `mypy` (`[tool.mypy]` in
`pyproject.toml`) and a complexity limit of 15: the functions already over it
carry `# noqa: C901` on their definition, so drop the marker when you split one.
A `tests/conftest.py` autouse fixture points
the plate-gate config at a per-test path, so a calibrated `configs/alpr.json` on
the dev box never changes what the tests see.

All subcommands wired: `run`/`batch` go through the asyncio runtime,
`pull`/`export-engine` ship sessions and build engines,
`alpr-run`/`alpr-score`/`alpr-label`/`alpr-report` available under the
`alpr` extra (**`uv sync --extra alpr --extra dev`** — extras are not
additive, so syncing with only `alpr` *uninstalls* pytest/ruff/mypy),
`vehicles` runs the
plate-anchored aggregator. Place the bespoke detector at
`src/streettracker/analysis/alpr/models/license_plate_detector.pt`
(gitignored).

## Showcase website

`streettracker showcase` (`src/streettracker/web/`) serves the plate-read cars joined to DVSA
make, model, year and colour, a per-car metadata editor and the `/stats` traffic page, on
<http://127.0.0.1:8090/>. Full reference: [`docs/sites.md`](docs/sites.md).

- **Identified cars only**, rebuilt live from `analysis.vehicles.build_vehicles` and merged across
  sessions by canonical plate. "Regulars" are cars seen on at least two calendar dates.
- **The only state it writes** is `<output-root>/showcase_metadata.json` (name, owner, notes,
  favourite), keyed by plate so a tag follows a car across sessions.
- **Local by default.** It binds `127.0.0.1` because the pages show plates and personal tags;
  `--host 0.0.0.0` opts into LAN exposure.
- **`/stats` counts every car track**, not just plated ones. One track is about one pass, and
  `class_suspect` tracks are excluded. Speed is in px/s unless `configs/showcase.json` sets a
  calibration; mph is approximate (one global factor, no perspective).

## Operator control panel

`streettracker control` (`src/streettracker/control/`) is the dev-box web app the operator uses
to run the day-to-day procedures without Claude attached, on <http://localhost:8095/>. Full
reference: [`docs/sites.md`](docs/sites.md).

- **Access split.** It binds `0.0.0.0` so the dashboards are viewable on the LAN, but every
  control action is localhost-only, enforced server-side (`server._require_local`).
- **Jobs** are subprocesses on two serial lanes that run side by side: a GPU lane (`alpr-run`,
  `alpr-colour`, the CNN commands, corpus build, training, `export-engine`) and a net lane
  (`pull`, `dvsa-label`, `vehicles`, `people`). Two GPU jobs never overlap.
- **Progress comes from stdout.** `control/progress.py` parses each command's printed lines, so
  rewording a progress message breaks its progress bar. Change the parser and its test with it.
- **Playbooks** (`control/playbooks.py`): enrich, build-train, roll, promote, reinfer, rescore,
  relabel and platecheck. **roll** (restarts the Orin service, then pulls the closed session) and
  **promote** (replaces the production model, then publishes the release) are destructive and
  confirm-gated.
- **Promote needs a head-to-head.** The recommendation reads the run's `compare.json`; a run with
  no fresh report can't be promoted with one click.
- **A failed job offers a "Copy Claude prompt" button** (`control/prompts.py`), and history
  survives a restart in `control_jobs.json` / `control_playbooks.json`.

## Vehicles aggregator (`streettracker vehicles`)

Folds per-track records + per-track best ALPR reads into a per-vehicle
view keyed by canonical plate. Run with
`uv run streettracker vehicles output/<session>`; writes
`<session>_vehicles.json`.

| Vehicle field                         | Notes                                                                                      |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| `plate` / `plate_conf`                | canonical plate (highest-conf variant); `null` for unread cars                             |
| `n_visits`, `track_ids`               | tracks attributed to this vehicle                                                          |
| `first_seen` / `last_seen`            | ISO timestamps                                                                             |
| `gap_minutes_max` / `gap_minutes_min` | between consecutive visits; 0 if `n_visits==1`; can be negative when BotSORT ID-switches   |
| `directions` / `colors`               | per-visit histograms                                                                       |
| `visits`                              | inline array with each visit's TrackRecord fields + best-read snap filename                |
| `plate_variants`                      | `[(str, max_conf), ...]` of OCR variants collapsed into the canonical via fuzzy clustering |

Fuzzy clustering uses rapidfuzz `fuzz.ratio` with default threshold 85,
same-length only. Catches 1-char OCR diffs on 7-char UK plates (ratio
85.7) but not 6-char ones (ratio 83.3). Dropping to `--fuzzy-ratio 80`
catches 6-char one-char diffs too but **over-merges across UK regional
prefixes** in this scene (an `LX7751` cluster absorbed 8 distinct
Newcastle-area plates whose 4th-6th chars happened to be close). Use
`--no-fuzzy` for strict-string equality.

A **DVSA-distinct veto** (2026-06-11) guards the opposite failure: two
REAL cars one character apart (`GU65UGK`, a red VW UP, was absorbed by
`GU65UGM`, a white VW GOLF — 16 such pairs corpus-wide). A merge is
vetoed when both spellings have DVSA rows with different
`primary_colour` groups AND each spelling's own observed colours have
differing majorities with at least one side matching its register
(`make_distinct_vehicle_checker` in `vehicles.py`; applies within-
session, cross-session, and in the showcase pool). One-car OCR-variant
merges are protected by the observed-colour check — both spellings of
one car see the same colours, even when the misread string resolves to
some other real vehicle on the register.

The aggregator **does not reject temporally-overlapping merges** — a
real example on `session_20260526_124704` (tracks 1516 `LD22BMG` R→L
and 1517 `LD22BWG` L→R, visits overlapping by 8s) showed both 4K snaps
were the same silver hatchback. BotSORT ID-switched mid-transit (brief
detection gap spawning a new track ID before the old one finalized),
and the new track's first-few-frames motion vector mislabelled the
direction. Earlier code rejected such merges; that produced more false
negatives than the heuristic caught true positives.

## Deploying to the Orin

The Orin runs the tracker from the checkout at `~/streettracker` under `streettracker.service`,
which passes `--no-sync`. A code-only deploy is: merge to `main`, `git pull` on the Orin, restart.
The panel's **roll** playbook does the restart and pulls the closed session. Afterwards, update
"Live on the Orin" above with the new commit and the rollback commit.

- **Don't run a bare `uv sync` on the Orin** without checking the Jetson wheel index first (see
  [Compatibility rules](#compatibility-rules)). If `.venv` was rebuilt, the system-TensorRT
  symlinks are gone and the service restart-loops on `No module named 'tensorrt'`: run
  `scripts/setup_orin.sh --symlinks-only`, then restart.
- **Back up `configs/camera.json` on the device** (`camera.json.bak.<UTC timestamp>`) before
  changing it. It holds the camera credentials, the traced polygon and the tuned snap gate, and
  it is not in the repo.

### Schema-additive config changes

The strict JSON loader is intentionally fatal on unknown keys (catches
typos). Schema-additive changes to `camera.json` (new field in
`SnapGateSpec`, etc.) **must land on `main` and be `git pull`'d on the
Orin BEFORE the corresponding camera.json edit**, or the service
crash-loops. Hit this on 2026-05-27 with `pipeline_interval_ms_by_direction`
(~60s downtime before rollback to the `camera.json.bak.<ts>` snapshot
restored the previous config and the service came back up).

Right deploy order for schema-additive config changes:

1. Push branch → open PR → CI green → merge to main.
2. `ssh streettracker@orin "cd ~/streettracker && git pull"` (+
   `scripts/setup_orin.sh --symlinks-only` if `.venv` was rebuilt — see
   [Recovery from a venv rebuild](docs/deployment.md#repo--venv-setup)).
3. Restart with the OLD `camera.json` — verify the new code parses
   unchanged.
4. THEN edit `camera.json`, scp, restart, verify.

Value changes to existing keys (e.g. Step 11's `t_usable_frac` trim)
are safe to deploy directly.

A fresh install on a wiped Orin, what to back up, dev-box disaster recovery and the JetPack 7
plan are in [`docs/deployment.md`](docs/deployment.md).

**This repo is public.** Secrets (`configs/dvsa.json`, `camera.json`), plate or person imagery
and per-session JSON never go into the repo, a release or a published page. `model-release`
refuses a checkpoint that holds plate-shaped strings or paths.

## Snap gate (road polygon + axis triggers)

`device/snap_planner.py` decides when to fire the 4K snapshot for each
tracked vehicle. Live deployment mode is **`RoadGate`**: an
operator-traced road polygon plus axis triggers. Each trigger fires at
most once per track; pipeline mode adds continuous in-band fires on top.

### Reolink resolution gotcha

The "4K" main snap on this install is actually **4512×2512**, not
3840×2160. Surfaced during Step 10's ghost-mask deploy: an early draft
hardcoded 3840×2160 as the scaling denominator and the mask landed
1.175× off-position. The fix reads `source_size` from the mask JSON
and scales `w_img / source_size[0]`. If Reolink firmware changes the
main-stream resolution, only update `source_size` in
`.claude/ghost_mask.json` — the rect coords stay.

### Pipeline mode (the dominant capture mechanism)

Trigger fires alone cap at ~3 per car. Since 2026-05-23 the live config
also runs **pipeline mode**: while a track sits inside the polygon's
`t_usable` band, fire a 4K snap every `pipeline_interval_ms` of
wall-clock time, capped at `pipeline_max_per_track`. Pipeline fires
have their own per-track counter (`pipeline_fires`) but share the
snapshotter's `max_concurrent` HTTP semaphore with trigger fires.
Pipeline fires dominate trigger fires by ~10× at the current settings
(400 ms / 15-cap).

`pipeline_interval_ms: 0` (the default in `RoadGateConfig`) disables
pipeline mode entirely. Don't blindly re-deploy the local
`.claude/snap_gate.json` artifact without preserving the live config's
`pipeline_interval_ms` / `pipeline_max_per_track` / `pipeline_interval_ms_by_direction`
fields — see [Snap-gate config layout](#snap-gate-config-layout).

### Snap-gate config layout

The live `~/streettracker/configs/camera.json`'s `snapshot.snap_gate`
block contains all of:

```jsonc
{
  "polygon_frac": [[x, y], ...],
  "trigger_t_prime": [0.05, 0.20, 0.35, 0.65, 0.80, 0.95],
  "trigger_directions": ["forward", "forward", "forward",
                         "reverse", "reverse", "reverse"],
  "t_usable_frac": [0.10, 0.45],
  "pipeline_interval_ms": 400,
  "pipeline_max_per_track": 15,
  "pipeline_interval_ms_by_direction": {},
  "pipeline_t_usable_by_direction": {
    "forward": [0.10, 0.20],
    "reverse": [0.30, 0.60]
  }
}
```

`pipeline_t_usable_by_direction` gates pipeline fires per motion direction in absolute t_norm
coordinates (the same space as `t_usable_frac`; trigger t' values are unaffected). A direction
missing from the dict, and any frame whose direction isn't known yet, falls back to
`t_usable_frac`. `forward` is R→L (front plate visible) and `reverse` is L→R (rear plate).
`trigger_t_prime` values are fractions of the usable band, not of the whole axis, so they move
when `t_usable_frac` changes.

- The reverse band's lower edge (0.30) was tuned on live soaks. The forward band `[0.10, 0.20]`
  has not been re-derived with fullframe crops (review, "R→L pipeline band").
- `pipeline_interval_ms_by_direction` is `{}` (a uniform 400 ms). The 2026-05-27
  `{forward:300, reverse:400}` setting was dropped and has not been restored.
- `min_sharpness=100.0` essentially never trips; don't lower it.

The reasons behind each value are in
[`docs/history/anpr_tuning.md`](docs/history/anpr_tuning.md).

The local artifact `.claude/snap_gate.json` historically only carried
`polygon_frac` / `trigger_t_prime` / `trigger_directions` / `t_usable_frac`.
**Always preserve the live `pipeline_*` fields when scp'ing** — pull
the live config, edit only what you need, scp back. See [Adjusting the
snap_gate](docs/snap_gate.md#adjusting-the-snap_gate).

The live geometry table (polygon, triggers, ghost mask), the scene artefacts in `.claude/`, how
to move a trigger or re-sketch the road, and the planner's modes are in
[`docs/snap_gate.md`](docs/snap_gate.md).
