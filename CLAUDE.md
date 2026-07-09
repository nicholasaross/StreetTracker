# CLAUDE.md <img src="docs/assets/st26trk.svg" alt="ST26 TRK" height="40" align="right"/>

Guidance for Claude Code when working in StreetTracker.

## ⮕ Start here — 2026-06-13 checkpoint

Everything from the last sprint is **merged (#57–#64) and on `main`** (clean
tree). The detailed history lives in the sections below; this block is the
"resume from cold" summary.

**Live on the Orin** (#63 runtime bundle deployed 2026-06-13, service active,
NRestarts=0):
- **per-direction pipeline bands** `pipeline_t_usable_by_direction =
  {forward(R→L):[0.10,0.20], reverse(L→R):[0.30,0.60]}` (reverse lower edge
  nudged 0.25→0.30 on 2026-06-19, **validated 2026-06-22** on the ~72h
  `session_20260619_111111` soak — L→R per-car 73-82 % → **92.3 %**, net
  per-car ~48 % → **57.9 %**). **Post-deploy verdict
  (2026-06-19, two ~70h soaks) falsified the R→L half** of the 2026-06-11
  re-analysis: L→R is the win (per-car ~73-82 %), R→L is camera-geometry-capped
  at ~15-20 % at *every* position (the "R→L reads far" finding was a
  motion-window-hint artifact that completion-time bboxes corrected; honest net
  per-car ~48 %, not the inflated 63.9 %). R→L now needs hardware, not tuning.
  Detail in the [snap-gate config layout](#snap-gate-config-layout) section.
- **`vehicle_classes = [0,1,2,3,5,7]`** (person+car+bike+moto+bus+truck; was
  `[0,2]`).
- **completion-time bbox capture** (`TrackRecord.main_snap_bboxes_done`) —
  sessions recorded *after* the deploy carry the bbox at snap-landing time, and
  `alpr-run` prefers it (exact crops, no extrapolation).
- rollback: on-device `configs/camera.json.bak.20260613T075913Z`, git `523c820`.

**State:** `session_20260608_125639` pulled (56 GB) + enriched (alpr + dvsa =
1,174 new labels, 59 parked beacons suppressed). Showcase live on **:8090**
(~2,700 cars).

**Make-model: CLOSED 2026-06-14.** Corpus rebuilt (`runs/uk_crops_0614_512` =
**32,777 crops / 36 makes**, ~4× the 0611 corpus after folding in the new
session), B5@456 retrained **make@1 0.349 → 0.402** (+5.3 pp on a trustworthy
6,570-crop val; plateaued 0.39-0.40 over epochs 16-30, not a fluke), **promoted**
(`runs/uk_make_0614_b5/best.pt` → `…/models/makemodel_b0.pt`; prior 0.349/0.303
backed up beside it), and **all 17 sessions re-inferred** with it + showcase
refreshed. Data-is-the-lever, confirmed again (4× data → +5.3 pp). The
`*_0613_*` runs are dead partials from a sleep-killed overnight job — ignore.

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

**Next steps (updated 2026-07-07), priority order:**
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
4. **Make classifier — B6 SHIPPED, VLM FALSIFIED (2026-07-08):**
   B6@528 trained on `uk_crops_0707_576` (45 makes / 3,942 cars /
   47,881 crops) → **make@1 0.451**, promoted over B5's 0.441. VLM
   bake-off DONE — **Qwen3-VL 8B loses decisively** (0.15-0.18 vs B6's
   ~0.29-0.30 per-car-largest-crop on 400 val cars; ~15 s/image vs ms)
   and shows its own OOD attractor (MINI 23 %) — no pipeline role at
   this size. **Positive surprise: B6 largely fixed the June OOD
   VW-collapse** (VW max-share on unlabelled tracks: B5 19 % → B6 11 %,
   plausible Ford-topped mix) — corpus growth + capacity, not
   structural after all. Bake-off methodology gotchas recorded in
   `.claude/vlm_bakeoff.py` (cross-corpus val leakage: 276/400 of the
   0707 val cars were in B5's train set; per-car-largest is a harder
   metric than the trainer's per-crop val). **Body-type coarse target —
   infra SHIPPED (2026-07-08):** `analysis/makemodel/bodytype.py` maps
   the DVSA *model* string (prefix-matched, trim-suffix-tolerant) to a
   silhouette class {hatchback, saloon, suv, mpv, van, pickup, coupe} at
   **90 % of labelled cars** (45 % hatchback / 32 % suv / ~8 % van+saloon
   / ...). `makemodel-build-uk` now tags `body_type` per crop; train it
   with `makemodel-train-uk <corpus> --target body_type` (the make path
   is byte-identical when `--target make`). The DVSA "UNKNOWN" placeholder
   is now filtered in `normalize_make` (PR #76, #77). **VERDICT
   (2026-07-09, `runs/uk_body_0708_b0`, b0@384, 20ep early-stop @12):
   body_type@1 = 0.694 per-crop / 0.667 per-car — beats make (0.45) by
   ~+24 pp but short of the 80-90 % hope.** The win is CONCENTRATED: the
   two dominant, visually-distinct classes — **hatchback + suv, 77 % of
   cars — read ~0.75 per-car**; the minority classes are weak (saloon
   0.37 / mpv 0.42 / van 0.59 / coupe 0.43 / pickup 0.21 recall),
   capped by rear-view ambiguity + tiny support + label noise (one model
   name → predominant body style mislabels estates/variants), and mostly
   confuse INTO hatchback. So body type is a usable coarse "hatchback vs
   suv vs other" tag for the majority, NOT a clean win — surface it only
   if that split is wanted; inference/showcase wiring is an unbuilt,
   optional follow-up. Reproduce: `.claude/bodytype_eval.py`. Ops: the
   8B VLM spills
   ~17 GB commit to system RAM on the 10 GB 3080 — never run it beside
   panel jobs (it starved an `alpr-run` to death on 2026-07-08).
5. **R→L hardware:** 2nd discreet camera on the approach — the only
   remaining ANPR lever (R→L geometry-capped ~23 %). Needs multi-camera
   architecture first: RoadGate assumes one road/one axis, so per-camera
   gates + cross-camera track fusion before any purchase.
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
  pull` *will* rebuild at the new pin and is fatal if torch won't
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
│   └── output.py           # EventLog, save_json, file-path helpers
├── inference/              # YOLO + BotSORT via Ultralytics
├── sources/                # RTSP (FFmpeg), file (NVDEC on Orin)
├── device/                 # Orin-only: live runtime, snapshotter, dashboard, IR
├── analysis/               # off-device: ALPR, recolor, vehicles, makemodel
└── cli/                    # `streettracker` entry + subcommands
```

Single import root: `from streettracker.common.schema import TrackRecord`.

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

## Output schema (preserved from NanoTracker)

Per finalized track:

| File | Quality | Use |
|---|---|---|
| `{prefix}_{id}.jpg` | q=85, ~80px | dashboard tile |
| `{prefix}_{id}_hq.jpg` | q=95, ~250px | quick color/silhouette |
| `{prefix}_{id}_main_{N}.jpg` | 4K Reolink HTTP | ALPR / make-model |

`{prefix}` is `vehicle` or `person`. `N` is `1..max_snaps_per_track`.

Session files:
- `{session}_events.jsonl` — appended line-per-track (crash-safe)
- `{session}_data.json` — array of records, written at session end
- `{session}_meta.json` — session-level metadata + IR periods + snap_stats
- `{session}_hourly.json` — per-hour rollup
- `{session}_summary.html` + `index.html` — dashboard + auto-redirect
- `{session}_alpr.json` + `{session}_alpr_by_track.json` — per-image
  + per-track ALPR rollup (after running `alpr-run`)
- `{session}_vehicles.json` — per-vehicle plate-anchored aggregation
  (after running `vehicles`); carries DVSA `make`/`model`/`year` once
  `dvsa-label` has harvested for the session
- `{session}_dvsa_labels.json` — DVSA MOT `make`/`model`/`year` per
  plate (after `dvsa-label`); `dvsa-apply` folds it onto `data.json` +
  `events.jsonl` per-track records
- `cross_session_repeats.json` — repeat vehicles pooled across a cohort
  of sessions (after `vehicles --across`; written to the output root)
- `{session}_makemodel.json` + `{session}_makemodel_by_track.json` —
  per-image top-k + per-track confidence-weighted CNN make/model
  predictions (after `makemodel`; `make_model_source="cnn"`)
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
  soaks).

JSON record fields: see `common/schema.py` (`TrackRecord`, `SessionMeta`).

## Common tasks

```bash
uv run pytest                                            # tests
uv run ruff check src/ tests/                            # lint
uv run ruff format src/ tests/                           # format
uv run mypy src/                                         # type-check
uv run streettracker batch sample.mp4                    # batch dev-box
uv run streettracker export-engine yolov8m.pt            # build TRT on device
uv run streettracker alpr-run output/<session>           # offline ALPR
uv run streettracker dvsa-label output/<session>         # DVSA make/model harvest (needs configs/dvsa.json)
uv run streettracker dvsa-apply output/<session>         # fold DVSA make/model onto per-track records
uv run streettracker vehicles output/<session>           # per-vehicle aggregation (+ DVSA make/model)
uv run streettracker vehicles output/<a> --across output/<b> ...  # cross-session repeat vehicles
uv run streettracker people output/<session>             # person activity enrichment (dog walkers / joggers / cyclists)
uv run streettracker showcase --output-root output       # local website: enriched + recurring cars (http://127.0.0.1:8090/)
uv run streettracker makemodel output/<session>          # CNN make/model inference -> _makemodel.json
# Mine the Orin -> grow the UK make-classifier corpus (run pull from PowerShell):
uv run streettracker pull --session <S> --only-main      # pull a session's 4K snaps from the Orin
uv run streettracker makemodel-build-uk runs/uk_crops --output-size 512  # DVSA-labelled UK make crops @512 (auto-discovers sessions)
uv run streettracker makemodel-train-uk runs/uk_crops --input-size 512   # train the UK make classifier (B0@512; +--backbone b4/b5). honest make@1 ~28% on 1229 cars (old "37.6%" was small-val optimism)
```

**Windows + Git Bash gotcha for `streettracker pull`**: MSYS rewrites
POSIX-looking arguments (`/home/...`) into Windows paths before Python
sees them. Either run from PowerShell / cmd, or prefix with
`MSYS_NO_PATHCONV=1` and pass `--key` as a Windows-style path.

## Showcase website

`streettracker showcase` serves a local website (`src/streettracker/web/`:
`aggregate.py` + `metadata.py` + `stats.py` + `server.py` + jinja2
`templates/`) that browses the *enriched* cars across every session in an
output root — plate-read (ANPR) cars joined to DVSA make/model/year/colour —
with the regularly-appearing ones featured, plus a per-car metadata editor
("this is my car", "this is Shaun's car"), and a traffic-statistics page.

```bash
uv run streettracker showcase --output-root output   # http://127.0.0.1:8090/
```

- **Identified cars only.** `web/aggregate.build_showcase` pools the plated
  subset from `analysis.vehicles.build_vehicles` (rebuilt **live** — the
  on-disk `*_vehicles.json` can be stale, e.g. written before `dvsa-label`)
  into one card per physical car, keyed by canonical plate and merged across
  sessions with the same fuzzy clustering the `vehicles --across` cohort uses.
  It also surfaces the richer DVSA fields (`primary_colour`/`fuel_type`/
  `engine_size_cc`) read straight from `<session>_dvsa_labels.json`.
  "Regulars" = the `different-day` kind (seen on ≥2 calendar dates).
- **User metadata** (name / owner / notes / favourite) persists atomically to
  `<output-root>/showcase_metadata.json`, keyed by plate so a tag follows a
  car across every session. This is the *only* state the site writes;
  everything else is read-only upstream data.
- **Local-only by default** (binds `127.0.0.1`; the page shows plate data +
  personal tags). `--host 0.0.0.0` opts into LAN exposure. `POST /api/refresh`
  re-aggregates in place after new sessions are pulled. Images are served from
  the output root through a `.jpg`-only, single-segment-validated route.
- **Statistics page** (`/stats`, `web/stats.build_stats`) — traffic analytics
  over **all** vehicle tracks (`class_name=="car"`, not just the plated subset;
  bucketed by camera-local `time_start`, so a multi-date session splits
  correctly): daily L→R/R→L journeys (click a day → its 15-min profile),
  day-of-week histogram, weekday×hour heatmap, speed distribution + fastest
  cars, and make/colour mix. A **People** section aggregates person tracks
  (footfall, dwell, heatmap) and — where sessions carry `_people.json` —
  rolls up walker/jogger/cyclist/dog-walk counts (cyclists classified since
  2026-06-13, dog walks since 2026-07-07; earlier jogger counts include
  unfiltered riders, ~+4pp). Charts are dependency-free SVG/CSS. One car track
  ≈ one pass (BotSORT can split). **Speed**: `speed_px_s` is inference-frame
  pixels, shown as px/s unless a calibration is set, then mph. Calibrate via
  `configs/showcase.json` (`{"m_per_px": X}` or `{"road_length_m": D}` →
  `m_per_px = D/801`, the traced road's travel-axis pixel length) or the
  `--m-per-px` / `--road-length-m` flags; see `configs/showcase.example.json`.
  mph is approximate (single global factor, ignores perspective).

## Operator control panel

`streettracker control` serves a standalone web app (`src/streettracker/control/`:
`server.py` + `orin.py` + `introspect.py` + jinja2 `templates/`) on the dev box
that lets the **operator drive the day-to-day procedures without Claude attached
— zero tokens once running**. It is a sibling to the showcase site (separate
port, separate concern), built on the same aiohttp + jinja2 + dependency-free
SVG stack.

```bash
uv run streettracker control --output-root output   # http://localhost:8095/
pwsh -NoProfile -File scripts/control_panel.ps1      # detached launcher (opens browser)
```

- **Hosting = on-demand.** `scripts/control_panel.ps1` starts the panel detached
  (outlives the launching terminal, runs outside any Claude helper tree). Ends at
  logout/reboot or when you close it.
- **Access split.** Binds `0.0.0.0` so dashboards are LAN-viewable (phone / wall
  screen), but every control action is **localhost-only**, enforced server-side
  (`server._require_local`) — a LAN viewer can watch but can't restart the camera
  service or promote a model. `--host 127.0.0.1` to keep it fully local.
- Background pollers refresh the Orin (~45 s, non-fatal/timeout-bounded) and
  rescan local `output/`+`runs/` (~30 s); model metadata is read sidecar-first
  (`makemodel_b0.meta.json`, written at promotion) then a one-off torch fallback.
  Introspection reads are mtime-cached so a closed session is parsed once.

**Phase 1 — decision radiator (shipped, read-only).** Orin live-session status
(service state, live session, runtime, images banked, on-demand pull size/ETA via
`remote_inventory`), local session inventory + enrichment badges (ALPR / DVSA /
vehicles / make-model), corpus stats (`runs/uk_crops_*/manifest.json`),
production-model metadata, training-run make@1 charts, and **retrain / promote
recommendations with the evidence numbers** (`introspect.retrain_recommendation`
/ `promote_recommendation`; defaults: retrain if +15 % cars / +3 makes / +5k
crops vs the model's recorded training corpus, promote if a run beats production
make@1 by ≥0.5 pp). Reuses `cli.pull` SSH/inventory helpers (made non-fatal +
timeout-bounded in `control/orin.py`), `common.output`, and
`analysis.snap_assets`. Tests: `tests/test_control/`.

**Phase 2 — in progress (process control).** Shipped + tested:
- `control/progress.py` — pure per-command **stdout progress parsers**
  (`alpr`/`makemodel` `N/total`; `makemodel-train-uk` per-epoch loss/make@1 rows
  for the live curve; pull/dvsa/build counts + summaries) + `estimate_eta`. No
  clock/IO, fully unit-tested.
- `control/jobs.py` — a **subprocess job runner**: streams stdout into the
  parser + a 500-line ring buffer, classifies the exit, holds an in-process
  `SetThreadExecutionState` **wake-lock** while busy, and cancels via Windows
  `taskkill /T`. **Per-lane execution**: a serial **GPU lane** (alpr / makemodel
  / build / train / export-engine) + a serial **net lane** (pull / dvsa /
  vehicles) run concurrently, so a pull overlaps a train but two GPU jobs never
  do (`JobSpec.lane` overrides). In-memory (no on-disk history yet).
- `control/prompts.py` — **`build_prompt`** turns a finished job into a
  complete, self-contained Claude Code prompt (project, command, cwd, exit code,
  captured output tail, command-specific pointer); `summarize_issue` decides
  *whether* to surface one (non-zero exit, or a traceback / OOM / config-error /
  … marker even on a clean exit).
- server `/api/jobs` (GET list, POST submit [localhost + kind-allowlisted],
  GET `{id}`, POST `{id}/cancel` [localhost]) + a **Processes** dashboard panel:
  a localhost run-bar, live job cards with progress bars/ETA, a **live training
  chart** (loss + make@1 per epoch + pre-training facts, rendered from the parsed
  `epochs` metrics for `makemodel-train-uk` jobs), and — when a job fails or
  looks off — a **"Copy Claude prompt"** button so the operator gets a
  paste-ready help request with zero hand-writing. The localhost guard +
  `POST /api/refresh` from Phase 1 pre-staged this access model.
- **Pull byte-ETA.** The runner forces unbuffered child stdout
  (`PYTHONUNBUFFERED`) so lines stream live, and a per-job **dir-watcher**
  samples the local session dir as it fills toward the remote byte total — a
  real transfer progress bar + ETA, not a spinner. `pull` prints a
  machine-readable `size_bytes` (the **main-snap** payload under `--only-main`,
  via `RemoteInventory.main_bytes`, so the denominator matches what's copied);
  `PullParser` turns the session/target/size_bytes lines into a generic
  `watch` directive the runner consumes.
- `control/playbooks.py` — a **multi-step playbook engine** (`PlaybookRunner`):
  ordered steps run sequentially, stopping on the first failure (rest →
  *skipped*); each step is a **job** (run on the shared `JobRunner`, so it
  inherits lanes + live progress + the copy-prompt) or an in-process **action**
  (for SSH/file steps). Playbooks (`PLAYBOOKS` registry; `build_playbook(name,
  ctx, …)` dispatches, `PlaybookContext` carries paths + device config):
  - **enrich** (`alpr-run` → `dvsa-label` → `dvsa-apply` → `vehicles` →
    `makemodel` → `people`) and **build-train** (`makemodel-build-uk` →
    `makemodel-train-uk`, dated dirs) — pure job-chains.
  - **roll** (action: `orin.restart_service` finalises the live session + starts
    a new one, verifies the handover + counts finalised tracks → then a `pull`
    job of the closed session) and **promote** (action: back up `makemodel_b0.pt`
    → copy the best run's `best.pt` in → write the metadata sidecar + corpus
    fingerprint) — **destructive**, so the submit route refuses them without
    `confirm=true` and the UI shows a confirm dialog. `roll` resolves the live
    session over SSH at submit and refuses if the device is down/idle.
  - **reinfer** (one `makemodel` job per local session → an action that POSTs
    the showcase's `/api/refresh`) — re-runs the classifier everywhere + updates
    the showcase.
  - `/api/playbooks` routes (localhost submit/cancel; name + session validated,
    destructive-confirm gated) + a **Playbooks** dashboard panel: step list with
    live status, the running/failed step's job **inlined** (progress bar /
    training chart / copy-prompt), and Cancel. (Fixed a cancel race: a cancel
    landing during a job's spawn window now reaps the process once it exists.)
- `control/history.py` — **on-disk job/playbook history**: each finished item's
  terminal snapshot (incl. the failed-job copy-prompt) is appended to a capped,
  atomically-written JSON list under the output root (`control_jobs.json`,
  `control_playbooks.json`), and the runners' `snapshots()` merge prior-run
  history with live items — so the panel's recent activity (and prompts) survive
  a restart.
- **`/training` full-page view** (`templates/training.html`) — picks the current
  `makemodel-train-uk` job and shows pre-training facts (backbone / input /
  makes / train+val crops), a large live **loss + make@1 curve** with the
  production make@1 as a dashed reference line, a per-epoch table, the training
  corpus's per-make distribution (brand logos), and — once finished — a
  candidate-vs-production verdict + a one-click **Promote** (localhost,
  confirm-gated, run derived from the checkpoint path). Top-bar nav switches
  Dashboard ↔ Training. The shared brand-mark + `trainChart` helpers live in
  `base.html` so both pages use them.

**Phase 2/3 — complete** for the planned scope. The panel covers the full
operator loop tokenlessly: radiator → run individual processes or one-click
playbooks (enrich / build-train / roll / promote / reinfer) with live
progress + ETAs + a copy-paste help prompt on failure, plus a dedicated live
training page. Possible future polish (not required): per-session enrich/pull
shortcut buttons on the sessions table, and surfacing `control_*.json` in
`.gitignore`.

## Migration status

Clean-slate replacement for VehicleTracker + NanoTracker. **Migration
complete** — both ancestor repos archived on GitHub 2026-07-07.

| Phase | Scope | Status |
|---|---|---|
| 0 | repo init + pyproject + CI + configs | done |
| 1 | `common/`: schema, color, output, hourly, summary | done |
| 2 | `inference/` (Ultralytics) + `sources/` (RTSP, file) | done |
| 3 | `device/`: live runtime, snapshotter, dashboard, IR | done — PRs #7/#8/#10/#11/#12/#13 |
| 4a | `analysis/`: recolor + debug-color | done |
| 4b | `analysis/alpr/` wholesale port | done |
| 5 | CLI: `pull`, `export-engine`, `setup_orin.sh`, systemd | done |
| 6 | (opt) Nano archive role | not started |
| 7 | cutover: enable systemd on Orin + decommission Nano + archive old repos | **done** — Orin live since 2026-05-22; `VehicleTracker` + `NanoTracker` archived 2026-07-07 with superseded-by banners |

Tests at HEAD: **606 passing on Python 3.10, ruff clean.**

All subcommands wired: `run`/`batch` go through the asyncio runtime,
`pull`/`export-engine` ship sessions and build engines,
`alpr-run`/`alpr-score`/`alpr-label`/`alpr-report` available under the
`alpr` extra (`uv sync --extra alpr`), `vehicles` runs the
plate-anchored aggregator. Place the bespoke detector at
`src/streettracker/analysis/alpr/models/license_plate_detector.pt`
(gitignored).

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

## ANPR tuning loop

**Current status (updated 2026-06-19 — per-direction band verdict in; core table from 2026-05-27):**

| Layer | Status |
|---|---|
| 4K capture coverage | Solved. 98 % cars get ≥1 snap. Pipeline mode dominates (`pipeline_interval_ms=400`, `pipeline_max_per_track=15`); trigger geometry is minor. |
| Ghost-plate aliasing | Solved (Step 10). Parked-car mask + padding cap eliminate all 5 ghost plates (138 tracks). |
| Per-image high-conf | **Honest post-verdict (2026-06-19, two ~70h post-deploy soaks, completion-time bboxes): L→R 73-76 % mid-road; R→L ~12-13 %/image, ~15-20 %/car — a camera-geometry ceiling.** History: the 06-10 triage (Step 16) correctly found 96 % of R→L failures were snap-latency stale-bbox, but the 06-11 motion-window-hint re-run **overcorrected** — its "41-46 % → 63.9 % pooled, R→L 69.0 % > L→R 58.6 %, asymmetry inverted" was a forward-extrapolation artifact. Completion-time bboxes (exact landing crops, no extrapolation) on the post-deploy soaks restored the pre-Step-16 "R→L ~14 % ceiling at any position" — right all along. L→R reads well; R→L doesn't, at any landing position (`.claude/band_position_done.py`). |
| Per-car aliasing-free | **~78 %** — intrinsic floor of camera + scene, verified across two sessions. Further snap-budget tuning will not move it. |
| Misclassification | Confidence-weighted class voting (PR #23) defends single-frame flips; kinematics guardrail (2026-07-07) flags car-shaped "persons" (`class_suspect`, median bbox w/h ≥ 1.5) — excluded from people analytics. |
| Direction-aware throttling | **NOT live.** Deployed 2026-05-27 (`pipeline_interval_ms_by_direction={forward:300, reverse:400}`) but dropped from the live config (likely the 2026-06-13 splice); now `{}` (uniform 400 ms). Not restored — premise (pack R→L's clean-read window) falsified: R→L is camera-capped ~15-20 % at any position (2026-06-19 verdict). |
| Dataset-level pivot | `vehicles` aggregator (Step 12) + fuzzy clustering (Step 14). **Make/model DVSA-first prong shipped** (PRs #41/#42/#43): `dvsa-label`→`dvsa-apply`→`vehicles` (+`--across`); covers the readable ~25-30 % of cars. **Universal CNN: CompCars trained (#46) but failed UK validation** (domain gap, ~1 %); **pivoted to a UK-native make classifier** (#47) trained on DVSA-auto-labelled local crops — the earlier "~21 % task-bound ceiling" was **wrong** — it was **resolution-bound** (the cropper hardcoded 224 px output, capping every prior lever). Lifting input resolution breaks it: **B0 @224/384/512 = 22 / 32 / 38 % make@1** (2026-06-03); bigger backbones (B4/B5) overfit 800 cars and don't beat B0. **Production UK model = EfficientNet-B5 @456** (make@1 **0.303**, 29 makes; B0 0.271 < B4 0.284 < B5 — bigger backbones win at 1,229 cars; `make_model_source="cnn"`). The old "37.6 %" was small-val optimism (audit: baseline scored 0.207 on unseen cars). Data IS a lever. See [Make/model classification](#makemodel-classification). Learned recolor + visual re-id still to do. |

**Conclusion:** ANPR coverage objective is met. The 78 % aliasing-free
floor is camera-geometry-bound (oblique angles, motion blur, occlusion),
not solvable by more snaps. Pivoted to dataset-level enrichment.
Capture-side read-rate tuning is likewise exhausted — near-camera band
position and camera exposure/shutter (Step 15) were both tried and
falsified.

**Step 16 addendum (2026-06-11) — SUPERSEDED by the 2026-06-19 verdict.**
The 06-11 motion-window-hint re-run claimed the honest per-snapped-car
canonical rate was 63.9 % (R→L 69.0 % > L→R 58.6 %, asymmetry inverted) and
that every pre-Step-16 position/band conclusion was a stale-hint artifact
needing re-derivation. **That re-derivation is now done** — the per-direction
band was deployed 06-13 and two ~70h post-deploy soaks assessed with
completion-time bboxes: the 63.9 %/inversion was ITSELF a forward-extrapolation
artifact of the hint window. **Honest reality: per-snapped-car canonical ~48 %
(L→R ~76 %, R→L ~16-20 %)** — L→R reads well, R→L is camera-geometry-capped at
~15-20 % at every landing position, exactly the pre-Step-16 "R→L ~14 % ceiling"
(right all along). Capture-side read-rate tuning is **exhausted** (band position
+ exposure + offline crop method all tried + falsified); R→L now needs hardware
(a 2nd discreet camera on the approach), not tuning. The Anti-Smearing
falsification stands (same-hint A/B). Full verdict: `.claude/verdict_band_0613.py`
+ `.claude/band_position_done.py`.

### Make/model classification

The dataset-level enrichment pivot. Two prongs:

- **DVSA-first (shipped, PRs #41/#42/#43).** The DVSA MOT API returns
  ground-truth `make`/`model`/`year` from a plate. Pipeline:
  `dvsa-label <session>` harvests `_dvsa_labels.json` (needs
  `configs/dvsa.json`; canonical-plate-filtered so OCR garbage doesn't
  bill the API) → `dvsa-apply <session>` writes make/model onto each
  car's `data.json` + `events.jsonl` record → `vehicles` joins it
  per-vehicle, and `vehicles <dir> --across <dirs>` pools repeat
  vehicles across sessions (same-day / different-day). `TrackRecord` +
  `Vehicle` carry `make`/`model`/`year`/`make_model_source` (= `"dvsa"`).
  Coverage is the readable, ≥3yr-old subset (~25-30 % of cars on this
  scene): newer cars 404 (no MOT record yet), unread plates aren't
  covered at all.
- **Universal CNN (the unreadable majority) — pivoted to UK-native
  training (2026-06-02).** `docs/makemodel_design.md` has the original
  CompCars plan, now partly superseded by the trajectory below:
  1. **CompCars CNN (PR #46).** EfficientNet-B0 fine-tuned on CompCars'
     surveillance subset → **98.8 % CompCars-val** make@1. **But ~1 % on
     real UK cars** — a domain gap (CompCars = zoomed *frontal* Chinese
     toll-camera shots; this scene = *rear/oblique* UK street views),
     plus UK makes CompCars never had (Vauxhall/Tesla/MG/Mini…). A dead
     end as a UK predictor. The inference CLI (`streettracker makemodel`)
     + the lifted `analysis/snap_assets.py` helpers are reusable infra.
  2. **UK-native classifier (PR #47).** Train on THIS scene instead: a
     DVSA plate→make lookup auto-labels the car's own crops. Pipeline:
     `pull --only-main` (Orin→local) → `alpr-run … --pipeline preferred
     --pre-crop --ghost-mask .claude/ghost_mask.json` → `dvsa-label` →
     `makemodel-build-uk` (crops; leakage-safe **by-car** split) →
     `makemodel-train-uk`. Make-only for now (model-level too sparse).
  - **Status (2026-06-03): RESOLUTION-bound, not task-bound — verdict
    resolved, CNN is viable.** The earlier "~21 % task-bound ceiling"
    was an artifact: `VehicleCropper` hardcoded `output_size=224`, so
    *every* prior lever (tuning, 2× data, best-view) was silently capped
    at 224 px input while the source cars were 400-800 px (p50 574). Re-
    extracting higher-res (`makemodel-build-uk --output-size`) + training
    there (`makemodel-train-uk --input-size`) breaks it cleanly.
    Resolution ladder (same 800-car corpus, by-car split, seed 0):

    | Config | make@1 | note |
    |---|---|---|
    | B0 @224 (old "ceiling") | 21.7 % | — |
    | B0 @384 | 32.2 % | +10.5 pp |
    | **B0 @512** | **37.6 %** | best + cheapest |
    | B4 @384 | 29.5 % | overfit, < B0@384 |
    | B5 @456 | 37.7 % | overfit, ≈ B0@512 |

    Resolution is the lever (+16 pp 224→512, decelerating). **Backbone
    capacity is saturated** — B4/B5 (`--backbone`) overfit the small
    corpus (train loss → ~0.04 while val stalls) and don't beat B0, so
    **B0 is right-sized**. The three "negatives" at 224 were all
    resolution-capped, not task limits — best-view collapsed to 9-10 %
    precisely because fewer-crops-at-224 starves a resolution-starved
    model further. **EfficientNet-B5 @456 is the production UK model**
    (make@1 0.303 on 1,229 cars — see the 2026-06-08 UPDATE below: B5 0.303
    > B4 0.284 > B0 0.271; the 06/03 B0@512 "37.6 %" was small-val-inflated,
    honest ≈28 %); predictions carry `make_model_source = "cnn"`, distinct
    from the `"dvsa"` ground truth.
  - **Only untapped lever is data.** B5's overfitting means capacity is
    data-starved: growing the 800-car corpus is what would let a bigger
    backbone push past 37.6 % (a collection effort — mine more Orin
    sessions — not a tuning knob). Higher-res on B0 (640+) keeps
    decelerating; a coarser target (body-type) is still untried. Banked
    + now productive: the 800-car corpus, the full mining pipeline, and
    the `--output-size` / `--input-size` / `--backbone` / `--top-by-area`
    flags.
  - **Mining the Orin grows the corpus** (`pull` snap-bearing
    sessions → `alpr-run` → `dvsa-label` → `makemodel-build-uk` +
    retrain). It grew 523→800 cars without lifting make@1 — but only
    because that was measured *at 224 px*, where resolution (not data)
    was the bottleneck. Now that resolution is lifted and bigger
    backbones overfit 800 cars, **data is the next lever** to push past
    37.6 %. Ops knowledge: **the Orin prunes 4K snaps after ~1 week**
    (keeps JSON) — pull + process within the week or the training images
    are gone. Don't run `alpr-run` on the Orin (compute-bound running the
    live tracker); pull to the dev-box 3080.

  - **UPDATE 2026-06-08 — the 37.6 % was small-val optimism; data IS a
    lever (it lifted *honest* make@1 ~21 → ~28 %).** Pulled + enriched 2
    big new sessions (incl. the 3-day `session_20260601_173815`); corpus
    grew **539 → 1,229 cropped cars / 24 → 29 makes / 3,952 → 9,841
    crops**. Retrained B0@512 b16: **make@1 0.271 (29 makes) / 0.284 (24
    makes, apples-to-apples)** — *looked* like a drop from 0.376, but a
    **leakage-free audit** (`runs/eval_gen.py`) shows the 06/03 baseline
    reproduces 0.3755 on its own ~108-car val yet scores only **0.207 on
    659 unseen cars** — so 0.376 was a small-val artifact and the true
    cross-session make@1 rose ~0.21 → ~0.28 on the bigger, more honest
    val. **Lesson: size the val (>~200 cars) before trusting make@1.** The
    `--backbone b4/b5` re-test on 1,229 cars **reverses the old "overfits
    800 cars" verdict** — at 2.3× data capacity pays off: **B0 0.271 < B4
    0.284 < B5 0.303** (monotonic, 29-make val). **B5@456 is now the
    default model** (make@1 0.303). Inference is **UK-aware**:
    `streettracker makemodel` auto-detects the make-only checkpoint + reads
    its `arch` + `input_size` from the checkpoint, so any backbone is a
    drop-in default (`make_model_source="cnn"`).

### Step trajectory

| # | Date | What | Headline result | Source |
|---|---|---|---|---|
| 1 | 05-22 | Observability — `snap_stats` in `_meta.json` (latency p50/p90/p99, blur skips, HTTP counters) | enabled tuning | PR #16 |
| 6 | 05-25 | First ALPR measurement on 27.3h soak (880 tracks) | 59 % per-car @ conf≥0.95 (preferred pipeline). **Bespoke pipeline contributes no useful signal — disagreements with preferred at 98/108 cases; preferred is conf ≥0.99 clean UK plates, bespoke is truncated/garbled.** | — |
| 7 | 05-25 | Vehicle pre-crop wrapper (`PreCropDetector` + `--pre-crop` flag, detector default bumped `yolo-v9-t-384` → `yolo-v9-t-640`) | 59 → **91.5 %** per-car ghost-filtered. **Aliasing new bottleneck:** `FD61PVX` parked car aliased onto 363/410 tracks via largest-vehicle heuristic. | PR (in repo) |
| 8 | 05-25 | BotSORT bbox pipe — runtime persists per-snap sub-stream bbox in `TrackRecord.main_snap_bboxes`; `alpr-run` reads it back as `bbox_hint` for `PreCropDetector` | Predicted ≥ 95 %; actual **~78.5 %**. Bbox correctly targets tracked car, but in late R→L snaps the bbox grows wide enough to physically encompass the adjacent parked `FD61PVX` car. | PR #28 |
| 9 | 05-26 | Re-soak measurement on 15h Step-8 build | 78.5 % verified. `FD61PVX` still in 106/247 tracks. **Asymmetric by direction:** L→R 93 %, R→L 43 % per-image. | — |
| 10 | 05-26 | Ghost mask (zero-fill parked-car rect before any detector sees pixels) + `PreCropDetector(pad_max_px=30)` cap on padding | All 5 ghost plates eliminated. Strict per-car aliasing-free (top read is correct) **55.4 → 78.9 %** (+23.5 pp). The 78.9 % is the **true** aliasing-free floor — Step 9's 78.5 % estimate was correct but unverifiable until the mask proved it. | — |
| 11 | 05-26 | `t_usable_frac` trim `[0.10, 0.67] → [0.10, 0.45]` (snap-firing band shrunk past `FD61PVX` zone) | Per-image **+13 pp** (L→R +17 pp, R→L +3 pp). Snap budget redistributed (mean fires/track 1.19 → 1.45, `pipeline_budget_exhausted` 954 → 665). **Per-car aliasing-free flat at ~78 %** — confirms intrinsic floor; cars that *could* be captured in mid-band already were. | — |
| 12 | 05-26 | `streettracker vehicles` plate-anchored aggregator — folds `data.json` + `alpr_by_track.json` into per-vehicle records with `n_visits`, `gap_minutes_max/min`, direction + color histograms, inline visit list | Step 10 session: 3 recurring. Step 11 session: 6 recurring including `HX18MYJ` going R→L then L→R 105 min later. | — |
| 13a | 05-27 | Direction-aware `pipeline_interval_ms_by_direction` runtime — fire faster in one direction's narrow clean-read window | Deployed `{forward:300, reverse:400}` (R→L 1.33× rate, L→R unchanged). **Later dropped from the live config + not restored — premise falsified 2026-06-19: R→L is camera-capped, there's no clean-read window to pack.** | PRs #30/#31 |
| 13b | 05-27 | Multi-frame plate consensus (`analysis/alpr/consensus.py`) — confidence-weighted character voting across a track's reads | **Negative result on this scene** (-43 pp vs best-of-N at conf 0.9). Per-image reads of one track frequently capture DIFFERENT physical plates (parked cars vs tracked car vs mask leakage), so voting dilutes. Primitive kept as infra. | — |
| 14 | 05-27 | Fuzzy plate clustering in `vehicles.py` (rapidfuzz, ratio default 85, same-length only); no temporal-overlap rejection | `LD22BWG`/`LD22BMG` correctly merged (BotSORT ID-switch on same silver hatchback); Step 11 recurring 6 → 8. | PRs #33/#34 |
| 15 | 05-31 | Reolink **Anti-Smearing** exposure + shutter cap (`125→32`) to freeze near-camera motion blur; validation soak ran a widened `[0.10,0.60]` band, assessed + reverted 06-01 | **Falsified.** Canonical read-rate ~halved (matched midday daylight 48.5 → 25.1 %, -23 pp, both directions); near-camera zone did not recover. Faster shutter → higher gain → sensor noise costs more than the motion blur it removes (mid-road reads were never blur-limited). Reverted ISP to Auto/125 + band to `[0.10,0.45]`. **Capture-side levers (band position + exposure) exhausted.** | — |
| 16 | 06-10 | **Stale-bbox fix.** Operator triage of 99 R→L failed snaps (`.claude/triage_rl.py` labelling site) → **96 % were snap latency**: the 4K HTTP snap lands ~0.7-1.3 s after fire (p50 710 ms), the car exits its fire-time bbox (+≤30 px pad) and the plate detector was shown empty road; the fastest cars exit the *frame* (the 11 % "no car" bucket had the highest host speeds). Zero smeared, zero sharp-unread — optics/OCR were never the limit. Fix: **motion-window hints** (`snap_assets.resolve_bbox_hint_window` — union of fire-time bboxes `i..i+3` ≈ the latency horizon; linear extrapolation at end-of-track, shift capped at 2.5 bbox-dims) + **vehicle stage inside the window** (`PreCropDetector(vehicle_stage_in_hint=True)`; window-only A/B *lost* 15 pp R→L detection to input downscale — tight vehicle crops restore plate pixels). `alpr-run --hint-lookahead N` (default 3, 0 = legacy). | A/B on `session_20260530_165958` (`.claude/measure_hint_window_ab.py`): **canonical cars/session 43 → 99 (+130 %)**; L→R detection 49 → 90 %, canonical/image 32.4 → 75.7 %; R→L canonical cars 4 → 11 (~2.8×). **[CORRECTED 2026-06-19: the R→L gain here was a forward-extrapolation artifact of the hint window — completion-time bboxes on two post-deploy soaks show R→L is camera-capped ~15-20 % at *every* landing position; only the L→R gain was real. The "per-direction band is the next lever" follow-up was deployed 06-13 and confirmed to help ONLY L→R.]** Parked-beacon count flat — no aliasing cost. Related same-day work: **parked-plate beacon suppression** (PRs #57/#58) — a parked Duster read 99×/69 tracks had become the showcase's phantom #1 "regular" (55 visits); now suppressed into a `parked_episodes` record, with `dvsa-label` harvests scrubbed (replace + orphan-clear `track_ids` semantics) and the corpus rebuilt (1,311 cars / 8,067 crops / 29 makes in `runs/uk_crops_0610_512`). | PR #59 |

Aggregation scripts that re-produce each measurement table live at
`.claude/aggregate_step{8,10,11}.py`, `.claude/measure_consensus.py`,
and `.claude/measure_hint_window_ab.py` (Step 16 A/B; expects the
`_alpr.baseline_prewindow.json` backup next to the session's live
output). The Step 16 failure-triage labelling site is
`.claude/triage_rl.py` (`--select` builds the sample, then serves on
:8091; labels persist to `.claude/triage/labels.json`).

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

### Latest snap_stats (Step 11 soak, `session_20260526_124704`, 5.5h)

| HTTP | Latency (ms) | Blur | Pipeline | Mean fires/track |
|---|---|---|---|---|
| 2381 / 2381 / 0 | p50=710 p90=957 p99=1221 max=2133 | skipped=0 | fires=2921 throttled=8565 exhausted=665 | 1.45 |

`min_sharpness=100.0` essentially never trips (`blur_skipped=0` in both
recent soaks); do not lower.

### Bugs hardened during the tuning loop

**asset_prefix split-flip** ([PR #21](https://github.com/nicholasaross/StreetTracker/pull/21)).
BotSORT reassigns a track's class as evidence accumulates, so
`track.class_id` can flip mid-life. The 4K snap path was built from
`class_id` at fire time, while the TrackRecord + tile/HQ were built
from it at finalize time. A flip between left the on-disk filenames
disagreeing with `record.asset_prefix` (26/880 tracks ≈ 3 % in the
soak). Fix: `BufferedTrack` records the fire-time prefix per
`snap_index` and a `final_prefix` slot. `finalize_track` locks
`final_prefix`, sweeps already-saved snaps whose fire prefix differs,
and renames them; the snap `_on_done` callback handles late completions.
Live on Orin since 2026-05-25.

**Confidence-weighted class voting** (PR #23). PR #21 made the *file
prefix* consistent with the *finalize-time class*, but the
finalize-time class itself came from "most-recent-detection wins".
A single stray YOLO frame at the end of a track could corrupt the
entire record. Fix: `BufferedTrack.class_votes: dict[class_id ->
sum(detection_score)]`; `class_id = argmax(class_votes)` always.
Three new tests cover tie-break (insertion-order), majority resilience,
and accumulated-evidence flipping. Doesn't fix the rare case where YOLO
is *consistently* wrong on a scene (e.g. parked grey Prius+ labelled
person) — that case is now covered by the **kinematics guardrail**
(2026-07-07): `compute_attributes` sets `TrackRecord.class_suspect=True`
on a "person" whose median bbox aspect (w/h) ≥ 1.5. Measured on six
completion-bbox soaks (10,691 person / 15,254 car tracks): person p99 =
0.82, rider max = 1.44, car p25 = 1.40 — so 1.5 catches the
misclassified parked cars (all wide AND slow; speed added no signal)
with zero rider collateral. Suspects keep their class + assets but are
excluded from `web/stats.py` people+car counts and `analysis/people.py`
(summary `n_suspect_excluded`). Old sessions lack the field (treated
as not-suspect).

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
   [Recovery from a venv rebuild](#repo--venv-setup)).
3. Restart with the OLD `camera.json` — verify the new code parses
   unchanged.
4. THEN edit `camera.json`, scp, restart, verify.

Value changes to existing keys (e.g. Step 11's `t_usable_frac` trim)
are safe to deploy directly.

### Vehicles aggregator (`streettracker vehicles`)

Folds per-track records + per-track best ALPR reads into a per-vehicle
view keyed by canonical plate. Run with
`uv run streettracker vehicles output/<session>`; writes
`<session>_vehicles.json`.

| Vehicle field | Notes |
|---|---|
| `plate` / `plate_conf` | canonical plate (highest-conf variant); `null` for unread cars |
| `n_visits`, `track_ids` | tracks attributed to this vehicle |
| `first_seen` / `last_seen` | ISO timestamps |
| `gap_minutes_max` / `gap_minutes_min` | between consecutive visits; 0 if `n_visits==1`; can be negative when BotSORT ID-switches |
| `directions` / `colors` | per-visit histograms |
| `visits` | inline array with each visit's TrackRecord fields + best-read snap filename |
| `plate_variants` | `[(str, max_conf), ...]` of OCR variants collapsed into the canonical via fuzzy clustering |

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

## Fresh deployment procedure

Canonical reference for any redeploy on a wiped Orin (new chassis, SD
swap, or JetPack upgrade — see [JetPack 7 upgrade plan](#jetpack-7-upgrade-plan)
for JP7 deltas).

### Backups to keep off-device

Three things are not in the repo and not portable from a fresh OS:

| File | Why |
|---|---|
| `~/streettracker/configs/camera.json` | Reolink credentials + operator-traced road polygon + per-install snap_gate tuning. Gitignored. |
| `/etc/sudoers.d/streettracker-svc` | Lets the `streettracker` user run `systemctl * streettracker.service` without password prompt. |
| `~/.ssh/authorized_keys` for `streettracker` | SSH keys for admin. |

The TRT engine (`yolov8m.engine`) is deliberately *not* on this list —
it's GPU-architecture-bound and must be rebuilt on the target device.

### One-time host setup

```bash
# As an admin user on the freshly-flashed Orin:
sudo useradd -m -s /bin/bash streettracker
sudo usermod -aG video,sudo,systemd-journal streettracker
# video for nvidia; systemd-journal so streettracker can read its own
# service logs (without this, `journalctl -u streettracker` looks
# empty rather than erroring).
sudo mkdir -p /home/streettracker/.ssh
sudo cp ~/.ssh/authorized_keys /home/streettracker/.ssh/
sudo chown -R streettracker:streettracker /home/streettracker/.ssh
sudo chmod 700 /home/streettracker/.ssh
sudo chmod 600 /home/streettracker/.ssh/authorized_keys

# Scoped NOPASSWD for service management.
echo "streettracker ALL=(ALL) NOPASSWD: /bin/systemctl * streettracker.service" \
  | sudo tee /etc/sudoers.d/streettracker-svc
sudo chmod 440 /etc/sudoers.d/streettracker-svc
```

### Repo + venv setup

```bash
ssh streettracker@orin
git clone https://github.com/nicholasaross/StreetTracker.git ~/streettracker
cd ~/streettracker
scripts/setup_orin.sh         # idempotent: apt deps + uv + uv sync +
                              # tensorrt symlinks + bashrc LD_LIBRARY_PATH
                              # + nvpmodel
```

Handles both JP6 (Ubuntu 22.04 / Py 3.10) and JP7 (Ubuntu 24.04 / Py
3.12) via uv's `.python-version` honouring.

**Recovery from a venv rebuild.** A `uv sync` that recreates `.venv`
(e.g. after a `.python-version` change) wipes the system-tensorrt
symlinks. The systemd service then fails with `ModuleNotFoundError: No
module named 'tensorrt'` and restart-loops. Recovery:

```bash
ssh streettracker@orin "cd ~/streettracker && scripts/setup_orin.sh --symlinks-only && sudo -n systemctl restart streettracker.service"
```

No sudo/apt/network needed — `--symlinks-only` runs the symlink step in
isolation. Hit this on 2026-05-25 during the 3.10 → 3.11 → 3.10 pin
churn, hence the dedicated flag.

### Per-deploy files

```bash
scp <backup>/camera.json streettracker@orin:~/streettracker/configs/

cd ~/streettracker
uv run streettracker export-engine yolov8m.pt
# ~5 min on Orin Nano Super. The output filename must match
# `inference.engine_path` in camera.json — the load-time validator
# catches mismatches with a clear JSON-path error.
```

### Systemd install + smoke

```bash
sudo cp scripts/systemd/streettracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo -n systemctl enable --now streettracker.service
journalctl -u streettracker -f                  # Ctrl-C once frames flow
curl http://orin:8080/                          # "no sessions yet" page
                                                # or latest summary redirect
```

Roll back: `sudo -n systemctl disable --now streettracker.service`. SD
card unchanged; next `enable --now` starts a fresh session.

## JetPack 7 upgrade plan

NVIDIA's JP7 for Orin Nano has shipped as **JetPack 7.2** (Jetson Linux
39.2, CUDA 13.2.1, TensorRT 10.16.2, Ubuntu 24.04, Py 3.12 native;
verified 2026-06-07). Before flashing, five repo edits have to land or
the redeploy will fail at `uv sync` — and the hard prerequisite gating
all of it is **torch-wheel availability** (see the go/no-go gate below):

| What | Where | JP6 today | JP7 target |
|---|---|---|---|
| Python pin | `.python-version` | `3.10` | `3.12` |
| Jetson torch wheel index | `pyproject.toml` `[[tool.uv.index]]` | `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/` | `https://pypi.jetson-ai-lab.io/jp7/cu13x/+simple/` (CUDA 13.2 → `cu132`/`cu130`; **not published as of 2026-06-07** — gate-check via `scripts/check_jp7_wheel.ps1`) |
| cuDSS dep | `pyproject.toml` `nvidia-cudss-cu12` | `cu12` | `cu13` if JP7 ships CUDA 13.x (confirm) |
| CUDA lib path | `scripts/setup_orin.sh` `CUDA_LIB=...` | `/usr/local/cuda-12.6/lib64` | auto-detect or hardcode `cuda-13.x` |
| Systemd LD_LIBRARY_PATH | `scripts/systemd/streettracker.service` Environment | `cuda-12.6/lib64` + `python3.10/site-packages/nvidia/cu12/lib` | both paths shift |

**Go/no-go gate.** JP7.2 is out, but **don't flash until the wheel gate
is green**: run `scripts/check_jp7_wheel.ps1` from the dev box (exit 0 =
a `jp7`/CUDA-13.x `cp312`+`aarch64` torch **and** torchvision wheel is
published on jetson-ai-lab — the only `sm_87`-safe source; the official
PyTorch SBSA `cu126` wheel has `sm_87` gaps → silent NaN). **NOT YET as
of 2026-06-07** (`jp7/` 404s); stay on JP6 meanwhile — the live service
is safe (`--no-sync` blocks resync against the decaying JP6 index).

**When the gate is green.** Spin up `claude/jp7-readiness`, make the five
edits, then flash and run the [Fresh deployment
procedure](#fresh-deployment-procedure). **Caveat — there is only ONE
Orin, no separate test box** (the earlier "verify on a test Orin" plan is
infeasible): flash JP7.2 onto a *second* NVMe and keep the current JP6
drive physically intact as the rollback + `output/` backstop; a
major-version flash also rewrites the module QSPI bootloader, so
reverting = re-flash QSPI to JP6 + reinsert the old drive (downtime, not
a swap). PR #15 hardening (forgiving stream-name lookup + engine-path
validation) means small config drift won't block the redeploy.

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

### Live geometry (current values)

| Field | Value | Origin |
|---|---|---|
| Polygon vertices | 23 fractional `(x, y)` | operator sketch, `.claude/sketch_me_done.png` |
| `t_usable_frac` | `[0.10, 0.45]` | Step 11 trim (was `[0.10, 0.67]`) |
| `trigger_t_prime` | `[0.05, 0.20, 0.35, 0.65, 0.80, 0.95]` (in usable-band coords) | 3 forward + 3 reverse triggers |
| `trigger_directions` | `["forward", "forward", "forward", "reverse", "reverse", "reverse"]` | asymmetric: forward = R→L approach, reverse = L→R depart |
| `pipeline_interval_ms` | `400` | base cadence |
| `pipeline_max_per_track` | `15` | per-track cap |
| `pipeline_interval_ms_by_direction` | `{}` (empty — uniform 400 ms both directions) | Step 13a deployed `{forward:300,reverse:400}` 2026-05-27 but it's **not in the live config** (dropped, likely the 2026-06-13 splice). Not restored — premise falsified (R→L camera-capped ~15-20 %, 2026-06-19). |
| `pipeline_t_usable_by_direction` | `{forward:[0.10,0.20], reverse:[0.30,0.60]}` | per-direction pipeline bands; reverse lower edge nudged 0.25→0.30 on 2026-06-19 (validated 2026-06-22: L→R per-car 92.3 %). See [config layout](#snap-gate-config-layout). |
| Ghost mask | rect `[750, 700, 960, 860]` at `source_size=[4512, 2512]` | covers parked `FD61PVX` car at 4K `(853, 780)` (`t_norm=0.593`) |

**`trigger_t_prime` values are in `[0, 1]` of the USABLE band, not
global** — so they automatically shrink with `t_usable_frac`. The
"reverse" triggers (placed for L→R entry) now fire at absolute t_norm
0.33 / 0.38 / 0.43 instead of 0.47 / 0.56 / 0.65 — moves L→R fires out
of the FD61PVX zone (intended secondary benefit of the Step 11 trim).

L→R = rear plate visible, R→L = front plate visible (visually confirmed
in 27.3h soak samples).

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

`pipeline_t_usable_by_direction` (optional, schema-additive) gates
PIPELINE fires per motion direction in **absolute t_norm coords** (same
space as `t_usable_frac`; trigger t' values are unaffected). Directions
absent from the dict — and frames whose motion direction isn't known
yet — fall back to `t_usable_frac`. The bands were the 2026-06-11 band
re-analysis prescription, deployed 2026-06-13. **Post-deploy verdict
(2026-06-19, two ~70h soaks, completion-time bboxes, 100 % done-bbox
coverage):** the L→R band is validated — rear plates read 73-76 % at
mid-road landings, per-car ~73-82 % (a genuine win). The R→L prescription
is **falsified**: front plates are camera-geometry-capped at ~15-20 % at
*every* landing position (`band_position_done.py`), so firing R→L far buys
nothing. The 2026-06-11 "R→L reads 56-76 % fired far" was a motion-window-
hint forward-extrapolation artifact; completion-time bboxes (exact landing
crops, no extrapolation) corrected it, restoring the pre-Step-16 ~14 %
geometry floor. The reverse lower edge was nudged **0.25→0.30 on 2026-06-19**
(L→R departs/recedes ~0.10 t_norm during the ~700 ms snap latency, so 0.25
fires landed in the weak far zone) and **validated 2026-06-22** on the ~72h
`session_20260619_111111` soak: L→R per-car 73-82 % → **92.3 %**, per-image
65.9 % → **77.4 %**, net per-car ~48 % → **57.9 %**. The landing histogram
confirms the mechanism — L→R reads peak **83-89 % at 0.20-0.40 t_norm**, exactly
where the raised 0.30 floor now lands them (vs 26-62 % below 0.20); R→L
unchanged at ~23 % per-car, as designed (the nudge only touches the reverse
band). And
`pipeline_interval_ms_by_direction` is **`{}`** (uniform 400 ms) — the
Step 13a `{forward:300,reverse:400}` throttle is not live and won't be
restored (same falsified R→L premise, and faster R→L firing only burns the
shared HTTP semaphore the readable L→R direction needs). Once deployed,
preserve these fields like the other `pipeline_*` fields when splicing
configs. Reproduce the verdict: `.claude/verdict_band_0613.py` +
`.claude/band_position_done.py`.

The local artifact `.claude/snap_gate.json` historically only carried
`polygon_frac` / `trigger_t_prime` / `trigger_directions` / `t_usable_frac`.
**Always preserve the live `pipeline_*` fields when scp'ing** — pull
the live config, edit only what you need, scp back. See [Adjusting the
snap_gate](#adjusting-the-snap_gate).

### Artifacts in `.claude/` (per-install, gitignored)

| File | Purpose |
|---|---|
| `live_frame.jpg` | Reference 4K snap of the live scene (fetched directly from Reolink). |
| `sketch_me.png` | 1200×668 downscale of the live frame, given to the operator to sketch on. |
| `sketch_me_done.png` | Operator's returned sketch — magenta (`#FF00FF`) outline of the visible road tarmac. |
| `road_polygon_user.json` | Polygon vertices extracted from the magenta sketch (fractional coords, resolution-independent). |
| `triggers_proposal.json` | Full trigger spec: polygon vertices, principal axis, centroid, raw t-range, `t_usable_orig`, `triggers_tprime`, `trigger_directions`, `trigger_labels`. |
| `triggers_proposal.jpg` | Visual overlay on the live frame — yellow road outline, dimmed excluded regions, coloured trigger lines. Re-render via `uv run python .claude/_render_triggers_overlay.py`. |
| `snap_gate.json` | Subset of `triggers_proposal.json` for shipping to `camera.json` under `snapshot.snap_gate`. |
| `ghost_mask.json` | Parked-car rects + `source_size`, fed to `alpr-run --ghost-mask`. |
| `aggregate_step{8,10,11}.py`, `measure_consensus.py` | Re-producible measurement scripts for each step's table. |

### Adjusting the snap_gate

Most operator changes ("move T2 toward camera", "drop T1", "add a
trigger between T2 and T3") only need t' values:

1. Edit `triggers_tprime` (and optionally `t_usable_orig`) in
   `.claude/triggers_proposal.json`. Values in `[0, 1]` over the usable
   band — t'=0 = distant edge, t'=1 = near edge.
2. Re-render: `uv run python .claude/_render_triggers_overlay.py`.
3. Regenerate `.claude/snap_gate.json` from the JSON
   (`polygon_frac=vertices_frac`, `trigger_t_prime=triggers_tprime`,
   `t_usable_frac=t_usable_orig`, `trigger_directions` if used).
4. Pull live config, splice the new fields in, scp back, restart:

```bash
scp streettracker@orin:~/streettracker/configs/camera.json /tmp/orin_live.json
# Edit (Python or jq) -- replace the snap_gate value-changes only;
# preserve pipeline_interval_ms / pipeline_max_per_track /
# pipeline_interval_ms_by_direction.
ssh streettracker@orin "cp ~/streettracker/configs/camera.json ~/streettracker/configs/camera.json.bak.$(date -u +%Y%m%dT%H%M%SZ)"
scp /tmp/orin_proposed.json streettracker@orin:~/streettracker/configs/camera.json
ssh streettracker@orin "sudo -n systemctl restart streettracker.service"
ssh streettracker@orin "journalctl -u streettracker.service --since '30 seconds ago' --no-pager | tail -15"
```

For schema-additive changes (new key in `SnapGateSpec`), see
[Schema-additive config changes](#schema-additive-config-changes).

### Re-sketching the road (camera moved, new view)

1. Pull a fresh 4K frame via the Reolink HTTP `cmd=Snap` endpoint →
   `.claude/live_frame.jpg`.
2. Downscale to 1200px wide → `.claude/sketch_me.png`.
3. Operator traces the visible road tarmac as a closed magenta
   (`#FF00FF`) outline → `.claude/sketch_me_done.png`.
4. Extract polygon: detect magenta pixels, dilate + fill, take the
   largest connected component, walk boundary radially from centroid,
   Ramer-Douglas-Peucker simplify (`eps≈6`) →
   `.claude/road_polygon_user.json`.
5. Compute principal axis (PCA on polygon vertices in pixel coords),
   propose initial triggers + `t_usable` band, render overlay,
   iterate with operator until approved. Continue from "Adjusting"
   step 3.

### Live planner behaviour

| `SnapPlannerConfig.road_gate` | Mode | Notes |
|---|---|---|
| `None`, `right_half_only=True` | Right-half zone-thirds | Pre-polygon fallback. Not used live. |
| `None`, `right_half_only=False` | Legacy peak/decay | Benchmark only. |
| `RoadGateConfig(...)` w/ `pipeline_interval_ms = 0` | Road polygon + trigger-only | Crossing semantics: each frame compares bbox-centre's t' with previous frame's; a not-yet-fired trigger between them fires *if its direction tag matches the motion sign*. After a fire, `prev_t_prime` advances to the trigger's t' so subsequent triggers in the same motion remain detectable one-per-frame. Asymmetric triggers let R→L and L→R each have their own early-capture trigger without one direction consuming the other's. |
| `RoadGateConfig(...)` w/ `pipeline_interval_ms > 0` | + pipeline mode | **Live deployment since 2026-05-23.** Trigger crossings as above. *Additionally*, while a track sits inside `t_usable`, fire a snap every `pipeline_interval_ms` (or per-direction override), capped at `pipeline_max_per_track`. Pipeline fires share `max_concurrent` with trigger fires; dominate by ~10×. See `consider_pipeline` in `snap_planner.py`. |
