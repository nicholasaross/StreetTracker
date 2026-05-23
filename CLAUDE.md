# CLAUDE.md

Guidance for Claude Code when working in StreetTracker.

## Project overview

StreetTracker unifies VehicleTracker (dev-box, file input) and NanoTracker
(Jetson Nano original, live RTSP) onto a single Python 3.12 + Ultralytics
+ TensorRT stack, targeting Jetson Orin Nano 8GB Super as the primary
device.

Pipeline:

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

## Compatibility rules

- **Python 3.12.** Pin via `.python-version`. uv manages the install.
- **No Python 3.6 hacks.** sys.path reorder, NamedTuple-for-dataclass,
  `# type:` comments — all gone. Use `@dataclass(slots=True)` and PEP-604
  unions (`X | None`).
- **TRT engines are not portable** across GPU architectures. Always
  build engines ON the target device (Orin or dev-box-with-matching-GPU).

## Architecture

```
src/streettracker/
├── common/                 # shared across runtime + analysis
│   ├── schema.py           # TrackRecord, SessionMeta @dataclass
│   ├── color.py            # COLOR_RANGES + vote_color()
│   ├── summary.py          # HTML dashboard generation
│   ├── hourly.py           # build_hourly_rollup()
│   └── output.py           # EventLog, save_json, file-path helpers
├── inference/              # YOLO + BotSORT via Ultralytics
├── sources/                # RTSP (FFmpeg), file (NVDEC on Orin)
├── device/                 # Orin-only: live runtime, snapshotter, dashboard, IR
├── analysis/               # off-device: ALPR, recolor, make/model, re-id
└── cli/                    # `streettracker` entry + subcommands
```

Single import root: `from streettracker.common.schema import TrackRecord`.

## Device runtime notes (Orin Nano 8GB Super)

- JetPack 6.x ships Ubuntu 22.04 / Python 3.10 → install Python 3.12 via
  `uv python install 3.12`. JetPack 7.x ships Ubuntu 24.04 / Python 3.12
  natively. uv handles both transparently.
- Ultralytics' built-in TRT path (`YOLO('best.engine')`) replaces
  NanoTracker's hand-rolled `trt_engine.py` (manual YOLOv8 decode + numpy
  NMS) and bespoke IoU tracker.
- Live RTSP from Reolink: same FFmpeg-backend workaround as NanoTracker
  (`cv2.CAP_FFMPEG` + `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`).
  Don't try GStreamer for live RTSP — it stalls on Reolink keyframes.
- MP4 input on Orin uses GStreamer + `nvv4l2decoder` (NVDEC). Works fine
  for file input; only the live-RTSP case is broken with cv2-GStreamer.

## Output schema (preserved from NanoTracker)

Per finalized track:

| File | Quality | Use |
|---|---|---|
| `{prefix}_{id}.jpg` | q=85, ~80px | dashboard tile |
| `{prefix}_{id}_hq.jpg` | q=95, ~250px | quick color/silhouette |
| `{prefix}_{id}_main_{N}.jpg` | 4K Reolink HTTP | ALPR / make-model |

`{prefix}` is `vehicle` or `person`. `N` is 1..max_snaps_per_track.

Session files:
- `{session}_events.jsonl` — appended line-per-track (crash-safe)
- `{session}_data.json` — array of records, written at session end
- `{session}_meta.json` — session-level metadata + IR periods
- `{session}_hourly.json` — per-hour rollup
- `{session}_summary.html` — dashboard
- `index.html` — auto-redirect to latest summary

JSON record fields: see `common/schema.py` (`TrackRecord`).

## Common tasks

- Run tests: `uv run pytest`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`
- Type check: `uv run mypy src/`
- Batch on dev box: `uv run streettracker batch sample.mp4`
- Build TRT engine on device: `uv run streettracker export-engine yolov8m.pt`

## Migration status

This repo is a clean-slate replacement for VehicleTracker + NanoTracker.
Source-of-truth scaffolding was developed in NanoTracker's
`claude/nano-orin-setup-plan-CCWUD` branch under `streettracker/` and
mirrored here phase by phase. All phases are code-complete; the
remaining cutover step is an operator hand-off (install + enable the
systemd unit on the Orin, decommission the old Nano).

| Phase | Scope | Status |
|---|---|---|
| 0 | repo init + pyproject + CI + configs | **done** |
| 1 | `common/`: schema, color, output, hourly, summary | **done** |
| 2 | `inference/` (Ultralytics runner) + `sources/` (RTSP, file) | **done** |
| 4a | `analysis/`: recolor + debug-color | **done** |
| 5 | CLI: `pull`, `export-engine`, `setup_orin.sh`, systemd | **done** |
| 4b | `analysis/alpr/` wholesale port | **done** |
| 3 | `device/`: live runtime, snapshotter, dashboard, IR | **done** — see [Phase 3 progress](#phase-3-progress) below |
| 6 | (opt) original Nano archive role | not started |
| 7 | cutover: enable systemd on Orin + decommission Nano + archive old repos | **mostly done** — Orin live since 2026-05-22; only the `VehicleTracker` + `NanoTracker` repo archival on GitHub is outstanding. See [Cutover status](#cutover-status) and [Fresh deployment procedure](#fresh-deployment-procedure). |

Tests at HEAD: **362 passing, ruff clean.**

Verify locally:

```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
```

CLI smoke:

```bash
uv run streettracker --help
uv run streettracker --version
uv run streettracker recolor --help
uv run streettracker debug-color --help
```

All subcommands are wired: `run` / `batch` go through the asyncio
runtime, `pull` / `export-engine` ship sessions and build engines,
and `alpr-run` / `alpr-score` / `alpr-label` / `alpr-report` are
available under the `alpr` extra (`uv sync --extra alpr`). Place the
bespoke detector at
`src/streettracker/analysis/alpr/models/license_plate_detector.pt`
(gitignored). `scripts/setup_orin.sh` and
`scripts/systemd/streettracker.service` are in place for device
install; see [Cutover](#cutover) for the install commands.

**Windows + Git Bash gotcha for `streettracker pull`**: MSYS rewrites
POSIX-looking arguments (`/home/...`) into Windows paths before Python
sees them, mangling `--remote-parent` and `--key`. Either run from
PowerShell / cmd, or prefix the invocation with `MSYS_NO_PATHCONV=1`
and pass `--key` as a Windows-style path.

## Phase 3 progress

Phase 3 (`device/`: live runtime, snapshotter, dashboard, IR) shipped
across six PRs landing on top of the Orin deploy hardening. The
architectural decisions below are locked in; re-deciding them
mid-stream caused churn during scoping, and they remain the contract
the runtime upholds.

### Decisions locked in

- **Asyncio throughout.** Entry is `asyncio.run(run_session(config))`.
  Blocking sources (`cv2.VideoCapture.read()`) go to a background
  thread that pushes into an `asyncio.Queue`. Blocking inference goes
  to `loop.run_in_executor`. HTTP is `aiohttp`. The systemd unit
  passes `--no-sync` so a transient `uv` resolver flap can't reinstall
  PyPI torch on top of the Jetson wheel.
- **Frozen dataclasses + strict JSON loader.** `common/config.py`
  rejects unknown keys with a JSON-path error. `from __future__ import
  annotations` + `typing.get_type_hints()` resolves string annotations
  at load time; that pattern is required because dataclasses store
  `f.type` as a string under PEP 563.
- **Graceful shutdown via `loop.add_signal_handler(SIGTERM, ...)`.**
  Bounded by a 30s timeout (configurable) so systemd won't SIGKILL
  mid-write. Stop the source first, drain active tracks through
  finalize, write summary HTML + data.json + meta.json + hourly.json,
  then `loop.stop()`.
- **Heavy unit tests + manual Orin smoke for the runtime loop.** No
  recorded-frames integration test in CI (would balloon the repo and
  CI deliberately skips torch / ultralytics). End-to-end validation
  happens by ssh'ing into the Orin against the live Reolink.

### Shipped manifest

| PR | What | Landed |
|---|---|---|
| 3a | `common/config.py` (frozen dataclasses, strict JSON loader) + aiohttp dep | [#7](https://github.com/nicholasaross/StreetTracker/pull/7) |
| 3b | `device/snapshotter.py` (aiohttp Reolink client, semaphore, retries, keep_days cleanup task) | [#8](https://github.com/nicholasaross/StreetTracker/pull/8) |
| 3c | `device/ir_detector.py` (R/G/B channel-diff + hysteresis → `IRPeriod` emission on day-resume) | [#10](https://github.com/nicholasaross/StreetTracker/pull/10) |
| 3d | `device/dashboard.py` (aiohttp.web static server + lifecycle helpers) | [#11](https://github.com/nicholasaross/StreetTracker/pull/11) |
| 3e | `device/runtime.py` + `device/track_buffer.py` + `cli/run.py` (asyncio loop integrating sources / inference / planner / snapshotter / finalize / EventLog / signal handlers / RTSP reconnect / idle HTML regen) | [#12](https://github.com/nicholasaross/StreetTracker/pull/12) |
| 3f | `cli/batch.py` + `enable_snapshotter` / `enable_dashboard` kwargs on `run_session` (file-source variant, no live-only subsystems) | [#13](https://github.com/nicholasaross/StreetTracker/pull/13) |

### Reference points in NanoTracker

The original Python 3.6 implementation that informed the port. Kept
here for cross-checking semantics if the runtime ever drifts:

- `nano_tracker.py:295-314` — `is_ir_frame()` + constants (PR 3c source)
- `nano_tracker.py:537-660` — `ReolinkSnapshotter` (PR 3b source)
- `nano_tracker.py:989-1037` — `start_http_server()` (PR 3d source)
- `nano_tracker.py:1250-1392` — config unpacking + finalize closure (PR 3e source)
- `nano_tracker.py:1394-1578` — `process_frame()` body (PR 3e source)
- `nano_tracker.py:1580-1673` — outer reconnect loop + shutdown `finally:` (PR 3e source)

## Cutover status

Phase 7 cutover happened **2026-05-22**:

| Step | Status |
|---|---|
| Orin: streettracker.service enabled + live | ✅ Active since 2026-05-22T15:42 BST |
| Old Jetson Nano: `nano_tracker.py` SIGTERM'd | ✅ Clean shutdown after 986,619 frames over 27.6h; in-flight session's final outputs flushed |
| Repo archival on GitHub (`VehicleTracker` + `NanoTracker`) | pending — wait ~a week of clean operation, then Settings → Archive on both repos with a "Superseded by StreetTracker as of `<date>`; this repo is read-only" one-liner on each CLAUDE.md |

Two config-mismatch bugs surfaced during the live cutover and were
hardened in [#15](https://github.com/nicholasaross/StreetTracker/pull/15):

1. **Stream-name lookup is now forgiving.** `cli/run.py` falls back to
   matching `stream.quality` when `nano.preferred_stream` doesn't
   match a `stream.name`. Matters for configs migrated from NanoTracker
   that carry descriptive stream names but the legacy `"sub"` /
   `"main"` token in `preferred_stream`.
2. **`engine_path` validated at config load.** Missing engine file
   now surfaces as `ConfigError: $.inference.engine_path: file ... does
   not exist (resolved to <abs>)` at startup instead of a Python
   traceback from inside Ultralytics after ~5s of engine-load attempts.

These mean a fresh deploy where you scp the live `configs/camera.json`
back into place will Just Work without manual edits even if your
engine target or stream names differ from the example.

## ANPR tuning loop

**Active focus.** Post-cutover work to improve 4K plate-capture
coverage. The runtime is live and healthy; we've shipped observability
([PR #16](https://github.com/nicholasaross/StreetTracker/pull/16)) and
the pipelining feature
([PR #18](https://github.com/nicholasaross/StreetTracker/pull/18)) that
together moved ≥3-cap coverage from **6% → 71%** of vehicles. Currently
running a tuning soak (2026-05-23 15:47 BST) with
`pipeline_max_per_track=15` and `snapshot.max_concurrent=3` to address
the two residual constraints the 2026-05-23 validation soak surfaced.

### Assessment baseline (2026-05-22, `session_20260522_154224`, ~1h12m of live traffic)

| Metric | Value |
|---|---|
| Total tracks (cars + persons) | 133 |
| Cars with any 4K cap | 34 / 63 (54%) |
| Cars reaching the configured max of 3 caps | 4 / 63 (6%) |
| L→R (rear plate, median 47 px/s) | 22 / 74 (29%) any cap; **3 reach 3 caps** |
| R→L (front plate, median 69 px/s) | 22 / 59 (37%) any cap; only 1 reaches 3 caps |

Persons mostly get zero captures (60 / 70). Correct — sidewalk
pedestrians are outside the road polygon by design.

Representative tracks (visually inspected via downsampled 4K samples):

| Track | Profile | Verdict |
|---|---|---|
| 153 | L→R Mercedes, 2 caps | **representative slow L→R**: rear plate visible, small but readable |
| 241 | R→L red hatchback, 107 px/s, 1 cap | **representative fast R→L**: car already at left edge of frame, plate region partially out of view |
| 516 | R→L Land Rover, 4 px/s, 131s in view | **not representative**: parked on the curb, being unloaded |

Plate quality when well-framed is excellent (4K JPEG sharp, readable
characters). The original dominant failure mode was therefore
**HTTP-snap latency**, not motion blur: by the time the Reolink JPEG
arrived, fast vehicles had moved past the plate-readable position.

### Step 1 (done): observability — [PR #16](https://github.com/nicholasaross/StreetTracker/pull/16)

Merged 2026-05-22. `SessionMeta.snap_stats` now lands in `*_meta.json`:

* `latency` — count / min / p50 / p90 / p99 / max ms per successful
  fire (covers Reolink HTTP roundtrip + disk write)
* `blur_skipped_frames` — cumulative frames where the planner's
  `min_sharpness` gate suppressed a fire
* `attempts / successes / failures / dropped` — pre-existing
  counters, now persisted in JSON rather than journal-only

Blur skips are also logged in the journal, throttled per-track per 5s.

### Step 2 (done): latency investigation — pivot from trigger-shift to pipelining

After the PR #16 soak (2026-05-23 07:34–09:46 BST, `session_20260523_073430`),
the latency numbers came back much larger than the original tuning math
assumed:

```
attempts 112  successes 112  failures 0  dropped 0  blur_skipped 0
latency   min 522.5  p50 630.5  p90 833.8  p99 1428.4  max 1725.3   ms
```

A 25-call `curl` baseline directly against `cmd=Snap` (runtime stopped)
decomposed the 643 ms total: **TTFB ≈ 370 ms (Reolink rendering the 4K
JPEG) + body ≈ 260 ms (~1.7 MB at ~50 Mbps over a gigabit LAN)**. The
runtime adds no measurable overhead. eno1 link is 1 Gb/s; network is
not the bottleneck. The camera is the floor.

Converting the 630 ms p50 to t' shifts using the polygon's usable-band
axis length of 456.5 sub-stream-px (sub-stream is 896×512, polygon
band span 606.94 sketch-px scaled component-wise):

```
R→L fast (forward)  shift = 0.0953   F1 (t'=0.05) → -0.0453  out of [0,1]
L→R slow (reverse)  shift = 0.0649   R1 (t'=0.95) →  1.0149  out of [0,1]
```

No purely-tuning trigger edit recovers all six triggers — F1 and R1 get
pushed out of the band. That was the trigger that the original
"Step 3 (tune)" recipe was meant to apply; **deferring it** in favour
of pipelining (below) was the right call. The recipe itself is still
valid and is preserved under [Trigger-shift recipe (deferred)](#trigger-shift-recipe-deferred)
in case ANPR's residual gaps after pipelining warrant revisiting it.

An endurance probe (200 calls, 2 concurrent loops sustained 85 s)
closed the safety question for sustained pulls:

```
http_codes  {200: 200}   non-jpeg  0   no truncation
total ms    min 565  p50 834  p90 968  p99 1134  max 1154
TTFB ms     min 297  p50 551  p90 693  p99 896   max 909
2-concurrent overhead vs single-call: 1.30x  (4-concurrent was 2.4x)
no latency drift across five 20 s buckets
```

Throughput jumps from 1.5 fires/s (single) to 2.36 fires/s (2-concurrent).
The win was clear: instead of trying to time-shift triggers to absorb
camera latency, **keep the camera continuously busy** so each vehicle
gets multiple captures and ALPR has more chances per transit.

### Step 3 (done): pipelining — [PR #18](https://github.com/nicholasaross/StreetTracker/pull/18)

Merged 2026-05-23. `SnapPlanner.consider_pipeline(...)` runs after each
frame's normal `consider()` and fires opportunistically every
`pipeline_interval_ms` (default 0 = off, set to 400 on the live deploy)
while a track sits in the usable t-band. Pipeline fires:

* ignore `area_threshold_frac` — a tiny sub-stream bbox still produces
  a full 4K main snap that ANPR can read, even at the polygon's
  distant edge
* ignore `post_fire_cooldown_frames` — the wall-clock interval is the
  throttle
* use a separate counter so the trigger budget is untouched
* share a combined `snap_index = fires_committed + pipeline_fires`
  for filename ordinals so no `_main_N.jpg` collision is possible

Config knobs (under `snapshot.snap_gate`):

| Knob | Default | Live deploy (2026-05-23) | Tuning soak (2026-05-23 15:47) |
|---|---|---|---|
| `pipeline_interval_ms` | `0` (off) | `400` | `400` |
| `pipeline_max_per_track` | `10` | `10` | `15` |
| `snapshot.max_concurrent` | `2` | `2` | `3` |

New `snap_stats` keys (always emitted when `snap_gate` is configured):

* `pipeline_fires` — aggregate successful pipeline submits
* `pipeline_throttled` — frames where the interval blocked a fire
* `pipeline_budget_exhausted` — frames where `pipeline_max_per_track`
  blocked a fire
* `fires_per_track` — `{n, p50, p90, max, mean, zero}` distribution
  computed at session end from per-track state captured before
  `forget()`
* `time_in_band_ms_per_track` — same shape, `*_ms` keys, sets the
  ceiling on possible fires per track

### Step 4 (done): validation soak — 2026-05-23, 3 h 50 min, 289 vehicles

Two sessions:

| Session | Duration | Vehicles | Trigger fires | Pipeline fires | Drops | Latency p50 |
|---|---|---|---|---|---|---|
| `session_20260523_114752` | 1 h 04 m | 69 | 38 | 430 | 14 | 1007.7 ms |
| `session_20260523_125127` | 2 h 46 m | 220 | 113 | 1128 | 146 | 1048.1 ms |

Combined `snap_stats`-style summary:

```
attempts 1563  successes 1563  failures 0  dropped 160 (9.3%)
pipeline_fires 1558 (91% of all fires)
pipeline_throttled 4577   pipeline_budget_exhausted 2937
blur_skipped 1
latency (combined) ~p50 1030 ms   ~p90 1557 ms   max 2375.6 ms
```

Caps-per-vehicle distribution against the baseline:

| Caps | Pipelined (2026-05-23) | Baseline (2026-05-22) |
|---|---|---|
| 0 | 58 / 289 (20%) | 29 / 63 (46%) |
| 1-2 | 27 (9%) | 26 (41%) |
| 3-5 | 62 (21%) | 4 (6%) |
| 6-9 | 97 (34%) | 0 |
| 10+ | 45 (16%) | 0 (was capped at 3) |

**≥3 caps: 71% vs 6% baseline.** Direction balance also closed:
L→R 5.42 caps/track vs R→L 4.86 (was 29% vs 37% asymmetry).

Three readings from the new counters worth remembering for future
diagnostic work:

* `pipeline_budget_exhausted / pipeline_fires ≈ 1.88` — for every
  pipeline fire we allowed, ~1.88 were blocked by the per-track cap.
  The cap of 10 was the dominant rate limiter for slow / long-dwell
  vehicles, not the interval.
* `dropped / attempts = 9.3%` — `max_concurrent=2` was the bottleneck
  during multi-track bursts. Each drop is one specific frame's fire
  that didn't happen; the interval-throttled per-track behaviour means
  the next frame is still eligible, so it doesn't compound.
* `latency p50 1030 ms` vs the endurance probe's 830 ms prediction —
  real traffic bursts have more contention than two-loop synthetic
  pulls. The system is stable at this floor (no drift across
  3 h 50 min).

`fires_per_track.n = 215` (session 1) and `738` (session 2) include
*all* planner-tracked entities — vehicles AND pedestrians that briefly
entered the polygon. Most pedestrians never reach the band, so
`fires_per_track.p50 = 0` in both. **For tuning decisions read the
distribution from `events.jsonl` directly (vehicles only)**, not
`snap_stats.fires_per_track`.

### Step 5 (in progress): tuning soak — `pipeline_max_per_track=15` + `max_concurrent=3`

Live since 2026-05-23 15:47:07 BST (`session_20260523_154707`). The
two changes address the two residual constraints Step 4 identified:

* Raising `pipeline_max_per_track` from 10 → 15 should unblock the ~16%
  of vehicles that hit the cap during Step 4. Expected effect: the
  `10+ caps` bucket re-distributes some weight into a new `13-15`
  range; `pipeline_budget_exhausted` drops proportionally.
* Raising `snapshot.max_concurrent` from 2 → 3 should cut the drop
  rate. The endurance probe only tested 2 vs 4 concurrent (1.30x vs
  2.4x single-call latency); 3 is interpolated to ~1.7x, so we predict
  `latency.p50_ms` climbs from ~1030 ms to ~1100 ms in exchange for
  the drop rate halving.

**Success criteria for the next 1-2 h soak (read against this section's
Step 4 numbers):**

| Metric | Step 4 baseline | Target | Concern threshold |
|---|---|---|---|
| `dropped / attempts` | 9.3% | ≤ 5% | > 8% means `max_concurrent=3` didn't help — back off |
| `fires_per_track` 10+ bucket | 16% | ≤ 8% (vehicles redistribute up) | unchanged means cap wasn't the limiter |
| `pipeline_budget_exhausted / pipeline_fires` | 1.88 | ≤ 1.0 | unchanged means cap is still dominant — try 20 |
| `latency.p50_ms` | ~1030 | ≤ 1150 | > 1300 means 3-concurrent overloaded the Reolink |
| `latency.max_ms` | 2375.6 | ≤ 2600 | > 3000 means tail latency is degrading |
| ≥3 caps coverage | 71% | ≥ 75% | < 71% means a regression — investigate |

Read with the standard recipe:

```bash
ssh streettracker@orin "sudo -n systemctl stop streettracker.service && \
  jq .snap_stats ~/streettracker/output/session_*/session_*_meta.json | tail -1 && \
  sudo -n systemctl start streettracker.service"
```

The session label to compare against in `events.jsonl` distributions
is `session_20260523_125127` for the heaviest sample (220 vehicles).
Use the same shell pipeline:

```bash
jq -r "(.main_snaps | length)" session_*_events.jsonl \
  | sort -n | uniq -c | awk "{print \"  \" \$2 \" caps: \" \$1 \" tracks\"}"
```

### Trigger-shift recipe (deferred)

The Step 2 analysis showed a pure trigger-placement edit can't recover
all six operator-traced triggers from 630 ms of camera latency — F1
and R1 get pushed out of the `[0, 1]` usable band. With pipelining
now producing 6+ caps for 49% of vehicles, the importance of those
specific trigger positions is much reduced. The recipe is preserved
here in case ANPR's residual gaps after the Step 5 tuning soak warrant
revisiting it.

1. Read `snap_stats.latency.p50_ms` from the soaked `_meta.json`.
2. Median traffic speed in sub-stream pixels per ms ≈ 0.05–0.07
   (47–69 px/s observed). Call this `v`. Use 0.069 (fast / R→L) for
   forward triggers, 0.047 (slow / L→R) for reverse.
3. Polygon usable-band axis length in sub-stream pixels is 456.5 for
   the current install (computed from `.claude/triggers_proposal.json`
   `t_min/t_max/t_usable_orig/main_axis_xy` plus the 1200×668→896×512
   scale). Call it `L`.
4. t'-unit shift = `(p50_ms × v) / L`.
5. **Forward triggers** (R→L approach, low t' values): *subtract* the
   shift.
6. **Reverse triggers** (L→R depart, high t' values): *add* the shift.

Apply via the existing recipe in
[Adjusting triggers without re-sketching](#adjusting-triggers-without-re-sketching):
edit `triggers_tprime` → render overlay → regenerate `snap_gate.json`
→ scp to Orin → restart unit.

If `blur_skipped_frames` is ≳ 20% of (`attempts` + `blur_skipped_frames`),
also drop `snapshot.min_sharpness` from 100.0 toward 50 or 30. (Current
data: 1 blur skip in 3 h 50 min of pipelined fires. Non-binding.)

### Open question: trigger direction convention

CLAUDE.md's [snap-gate section](#snap-gate-road-polygon--axis-triggers)
says R→L = front plate, L→R = rear plate. The original assessment's
visual spot-check on track 153 (L→R, rear plate visible) was
consistent, but track 516's plate was hard to call from the
downsampled frame. With pipelining producing 5–7 captures per
direction, this is now easy to verify: pick a recent vehicle from
`events.jsonl`, open its `vehicle_<id>_main_*.jpg` files, and confirm
the plate is on the side the convention predicts. Worth doing once
the Step 5 soak finishes.

## Fresh deployment procedure

Canonical reference for any future redeploy on a wiped Orin (e.g.
moving to a new chassis, swapping SD cards, or upgrading JetPack —
see [JetPack 7 upgrade plan](#jetpack-7-upgrade-plan) below for the
JP7-specific deltas).

### Backups to keep off-device

Three things are not in the repo and not portable from a fresh OS
install. Keep these in your usual backup:

| File | Why |
|---|---|
| `~/streettracker/configs/camera.json` | Reolink credentials + your operator-traced road polygon + per-install snap_gate tuning. Gitignored on purpose. |
| `/etc/sudoers.d/streettracker-svc` | Lets the `streettracker` user (and remote agents acting as them) run `systemctl * streettracker.service` without a password prompt. |
| `~/.ssh/authorized_keys` for the `streettracker` user | The SSH keys you use to administer the box. |

The TRT engine (`yolov8m.engine` or whatever you build) is
deliberately *not* on this list — it's GPU-architecture-bound and
must be rebuilt on the target device.

### One-time host setup

```bash
# As an admin user on the freshly-flashed Orin:
sudo useradd -m -s /bin/bash streettracker
sudo usermod -aG video,sudo,systemd-journal streettracker   # video for nvidia, journal so they can read their own service logs
sudo mkdir -p /home/streettracker/.ssh
sudo cp ~/.ssh/authorized_keys /home/streettracker/.ssh/
sudo chown -R streettracker:streettracker /home/streettracker/.ssh
sudo chmod 700 /home/streettracker/.ssh
sudo chmod 600 /home/streettracker/.ssh/authorized_keys

# Scoped NOPASSWD for service management (lets you / agents enable / restart
# the unit over SSH without an interactive sudo prompt).
echo "streettracker ALL=(ALL) NOPASSWD: /bin/systemctl * streettracker.service" \
  | sudo tee /etc/sudoers.d/streettracker-svc
sudo chmod 440 /etc/sudoers.d/streettracker-svc
```

`systemd-journal` membership is important — without it the
`streettracker` user can't run `journalctl -u streettracker.service`
on their own service (silent permission denial; the output of
`journalctl` looks empty rather than erroring).

**Already-deployed box missing the group?** If `journalctl -u
streettracker.service` as the `streettracker` user returns *"No
journal files were opened due to insufficient permissions"*, run
`sudo usermod -aG systemd-journal streettracker` from an admin
account, then have the streettracker user log out + back in (or
`newgrp systemd-journal`) for new sessions. The scoped sudoers entry
above only covers `systemctl * streettracker.service`, so the
streettracker user can't self-remediate. The live Orin hit this on
2026-05-23 — it was deployed before this recipe existed.

### Repo + venv setup

```bash
ssh streettracker@orin
git clone https://github.com/nicholasaross/StreetTracker.git ~/streettracker
cd ~/streettracker
scripts/setup_orin.sh         # idempotent: apt deps + uv + uv sync + tensorrt symlinks + bashrc LD_LIBRARY_PATH + nvpmodel
```

`setup_orin.sh` handles JetPack 6 (Ubuntu 22.04 / Python 3.10) and
JetPack 7 (Ubuntu 24.04 / Python 3.12) — both via uv's
`.python-version` honouring. See [JetPack 7 upgrade plan](#jetpack-7-upgrade-plan)
for the pyproject + setup-script changes that have to land *before*
the JP7 flash for that path to work.

### Per-deploy files

```bash
# Restore your backed-up config (gitignored).
scp <backup>/camera.json streettracker@orin:~/streettracker/configs/

# Build the TRT engine on THIS device — engines are not portable
# across GPU architectures.
cd ~/streettracker
uv run streettracker export-engine yolov8m.pt
# (~5 min on Orin Nano Super. The output filename must match
# `inference.engine_path` in your camera.json -- the load-time
# validator catches mismatches with a clear JSON-path error.)
```

### Systemd install

```bash
sudo cp scripts/systemd/streettracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo -n systemctl enable --now streettracker.service   # -n works thanks to the sudoers drop-in above
journalctl -u streettracker -f                          # tail; Ctrl-C once you see frame-progress lines
```

Smoke check: `curl http://orin:8080/` returns either the "no sessions
yet" page (~315 bytes) or the latest summary's `index.html` redirect
(~96 bytes).

### Roll back

```bash
sudo -n systemctl disable --now streettracker.service
```

That's it — the unit stops, the SD card is unchanged, and the next
`enable --now` starts a fresh session.

## JetPack 7 upgrade plan

NVIDIA's JetPack 7 release for Orin Nano (Jetson Linux 38.2,
Ubuntu 24.04, Python 3.12 native) is expected in mid-2026. At that
point the device will be wiped and reflashed. Before the flash, the
following repo-level changes have to land or the redeploy will fail
at `uv sync`:

| What | Where | Current (JP6) | JP7 target |
|---|---|---|---|
| Python pin | `.python-version` | `3.10` | `3.12` (matches OS native; uv still honours either, but matching avoids a redundant uv-installed interpreter) |
| Jetson torch wheel index URL | `pyproject.toml` `[[tool.uv.index]] name="jetson-ai-lab-jp6-cu126"` | `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/` | `https://pypi.jetson-ai-lab.io/jp7/cu130/+simple/` (or whatever Anibali's index publishes for JP7 — confirm at the time) |
| cuDSS dep | `pyproject.toml` dependency `nvidia-cudss-cu12` | `cu12` (CUDA 12 series) | `cu13` if JP7 ships CUDA 13.x — verify against the JP7 release notes |
| CUDA lib path | `scripts/setup_orin.sh` `CUDA_LIB=/usr/local/cuda-12.6/lib64` | `cuda-12.6` | Auto-detect via `ls -d /usr/local/cuda-*/lib64 \| sort -V \| tail -1`, OR hardcode `cuda-13.x` after confirming |
| Systemd LD_LIBRARY_PATH | `scripts/systemd/streettracker.service` Environment block | `cuda-12.6/lib64` + `python3.10/site-packages/nvidia/cu12/lib` | Both paths shift; consider auto-substitution at install time or document the manual edit. The `setup_orin.sh` rewriter for `~/.bashrc` could be extended to also emit the unit. |

**Recommended timing.** Don't pre-empt JP7 — NVIDIA hasn't published
exact paths yet. When the JP7 release lands:

1. Spin up a separate branch (`claude/jp7-readiness`).
2. Make the five edits above on the branch.
3. Verify on a *test* Orin (separate from the live one) — flash JP7,
   re-run setup_orin.sh, build an engine, run the service for an hour.
4. Once green, merge to main.
5. *Then* flash JP7 on the live Orin and run the [Fresh deployment
   procedure](#fresh-deployment-procedure) above.

The hardening fixes from [#15](https://github.com/nicholasaross/StreetTracker/pull/15)
mean steps 4-5 are unblocked even if the config you scp back has
slight drift from the example — the runtime will tell you exactly
which field needs attention.

## Snap gate (road polygon + axis triggers)

`device/snap_planner.py` decides when to fire the Reolink 4K snapshot
for each tracked vehicle. The active mode on the live deployment is
**`RoadGate`**: an operator-traced road polygon plus a list of
trigger lines along the polygon's principal axis. Each trigger fires
at most once per track; a vehicle traversing the gate yields up to N
spatially distinct plate captures.

### Artifacts in `.claude/` (per-install, not committed)

These files are the source of truth for the live gate config. They
were generated by an interactive session with the operator and should
be reused / edited rather than regenerated from scratch unless the
camera physically moves.

| File | Purpose |
|---|---|
| `.claude/live_frame.jpg` | Reference 4K snap of the live scene (fetched directly from the Reolink). Source for any new sketching. |
| `.claude/sketch_me.png` | 1200×668 downscale of the live frame, given to the operator to sketch on. |
| `.claude/sketch_me_done.png` | Operator's returned sketch — magenta (`#FF00FF`) outline of the visible road tarmac. |
| `.claude/road_polygon_user.json` | Polygon vertices extracted from the magenta sketch (`{source_size, vertices_frac}`). Fractional coords, resolution-independent. |
| `.claude/triggers_proposal.json` | Full trigger spec: polygon vertices, principal axis, centroid, raw t-range, **`t_usable_orig`** (the band of the polygon to actually use — trims the distant tip and Z2/near zone), **`triggers_tprime`** (list of t' values in `[0, 1]` over the usable band where snaps fire), and optionally **`trigger_directions`** (parallel list of `"forward"` / `"reverse"` / `"both"`; omitted = all `"both"`). |
| `.claude/triggers_proposal.jpg` | Visual overlay of the proposal on the live frame — yellow road outline, dimmed excluded regions, coloured trigger lines + circles. **Reference this image when discussing zone adjustments with the operator.** Re-render via `uv run python .claude/_render_triggers_overlay.py`. |
| `.claude/snap_gate.json` | Subset of `triggers_proposal.json` shipped to `~/NanoTracker/camera_config.json` on the Nano under `snapshot.snap_gate`. Format: `{polygon_frac, trigger_t_prime, t_usable_frac, trigger_directions?}` (the last field is optional and defaults to all `"both"`). |

### Adjusting triggers without re-sketching

Most operator changes ("move T2 slightly toward camera", "drop T1
entirely", "add a 4th trigger between T2 and T3") only need the t'
values to change.

1. Edit `triggers_tprime` (and optionally `t_usable_orig`) in
   `.claude/triggers_proposal.json`. Values are in `[0, 1]` along the
   usable band — t'=0 is the distant edge, t'=1 is the near edge.
2. Re-render `.claude/triggers_proposal.jpg` from the updated JSON so
   the visual stays in sync:
   `uv run python .claude/_render_triggers_overlay.py`.
3. Regenerate `.claude/snap_gate.json` from the JSON
   (`polygon_frac=vertices_frac`, `trigger_t_prime=triggers_tprime`,
   `t_usable_frac=t_usable_orig`, and `trigger_directions` if used).
4. `scp .claude/snap_gate.json claude@nano:~/NanoTracker/`, then merge
   into `camera_config.json` under `snapshot.snap_gate` and restart
   the tracker process. Keep the prior config as a timestamped backup.

### Re-sketching the road (camera moved, new view)

1. Pull a fresh 4K frame: `curl http://.../cmd=Snap...` via the Nano
   and `scp` it back to `.claude/live_frame.jpg`.
2. Downscale to 1200px wide → `.claude/sketch_me.png`.
3. Operator opens it in any image editor, traces the visible road
   tarmac as a closed magenta (`#FF00FF`) outline, saves to
   `.claude/sketch_me_done.png`.
4. Extract polygon: detect magenta pixels, dilate + fill, take the
   largest connected component, walk the boundary radially from the
   centroid, Ramer-Douglas-Peucker simplify (`eps≈6`) →
   `.claude/road_polygon_user.json`.
5. Compute principal axis (PCA on polygon vertices in pixel coords)
   and propose initial triggers + `t_usable` band — render overlay,
   iterate with operator until approved. Then continue from
   "Adjusting triggers" step 3.

### Live planner behaviour

| `SnapPlannerConfig.road_gate` | Mode | Notes |
|---|---|---|
| `None`, `right_half_only=True` | Right-half zone-thirds | Pre-polygon fallback. Not used on the live Nano. |
| `None`, `right_half_only=False` | Legacy peak/decay | Benchmark only. |
| `RoadGateConfig(...)` | Road polygon + axis triggers | **Live deployment mode.** Crossing semantics: each frame compares the bbox-centre's t' with the previous frame's; a not-yet-fired trigger between them fires *if its direction tag matches the motion sign* (`"forward"` = t' increasing = camera-approach side, `"reverse"` = t' decreasing = departure side, `"both"` = default). After a fire, `prev_t_prime` is advanced to the trigger's t' (not to `cur_tp`) so subsequent triggers in the same forward motion remain detectable one-per-frame. Asymmetric triggers let R→L (front plate) and L→R (rear plate) tracks each have their own early-capture trigger without one direction consuming the other's. |

Phase 7 cutover is operationally complete (Orin live since
2026-05-22); the remaining tail is repo archival on GitHub
(VehicleTracker + NanoTracker) after ~a week of clean operation.
Active work is in the [ANPR tuning loop](#anpr-tuning-loop) section
above — currently in Step 5 (post-PR-#18 tuning soak with
`pipeline_max_per_track=15` and `max_concurrent=3`).
