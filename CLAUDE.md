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

- **Python 3.11 today, 3.12 the JP7 target.** `.python-version` is
  pinned to `3.11`, inside `pyproject.toml`'s `requires-python =
  ">=3.10,<3.13"` band. The bump from 3.10 → 3.11 landed on 2026-05-25
  because the `alpr` extra's `onnxruntime-gpu>=1.18` has no Python
  3.10 wheels at recent versions. The eventual move to 3.12 is bundled
  with the JetPack 7 flash — see
  [JetPack 7 upgrade plan](#jetpack-7-upgrade-plan). uv installs
  whichever interpreter `.python-version` names; tests pass on 3.11.
- **Orin redeploys after the 3.10 → 3.11 bump** need a `uv sync` to
  rebuild the venv at 3.11. The systemd unit passes `--no-sync` so
  the running service won't auto-rebuild on `git pull`; do it
  explicitly during the next deploy window. Verify the Jetson torch
  wheel index
  (`https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`) actually has
  3.11 wheels before the first 3.11 deploy, since JP6 ships 3.10
  natively and the index may not stock 3.11.
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

Tests at HEAD: **396 passing on Python 3.11, ruff clean.**

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

**Status as of 2026-05-25: coverage objective met (97 % of cars get
a 4K snap); ALPR read-rate floor measured at 59 % high-confidence
plates per car.** The capture-coverage gap that motivated this loop
was closed by enabling **pipeline mode** on the live `snap_gate`
(see [Pipeline mode](#pipeline-mode-the-dominant-capture-mechanism)
below). The trigger-geometry tuning prescription preserved further
down turns out not to be the binding knob for this deployment —
pipeline-timer fires dominate the trigger-based fires by roughly an
order of magnitude. The remaining gap is plate *readability* not
plate *existence*; next decision is whether to (a) move on to
dataset-level analysis (re-id, recolor, make/model) on the 236
high-confidence reads, or (b) investigate the 30 % of cars whose 4K
snaps exist but didn't yield a read. See
[Step 6 (done): ALPR read-rate measurement](#step-6-done-alpr-read-rate-measurement).

### Soak completion (2026-05-25, `session_20260524_075630`, 27.3h, 880 tracks)

| Metric | Value | Read |
|---|---|---|
| HTTP attempts / successes / failures | 4522 / 4522 / 0 | Reolink endpoint solid. |
| Latency p50 / p90 / p99 / max | 688 / 937 / 1211 / 2423 ms | The "magic number" from Step 3 — real but no longer the binding constraint. |
| `blur_skipped_frames` | **2** | `min_sharpness=100.0` is essentially never tripping; do not lower. |
| `pipeline_throttled` / `pipeline_budget_exhausted` | 14850 / 2484 | Snap budget *is* binding, but coverage is still saturated. |
| Cars L→R any cap | 179 / 182 = **98 %** | Up from baseline's 29 %. |
| Cars L→R 3+ caps | 177 / 182 = **97 %** | Up from baseline's 4 %. |
| Cars R→L any cap | 215 / 218 = **99 %** | Up from baseline's 37 %. |
| Cars R→L 3+ caps | 208 / 218 = **95 %** | Up from baseline's 2 %. |

Visual review of six representative cars (slow/median/fast in each
direction) confirmed:

- **Every observed car has at least one plate-readable 4K snap.** ANPR
  coverage objective is met.
- The *first* snap in each track's sequence is consistently the gold
  capture for both directions. Later snaps either catch a receding /
  distant vehicle (L→R) or an empty frame post-exit / a wrong-angle
  close-up (R→L) — the 688 ms p50 latency landing pipeline fires
  outside the plate-readable window.
- The L→R = rear plate, R→L = front plate convention is visually
  correct (this resolves the original "open question" carried under
  Step 4 below).

Samples used for the inspection are at `.claude/soak_samples/` (52
files, ~65 MB; not committed).

### Pipeline mode (the dominant capture mechanism)

The 2026-05-22 baseline reflects a **trigger-only** snap_gate (6
triggers at fixed t' positions on the polygon principal axis, max ~3
fires per car). At some point on 2026-05-23 (Orin
`configs/camera.json` mtime `21:49 BST`) two extra fields were added
to the deployed `snap_gate` block:

```json
"pipeline_interval_ms": 400,
"pipeline_max_per_track": 15
```

Behaviour (verified in `src/streettracker/device/snap_planner.py` near
the `consider_pipeline` method): while a track is inside the
polygon's `t_usable` band, fire a 4K snap every
`pipeline_interval_ms` of wall-clock time, capped at
`pipeline_max_per_track` per track. Pipeline fires share the
snapshotter's `max_concurrent` HTTP semaphore but have their own
per-track counter (`pipeline_fires`) separate from the trigger pool
(`fires_committed`).

The local artifact at `.claude/snap_gate.json` represents the
*pre-pipeline-mode* config (same polygon + same 6 triggers, but no
`pipeline_*` fields). The two configs match on geometry; they diverge
only on whether pipeline mode is enabled. Don't blindly re-deploy the
local artifact without preserving the pipeline fields, or the gate
will silently fall back to trigger-only and coverage will collapse.

`pipeline_interval_ms: 0` is the default and disables pipeline mode
entirely (legacy trigger-only behaviour).

### Step 6 (done): ALPR read-rate measurement

Ran `streettracker alpr-run --pipeline both` against the soak's 4K
snaps on 2026-05-25. Setup:

1. `.python-version` bumped 3.10 → 3.11 (the `alpr` extra's
   `onnxruntime-gpu>=1.18` has no 3.10 wheels at recent versions).
   See [Compatibility rules](#compatibility-rules).
2. Session pulled to dev box via `tar | scp` (~6.9 GB; `rsync` is not
   on Git Bash for Windows).
3. `uv sync --extra alpr --extra dev`.
4. `uv run streettracker alpr-run output/session_20260524_075630
   --pipeline both` — CPU fallback for ONNX (TensorRT/CUDA libs not
   present locally), but throughput is fine; full run finished in
   under an hour.

**Floor against the 400-car soak population:**

| Read criterion | Count | Rate |
|---|---|---|
| Any OCR text from either pipeline | 267 / 400 | **66.8 %** |
| Preferred-pipeline read ≥ 0.95 conf | 236 / 400 | **59.0 %** |
| Bespoke-pipeline read ≥ 0.50 conf | 65 / 400 | 16.3 % |

Per-image rates (2391 vehicle-prefix 4K snaps processed):

| Pipeline | Any OCR | Conf ≥ 0.5 |
|---|---|---|
| bespoke (custom YOLO + EasyOCR) | 206 / 2391 = 8.6 % | 76 / 2391 = 3.2 % |
| preferred (open-image-models + fast-plate-ocr) | 380 / 2391 = 15.9 % | 380 / 2391 = 15.9 % |

**Pipeline comparison.** When both pipelines produce a read on the
same image (108 cases), they *disagree on the plate string 98 times
out of 108*. Spot-checking the disagreements (track 184: bespoke
`LGSRXH` vs preferred `LG15RXH`; track 294: bespoke `GXIGHXT` vs
preferred `G419HXT`; track 301: bespoke `HBY` vs preferred `LY05WBY`)
shows the preferred pipeline produces clean UK plates at conf ≥0.99,
while bespoke output is typically truncated or garbled at much lower
confidence. The bespoke pipeline as currently configured is not
contributing useful signal — likely retrain or replace as a separate
piece of work; for now treat preferred as the read-rate authority.

**Caveats on the 66.8 % / 59.0 % numbers.**

- 13 of the 400 cars have no vehicle-prefix 4K snaps on disk at all
  — 7 because the [asset_prefix split-flip bug](#known-issue-surfaced-during-the-soak-asset_prefix-split-flip--fixed-in-21)
  put their snaps under `person_<id>_main_*.jpg`, 6 because they had
  zero captures recorded. These are invisible to `alpr-run`'s
  glob-and-prefix discovery. If the 7 orphan-prefix tracks were
  rescued (one-off rename script suggested in PR #21 notes), they'd
  contribute up to 7 more reads, lifting the upper bound by ≤ 2 pp.
- 120 of the 387 cars with vehicle-prefix snaps yielded *zero* reads
  on either pipeline. Worth a follow-up sample: are they
  predominantly fast tracks with motion-blurred plates, oblique
  close-ups past the readable angle, or genuinely no-plate frames
  (delivery vans with rear obstructions, etc.)?

**Interpretation.** 59 % high-confidence read-rate per car is a
respectable baseline floor for a non-ANPR-purpose-built RTSP camera
running on a residential street. The plate-existence problem is
solved (97 % of cars had a 4K snap available); the *plate-readability*
gap is now the bottleneck. Two non-mutually-exclusive paths:

1. **Move on to dataset-level analysis.** ~59 % high-confidence plate
   reads across 400 cars in 27 h is enough data to start meaningful
   per-vehicle aggregation: re-id within session, recolor on the
   readable plates' HQ thumbnails, make/model on the 4K crops.
2. **Investigate the 30 % of cars with snaps but no reads** to see
   if a snap_planner tweak (later trim of `t_usable_frac`, raising
   `min_sharpness`) would convert any of them into readable snaps,
   or whether the gap is intrinsic to camera placement / vehicle
   types.

Outputs landed at:
- `output/session_20260524_075630/session_20260524_075630_alpr.json`
  (per-image detections; ~4782 entries: 2391 images × 2 pipelines)
- `output/session_20260524_075630/session_20260524_075630_alpr_by_track.json`
  (per-track best read across all snaps)
- `output/session_20260524_075630/alpr_crops/{bespoke,preferred}/`
  (cropped plate regions per detection)

### Known issue surfaced during the soak: asset_prefix split-flip — fixed in [#21](https://github.com/nicholasaross/StreetTracker/pull/21)

On at least one track (`track_id=4463` in this session), the dashboard
tile + HQ were written as `vehicle_4463_*` but the 4K main snaps as
`person_4463_main_*.jpg`. The data.json record carries `asset_prefix:
"vehicle"`, so the summary HTML and downstream tooling look for the
wrong prefix and miss the 4K captures entirely.

Root cause: BotSORT reassigns a track's class as evidence accumulates
([track_buffer.py:280](src/streettracker/device/track_buffer.py)), so
`track.class_id` can flip mid-life. The 4K snap path was built from
`track.class_id` at fire time, while the TrackRecord + tile/HQ were
built from it at finalize time. A flip between those moments left the
on-disk filenames disagreeing with `record.asset_prefix`. 26/880
tracks (~3%) in the soak were affected.

Fix: `BufferedTrack` now records the fire-time prefix per snap_index
and a `final_prefix` slot. `finalize_track` locks the final prefix,
sweeps already-saved snaps whose fire prefix differs, and renames
them; the snap `_on_done` callback handles snaps that complete after
finalize. Thumbnails reuse the same `final_prefix` so all on-disk
filenames and the record agree. Live on Orin since 2026-05-25.

Existing on-disk orphans in `session_20260524_075630` aren't
backfilled by this change — a one-off rename script can recover them
per-session if you want the dashboard / `alpr-run` to see them.

### Follow-up: confidence-weighted class voting (2026-05-25, not yet deployed)

Even with PR #21 in place, two tracks in the live
`session_20260525_121236` were flagged by the operator as obviously
misclassified: track 500 was a parked grey Toyota Prius+ labelled
`person`, track 976 was a pedestrian in white walking labelled `car`.
Visual inspection of dashboard tiles + 4K main snaps confirmed both.

Diagnosis: PR #21 made the *file prefix* consistent with the
*finalize-time class*, but the finalize-time class itself came from
"most-recent-detection wins"
([track_buffer.py:290-292](src/streettracker/device/track_buffer.py)).
A single stray YOLO frame at the end of an otherwise correctly
classified track was enough to corrupt the entire record. For tracks
500 / 976 specifically the model was *consistently* wrong (every saved
file is one-class — no flip between fire and finalize), so PR #21's
rename-on-finalize logic had nothing to fix; the underlying class
was just wrong.

Fix landed on a branch (not yet on Orin):

* `BufferedTrack.class_votes: dict[int, float]` — `class_id ->
  sum(detection_score)` over the track's life.
* `TrackBuffer.ingest` accumulates each frame's detection score into
  `class_votes` then sets `class_id = argmax(class_votes)` so
  fire-time and finalize-time reads always see the cumulative
  argmax, not the latest single detection.
* Tie-break is dict-insertion-order (the first class seen wins on
  ties; deterministic in CPython 3.7+).
* Three new tests cover: equal-confidence tie-break (first wins),
  five-frames-of-A-then-one-frame-of-B resists the flip, and
  one-low-conf-A vs four-high-conf-B does flip.

What this *does* fix: any class confusion where YOLO mostly agrees
with itself but emits one or two stray frames of the wrong class.
What this *doesn't* fix: tracks 500 / 976 themselves, since the
model is consistently wrong on those scenes. Those would need
either a heuristic guardrail at finalize (e.g. cross-check
`class_name` against `speed_px_s` + bbox aspect ratio) or a
detector retrain.

Tests: 396 passing on 3.11 (was 394; +3 new vote tests, −1 obsolete
"most-recent wins" assertion).

### Assessment baseline (2026-05-22, `session_20260522_154224`, ~1h12m of live traffic) — historical

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
characters). The dominant failure mode is therefore **HTTP-snap
latency**, not motion blur: by the time the Reolink JPEG arrives, fast
vehicles have moved past the plate-readable position.

### Step 1 (done): observability — [PR #16](https://github.com/nicholasaross/StreetTracker/pull/16)

Merged 2026-05-22. `SessionMeta.snap_stats` now lands in `*_meta.json`:

* `latency` — count / min / p50 / p90 / p99 / max ms per successful
  fire (covers Reolink HTTP roundtrip + disk write)
* `blur_skipped_frames` — cumulative frames where the planner's
  `min_sharpness` gate suppressed a fire
* `attempts / successes / failures / dropped` — pre-existing
  counters, now persisted in JSON rather than journal-only

Blur skips are also logged in the journal, throttled per-track per 5s.

### Step 2 (done): deploy + ~24 h soak

```bash
ssh streettracker@orin
cd ~/streettracker && git pull
sudo -n systemctl restart streettracker.service
journalctl -u streettracker -f      # confirm clean restart; Ctrl-C
```

The session JSON is only written at session shutdown, so to read mid-soak:

```bash
ssh streettracker@orin "sudo -n systemctl stop streettracker.service && \
  jq .snap_stats ~/streettracker/output/session_*/session_*_meta.json | tail -1 && \
  sudo -n systemctl start streettracker.service"
```

(Stopping triggers the runtime's graceful drain → final `_meta.json`
write, then we restart for the next session.)

### Step 3 (parked — wrong knob in pipeline mode): tune

**Parked 2026-05-25 after the soak.** This recipe assumes the dominant
capture mechanism is trigger-based fires at fixed t' positions. In
pipeline mode (see [Pipeline mode](#pipeline-mode-the-dominant-capture-mechanism)
above) trigger fires are a small minority of total fires — perfect t'
placement won't change coverage materially. Recipe is preserved here
for any future trigger-only deployment (`pipeline_interval_ms: 0`),
and also as the reference math if pipeline mode is ever swapped out.

A worked example against the soak's numbers: at `p50_ms = 688`,
median speeds 77 / 87 px/s, principal-axis usable length ≈ 456
sub-stream px, the formula produces shifts of **+0.116 t'** (L→R) and
**−0.132 t'** (R→L). Applied mechanically that would push the
outermost trigger on each side off the usable band, dropping from 6
triggers to 4 — net negative on coverage. So even in a trigger-only
config this loop wants a sanity check after computing the shift.

The actual analogue for pipeline mode is **`t_usable_frac` band
trimming** (shorten the band on the exit edge by the same px-offset
so pipeline fires can't land outside the polygon) and/or raising
`pipeline_interval_ms` to reduce budget pressure. Neither is needed
right now — coverage is saturated.

Read `snap_stats.latency.p50_ms` from the soaked `_meta.json`. Convert
to a t'-unit trigger offset:

1. Median traffic speed in sub-stream pixels per ms ≈ 0.05–0.07 px/ms
   (47–69 px/s observed). Call this `v`.
2. Pixel offset during latency = `p50_ms × v` (typically 15-30 px in
   the 896-wide sub-stream).
3. Polygon principal-axis length in sub-stream pixels. The local
   `.claude/triggers_proposal.json` doesn't ship `axis_endpoints_orig`
   directly; derive it from `t_min`, `t_max`, `main_axis_xy`, and
   `centroid_frac` (scale endpoints from the 1200×668 sketch space
   into the 896×512 sub-stream — separate x and y scale factors
   because the aspect ratios don't match exactly). Call it `L`.
4. t'-unit shift = `(p50_ms × v) / L`. Typically 0.02–0.05.
5. **Forward triggers** (R→L approach, low t' values): *subtract* the
   shift — fire earlier so the snap lands at the readable position.
6. **Reverse triggers** (L→R depart, high t' values): *add* the shift
   — same logic, motion sign is opposite.

If `blur_skipped_frames` is ≳ 20% of (`attempts` + `blur_skipped_frames`),
also drop `snapshot.min_sharpness` from 100.0 toward 50 or 30. A
borderline-blurry cap feeds ALPR better than no cap.

Apply changes via the existing recipe in
[Adjusting triggers without re-sketching](#adjusting-triggers-without-re-sketching):
edit `triggers_tprime` → render overlay → regenerate `snap_gate.json`
→ scp to Orin → restart unit.

### Step 4 (parked): re-soak ~1 h + diff against baseline

Contingent on Step 3 producing a config change worth validating.
Parked alongside Step 3 — see the soak completion table above for
the actual baseline-vs-soak diff.

### Resolved (was open question): trigger direction convention

The convention in the snap-gate section — R→L = front plate, L→R =
rear plate — is **visually confirmed** in the soak's 4K snaps. Median
L→R cars (e.g. track 498) show a readable rear yellow plate near
camera; median R→L cars (e.g. track 294) show the front grille +
plate in mid-frame. The asymmetric trigger placement (3 forward in
the approach half, 3 reverse in the depart half) is therefore aimed
at the correct plates by direction.

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
| Python pin | `.python-version` | `3.11` (bumped from `3.10` on 2026-05-25 to unblock the `alpr` extra) | `3.12` (matches OS native; uv still honours either, but matching avoids a redundant uv-installed interpreter) |
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
| `.claude/snap_gate.json` | Subset of `triggers_proposal.json` shipped to `~/streettracker/configs/camera.json` on the Orin under `snapshot.snap_gate`. Format: `{polygon_frac, trigger_t_prime, t_usable_frac, trigger_directions?, pipeline_interval_ms?, pipeline_max_per_track?}`. `trigger_directions` defaults to all `"both"` if absent. The two `pipeline_*` fields are required to enable [pipeline mode](#pipeline-mode-the-dominant-capture-mechanism) — the live deployment has them set (`400`, `15` respectively) but the local artifact currently doesn't; the artifact predates the pipeline-mode rollout. |

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
4. `scp .claude/snap_gate.json streettracker@orin:/tmp/`, then merge
   into `~/streettracker/configs/camera.json` under `snapshot.snap_gate`
   (preserving the live `pipeline_interval_ms` / `pipeline_max_per_track`
   fields if the local artifact still lacks them — see the [Pipeline
   mode](#pipeline-mode-the-dominant-capture-mechanism) note above) and
   `sudo -n systemctl restart streettracker.service`. Keep the prior
   config as a timestamped backup.

### Re-sketching the road (camera moved, new view)

1. Pull a fresh 4K frame: `curl http://.../cmd=Snap...` via the Orin
   (or directly against the Reolink HTTP endpoint with credentials)
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
| `None`, `right_half_only=True` | Right-half zone-thirds | Pre-polygon fallback. Not used in the live deployment. |
| `None`, `right_half_only=False` | Legacy peak/decay | Benchmark only. |
| `RoadGateConfig(...)` w/ `pipeline_interval_ms = 0` | Road polygon + axis triggers, trigger-only | Crossing semantics: each frame compares the bbox-centre's t' with the previous frame's; a not-yet-fired trigger between them fires *if its direction tag matches the motion sign* (`"forward"` = t' increasing = camera-approach side, `"reverse"` = t' decreasing = departure side, `"both"` = default). After a fire, `prev_t_prime` is advanced to the trigger's t' (not to `cur_tp`) so subsequent triggers in the same forward motion remain detectable one-per-frame. Asymmetric triggers let R→L (front plate) and L→R (rear plate) tracks each have their own early-capture trigger without one direction consuming the other's. This was the 2026-05-22 baseline configuration. |
| `RoadGateConfig(...)` w/ `pipeline_interval_ms > 0` | Road polygon + axis triggers + **pipeline mode** | **Live deployment mode since 2026-05-23.** Trigger crossings still fire as above. *Additionally*, while a track sits inside the `t_usable` band, the planner fires a snap every `pipeline_interval_ms` of wall-clock time, capped at `pipeline_max_per_track`. Pipeline fires use a separate per-track counter (`pipeline_fires`) but share the snapshotter's `max_concurrent` HTTP semaphore with trigger fires. Pipeline fires dominate trigger fires in volume by roughly an order of magnitude at the current settings (`400` ms / `15` cap). Documented in `snap_planner.py` `consider_pipeline`. |

Phase 7 cutover is operationally complete (Orin live since
2026-05-22); the remaining tail is repo archival on GitHub
(VehicleTracker + NanoTracker) after ~a week of clean operation.
The [ANPR tuning loop](#anpr-tuning-loop) section above is at a
natural decision point — Step 6 measured a 59 % high-confidence
read-rate floor; next move is either dataset-level analysis on the
236 readable plates or a follow-up on the 120 cars whose snaps
didn't yield a read. No specific work is committed-to yet.
