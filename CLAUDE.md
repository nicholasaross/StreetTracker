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
Source-of-truth scaffolding is currently developed in NanoTracker's
`claude/nano-orin-setup-plan-CCWUD` branch under `streettracker/` and
mirrored here phase by phase via `cp -a`. See NanoTracker's `CLAUDE.md`
"Active migration: StreetTracker" section for the recipe.

| Phase | Scope | Status |
|---|---|---|
| 0 | repo init + pyproject + CI + configs | **done** |
| 1 | `common/`: schema, color, output, hourly, summary | **done** |
| 2 | `inference/` (Ultralytics runner) + `sources/` (RTSP, file) | **done** |
| 4a | `analysis/`: recolor + debug-color | **done** |
| 5 | CLI: `pull`, `export-engine`, `setup_orin.sh`, systemd | **done** |
| 4b | `analysis/alpr/` wholesale port | **done** |
| 3 | `device/`: live runtime, snapshotter, dashboard, IR | **in progress** — see [Phase 3 progress](#phase-3-progress) below |
| 6 | (opt) original Nano archive role | not started |
| 7 | cutover: archive both old repos | not started |

Tests at HEAD: **248 passing, ruff clean.**

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

`run` / `batch` still print "not yet implemented" pending phase 3 / 5b.
`pull` and `export-engine` are wired; `scripts/setup_orin.sh` and
`scripts/systemd/streettracker.service` are in place for device install.
`alpr-run` / `alpr-score` / `alpr-label` / `alpr-report` are available
under the `alpr` extra (`uv sync --extra alpr`); place the bespoke
detector at `src/streettracker/analysis/alpr/models/license_plate_detector.pt`
(gitignored).

**Windows + Git Bash gotcha for `streettracker pull`**: MSYS rewrites
POSIX-looking arguments (`/home/...`) into Windows paths before Python
sees them, mangling `--remote-parent` and `--key`. Either run from
PowerShell / cmd, or prefix the invocation with `MSYS_NO_PATHCONV=1`
and pass `--key` as a Windows-style path.

## Phase 3 progress

Phase 3 (`device/`: live runtime, snapshotter, dashboard, IR) is
broken into 6 PRs plus a cutover step. The plan was scoped after the
Orin deploy hardening landed (see `git log` for the scoping
conversation in PR #6's predecessor session). Architectural decisions
are locked in — re-deciding them mid-stream causes churn:

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
  happens by ssh'ing into the Orin against the live Reolink after
  PR 3e merges.

### PR breakdown + status

| PR | What | Status |
|---|---|---|
| 3a | `common/config.py` (frozen dataclasses, strict JSON loader) + aiohttp dep + tests against `configs/camera.example.json` | **done** ([#7](https://github.com/nicholasaross/StreetTracker/pull/7)) |
| 3b | `device/snapshotter.py` (aiohttp Reolink client, semaphore, retries, keep_days cleanup task) | **done** ([#8](https://github.com/nicholasaross/StreetTracker/pull/8)) |
| 3c | `device/ir_detector.py` (sync frame analysis — R/G/B channel-diff + hysteresis → emits `IRPeriod`) | **next** |
| 3d | `device/dashboard.py` (aiohttp.web static server, lifecycle helpers) | pending |
| 3e | `device/runtime.py` + `cli/run.py` (asyncio loop integrating sources / inference / planner / snapshotter / finalize / EventLog / signal handlers / RTSP reconnect / idle HTML regen) | pending |
| 3f | `cli/batch.py` (file-source variant of runtime, no snapshotter/dashboard) | pending |
| — | Cutover: CLAUDE.md migration-table update + README + enable systemd on Orin + manual cutover from Nano | pending |

### Resuming PR 3c (IR detector)

Branch off main: `git checkout -b claude/phase-3c-ir-detector`.
NanoTracker's IR detection is the source-of-truth port reference:

- `nano_tracker.py:295-314` — `is_ir_frame()` plus the constants
  (`_IR_CHANNEL_DIFF_THR=8`, `_IR_SAMPLE_STRIDE=16`,
  `_IR_HYSTERESIS_FRAMES=30`). The check is sub-millisecond at 1080p
  via stride-sampling: max |R-G| and max |G-B| across the strided
  pixels; if both are below the threshold the frame is monochrome.
- `nano_tracker.py:1419-1444` — the hysteresis state machine.
  Maintain a rolling list of the last N readings. Flip to IR mode
  when **all** N say IR and we're not in IR yet; flip back when
  **none** say IR and we are. On both transitions emit (or close)
  an `IRPeriod` record. On entering IR, flush active tracks so the
  day/night boundary is a clean cut.

`common/schema.IRPeriod` already exists (start/end ISO strings +
`duration_s`). The detector should return `IRPeriod` instances as
side outputs of `update(frame, wall_time)` rather than mutate
shared state — the runtime loop (PR 3e) owns the period list.

API sketch:

```python
class IRDetector:
    def __init__(self, *,
                 channel_diff_threshold: int = 8,
                 sample_stride: int = 16,
                 hysteresis_frames: int = 30): ...

    @property
    def in_ir_mode(self) -> bool: ...

    def update(self, frame: np.ndarray, wall_time: float
              ) -> IRPeriod | None:
        """Returns a closed IRPeriod on day-resume, None otherwise."""
```

Tests: synthetic frames (uniform grayscale = IR; saturated colour =
day), hysteresis flapping protection (29 IR + 1 day stays in day
mode), period emission on day→IR→day, fresh detector starts in day
mode.

### Resuming PRs 3d–3f

When 3c is in, the next branch order is:
- `claude/phase-3d-dashboard` — independent of 3c, can be parallel
- `claude/phase-3e-runtime` — integrates 3a + 3b + 3c + 3d (bottleneck)
- `claude/phase-3f-batch` — derivative of 3e, file source

### Reference points in NanoTracker

The full module map is in PR #6's scoping conversation; the most
useful line ranges to keep open while porting 3e:

- `nano_tracker.py:537-660` — `ReolinkSnapshotter` (already ported in
  3b; cross-check semantics)
- `nano_tracker.py:989-1037` — `start_http_server()` (PR 3d source)
- `nano_tracker.py:1250-1392` — config unpacking, mutable state,
  `finalize()` nested closure (PR 3e)
- `nano_tracker.py:1394-1578` — `process_frame()` body, the actual
  per-frame work (PR 3e)
- `nano_tracker.py:1580-1673` — outer reconnect loop + shutdown
  `finally:` (PR 3e — adapt for asyncio)

### Hard rule once 3e lands

The Orin systemd unit's `ExecStart` is already
`uv run --no-sync streettracker run --config configs/camera.json` —
flipping the migration to live is one command:
`sudo systemctl enable --now streettracker.service`. Do NOT enable
the unit before 3e merges; the stub `streettracker run` exits with
status 2, and `Restart=on-failure` will flap it. The cutover step
explicitly belongs after 3e.

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

Until cutover (phase 7), VehicleTracker + NanoTracker remain
authoritative for their current targets.
