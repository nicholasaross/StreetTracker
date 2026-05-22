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
| 7 | cutover: enable systemd on Orin + decommission Nano + archive old repos | **in progress** — operator hand-off, see [Cutover](#cutover) below |

Tests at HEAD: **353 passing, ruff clean.**

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

## Cutover

The repo is code-complete; flipping the live deployment to it is an
operator hand-off. Two boxes are involved: the new Orin Nano 8GB
Super (where StreetTracker runs) and the original Jetson Nano (where
NanoTracker was running).

### Orin install

The systemd unit at
[`scripts/systemd/streettracker.service`](scripts/systemd/streettracker.service)
ships with `ExecStart=uv run --no-sync streettracker run --config
configs/camera.json` -- enabling the unit is the one command that
flips StreetTracker to the live consumer of the Reolink:

```bash
ssh streettracker@orin
cd ~/streettracker
git pull && uv sync
uv run streettracker export-engine yolov8m.pt   # one-time, on this GPU
cp configs/camera.example.json configs/camera.json
$EDITOR configs/camera.json                      # set IP + password
sudo cp scripts/systemd/streettracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now streettracker.service
journalctl -u streettracker -f                   # confirm first frames
```

Smoke check after ~30s: `curl http://orin:8080/` should serve either
the "no sessions yet" page or the first session's summary HTML.

### Old Nano decommission

NanoTracker on the original Jetson Nano never ran as a systemd unit;
stop the running process (likely in `tmux` / `screen`) and optionally
tar up the working output dir before the Orin overwrites the live
RTSP consumer slot.

### Repo archival (phase 7 tail)

Once StreetTracker has run for ~a week with no regressions, archive
the `VehicleTracker` and `NanoTracker` repos on GitHub (Settings →
Archive). Their CLAUDE.md files should get a one-line "Superseded by
StreetTracker as of `<date>`; this repo is read-only" header.

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

Cutover (phase 7) is in progress: once the Orin systemd unit is
enabled and StreetTracker has run cleanly for ~a week,
VehicleTracker + NanoTracker get archived (see
[Cutover](#cutover) above for the operator commands).
