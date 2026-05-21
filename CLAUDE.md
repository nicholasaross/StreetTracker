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
| 4b | `analysis/alpr/` wholesale port | pending |
| 5 | CLI: `pull`, `export-engine`, `setup_orin.sh`, systemd | pending |
| 3 | `device/`: live runtime, snapshotter, dashboard, IR | in progress (`device/snap_planner.py` landed with road-polygon + axis-trigger gate, mirroring NanoTracker's deployed copy) |
| 6 | (opt) original Nano archive role | not started |
| 7 | cutover: archive both old repos | not started |

Tests at HEAD: **118 passing, ruff clean.**

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

`run` / `batch` / `pull` / `export-engine` print
"not yet implemented" until their phases land.

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
