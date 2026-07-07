# StreetTracker <img src="docs/assets/st26trk.svg" alt="ST26 TRK" height="40" align="right"/>

Detect, track, and characterise vehicles and people from video files or a live
RTSP camera — then enrich the results with number-plate reads, make/model/year,
and a browsable web UI.

Successor to `VehicleTracker` (file-input, dev-box only) and `NanoTracker` (live
RTSP on the original Jetson Nano / JetPack 4.6.1 / Python 3.6), unified onto one
modern stack: **Python 3.10 · Ultralytics + BotSORT · TensorRT · uv-managed**.

**Status:** production. Running live on a Jetson Orin Nano 8GB Super watching a
street through a Reolink camera since 2026-05-22. `ruff` clean, **870** unit
tests green on the dev box. (Migration from the two legacy repos is
code-complete; the only open tail is the JetPack 7.2 device upgrade — see
[Roadmap](#roadmap).)

```
RTSP (live, Orin)  ─┐
                    ├─▶ FrameSource ─▶ YOLO + BotSORT (TensorRT) ─▶ per-track attributes
MP4 (batch, dev) ───┘                                               (direction · speed · colour)
                                                                          │
                       4K snapshots (operator-traced snap gate) ◀────────┤
                                   │                                      ▼
                                   ▼                            EventLog (JSONL, fsync)
                          ANPR ─▶ DVSA make/model/year                    │
                                   │                                      ▼
                                   └────▶ per-vehicle aggregation ─▶ summary HTML · showcase · stats
```

## Hardware targets

| Role | Hardware | Status |
|---|---|---|
| Primary device | Jetson Orin Nano 8GB Super (67 TOPS, JetPack 6.x) | **live** — production since 2026-05-22 |
| Dev box | Windows / Linux / macOS (CUDA optional; RTX 3080 here) | active — batch, ANPR, training |
| Original Jetson Nano | JetPack 4.6.1 | decommissioned 2026-05-22 (superseded) |

> JetPack 7.2 (CUDA 13.2 / Python 3.12) support for the Orin Nano has shipped;
> the device upgrade is staged but gated on torch-wheel availability — see
> [Roadmap](#roadmap).

## Features

- **Detection + tracking** — Ultralytics YOLO with integrated BotSORT, running the
  built-in TensorRT path (`YOLO('best.engine')`) on the Orin. Tracks vehicles
  *and* people; confidence-weighted class voting defends against single-frame
  label flips.
- **Per-track attributes** — direction, speed, colour (vote over the track),
  lane, first/last-seen timestamps.
- **4K snapshot capture** — an operator-traced road polygon + axis triggers +
  continuous "pipeline" fires drive on-demand 4512×2512 Reolink HTTP snapshots,
  tuned for plate-readability (≈98 % of cars get ≥1 snap).
- **ANPR** — offline plate reading over a session's snaps, with a parked-car
  "ghost mask", per-track bbox hints, and fuzzy clustering of OCR variants.
- **Make / model / year** — two prongs: DVSA MOT lookup from the plate
  (ground-truth, for the readable subset) and a UK-native EfficientNet make
  classifier for the unreadable majority.
- **Per-vehicle aggregation** — folds tracks + plate reads into one record per
  physical car (`n_visits`, gap times, direction/colour histograms), including
  repeat vehicles pooled across a cohort of sessions.
- **Web** — a local **showcase** site of the identified/recurring cars (with a
  per-car "this is my car" metadata editor) and a **traffic-statistics** page
  (journeys by day/hour, day-of-week + weekday×hour heatmaps, speed distribution,
  make/colour mix) — all dependency-free SVG/CSS.
- **Dashboard** — auto-regenerated HTML summary per session + hourly rollup,
  served live on the device at `http://<orin>:8080/`.
- **Operator control panel** — a token-free web cockpit (`streettracker
  control`) to drive the pull → enrich → train → promote loop with live
  progress, ETAs, a live training curve, and a copy-paste help prompt when a
  step fails. See [Web interfaces](#web-interfaces).

## Quick start — dev box

```bash
uv sync                                                  # Python 3.10 venv
cp configs/camera.example.json configs/camera.json       # tune as needed
uv run streettracker batch path/to/video.mp4 --config configs/camera.json
```

## Quick start — Orin Nano

```bash
scripts/setup_orin.sh                                    # idempotent: JetPack deps + uv + Python 3.10 + TensorRT symlinks
uv run streettracker export-engine yolov8m.pt            # build the TRT engine ON this GPU (not portable)
cp configs/camera.example.json configs/camera.json       # fill in Reolink IP + password + traced road polygon
uv run streettracker run --config configs/camera.json
```

To run as a managed service, install the unit at
`scripts/systemd/streettracker.service` — see the
[Fresh deployment procedure](CLAUDE.md#fresh-deployment-procedure) in `CLAUDE.md`.

## CLI

Runtime + device:

| Command | Purpose |
|---|---|
| `streettracker run --config <cfg>` | live RTSP capture (Orin) |
| `streettracker batch <video>` | file input (dev box) |
| `streettracker export-engine <model.pt>` | `.pt` → `.engine` via Ultralytics (run on the target GPU) |
| `streettracker pull --session <S>` | scp a session off the device |
| `streettracker recolor <session>` / `debug-color <session>` | rerun / inspect the colour heuristic |

ANPR (needs `uv sync --extra alpr`):

| Command | Purpose |
|---|---|
| `streettracker alpr-run <session>` | run the plate-reading pipeline over a session's snaps |
| `streettracker alpr-score` / `alpr-label` / `alpr-report <session>` | score vs labels · label interactively · render comparison HTML |

Enrichment + web:

| Command | Purpose |
|---|---|
| `streettracker dvsa-label <session>` | harvest DVSA MOT make/model/year per plate (needs `configs/dvsa.json`) |
| `streettracker dvsa-apply <session>` | fold DVSA fields onto each car's records |
| `streettracker vehicles <session> [--across <dirs>]` | per-vehicle aggregation (+ cross-session repeats) |
| `streettracker makemodel <session>` | CNN make/model inference → `_makemodel.json` |
| `streettracker makemodel-build-uk` / `makemodel-train-uk` | build a DVSA-labelled UK crop corpus · train the make classifier |
| `streettracker showcase --output-root output` | local website of enriched + recurring cars + stats (http://127.0.0.1:8090/) |
| `streettracker control --output-root output` | operator control panel: live radiator + one-click procedures (http://localhost:8095/) |

## Web interfaces

Three browsable UIs, all dependency-free SVG/CSS:

| UI | Host | Command | URL |
|---|---|---|---|
| **Live dashboard** | Orin (device) | served by `streettracker run` / the systemd service — no extra command | `http://<orin>:8080/` |
| **Showcase** | dev box | `uv run streettracker showcase --output-root output` | `http://127.0.0.1:8090/` |
| **Control panel** | dev box | `uv run streettracker control --output-root output` | `http://localhost:8095/` |

- **Live dashboard** — the running capture service auto-regenerates an HTML
  summary of the current session (track tiles + hourly rollup) and serves it on
  the device at port 8080. It comes up with `streettracker run`; nothing else to
  start.
- **Showcase** — a cross-session gallery of the identified + recurring cars
  (ANPR + DVSA make/model/year/colour) with a per-car "this is my car" metadata
  editor and a `/stats` traffic-analytics page. Local-only by default; add
  `--host 0.0.0.0` for LAN. (Speed→mph calibration via `configs/showcase.json`.)
- **Control panel** — an operator cockpit for running the day-to-day procedures
  (pull → ALPR → DVSA → vehicles → make/model, corpus rebuild, training, model
  promotion) **without an assistant attached, consuming zero tokens once
  running**: a live radiator (Orin status, session inventory, corpus/model
  retrain & promote recommendations) plus one-click playbooks with progress
  bars, ETAs, a live training loss curve (`/training`), and a copy-paste Claude
  prompt when a step fails. Dashboards bind `0.0.0.0` (LAN-viewable); every
  control action is localhost-only. Detached launcher (opens a browser):

  ```powershell
  pwsh -NoProfile -File scripts/control_panel.ps1
  ```

## Output

Per finalised track: a dashboard tile (`{prefix}_{id}.jpg`), an HQ crop
(`_hq.jpg`), and up to N 4K snaps (`_main_{N}.jpg`); `{prefix}` is `vehicle` or
`person`. Per session: `*_events.jsonl` (crash-safe, line-per-track),
`*_data.json`, `*_meta.json`, `*_hourly.json`, `*_summary.html`, plus the
analysis artifacts (`*_alpr*.json`, `*_dvsa_labels.json`, `*_vehicles.json`,
`*_makemodel*.json`). Record fields are defined in `common/schema.py`.

## Project layout

```
src/streettracker/
├── common/      # schema, colour, config (strict JSON), summary, hourly, output — shared
├── inference/   # YOLO + BotSORT via Ultralytics / TensorRT
├── sources/     # RTSP (FFmpeg, TCP) + file (NVDEC on Orin)
├── device/      # Orin-only: live runtime, snapshotter, snap planner, dashboard, IR
├── analysis/    # off-device: ALPR, recolour, vehicles, make/model
├── web/         # showcase site + stats (aggregate · metadata · stats · server)
└── cli/         # `streettracker` entry + subcommands
```

## Development

```bash
uv run pytest                        # tests (device-marked tests need a Jetson)
uv run ruff check src/ tests/        # lint
uv run ruff format src/ tests/       # format
uv run mypy src/                     # type-check
```

The runtime is asyncio throughout; config is loaded by a strict frozen-dataclass
JSON loader that rejects unknown keys. See **`CLAUDE.md`** for architecture
decisions, the ANPR/make-model tuning history, deployment + recovery procedures,
and the snap-gate geometry.

## Performance on the deployment scene

Scene-specific (oblique residential-street view), measured over multi-hour soaks:

- **4K snap coverage:** ≈98 % of cars get ≥1 snapshot; ≈67 % of person tracks.
- **Plate reads:** a clean canonical UK plate for ~58 % of cars net —
  rear plates (L→R here) read ~92 % per car; front plates are
  camera-geometry-capped ~23 % at every position, so the remaining lever
  is a second approach camera, not tuning.
- **DVSA make/model/year:** the readable, ≥3-year-old subset (~25–30 % of cars).
- **CNN make classifier:** EfficientNet-B5 @456 px, **44 % make@1** over 42
  makes for the unreadable majority — the corpus (43.8k crops) grows as a
  by-product of operation, and data remains the lever.

## Roadmap

The one open item is the **JetPack 7.2 device upgrade** (Orin Nano now supported;
brings CUDA 13.2 / TensorRT 10.16 / Python 3.12 / Ubuntu 24.04). It is **gated on
torch-wheel availability**: the jetson-ai-lab index has no `jp7` shelf yet, so a
flash today would leave the Orin with no `sm_87`-compatible PyTorch. Re-check the
gate from the dev box at any time:

```powershell
pwsh -ExecutionPolicy Bypass -File scripts/check_jp7_wheel.ps1   # exit 0 = ready, 1 = not yet
```

When it turns green, the upgrade follows the five repo edits + single-Orin flash
plan in [`CLAUDE.md` → JetPack 7 upgrade plan](CLAUDE.md#jetpack-7-upgrade-plan).

## License

MIT.
