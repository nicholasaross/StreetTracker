# StreetTracker

Detect, track, and summarize moving vehicles and persons from video files or
a live RTSP stream. Emits a sortable HTML dashboard plus JSON / JSONL for
downstream analysis.

Successor to `VehicleTracker` (file-input only, dev-box-only) and
`NanoTracker` (live RTSP on Jetson Nano original / JetPack 4.6.1 / Python
3.6). Unifies both onto a single modern stack: Python 3.12, Ultralytics +
BotSORT, TensorRT, uv-managed.

## Hardware targets

| Role | Hardware | Status |
|---|---|---|
| Primary device | Jetson Orin Nano 8GB Super (67 TOPS, JetPack 6.x / 7.x) | active |
| Dev box | Linux / macOS / Windows (CUDA optional) | active |
| Original Jetson Nano | JetPack 4.6.1 | optional archive/web-host role only |

## Quick start (dev box)

```bash
uv sync
cp configs/camera.example.json configs/camera.json   # tune as needed
uv run streettracker batch path/to/video.mp4 --config configs/camera.json
```

## Quick start (Orin Nano)

```bash
scripts/setup_orin.sh        # idempotent JetPack + uv + Python 3.12 install
uv sync
uv run streettracker export-engine yolov8m.pt        # build TRT engine on this GPU
cp configs/camera.example.json configs/camera.json   # fill in IP + password
uv run streettracker run --config configs/camera.json
```

To run as a systemd service on the Orin, see the
[Cutover section in `CLAUDE.md`](CLAUDE.md#cutover).

## CLI

- `streettracker run` — live RTSP capture (Orin-only)
- `streettracker batch <video>` — file input
- `streettracker pull` — scp session from device
- `streettracker recolor <session>` — rerun color heuristic on a closed session
- `streettracker export-engine` — `.pt` → `.engine` via Ultralytics
- `streettracker alpr-run <session>` — run ALPR pipelines on session snaps
- `streettracker alpr-score <session>` — score ALPR pipelines against labels
- `streettracker alpr-label <session>` — interactive plate labeling
- `streettracker alpr-report <session>` — render the comparison HTML

See `CLAUDE.md` for architecture and operational notes.
