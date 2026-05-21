#!/usr/bin/env bash
# One-shot Jetson Orin Nano setup for StreetTracker.  Idempotent.
#
# Assumes JetPack 6.x (Ubuntu 22.04, Python 3.10) OR JetPack 7.x
# (Ubuntu 24.04, Python 3.12) is already flashed and the network /
# SSH access work.  Run this script once per fresh flash:
#
#   ./scripts/setup_orin.sh
#
# What it does, in order:
#   1. apt update + install system packages (FFmpeg, GStreamer plugins,
#      v4l-utils, build essentials needed by some wheels)
#   2. install uv (the Python package manager StreetTracker uses)
#   3. install Python 3.12 via uv (transparent on JetPack 7.x; brings a
#      separate 3.12 alongside the system 3.10 on JetPack 6.x)
#   4. uv sync at the repo root — installs ultralytics, torch, cv2, etc.
#   5. lock the Orin into max-clock / max-power mode (10W/15W super)
#   6. verify FFmpeg + GStreamer + NVDEC paths
#
# We deliberately do NOT install pycuda or trtexec by hand — Ultralytics'
# built-in TRT path (`YOLO('best.engine')`) handles engine inference,
# and `streettracker export-engine` handles .pt -> .engine conversion.

set -euo pipefail

# ---- locate repo root (script lives in <repo>/scripts/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] repo:  $REPO_ROOT"
echo "[setup] kernel: $(uname -r)"
if [[ -f /etc/nv_tegra_release ]]; then
    echo "[setup] JetPack: $(head -1 /etc/nv_tegra_release)"
else
    echo "[setup] WARNING: /etc/nv_tegra_release not found — is this an Orin?"
fi

echo ""
echo "[setup] (1/6) apt update + install system packages"
sudo apt update
sudo apt install -y \
    build-essential cmake git curl ca-certificates \
    htop nano v4l-utils ffmpeg \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

echo ""
echo "[setup] (2/6) installing uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv into ~/.local/bin; make it visible to this shell.
    export PATH="$HOME/.local/bin:$PATH"
    grep -q '/.local/bin' ~/.bashrc 2>/dev/null || \
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "[setup]   uv already installed: $(uv --version)"
fi

echo ""
echo "[setup] (3/6) installing Python 3.12 via uv"
uv python install 3.12

echo ""
echo "[setup] (4/6) uv sync (ultralytics + torch + opencv)"
# `uv sync` reads pyproject.toml / uv.lock and provisions the venv.
# On Orin we want CUDA-enabled torch; the project's pyproject already
# pins the CUDA wheel index for non-darwin platforms.
uv sync

echo ""
echo "[setup] (5/6) locking Orin to max-clock / max-power"
# nvpmodel -m 0  = "MAXN" / "Super" mode on Orin Nano 8GB Super (40W ceiling)
# jetson_clocks  = lock GPU + CPU + EMC at max sustained frequencies
sudo nvpmodel -m 0 || echo "[setup]   nvpmodel not available (non-Jetson?) — skipping"
sudo jetson_clocks || echo "[setup]   jetson_clocks not available — skipping"

echo ""
echo "[setup] (6/6) verifying media pipeline"
if gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1; then
    echo "[setup]   OK: nvv4l2decoder (NVDEC) present"
else
    echo "[setup]   WARN: nvv4l2decoder missing — file input will fall back to CPU decode" >&2
fi
if uv run python -c "import cv2; assert 'FFMPEG' in cv2.getBuildInformation()" 2>/dev/null; then
    echo "[setup]   OK: OpenCV has FFmpeg backend (required for Reolink RTSP)"
else
    echo "[setup]   FAIL: OpenCV missing FFmpeg backend — RTSP capture will not work" >&2
    exit 1
fi
if uv run python -c "import ultralytics; print('   ultralytics', ultralytics.__version__)" 2>&1 | grep -q ultralytics; then
    echo "[setup]   OK: ultralytics importable inside uv venv"
else
    echo "[setup]   FAIL: ultralytics not importable" >&2
    exit 1
fi

echo ""
echo "[setup] Done."
echo "        Next:"
echo "          1. Copy a YOLO checkpoint to the Orin (e.g. yolov8m.pt)"
echo "          2. Build a TRT engine ON THIS DEVICE:"
echo "               uv run streettracker export-engine yolov8m.pt"
echo "          3. Drop your camera config under configs/ and start the runtime:"
echo "               uv run streettracker run --config configs/camera.json"
echo "        Optional: install the systemd unit so the tracker starts on boot:"
echo "          sudo cp scripts/systemd/streettracker.service /etc/systemd/system/"
echo "          sudo systemctl enable --now streettracker.service"
