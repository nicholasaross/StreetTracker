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
#   3. uv sync at the repo root — installs ultralytics, torch (jetson-ai-lab
#      cp310 wheel on aarch64 via pyproject's platform-aware sources),
#      cuDSS, onnx/onnxruntime, opencv, etc.  Reads .python-version (3.10
#      for the Orin path).
#   4. symlink system-installed TensorRT into the venv (JetPack ships
#      tensorrt at /usr/lib/python3.10/dist-packages; the venv can't
#      see it without a bridge).  Required for `streettracker export-engine`.
#   5. append LD_LIBRARY_PATH + OPENCV_FFMPEG_CAPTURE_OPTIONS exports to
#      ~/.bashrc so interactive shells and `uv run` find system cuDNN +
#      cuBLAS plus venv-bundled cuDSS.
#   6. lock the Orin into max-clock / max-power mode (15W / MAXN).
#   7. verify the full media + ML pipeline.
#
# We deliberately do NOT install pycuda or trtexec by hand — Ultralytics'
# built-in TRT path (`YOLO('best.engine')`) handles engine inference,
# and `streettracker export-engine` handles .pt -> .engine conversion
# via the system tensorrt symlinked in step 4.

set -euo pipefail

# ---- locate repo root (script lives in <repo>/scripts/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- Orin-only paths used by step 4 (tensorrt) and the verification
# step.  Both auto-detect; the script still runs (with warnings) on a
# non-Jetson aarch64 box, which is useful for dry-runs over SSH.
PY_VERSION="$(cat .python-version 2>/dev/null || echo 3.10)"
SYS_TRT_DIR="/usr/lib/python${PY_VERSION}/dist-packages"
CUDA_LIB="/usr/local/cuda-12.6/lib64"
SYS_LIB="/usr/lib/aarch64-linux-gnu"
VENV_NVIDIA_LIB="$REPO_ROOT/.venv/lib/python${PY_VERSION}/site-packages/nvidia/cu12/lib"

echo "[setup] repo:    $REPO_ROOT"
echo "[setup] python:  ${PY_VERSION} (from .python-version)"
echo "[setup] kernel:  $(uname -r)"
if [[ -f /etc/nv_tegra_release ]]; then
    echo "[setup] JetPack: $(head -1 /etc/nv_tegra_release)"
else
    echo "[setup] WARNING: /etc/nv_tegra_release not found — is this an Orin?"
fi

echo ""
echo "[setup] (1/7) apt update + install system packages"
sudo apt update
sudo apt install -y \
    build-essential cmake git curl ca-certificates \
    htop nano v4l-utils ffmpeg \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

echo ""
echo "[setup] (2/7) installing uv"
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
echo "[setup] (3/7) uv sync (ultralytics + torch + opencv + onnx stack)"
# `uv sync` reads pyproject.toml / uv.lock and provisions the venv.
# pyproject pins jetson-ai-lab-jp6-cu126 for aarch64 Linux so torch
# resolves to the cp310 sm_87 build instead of PyPI's sm_80/sm_90 wheel.
# `--extra dev` is intentionally omitted; install it on the dev box, not
# the Orin runtime.
uv sync

echo ""
echo "[setup] (4/7) symlinking system TensorRT into the venv"
# JetPack ships tensorrt as a system Python package at
# /usr/lib/python${PY_VERSION}/dist-packages/.  The uv-managed venv uses
# uv's own Python so it doesn't pick that up by default.  Symlinks are
# the simplest bridge — no `include-system-site-packages` pollution,
# only the four tensorrt namespaces are exposed.
SITE="$REPO_ROOT/.venv/lib/python${PY_VERSION}/site-packages"
if [[ -d "$SYS_TRT_DIR/tensorrt" && -d "$SITE" ]]; then
    # Glob the dist-info dirs since the version number is in their name.
    for pkg in tensorrt tensorrt_dispatch tensorrt_lean; do
        if [[ -e "$SYS_TRT_DIR/$pkg" ]]; then
            ln -sf "$SYS_TRT_DIR/$pkg" "$SITE/$pkg"
        fi
        for dist in "$SYS_TRT_DIR/${pkg}-"*.dist-info; do
            [[ -e "$dist" ]] && ln -sf "$dist" "$SITE/$(basename "$dist")"
        done
    done
    echo "[setup]   OK: tensorrt symlinked from $SYS_TRT_DIR -> $SITE"
else
    echo "[setup]   WARN: $SYS_TRT_DIR/tensorrt not found — engine builds will fail" >&2
fi

echo ""
echo "[setup] (5/7) appending Jetson library paths to ~/.bashrc"
# Jetson torch wheel needs system cuDNN + cuBLAS to be discoverable;
# venv-bundled libcudss must also be on the loader path.  Set these in
# .bashrc so interactive `uv run` works; the systemd unit duplicates
# them in its own Environment block (different process inheritance).
if ! grep -q 'STREETTRACKER LIBS' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF

# --- STREETTRACKER LIBS ---
# Required for the community Jetson torch wheel to find system cuDNN +
# cuBLAS plus the venv-bundled libcudss.  See pyproject.toml's
# tool.uv.sources comment for the wheel-architecture context.
export PATH="\$HOME/.local/bin:\$PATH"
export LD_LIBRARY_PATH="${CUDA_LIB}:${SYS_LIB}:${VENV_NVIDIA_LIB}"
# Reolink RTSP needs FFmpeg backend in TCP mode (cv2's GStreamer backend stalls).
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
EOF
    echo "[setup]   OK: ~/.bashrc updated"
else
    echo "[setup]   OK: ~/.bashrc already has STREETTRACKER LIBS section"
fi

echo ""
echo "[setup] (6/7) locking Orin to max-clock / max-power"
# nvpmodel -m 0  = MAXN / 15W on Orin Nano 8GB Super (sustained mode).
# `-m 2` would select MAXN_SUPER (40W) on firmware that exposes it.
# jetson_clocks  = lock GPU + CPU + EMC at max sustained frequencies.
sudo nvpmodel -m 0 || echo "[setup]   nvpmodel not available (non-Jetson?) — skipping"
sudo jetson_clocks || echo "[setup]   jetson_clocks not available — skipping"

echo ""
echo "[setup] (7/7) verifying media + ML pipeline"
# Use the LD_LIBRARY_PATH we just added so the Python checks below see
# the same environment a future `uv run` will get.
export LD_LIBRARY_PATH="${CUDA_LIB}:${SYS_LIB}:${VENV_NVIDIA_LIB}:${LD_LIBRARY_PATH:-}"

if gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1; then
    echo "[setup]   OK: nvv4l2decoder (NVDEC) present"
else
    echo "[setup]   WARN: nvv4l2decoder missing — file input will fall back to CPU decode" >&2
fi

if uv run --no-sync python - <<'PY' >/dev/null 2>&1
import cv2
assert "FFMPEG" in cv2.getBuildInformation()
PY
then
    echo "[setup]   OK: OpenCV has FFmpeg backend (required for Reolink RTSP)"
else
    echo "[setup]   FAIL: OpenCV missing FFmpeg backend — RTSP capture will not work" >&2
    exit 1
fi

if uv run --no-sync python - <<'PY' >/dev/null 2>&1
import ultralytics
import torch
assert torch.cuda.is_available()
# Force one real CUDA op so libcudss / cuBLAS / cuDNN linkage is exercised.
x = torch.randn(64, 64, device="cuda")
_ = (x @ x.T).sum().item()
PY
then
    echo "[setup]   OK: torch.cuda usable (matmul exercised) + ultralytics importable"
else
    echo "[setup]   FAIL: torch.cuda exercise failed — check LD_LIBRARY_PATH and the cuDSS / cuDNN install" >&2
    exit 1
fi

if uv run --no-sync python -c "import tensorrt as trt; print('   tensorrt', trt.__version__)" 2>&1 | grep -q tensorrt; then
    echo "[setup]   OK: tensorrt importable (engine builds will work)"
else
    echo "[setup]   FAIL: tensorrt not importable — symlinks in step 4 may have failed" >&2
    exit 1
fi

echo ""
echo "[setup] Done."
echo "        Next:"
echo "          1. Drop your camera config under configs/ (gitignored)."
echo "          2. Build a TRT engine ON THIS DEVICE:"
echo "               uv run --no-sync streettracker export-engine yolov8m.pt"
echo "          3. Install the systemd unit (kept disabled until phase 3 lands):"
echo "               sudo cp scripts/systemd/streettracker.service /etc/systemd/system/"
echo "               sudo systemctl daemon-reload"
echo "        When \`streettracker run\` is implemented:"
echo "          sudo systemctl enable --now streettracker.service"
