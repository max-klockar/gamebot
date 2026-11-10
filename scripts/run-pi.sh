#!/usr/bin/env bash
# Run gamebot on the Pi with root for WS2812 DMA, keeping the HDMI display.
set -euo pipefail
cd "$(dirname "$0")/.."
export DISPLAY="${DISPLAY:-:0}"
# Allow root to draw on the logged-in user's X/Wayland display session.
if command -v xhost >/dev/null 2>&1; then
  xhost +SI:localuser:root >/dev/null 2>&1 || true
fi
exec sudo -E "$(pwd)/.venv/bin/gamebot" "$@"
