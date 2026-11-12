#!/usr/bin/env bash
# Run gamebot (or any venv python command) on the Pi with root for WS2812 DMA.
#
#   ./scripts/run-pi.sh --camera picam --projector hdmi --elo 500
#   ./scripts/run-pi.sh python -m gamebot.piece_detect_demo
#   ./scripts/run-pi.sh -m gamebot.piece_detect_demo
set -euo pipefail
cd "$(dirname "$0")/.."
export DISPLAY="${DISPLAY:-:0}"
# Allow root to draw on the logged-in user's X/Wayland display session.
if command -v xhost >/dev/null 2>&1; then
  xhost +SI:localuser:root >/dev/null 2>&1 || true
fi

PY="$(pwd)/.venv/bin/python"
if [[ "${1:-}" == "python" ]]; then
  shift
  exec sudo -E "$PY" "$@"
elif [[ "${1:-}" == "-m" ]]; then
  exec sudo -E "$PY" "$@"
else
  exec sudo -E "$(pwd)/.venv/bin/gamebot" "$@"
fi
