#!/usr/bin/env bash
# Run gamebot (or any venv python command) on the Pi with root for WS2812 DMA.
#
#   ./scripts/run-pi.sh --camera picam --projector hdmi --elo 500
#   ./scripts/run-pi.sh python -m gamebot.piece_detect_demo
#   ./scripts/run-pi.sh -m gamebot.piece_detect_demo
set -euo pipefail
cd "$(dirname "$0")/.."

REAL_USER="${SUDO_USER:-$USER}"
REAL_UID="$(id -u "$REAL_USER" 2>/dev/null || id -u)"
export DISPLAY="${DISPLAY:-:0}"
# Prefer X11 for pygame-as-root — Wayland sockets under /run/user/<uid> reject root.
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"

# Authorize root on the logged-in user's display (must run as that user).
allow_root_on_display() {
  if ! command -v xhost >/dev/null 2>&1; then
    return 0
  fi
  local user_runtime="/run/user/${REAL_UID}"
  if [[ "$(id -u)" -eq 0 && "$REAL_USER" != "root" ]]; then
    if [[ -d "$user_runtime" ]]; then
      sudo -u "$REAL_USER" env DISPLAY="$DISPLAY" XDG_RUNTIME_DIR="$user_runtime" \
        xhost +SI:localuser:root >/dev/null 2>&1 || true
    fi
  else
    env DISPLAY="$DISPLAY" XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$user_runtime}" \
      xhost +SI:localuser:root >/dev/null 2>&1 || true
  fi
}

# Root needs a runtime dir it owns; keeping the user's /run/user/N makes SDL complain.
ensure_root_runtime() {
  mkdir -p /run/user/0
  chmod 700 /run/user/0
  export XDG_RUNTIME_DIR=/run/user/0
  # Don't let SDL pick up the session Wayland socket.
  unset WAYLAND_DISPLAY || true
}

allow_root_on_display

PY="$(pwd)/.venv/bin/python"
GB="$(pwd)/.venv/bin/gamebot"

run_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    ensure_root_runtime
    exec "$@"
  fi
  # Fresh root env: pass display vars explicitly (sudoers often strips -E).
  exec sudo env \
    DISPLAY="$DISPLAY" \
    SDL_VIDEODRIVER="$SDL_VIDEODRIVER" \
    XDG_RUNTIME_DIR=/run/user/0 \
    PATH="$PATH" \
    bash -c 'mkdir -p /run/user/0 && chmod 700 /run/user/0; unset WAYLAND_DISPLAY; exec "$@"' -- "$@"
}

if [[ "${1:-}" == "python" ]]; then
  shift
  run_cmd "$PY" "$@"
elif [[ "${1:-}" == "-m" ]]; then
  run_cmd "$PY" "$@"
else
  run_cmd "$GB" "$@"
fi
