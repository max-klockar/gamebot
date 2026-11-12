#!/usr/bin/env bash
# One-shot update on the Pi (or laptop): pull gamebot + chess plugin, reinstall only if needed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Chess plugin lives next to gamebot under different folder names.
PLUGIN=""
for cand in \
  "$ROOT/../gamebot-plugin-chess" \
  "$ROOT/../chess-plugin" \
  "$HOME/src/gamebot-plugin-chess" \
  "$HOME/src/chess-plugin" \
  "$HOME/dev/schack/chess-plugin"
do
  if [[ -d "$cand/.git" ]]; then
    PLUGIN="$(cd "$cand" && pwd)"
    break
  fi
done

VENV_PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
if [[ ! -x "$VENV_PY" ]]; then
  echo "missing venv at $ROOT/.venv — create it first"
  exit 1
fi

# Returns 0 if HEAD moved after pull.
pull_changed() {
  local dir="$1"
  local before after
  before="$(git -C "$dir" rev-parse HEAD)"
  git -C "$dir" pull --ff-only
  after="$(git -C "$dir" rev-parse HEAD)"
  [[ "$before" != "$after" ]]
}

importable() {
  local mod="$1"
  "$VENV_PY" -c "import ${mod}" >/dev/null 2>&1
}

echo "==> gamebot  ($ROOT)"
GAMEBOT_CHANGED=0
if pull_changed "$ROOT"; then
  GAMEBOT_CHANGED=1
fi

PLUGIN_CHANGED=0
if [[ -n "$PLUGIN" ]]; then
  echo "==> chess plugin  ($PLUGIN)"
  if pull_changed "$PLUGIN"; then
    PLUGIN_CHANGED=1
  fi
else
  echo "==> chess plugin not found beside gamebot (skipped)"
fi

if [[ "$PLUGIN_CHANGED" -eq 1 ]] || { [[ -n "$PLUGIN" ]] && ! importable chess_plugin; }; then
  echo "==> pip install -e chess plugin"
  "$PIP" install -e "$PLUGIN" -q
elif [[ -n "$PLUGIN" ]]; then
  echo "==> chess plugin unchanged — skip pip"
fi

if [[ "$GAMEBOT_CHANGED" -eq 1 ]] || ! importable gamebot; then
  echo "==> pip install -e gamebot"
  "$PIP" install -e "$ROOT" -q
else
  echo "==> gamebot unchanged — skip pip"
fi

# Optional hardware docs repo
for hw in "$ROOT/../gamebot-hardware" "$HOME/src/gamebot-hardware"; do
  if [[ -d "$hw/.git" ]]; then
    echo "==> hardware  ($hw)"
    git -C "$hw" pull --ff-only || true
    break
  fi
done

echo
echo "HEAD gamebot:  $(git -C "$ROOT" log -1 --oneline)"
if [[ -n "$PLUGIN" ]]; then
  echo "HEAD chess:    $(git -C "$PLUGIN" log -1 --oneline)"
fi
echo "done. example:"
echo "  ./scripts/run-pi.sh --camera picam --projector hdmi --elo 500"
echo "  ./scripts/run-pi.sh python -m gamebot.capture_frames"
echo "  ./scripts/run-pi.sh python -m gamebot.capture_sets"
echo "  ./scripts/run-pi.sh python -m gamebot.piece_detect_demo"
