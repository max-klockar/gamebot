#!/usr/bin/env bash
# One-shot update on the Pi (or laptop): pull gamebot + chess plugin, reinstall.
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
    PLUGIN="$cand"
    break
  fi
done

VENV_PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
if [[ ! -x "$VENV_PY" ]]; then
  echo "missing venv at $ROOT/.venv — create it first"
  exit 1
fi

echo "==> gamebot  ($ROOT)"
git -C "$ROOT" pull --ff-only

if [[ -n "$PLUGIN" ]]; then
  echo "==> chess plugin  ($PLUGIN)"
  git -C "$PLUGIN" pull --ff-only
  echo "==> pip install -e chess plugin"
  "$PIP" install -e "$PLUGIN" -q
else
  echo "==> chess plugin not found beside gamebot (skipped)"
fi

echo "==> pip install -e gamebot"
"$PIP" install -e "$ROOT" -q

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
echo "  .venv/bin/python -m gamebot.piece_detect_demo --camera picam --projector hdmi"
