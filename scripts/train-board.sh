#!/usr/bin/env bash
# Train board playing-field detector (class: board) from LabelMe exports.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

.venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue \
  --out datasets/board --classes datasets/board_classes.txt --require-label

# Prefer images with slight rotation / perspective in the set — see docs/vision-strategy.md
exec .venv/bin/yolo detect train \
  data=datasets/board/data.yaml \
  model=yolov8n.pt \
  epochs="${EPOCHS:-60}" \
  imgsz=640 \
  device="${DEVICE:-mps}" \
  box="${BOX:-7.5}" \
  project=runs/board \
  name="${NAME:-yolov8n_board}" \
  exist_ok=True
