#!/usr/bin/env bash
# Train piece detector (white/black) from LabelMe exports.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

.venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue \
  --out datasets/pieces --classes datasets/piece_classes.txt --require-label

# Pretrained yolov8n; Ultralytics aug is on by default.
# box=7.5 is CIoU default — raise if boxes converge poorly (e.g. box=10).
exec .venv/bin/yolo detect train \
  data=datasets/pieces/data.yaml \
  model=yolov8n.pt \
  epochs="${EPOCHS:-80}" \
  imgsz=640 \
  device="${DEVICE:-mps}" \
  box="${BOX:-7.5}" \
  project=runs/pieces \
  name="${NAME:-yolov8n_pieces}" \
  exist_ok=True
