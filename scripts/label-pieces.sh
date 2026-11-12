#!/usr/bin/env bash
# Launch LabelMe on capture sets (laptop). Classes: white / black.
#
#   ./scripts/label-pieces.sh              # all images in captures/sets
#   ./scripts/label-pieces.sh --new-only   # only images without a LabelMe JSON yet
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NEW_ONLY=0
SRC="captures/sets"
for arg in "$@"; do
  case "$arg" in
    --new-only) NEW_ONLY=1 ;;
    -*)
      echo "unknown flag: $arg (supported: --new-only)"
      exit 1
      ;;
    *) SRC="$arg" ;;
  esac
done

if [[ ! -d "$SRC" ]]; then
  echo "missing $SRC — scp captures from the Pi first"
  exit 1
fi

if [[ ! -x "$ROOT/.venv/bin/labelme" ]]; then
  echo "==> pip install labelme"
  "$ROOT/.venv/bin/pip" install -q "labelme>=5.4"
fi

# LabelMe only lists one directory (non-recursive) — flatten via symlinks.
QUEUE="$ROOT/captures/labelme_queue"
mkdir -p "$QUEUE"
# Refresh image symlinks only (LabelMe JSON in this folder is kept).
find "$QUEUE" -maxdepth 1 -type l -delete 2>/dev/null || true
count=0
skipped=0
while IFS= read -r -d '' img; do
  # Skip classical-debug overlays (…/detect_out/*)
  case "$img" in
    */detect_out/*|*_detect.jpg) continue ;;
  esac
  set_name="$(basename "$(dirname "$img")")"
  base="$(basename "$img")"
  stem="${base%.*}"
  link="$QUEUE/${set_name}__${base}"
  json="$QUEUE/${set_name}__${stem}.json"
  if [[ "$NEW_ONLY" -eq 1 && -f "$json" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  ln -sfn "$(cd "$(dirname "$img")" && pwd)/$(basename "$img")" "$link"
  count=$((count + 1))
done < <(find "$SRC" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0)

CLASSES="$ROOT/datasets/piece_classes.txt"
if [[ "$count" -eq 0 ]]; then
  echo "nothing to label (new-only=$NEW_ONLY, skipped_labeled=$skipped)"
  exit 0
fi

echo "LabelMe queue: $QUEUE ($count images to open; skipped_labeled=$skipped, new-only=$NEW_ONLY)"
echo "  classes: $(tr '\n' ' ' < "$CLASSES")"
echo "  Create rectangle → pick white or black → Next (D) / Prev (A)"
echo "  Save JSON beside each image (Ctrl+S)."
echo
echo "When done:"
echo "  .venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue --out datasets/pieces"
echo
exec "$ROOT/.venv/bin/labelme" "$QUEUE" --labels "$CLASSES"
