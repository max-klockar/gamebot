#!/usr/bin/env bash
# Launch LabelMe on capture sets (laptop). Classes: white / black.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC="${1:-captures/sets}"
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
while IFS= read -r -d '' img; do
  # Skip classical-debug overlays (…/detect_out/*)
  case "$img" in
    */detect_out/*|*_detect.jpg) continue ;;
  esac
  set_name="$(basename "$(dirname "$img")")"
  link="$QUEUE/${set_name}__$(basename "$img")"
  ln -sfn "$(cd "$(dirname "$img")" && pwd)/$(basename "$img")" "$link"
  count=$((count + 1))
done < <(find "$SRC" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0)

CLASSES="$ROOT/datasets/piece_classes.txt"
echo "LabelMe queue: $QUEUE ($count images)"
echo "  classes: $(tr '\n' ' ' < "$CLASSES")"
echo "  Create rectangle → pick white or black → Next (D) / Prev (A)"
echo "  Save JSON beside each image (Ctrl+S)."
echo
echo "When done:"
echo "  .venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue --out datasets/pieces"
echo
# Newer labelme: omit image blobs by default (use --with-image-data only if you need them).
exec "$ROOT/.venv/bin/labelme" "$QUEUE" --labels "$CLASSES"
