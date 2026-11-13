#!/usr/bin/env bash
# Launch LabelMe for vision labels: white / black / board.
#
#   ./scripts/label-pieces.sh                 # all images
#   ./scripts/label-pieces.sh --new-only      # no JSON yet
#   ./scripts/label-pieces.sh --missing-board # has JSON but no board shape (add board field)
#
# See docs/vision-strategy.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NEW_ONLY=0
MISSING_BOARD=0
SRC="captures/sets"
for arg in "$@"; do
  case "$arg" in
    --new-only) NEW_ONLY=1 ;;
    --missing-board) MISSING_BOARD=1 ;;
    -*)
      echo "unknown flag: $arg (supported: --new-only, --missing-board)"
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

has_board_label() {
  local json="$1"
  [[ -f "$json" ]] || return 1
  "$ROOT/.venv/bin/python" - "$json" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
for s in data.get("shapes") or []:
    if str(s.get("label") or "").strip().lower() == "board":
        raise SystemExit(0)
raise SystemExit(1)
PY
}

QUEUE="$ROOT/captures/labelme_queue"
mkdir -p "$QUEUE"
find "$QUEUE" -maxdepth 1 -type l -delete 2>/dev/null || true
count=0
skipped=0
while IFS= read -r -d '' img; do
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
  if [[ "$MISSING_BOARD" -eq 1 ]]; then
    # Only images that already have labels but lack a board playing-field shape
    if [[ ! -f "$json" ]] || has_board_label "$json"; then
      skipped=$((skipped + 1))
      continue
    fi
  fi
  ln -sfn "$(cd "$(dirname "$img")" && pwd)/$(basename "$img")" "$link"
  count=$((count + 1))
done < <(find "$SRC" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0)

CLASSES="$ROOT/datasets/label_classes.txt"
if [[ "$count" -eq 0 ]]; then
  echo "nothing to label (new-only=$NEW_ONLY missing-board=$MISSING_BOARD skipped=$skipped)"
  exit 0
fi

echo "LabelMe queue: $QUEUE ($count images; skipped=$skipped)"
echo "  classes: $(tr '\n' ' ' < "$CLASSES")"
echo "  pieces: rectangle around whole piece → white|black"
echo "  board:  polygon (prefer 4 corners) or rectangle around the 8x8 FIELD"
echo "  Slightly rotated / trapezoid boards are GOOD — label them."
echo
echo "When done:"
echo "  .venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue --out datasets/pieces --classes datasets/piece_classes.txt"
echo "  .venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue --out datasets/board --classes datasets/board_classes.txt"
echo
exec "$ROOT/.venv/bin/labelme" "$QUEUE" --labels "$CLASSES"
