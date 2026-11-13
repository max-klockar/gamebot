# Vision strategy (chess on projected / physical board)

Goal: stable **piece → square** mapping for turn-based play on the Pi
(overhead Picam + projector), without needing perfect photo→FEN every frame.

## Pipeline

```
camera frame
    │
    ├─ (A) Board region ──► crop / quad
    │         │
    │         ▼
    │      warp to square → 8×8 grid (math)
    │
    └─ (B) Piece detector (white / black) inside board region
              │
              ▼
           piece center → square index
              │
              ▼
           diff vs python-chess legal moves
```

Two **separate** YOLO-nano models (same LabelMe files, filtered exports):

| Model | Classes | Role |
|-------|---------|------|
| Board | `board` | Find the **playing field** (64-square area), not each square |
| Pieces | `white`, `black` | Find pieces; run on board crop when possible |

Calibration (`~/.gamebot/calibration.json`) remains the preferred board pose for
**projected** play. The board model is backup when cal is missing, the physical
board moved, or we want a crop before piece inference.

## Labeling

Use LabelMe with classes in `datasets/label_classes.txt`:

```bash
./scripts/label-pieces.sh                 # all / new images
./scripts/label-pieces.sh --missing-board # reopen shots that need a board box
```

- **Pieces:** rectangle around the whole piece (rough is fine). Overhead → use
  box **center** as the square sample point.
- **Board:** one shape around the **inner playing field** (the 64 squares), not
  the wooden frame / logo margin if you can avoid it.
  - Prefer a **4-point polygon** (corners of the field) when the board is
    rotated or trapezoid in the image.
  - A loose axis-aligned rectangle is OK for the board *detector*; accurate
    8×8 splitting still needs a quad (see below).

Export datasets:

```bash
# pieces only
.venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue \
  --out datasets/pieces --classes datasets/piece_classes.txt

# board only
.venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue \
  --out datasets/board --classes datasets/board_classes.txt
```

Polygon board labels also write `datasets/board/quads/*.json` (4 corners) for
perspective math / later refinement.

## Rotation & perspective (important)

- Capture at **slight angles** on purpose. If every train image is perfectly
  axis-aligned, the board model overfits to “level rectangles” and fails when
  the table is rotated a few degrees.
- The camera is often closer to one side → the field is a **trapezoid**
  (near edge wider than far). You must **not** slice an axis-aligned bbox into
  equal pixel rows/cols and call that chess squares.
- Correct approach:
  1. Get a **quad** (4 corners of the playing field) — from polygon label,
     cal homography, or refine corners inside the board AABB crop.
  2. `getPerspectiveTransform` → warp to a square canonical board.
  3. Divide that square into an 8×8 grid in warped space.
  4. Map piece centers through the same homography, then into cells.

Standard YOLO boxes are always axis-aligned; that only localizes the region.
Grid accuracy comes from the **quad + warp**, not from the AABB alone.

## Training notes

- Start from **pretrained** `yolov8n.pt` (not from scratch).
- Ultralytics already applies augmentation (mosaic, flip, HSV, translate, etc.).
  Keep capturing real variation (angles, lighting, projector on/off) rather than
  relying on aug alone. ~50–200 images is a sensible band; 15 was a smoke test.
- Default box loss is CIoU-style; if convergence is poor, try raising `box=`
  (e.g. `box=10`) before inventing a custom head.
- **False positives outside the board** (arms, laptop): at inference, **reject**
  any piece whose center lies outside the board quad. Optionally train the
  piece model on **board-cropped** images only so outside pixels are never
  positive examples. That matches “predictions outside the board region should
  hurt” without a custom loss on day one.
- Do **not** train `board` + `white`/`black` as one multi-class detector at
  first — scale imbalance (huge board vs tiny pieces) fights you. Two models.

## Runtime loop (play)

1. Settle frame after a human move.
2. Board quad = cal if healthy, else board YOLO (+ corner refine if needed).
3. Piece YOLO on full frame or board crop; keep detections inside quad.
4. Occupancy / centers → squares → legal-move filter → accept move.
5. On failures: `capture_frames` / failure snap → LabelMe `--new-only` → retrain
   (pseudo-labels later: predict → quick human fix → add to set).

## Commands (laptop)

```bash
# label
./scripts/label-pieces.sh --missing-board

# export + train
.venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue \
  --out datasets/board --classes datasets/board_classes.txt
./scripts/train-board.sh

.venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue \
  --out datasets/pieces --classes datasets/piece_classes.txt
./scripts/train-pieces.sh
```
