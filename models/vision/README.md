# Vision weights

Exported from the laptop [`gamebot-vision`](../../gamebot-vision) repo:

- `board.onnx` — **Pi runtime** board AABB (onnxruntime; ~12 MB)
- `pieces.onnx` — **Pi runtime** white/black pieces (onnxruntime; ~12 MB)
- `board.pt` / `pieces.pt` — train / re-export source (do **not** install Ultralytics on the Pi)
- `board_pose.pt` — optional; keypoints alone are not grid-accurate

**Runtime:** board ONNX → classical corner refine (`gamebot.board_detect`).
Piece ONNX → square occupancy / side colour (`gamebot.piece_detect` + chess-plugin).

Refresh from laptop:

```bash
cd ../gamebot-vision && ./scripts/export-weights.sh ../gamebot
# writes board.onnx + pieces.onnx when ultralytics is available
```
