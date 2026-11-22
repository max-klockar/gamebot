# Vision weights

Exported from the laptop [`gamebot-vision`](../../gamebot-vision) repo:

- `board.onnx` — **Pi runtime** board AABB (onnxruntime; ~12 MB)
- `board.pt` — train / re-export source (do **not** install Ultralytics on the Pi)
- `board_pose.pt` — optional; keypoints alone are not grid-accurate
- `pieces.pt` — white/black piece detector (future ONNX export)

**Runtime:** ONNX finds the board box → OpenCV chessboard / contour refine snaps
the 4 playing-field corners (`gamebot.board_detect`).

Refresh from laptop:

```bash
cd ../gamebot-vision && ./scripts/export-weights.sh ../gamebot
# also writes board.onnx from board.pt when ultralytics is available
```
