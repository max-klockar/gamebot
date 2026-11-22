# Vision weights

Exported from the laptop [`gamebot-vision`](../../gamebot-vision) repo:

- `board.pt` — AABB playing-field detector (ROI for classical corner refine)
- `board_pose.pt` — pose model; used only as AABB fallback if `board.pt` misses
- `pieces.pt` — white/black piece detector

**Runtime:** YOLO finds the board box → OpenCV chessboard / contour refine snaps
the 4 playing-field corners (see `gamebot.board_detect`). Pose keypoints alone
are not precise enough for the 8×8 grid.

Refresh from laptop:

```bash
cd ../gamebot-vision && ./scripts/export-weights.sh ../gamebot
```
