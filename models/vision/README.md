# Vision weights

Exported from the laptop [`gamebot-vision`](../../gamebot-vision) repo:

- `board_pose.pt` — board + **4 corner keypoints** (TL, TR, BR, BL) → quad / homography
- `board.pt` — AABB playing-field detector (backup)
- `pieces.pt` — white/black piece detector

Refresh from laptop:

```bash
cd ../gamebot-vision && ./scripts/export-weights.sh ../gamebot
```
