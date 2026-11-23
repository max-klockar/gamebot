"""Lock physical board with YOLO+mech and write a debug overlay.

  python -m gamebot.board_detect_demo path/to.jpg
  python -m gamebot.board_detect_demo --camera   # one Picamera frame (Pi)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gamebot.board_detect import default_weights_dir, detect_physical_board_quad


def _overlay(frame: np.ndarray, result) -> np.ndarray:
    vis = frame.copy()
    if result is None:
        cv2.putText(vis, "NO BOARD", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return vis
    x0, y0, x1, y1 = result.aabb
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
    cv2.polylines(vis, [result.quad.astype(np.int32)], True, (0, 255, 0), 3)
    names = ("TL", "TR", "BR", "BL")
    for name, pt in zip(names, result.quad):
        p = tuple(pt.astype(int))
        cv2.circle(vis, p, 7, (0, 255, 255), -1)
        cv2.putText(vis, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, result.method, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return vis


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO + mechanical board lock demo")
    p.add_argument("image", type=Path, nargs="?", help="still image path")
    p.add_argument("--camera", action="store_true", help="grab one live camera frame")
    p.add_argument("--out", type=Path, default=Path("board_lock_demo.jpg"))
    p.add_argument("--weights", type=Path, default=None)
    args = p.parse_args()

    if args.camera:
        from gamebot.adapters import create_camera
        from gamebot.config import AppConfig

        cfg = AppConfig()
        cam = create_camera("picam", cfg.camera_size, None, zoom=cfg.camera_zoom)
        cam.open()
        try:
            frame = cam.read()
            for _ in range(4):
                frame = cam.read()
        finally:
            cam.close()
    elif args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
    else:
        raise SystemExit("pass an image path or --camera")

    weights = args.weights or default_weights_dir()
    print(f"weights: {weights} (prefer board.onnx + onnxruntime on Pi)")
    result = detect_physical_board_quad(frame, weights_dir=weights)
    if result is None:
        print("no board")
    else:
        print(f"method={result.method} quad=\n{result.quad}")
    out = args.out.expanduser()
    cv2.imwrite(str(out), _overlay(frame, result))
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
