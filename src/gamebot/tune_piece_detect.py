#!/usr/bin/env python3
"""Run piece detection on captured stills and write annotated overlays.

Intended for laptop iteration against Pi photos:

  python -m gamebot.tune_piece_detect captures/
  python -m gamebot.tune_piece_detect captures/20261112-195000_01.jpg --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gamebot.calibration import CalibrationData


def _load_roi(image: Path, explicit: Path | None) -> np.ndarray | None:
    if explicit is not None:
        data = json.loads(explicit.read_text())
        corners = data.get("board_corners_camera")
        return None if corners is None else np.asarray(corners, dtype=np.float64)

    side = image.with_suffix(".json")
    if side.is_file():
        data = json.loads(side.read_text())
        corners = data.get("board_corners_camera")
        if corners is not None:
            return np.asarray(corners, dtype=np.float64)

    cal = Path.home() / ".gamebot" / "calibration.json"
    if cal.is_file():
        try:
            return CalibrationData.load(cal).board_corners_camera
        except Exception:
            return None
    return None


def _annotate(
    frame: np.ndarray,
    whites: list,
    blacks: list,
    roi: np.ndarray | None,
) -> np.ndarray:
    out = frame.copy()
    if roi is not None:
        cv2.polylines(out, [roi.astype(np.int32)], True, (0, 200, 140), 2)
    for i, b in enumerate(whites, start=1):
        x, y = int(b.xy[0]), int(b.xy[1])
        cv2.rectangle(out, (x - 22, y - 22), (x + 22, y + 22), (255, 255, 255), 2)
        cv2.putText(out, f"W{i}", (x - 14, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, b in enumerate(blacks, start=1):
        x, y = int(b.xy[0]), int(b.xy[1])
        cv2.rectangle(out, (x - 22, y - 22), (x + 22, y + 22), (40, 40, 255), 2)
        cv2.putText(out, f"B{i}", (x - 14, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 255), 2)
    label = f"W={len(whites)} B={len(blacks)}"
    cv2.putText(out, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


def _images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def main() -> None:
    p = argparse.ArgumentParser(description="Tune/debug piece detection on stills")
    p.add_argument("path", type=Path, help="Image file or directory of captures")
    p.add_argument("--roi-json", type=Path, default=None, help="Override ROI sidecar")
    p.add_argument("--full-frame", action="store_true", help="Ignore board ROI")
    p.add_argument("--out-dir", type=Path, default=None, help="Where to write *_detect.jpg")
    p.add_argument("--show", action="store_true", help="Open a preview window")
    p.add_argument("--max-each", type=int, default=16)
    args = p.parse_args()

    from chess_plugin.vision import find_piece_blobs

    images = _images(args.path.expanduser())
    if not images:
        raise SystemExit(f"no images under {args.path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = images[0].parent / "detect_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"skip unreadable {img_path}")
            continue
        roi = None if args.full_frame else _load_roi(img_path, args.roi_json)
        whites, blacks = find_piece_blobs(frame, roi, max_each=args.max_each)
        print(f"{img_path.name}: W={len(whites)} B={len(blacks)}")
        for i, b in enumerate(whites, start=1):
            print(f"  W{i}=({b.xy[0]:.0f},{b.xy[1]:.0f}) area={b.area:.0f} score={b.score:.3f} mean={b.mean_gray:.0f}")
        for i, b in enumerate(blacks, start=1):
            print(f"  B{i}=({b.xy[0]:.0f},{b.xy[1]:.0f}) area={b.area:.0f} score={b.score:.3f} mean={b.mean_gray:.0f}")
        annotated = _annotate(frame, whites, blacks, roi)
        out_path = out_dir / f"{img_path.stem}_detect.jpg"
        cv2.imwrite(str(out_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"  -> {out_path}")
        if args.show:
            cv2.imshow("tune_piece_detect", annotated)
            if cv2.waitKey(0) in (ord("q"), 27):
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
