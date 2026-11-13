#!/usr/bin/env python3
"""Convert LabelMe JSON → YOLO detection dataset(s).

Same LabelMe files can hold `white` / `black` / `board`. Export separately:

  # pieces
  .venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue \\
      --out datasets/pieces --classes datasets/piece_classes.txt

  # board playing-field
  .venv/bin/python -m gamebot.labelme_to_yolo captures/labelme_queue \\
      --out datasets/board --classes datasets/board_classes.txt

Board polygons also write quads/*.json (4 corners) for perspective 8×8 math.
See docs/vision-strategy.md.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np


DEFAULT_CLASSES = ("white", "black")


def _load_classes(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_CLASSES)
    names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not names:
        raise SystemExit(f"no classes in {path}")
    return names


def _rect_to_yolo(
    points: list[list[float]],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def _normalize_points(shape: dict) -> list[list[float]] | None:
    shape_type = shape.get("shape_type") or "rectangle"
    pts = shape.get("points") or []
    if shape_type == "circle" and len(pts) >= 2:
        (cx0, cy0), (px, py) = pts[0], pts[1]
        rad = float(np.hypot(float(px) - float(cx0), float(py) - float(cy0)))
        return [
            [float(cx0) - rad, float(cy0) - rad],
            [float(cx0) + rad, float(cy0) + rad],
        ]
    if shape_type in {"rectangle", "polygon"} and len(pts) >= 2:
        return [[float(p[0]), float(p[1])] for p in pts]
    return None


def _quad_from_points(pts: list[list[float]]) -> list[list[float]] | None:
    """Best-effort 4 corners for perspective (polygon preferred)."""
    if len(pts) == 4:
        return pts
    if len(pts) < 4:
        # AABB corners from rectangle (axis-aligned — weaker under tilt)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    # Many points: take convex hull extremes via min-area rect
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    import cv2

    (_c, _s, _a) = cv2.minAreaRect(arr)
    box = cv2.boxPoints((_c, _s, _a))
    return [[float(x), float(y)] for x, y in box]


def _shapes_to_lines_and_quads(
    data: dict,
    class_to_id: dict[str, int],
) -> tuple[list[str], list[dict]]:
    img_w = int(data["imageWidth"])
    img_h = int(data["imageHeight"])
    lines: list[str] = []
    quads: list[dict] = []
    for shape in data.get("shapes") or []:
        label = str(shape.get("label") or "").strip().lower()
        if label not in class_to_id:
            continue  # other classes in the same JSON (e.g. board when exporting pieces)
        pts = _normalize_points(shape)
        if pts is None:
            continue
        xc, yc, bw, bh = _rect_to_yolo(pts, img_w, img_h)
        lines.append(f"{class_to_id[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        if label == "board":
            quad = _quad_from_points(pts)
            if quad is not None:
                quads.append(
                    {
                        "label": "board",
                        "quad": quad,  # TL-ish order not guaranteed; consumer may reorder
                        "imageWidth": img_w,
                        "imageHeight": img_h,
                        "shape_type": shape.get("shape_type"),
                    }
                )
    return lines, quads


def _resolve_image(json_path: Path, data: dict) -> Path:
    image_path = Path(data.get("imagePath") or "")
    if not image_path.is_absolute():
        image_path = (json_path.parent / image_path).resolve()
    if image_path.is_file():
        return image_path
    alt = json_path.with_suffix(image_path.suffix if image_path.suffix else ".jpg")
    if alt.is_file():
        return alt.resolve()
    raise FileNotFoundError(f"{json_path.name}: missing image {data.get('imagePath')}")


def main() -> None:
    p = argparse.ArgumentParser(description="LabelMe JSON → YOLO dataset")
    p.add_argument("src", type=Path, help="Folder of LabelMe JSON (+ images), searched recursively")
    p.add_argument("--out", type=Path, default=Path("datasets/pieces"), help="YOLO dataset root")
    p.add_argument("--classes", type=Path, default=Path("datasets/piece_classes.txt"))
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of images for val split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--require-label",
        action="store_true",
        help="Skip images that have none of the requested class boxes",
    )
    args = p.parse_args()

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    classes = _load_classes(args.classes if args.classes.is_file() else None)
    class_to_id = {n.lower(): i for i, n in enumerate(classes)}

    labelme_files: list[Path] = []
    for jp in sorted(src.rglob("*.json")):
        try:
            data = json.loads(jp.read_text())
        except Exception:
            continue
        if "shapes" in data and "imageWidth" in data:
            labelme_files.append(jp)
    if not labelme_files:
        raise SystemExit(f"no LabelMe JSON under {src} — run labelme first")

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    quads_dir = out / "quads"
    if "board" in class_to_id:
        quads_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[Path, Path, list[str], list[dict], str]] = []
    for jp in labelme_files:
        data = json.loads(jp.read_text())
        lines, quads = _shapes_to_lines_and_quads(data, class_to_id)
        if args.require_label and not lines:
            continue
        if not lines and not args.require_label:
            # still allow empty label files only if image is in set; skip empties by default
            continue
        img_path = _resolve_image(jp, data)
        try:
            rel_parent = jp.parent.name
            stem = f"{rel_parent}__{jp.stem}"
        except Exception:
            stem = jp.stem
        items.append((jp, img_path, lines, quads, stem))
        print(f"{jp.name} → {len(lines)} boxes" + (f", {len(quads)} board quad(s)" if quads else ""))

    if not items:
        raise SystemExit(f"no images with classes {classes} under {src}")

    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n_val = max(1, int(round(len(items) * args.val_frac))) if len(items) > 1 else 0
    val_idx = set(order[:n_val])

    total_boxes = 0
    for i, (_jp, img_path, lines, quads, stem) in enumerate(items):
        split = "val" if i in val_idx else "train"
        dest_img = out / "images" / split / f"{stem}{img_path.suffix.lower()}"
        dest_lbl = out / "labels" / split / f"{stem}.txt"
        shutil.copy2(img_path, dest_img)
        dest_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        total_boxes += len(lines)
        if quads and "board" in class_to_id:
            (quads_dir / f"{stem}.json").write_text(json.dumps(quads[0], indent=2) + "\n")

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                "val: images/val",
                f"names: {classes}",
                f"nc: {len(classes)}",
                "",
            ]
        )
    )

    n_train = len(items) - n_val
    print()
    print(f"images: {len(items)}  boxes: {total_boxes}")
    print(f"train: {n_train}  val: {n_val or len(items)}")
    print(f"wrote {yaml_path}")
    if "board" in class_to_id:
        print(f"board quads: {quads_dir}")
    print(
        "next:  .venv/bin/yolo detect train "
        f"data={out}/data.yaml model=yolov8n.pt epochs=80 imgsz=640 device=mps"
    )


if __name__ == "__main__":
    main()
