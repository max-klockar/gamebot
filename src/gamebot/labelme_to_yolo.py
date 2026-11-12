#!/usr/bin/env python3
"""Convert LabelMe rectangle JSON → YOLO detection dataset.

Typical flow:
  1. .venv/bin/labelme captures/sets --labels datasets/piece_classes.txt --nodata
  2. .venv/bin/python -m gamebot.labelme_to_yolo captures/sets --out datasets/pieces

YOLO label line:  class_id x_center y_center width height  (all normalized 0–1)
"""

from __future__ import annotations

import argparse
import json
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


def convert_one(
    json_path: Path,
    class_to_id: dict[str, int],
    images_out: Path,
    labels_out: Path,
    *,
    copy_images: bool,
) -> int:
    data = json.loads(json_path.read_text())
    img_w = int(data["imageWidth"])
    img_h = int(data["imageHeight"])
    image_path = Path(data.get("imagePath") or "")
    if not image_path.is_absolute():
        image_path = (json_path.parent / image_path).resolve()
    if not image_path.is_file():
        # Common: imagePath is just the filename beside the JSON
        alt = json_path.with_suffix(image_path.suffix if image_path.suffix else ".jpg")
        if alt.is_file():
            image_path = alt
        else:
            raise FileNotFoundError(f"{json_path.name}: missing image {data.get('imagePath')}")

    lines: list[str] = []
    for shape in data.get("shapes") or []:
        label = str(shape.get("label") or "").strip().lower()
        if label not in class_to_id:
            raise SystemExit(f"{json_path}: unknown label {label!r} (expected {sorted(class_to_id)})")
        shape_type = shape.get("shape_type") or "rectangle"
        pts = shape.get("points") or []
        if shape_type == "circle" and len(pts) >= 2:
            # LabelMe circle: center + point on circumference → axis-aligned box
            (cx0, cy0), (px, py) = pts[0], pts[1]
            rad = float(np.hypot(float(px) - float(cx0), float(py) - float(cy0)))
            pts = [
                [float(cx0) - rad, float(cy0) - rad],
                [float(cx0) + rad, float(cy0) + rad],
            ]
        elif shape_type in {"rectangle", "polygon"}:
            if len(pts) < 2:
                continue
        else:
            continue
        # rectangle / polygon / circle → axis-aligned YOLO box
        xc, yc, bw, bh = _rect_to_yolo(pts, img_w, img_h)
        lines.append(f"{class_to_id[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    stem = json_path.stem
    # Keep set folder in the name so duplicates across sets don't collide
    try:
        rel_parent = json_path.parent.name
        out_stem = f"{rel_parent}__{stem}"
    except Exception:
        out_stem = stem

    dest_img = images_out / f"{out_stem}{image_path.suffix.lower()}"
    dest_lbl = labels_out / f"{out_stem}.txt"
    if copy_images:
        shutil.copy2(image_path, dest_img)
    else:
        if dest_img.exists() or dest_img.is_symlink():
            dest_img.unlink()
        dest_img.symlink_to(image_path.resolve())
    dest_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="LabelMe JSON → YOLO dataset")
    p.add_argument("src", type=Path, help="Folder of LabelMe JSON (+ images), searched recursively")
    p.add_argument("--out", type=Path, default=Path("datasets/pieces"), help="YOLO dataset root")
    p.add_argument("--classes", type=Path, default=Path("datasets/piece_classes.txt"))
    p.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of images for val split")
    args = p.parse_args()

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    classes = _load_classes(args.classes if args.classes.is_file() else None)
    class_to_id = {n.lower(): i for i, n in enumerate(classes)}

    json_files = sorted(src.rglob("*.json"))
    # Skip capture sidecars that aren't LabelMe (no shapes key / no imageWidth)
    labelme_files: list[Path] = []
    for jp in json_files:
        try:
            data = json.loads(jp.read_text())
        except Exception:
            continue
        if "shapes" in data and "imageWidth" in data:
            labelme_files.append(jp)

    if not labelme_files:
        raise SystemExit(f"no LabelMe JSON under {src} — run labelme first")

    images_all = out / "images" / "all"
    labels_all = out / "labels" / "all"
    images_all.mkdir(parents=True, exist_ok=True)
    labels_all.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    stems: list[str] = []
    for jp in labelme_files:
        n = convert_one(jp, class_to_id, images_all, labels_all, copy_images=args.copy_images)
        total_boxes += n
        try:
            rel_parent = jp.parent.name
            stems.append(f"{rel_parent}__{jp.stem}")
        except Exception:
            stems.append(jp.stem)
        print(f"{jp.relative_to(src)} → {n} boxes")

    # Simple deterministic split
    stems_sorted = sorted(set(stems))
    n_val = max(1, int(round(len(stems_sorted) * args.val_frac))) if len(stems_sorted) > 1 else 0
    val_set = set(stems_sorted[:n_val])
    train_set = [s for s in stems_sorted if s not in val_set]

    def write_list(name: str, ids: list[str]) -> None:
        lines = []
        for s in ids:
            # find image file with any common suffix
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                cand = images_all / f"{s}{ext}"
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            lines.append(str(img.resolve()))
        (out / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    write_list("train", train_set)
    write_list("val", sorted(val_set) if val_set else train_set)

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: train.txt",
                "val: val.txt",
                f"names: {classes}",
                f"nc: {len(classes)}",
                "",
            ]
        )
    )

    print()
    print(f"images: {len(stems_sorted)}  boxes: {total_boxes}")
    print(f"train: {len(train_set)}  val: {len(val_set) or len(train_set)}")
    print(f"wrote {yaml_path}")
    print("next:  .venv/bin/yolo detect train data=datasets/pieces/data.yaml model=yolov8n.pt epochs=80 imgsz=640")


if __name__ == "__main__":
    main()
