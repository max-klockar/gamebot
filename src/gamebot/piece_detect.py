"""White/black piece detector: YOLO ONNX (onnxruntime) for the Pi.

Class ids match training: 0=white, 1=black. Centers map to board squares via
the locked camera quad (same warp as occupancy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gamebot.board_detect import _letterbox, default_weights_dir

_ONNX_SESSION = None
_ONNX_PATH: Path | None = None

CLASS_WHITE = 0
CLASS_BLACK = 1
CLASS_NAMES = {CLASS_WHITE: "white", CLASS_BLACK: "black"}


@dataclass(frozen=True)
class PieceDet:
    xy: tuple[float, float]  # center in full-frame camera px
    cls: int  # 0=white, 1=black
    conf: float
    xyxy: tuple[float, float, float, float]


def _session(weights: Path):
    global _ONNX_SESSION, _ONNX_PATH
    weights = weights.resolve()
    if _ONNX_SESSION is not None and _ONNX_PATH == weights:
        return _ONNX_SESSION
    import onnxruntime as ort

    _ONNX_SESSION = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
    _ONNX_PATH = weights
    return _ONNX_SESSION


def _decode_yolo_pred(
    pred: np.ndarray,
    *,
    r: float,
    left: float,
    top: float,
    conf: float,
    nms_iou: float,
) -> list[PieceDet]:
    """YOLOv8 ONNX (1, 4+nc, N) or (4+nc, N) → PieceDet list after NMS."""
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2 or pred.shape[0] < 6:
        return []
    # rows: cx,cy,w,h, class0, class1, ...
    boxes = pred[:4]
    cls_scores = pred[4:]
    best_cls = np.argmax(cls_scores, axis=0)
    best_sc = cls_scores[best_cls, np.arange(cls_scores.shape[1])]
    mask = best_sc >= conf
    if not np.any(mask):
        return []
    boxes = boxes[:, mask]
    best_cls = best_cls[mask]
    best_sc = best_sc[mask]
    cx, cy, bw, bh = boxes
    x1 = (cx - bw / 2 - left) / r
    y1 = (cy - bh / 2 - top) / r
    w = bw / r
    h = bh / r
    idxs = cv2.dnn.NMSBoxes(
        bboxes=np.stack([x1, y1, w, h], axis=1).tolist(),
        scores=best_sc.tolist(),
        score_threshold=conf,
        nms_threshold=nms_iou,
    )
    if idxs is None or len(idxs) == 0:
        return []
    out: list[PieceDet] = []
    for i in np.array(idxs).reshape(-1):
        i = int(i)
        xa, ya, wa, ha = float(x1[i]), float(y1[i]), float(w[i]), float(h[i])
        out.append(
            PieceDet(
                xy=(xa + wa * 0.5, ya + ha * 0.5),
                cls=int(best_cls[i]),
                conf=float(best_sc[i]),
                xyxy=(xa, ya, xa + wa, ya + ha),
            )
        )
    return out


def detect_pieces_onnx(
    frame_bgr: np.ndarray,
    weights: Path,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    nms_iou: float = 0.45,
) -> list[PieceDet]:
    sess = _session(weights)
    best: list[PieceDet] = []
    best_n = -1
    for swap_rb in (True, False):
        lb, r, left, top = _letterbox(frame_bgr, imgsz)
        rgb = lb[:, :, ::-1] if swap_rb else lb
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
        dets = _decode_yolo_pred(pred, r=r, left=left, top=top, conf=conf, nms_iou=nms_iou)
        if len(dets) > best_n:
            best_n = len(dets)
            best = dets
            if best_n >= 28:  # good enough — skip second colour order
                break
    return best


def detect_pieces(
    frame_bgr: np.ndarray,
    *,
    weights_dir: Path | None = None,
    conf: float = 0.25,
    search_roi: np.ndarray | None = None,
) -> list[PieceDet]:
    """Detect white/black pieces. Optional board quad crops for speed/accuracy."""
    weights_dir = Path(weights_dir) if weights_dir is not None else default_weights_dir()
    onnx = weights_dir / "pieces.onnx"
    if not onnx.exists():
        return []
    crop = frame_bgr
    ox = oy = 0.0
    if search_roi is not None and len(np.asarray(search_roi).reshape(-1)) >= 8:
        q = np.asarray(search_roi, dtype=np.float64).reshape(4, 2)
        x0 = max(0, int(q[:, 0].min()) - 20)
        y0 = max(0, int(q[:, 1].min()) - 20)
        x1 = min(frame_bgr.shape[1], int(q[:, 0].max()) + 20)
        y1 = min(frame_bgr.shape[0], int(q[:, 1].max()) + 20)
        if x1 > x0 + 40 and y1 > y0 + 40:
            crop = frame_bgr[y0:y1, x0:x1]
            ox, oy = float(x0), float(y0)
    try:
        dets = detect_pieces_onnx(crop, onnx, conf=conf)
    except ImportError:
        print("piece detect: onnxruntime missing")
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"piece onnx failed ({exc})")
        return []
    if ox == 0 and oy == 0:
        return dets
    return [
        PieceDet(
            xy=(d.xy[0] + ox, d.xy[1] + oy),
            cls=d.cls,
            conf=d.conf,
            xyxy=(d.xyxy[0] + ox, d.xyxy[1] + oy, d.xyxy[2] + ox, d.xyxy[3] + oy),
        )
        for d in dets
    ]


def pieces_available(weights_dir: Path | None = None) -> bool:
    weights_dir = Path(weights_dir) if weights_dir is not None else default_weights_dir()
    return (weights_dir / "pieces.onnx").exists()
