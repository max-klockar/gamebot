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
_LOGGED_WEIGHTS = False
_ZERO_LOG_I = 0
_LAST_DET_LOG: tuple[int, str] | None = None

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
    global _ONNX_SESSION, _ONNX_PATH, _LOGGED_WEIGHTS
    weights = weights.resolve()
    if _ONNX_SESSION is not None and _ONNX_PATH == weights:
        return _ONNX_SESSION
    import onnxruntime as ort

    _ONNX_SESSION = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
    _ONNX_PATH = weights
    if not _LOGGED_WEIGHTS:
        print(f"piece detect: loaded {weights}")
        _LOGGED_WEIGHTS = True
    return _ONNX_SESSION


def _normalize_pred(pred: np.ndarray) -> np.ndarray:
    """Return (4+nc, N) float32 array."""
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        return np.zeros((6, 0), dtype=np.float32)
    # Ultralytics export is (4+nc, N). Some runtimes flip to (N, 4+nc).
    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 64:
        return pred.astype(np.float32)
    if pred.shape[1] <= 64 and pred.shape[0] > pred.shape[1]:
        return pred.T.astype(np.float32)
    return pred.astype(np.float32)


def _decode_yolo_pred(
    pred: np.ndarray,
    *,
    r: float,
    left: float,
    top: float,
    conf: float,
    nms_iou: float,
) -> tuple[list[PieceDet], float]:
    """YOLOv8 ONNX → (detections, max_class_score)."""
    pred = _normalize_pred(pred)
    if pred.shape[0] < 6 or pred.shape[1] == 0:
        return [], 0.0
    boxes = pred[:4]
    cls_scores = pred[4:]
    peak = float(cls_scores.max()) if cls_scores.size else 0.0
    best_cls = np.argmax(cls_scores, axis=0)
    best_sc = cls_scores[best_cls, np.arange(cls_scores.shape[1])]
    mask = best_sc >= conf
    if not np.any(mask):
        return [], peak
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
        return [], peak
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
    return out, peak


def detect_pieces_onnx(
    frame_bgr: np.ndarray,
    weights: Path,
    *,
    conf: float = 0.15,
    imgsz: int = 640,
    nms_iou: float = 0.45,
) -> tuple[list[PieceDet], float]:
    """Returns (dets, max_raw_score across colour/swap tries)."""
    sess = _session(weights)
    best: list[PieceDet] = []
    best_n = -1
    peak_all = 0.0
    for swap_rb in (True, False):
        lb, r, left, top = _letterbox(frame_bgr, imgsz)
        rgb = lb[:, :, ::-1] if swap_rb else lb
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
        dets, peak = _decode_yolo_pred(
            pred, r=r, left=left, top=top, conf=conf, nms_iou=nms_iou
        )
        peak_all = max(peak_all, peak)
        if len(dets) > best_n:
            best_n = len(dets)
            best = dets
            if best_n >= 28:
                break
    return best, peak_all


def detect_pieces(
    frame_bgr: np.ndarray,
    *,
    weights_dir: Path | None = None,
    conf: float = 0.15,
    search_roi: np.ndarray | None = None,
) -> list[PieceDet]:
    """Detect white/black pieces. Tries board crop then full frame if crop is empty."""
    weights_dir = Path(weights_dir) if weights_dir is not None else default_weights_dir()
    onnx = weights_dir / "pieces.onnx"
    if not onnx.exists():
        print(f"piece detect: missing {onnx}")
        return []

    crops: list[tuple[np.ndarray, float, float, str]] = []
    if search_roi is not None and len(np.asarray(search_roi).reshape(-1)) >= 8:
        q = np.asarray(search_roi, dtype=np.float64).reshape(4, 2)
        x0 = max(0, int(q[:, 0].min()) - 40)
        y0 = max(0, int(q[:, 1].min()) - 40)
        x1 = min(frame_bgr.shape[1], int(q[:, 0].max()) + 40)
        y1 = min(frame_bgr.shape[0], int(q[:, 1].max()) + 40)
        if x1 > x0 + 40 and y1 > y0 + 40:
            crops.append((frame_bgr[y0:y1, x0:x1], float(x0), float(y0), "board-crop"))
    crops.append((frame_bgr, 0.0, 0.0, "full-frame"))

    try:
        best: list[PieceDet] = []
        best_peak = 0.0
        best_tag = ""
        for crop, ox, oy, tag in crops:
            dets, peak = detect_pieces_onnx(crop, onnx, conf=conf)
            best_peak = max(best_peak, peak)
            if ox or oy:
                dets = [
                    PieceDet(
                        xy=(d.xy[0] + ox, d.xy[1] + oy),
                        cls=d.cls,
                        conf=d.conf,
                        xyxy=(
                            d.xyxy[0] + ox,
                            d.xyxy[1] + oy,
                            d.xyxy[2] + ox,
                            d.xyxy[3] + oy,
                        ),
                    )
                    for d in dets
                ]
            if len(dets) > len(best):
                best = dets
                best_tag = tag
            if len(best) >= 16:
                break
        global _ZERO_LOG_I, _LAST_DET_LOG
        if not best:
            _ZERO_LOG_I += 1
            _LAST_DET_LOG = None
            if _ZERO_LOG_I <= 3 or _ZERO_LOG_I % 25 == 0:
                print(
                    f"piece detect: 0 boxes (conf≥{conf:.2f}, peak_score≈{best_peak:.3f}, "
                    f"frame={frame_bgr.shape[1]}x{frame_bgr.shape[0]})"
                )
        else:
            _ZERO_LOG_I = 0
            sig = (len(best), best_tag)
            if sig != _LAST_DET_LOG:
                _LAST_DET_LOG = sig
                print(f"piece detect: {len(best)} boxes via {best_tag} (peak≈{best_peak:.3f})")
        return best
    except ImportError:
        print("piece detect: onnxruntime missing")
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"piece onnx failed ({exc})")
        return []


def pieces_available(weights_dir: Path | None = None) -> bool:
    weights_dir = Path(weights_dir) if weights_dir is not None else default_weights_dir()
    return (weights_dir / "pieces.onnx").exists()
