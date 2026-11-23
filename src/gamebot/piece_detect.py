"""White/black piece detector: YOLO ONNX (onnxruntime) for the Pi.

Class ids match training: 0=white, 1=black. The standing point is the box
center pulled toward the image center (a straight-down camera throws the
head farther out than the feet), then mapped through the locked board quad.
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

# Share of the way from the box center to its inner radial edge.
# The head projects outward; the base is inward of the center. The box is
# square and the piece is round, so that inner corner sits outside the piece.
# Stay closer to the box middle than to the corner.
FOOT_BIAS = 0.4
# A second box on the same piece: high IoU, or one box mostly inside the other
# (IoU stays modest when a tight box sits in a loose one).
DUP_IOU = 0.55
DUP_CONTAIN = 0.72


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


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return (IoU, intersection / smaller area)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    union = area_a + area_b - inter
    iou = inter / union if union > 0.0 else 0.0
    contain = inter / smaller if smaller > 0.0 else 0.0
    return iou, contain


def suppress_duplicate_pieces(
    dets: list[PieceDet],
    *,
    iou_thr: float = DUP_IOU,
    contain_thr: float = DUP_CONTAIN,
) -> tuple[list[PieceDet], int]:
    """Drop a lower-confidence box that is the same piece a second time.

    Neighbouring pieces overlap a little and stay. A second box on one piece
    either covers it (high IoU) or nests inside the first (high containment).
    """
    ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: list[PieceDet] = []
    dropped = 0
    for d in ordered:
        if any(
            (iou >= iou_thr or contain >= contain_thr)
            for iou, contain in (_overlap(d.xyxy, k.xyxy) for k in kept)
        ):
            dropped += 1
            continue
        kept.append(d)
    return kept, dropped


def standing_xy(
    xyxy: tuple[float, float, float, float],
    frame_wh: tuple[int, int],
    *,
    bias: float = FOOT_BIAS,
) -> tuple[float, float]:
    """Where the piece stands, given an overhead box.

    The head is closer to the lens, so it lands farther from the image center
    than the feet. Move from the box center toward the image center, along
    that ray, by ``bias`` of the distance to the box's inner edge. Never step
    past the image center.
    """
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    ox = float(frame_wh[0]) * 0.5
    oy = float(frame_wh[1]) * 0.5
    vx, vy = cx - ox, cy - oy
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1.0 or bias <= 0.0:
        return (cx, cy)
    ux, uy = vx / norm, vy / norm
    projs = [(px - ox) * ux + (py - oy) * uy for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    inward = max(0.0, (cx - ox) * ux + (cy - oy) * uy - min(projs)) * bias
    inward = min(inward, norm)
    return (cx - ux * inward, cy - uy * inward)


def _with_standing_points(dets: list[PieceDet], frame_wh: tuple[int, int]) -> list[PieceDet]:
    return [
        PieceDet(
            xy=standing_xy(d.xyxy, frame_wh),
            cls=d.cls,
            conf=d.conf,
            xyxy=d.xyxy,
        )
        for d in dets
    ]


def detect_pieces(
    frame_bgr: np.ndarray,
    *,
    weights_dir: Path | None = None,
    conf: float = 0.15,
    search_roi: np.ndarray | None = None,
) -> list[PieceDet]:
    """Detect white/black pieces on the full frame.

    The piece model is trained on full frames. A board crop changes the scale
    and used to disagree with the full-frame count, so it is not run.
    ``search_roi`` is accepted and ignored; occupancy still rejects centers
    that fall outside the board quad.
    """
    del search_roi
    weights_dir = Path(weights_dir) if weights_dir is not None else default_weights_dir()
    onnx = weights_dir / "pieces.onnx"
    if not onnx.exists():
        print(f"piece detect: missing {onnx}")
        return []

    try:
        dets, peak = detect_pieces_onnx(frame_bgr, onnx, conf=conf)
        dets, dropped = suppress_duplicate_pieces(dets)
        frame_wh = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        dets = _with_standing_points(dets, frame_wh)
        global _ZERO_LOG_I, _LAST_DET_LOG
        if not dets:
            _ZERO_LOG_I += 1
            _LAST_DET_LOG = None
            if _ZERO_LOG_I <= 3 or _ZERO_LOG_I % 25 == 0:
                print(
                    f"piece detect: 0 boxes (conf≥{conf:.2f}, peak_score≈{peak:.3f}, "
                    f"frame={frame_wh[0]}x{frame_wh[1]})"
                )
        else:
            _ZERO_LOG_I = 0
            sig = (len(dets), dropped)
            if sig != _LAST_DET_LOG:
                _LAST_DET_LOG = sig
                extra = f", dropped {dropped} overlaps" if dropped else ""
                print(f"piece detect: {len(dets)} boxes (peak≈{peak:.3f}{extra})")
        return dets
    except ImportError:
        print("piece detect: onnxruntime missing")
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"piece onnx failed ({exc})")
        return []


def pieces_available(weights_dir: Path | None = None) -> bool:
    weights_dir = Path(weights_dir) if weights_dir is not None else default_weights_dir()
    return (weights_dir / "pieces.onnx").exists()
