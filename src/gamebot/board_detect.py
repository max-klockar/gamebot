"""Physical board quad: YOLO AABB (ONNX) + classical corner refine.

Pi-friendly: uses `board.onnx` + onnxruntime (no Torch / Ultralytics).
Laptop may still fall back to `board.pt` via ultralytics if ONNX is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gamebot.calibration import detect_chessboard_outer_corners, order_quad_tl_tr_br_bl, quad_looks_sane
from gamebot.paths import gamebot_home

_ONNX_SESSION = None
_ONNX_PATH: Path | None = None


@dataclass(frozen=True)
class BoardQuadResult:
    quad: np.ndarray  # (4,2) TL,TR,BR,BL full-image coords
    method: str
    aabb: tuple[int, int, int, int]  # x0,y0,x1,y1 ROI


def default_weights_dir() -> Path:
    env = os.environ.get("GAMEBOT_VISION_WEIGHTS")
    if env:
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parents[2]
    cand = repo / "models" / "vision"
    if any((cand / n).exists() for n in ("board.onnx", "board.pt", "board_pose.pt")):
        return cand
    return gamebot_home() / "models" / "vision"


def _letterbox(im: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    dw, dh = new_shape - nw, new_shape - nh
    left = int(round(dw / 2 - 0.1))
    top = int(round(dh / 2 - 0.1))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, r, left, top


def _onnx_session(weights: Path):
    global _ONNX_SESSION, _ONNX_PATH
    weights = weights.resolve()
    if _ONNX_SESSION is not None and _ONNX_PATH == weights:
        return _ONNX_SESSION
    import onnxruntime as ort

    _ONNX_SESSION = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
    _ONNX_PATH = weights
    return _ONNX_SESSION


def _onnx_pred_aabb(
    sess,
    frame_bgr: np.ndarray,
    *,
    conf: float,
    pad_frac: float,
    imgsz: int,
    swap_rb: bool,
) -> tuple[tuple[int, int, int, int] | None, float]:
    """Run one ONNX forward. Returns (aabb_or_None, max_conf)."""
    lb, r, left, top = _letterbox(frame_bgr, imgsz)
    rgb = lb[:, :, ::-1] if swap_rb else lb
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    if pred.ndim != 2 or pred.shape[0] < 5:
        return None, 0.0
    scores = pred[4]
    max_conf = float(scores.max()) if scores.size else 0.0
    mask = scores >= conf
    if not np.any(mask):
        return None, max_conf
    boxes = pred[:4, mask].T
    scores_m = scores[mask]
    cx, cy, bw, bh = boxes.T
    x1 = (cx - bw / 2 - left) / r
    y1 = (cy - bh / 2 - top) / r
    w = bw / r
    h = bh / r
    idxs = cv2.dnn.NMSBoxes(
        bboxes=np.stack([x1, y1, w, h], axis=1).tolist(),
        scores=scores_m.tolist(),
        score_threshold=conf,
        nms_threshold=0.45,
    )
    if idxs is None or len(idxs) == 0:
        return None, max_conf
    i = int(np.array(idxs).reshape(-1)[0])
    xa, ya, wa, ha = float(x1[i]), float(y1[i]), float(w[i]), float(h[i])
    pad_x, pad_y = wa * pad_frac, ha * pad_frac
    fh, fw = frame_bgr.shape[:2]
    aabb = (
        max(0, int(xa - pad_x)),
        max(0, int(ya - pad_y)),
        min(fw, int(xa + wa + pad_x)),
        min(fh, int(ya + ha + pad_y)),
    )
    return aabb, max_conf


def yolo_board_aabb_onnx(
    weights: Path,
    frame_bgr: np.ndarray,
    *,
    conf: float = 0.08,
    pad_frac: float = 0.12,
    imgsz: int = 640,
) -> tuple[int, int, int, int] | None:
    """YOLOv8 detect ONNX → padded AABB. Tries BGR→RGB and raw (Picamera quirks)."""
    sess = _onnx_session(weights)
    best: tuple[int, int, int, int] | None = None
    best_conf = -1.0
    for swap in (True, False):
        aabb, max_c = _onnx_pred_aabb(
            sess, frame_bgr, conf=conf, pad_frac=pad_frac, imgsz=imgsz, swap_rb=swap
        )
        if max_c > best_conf:
            best_conf = max_c
        if aabb is not None:
            return aabb
        best = best or aabb
    return best


def yolo_board_aabb_ultralytics(
    weights: Path,
    frame_bgr: np.ndarray,
    *,
    conf: float = 0.25,
    pad_frac: float = 0.12,
) -> tuple[int, int, int, int] | None:
    from ultralytics import YOLO

    res = YOLO(str(weights)).predict(frame_bgr, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    bi = int(res.boxes.conf.cpu().numpy().argmax())
    x1, y1, x2, y2 = map(float, res.boxes.xyxy.cpu().numpy()[bi])
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    h, w = frame_bgr.shape[:2]
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(w, int(x2 + pad_x)),
        min(h, int(y2 + pad_y)),
    )


def yolo_board_aabb(
    frame_bgr: np.ndarray,
    *,
    weights_dir: Path,
    conf: float = 0.12,
    pad_frac: float = 0.12,
) -> tuple[int, int, int, int] | None:
    """Prefer board.onnx (Pi); optional .pt via ultralytics on laptop."""
    onnx = weights_dir / "board.onnx"
    if onnx.exists():
        try:
            return yolo_board_aabb_onnx(onnx, frame_bgr, conf=conf, pad_frac=pad_frac)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"board onnx failed ({exc}); trying .pt / classical")
    for name in ("board.pt", "board_pose.pt"):
        pt = weights_dir / name
        if not pt.exists():
            continue
        try:
            return yolo_board_aabb_ultralytics(pt, frame_bgr, conf=max(conf, 0.2), pad_frac=pad_frac)
        except ImportError:
            break
        except Exception as exc:  # noqa: BLE001
            print(f"board pt failed ({exc})")
            break
    return None


def refine_wood_contour(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Non-table (wood) mask → quad. Fallback when pieces block the inner grid."""
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    table = cv2.inRange(hsv, (78, 35, 60), (125, 255, 255))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    board = cv2.bitwise_and(cv2.bitwise_not(table), (gray > 35).astype(np.uint8) * 255)
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    board = cv2.morphologyEx(board, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.12 * w * h:
        return None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = approx.reshape(4, 2).astype(np.float64)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64)
    ordered = order_quad_tl_tr_br_bl(quad)
    if not quad_looks_sane(ordered, (w, h), min_area=0.12 * w * h, min_edge=min(w, h) * 0.2):
        return None
    # Reject tall/skinny masks (table edge, projected strip) — board is ~square.
    edges = [float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])) for i in range(4)]
    if max(edges) / max(min(edges), 1e-6) > 1.55:
        return None
    c = ordered.mean(axis=0)
    return ordered + (c - ordered) * 0.035


def _ang_diff(a: float, b: float) -> float:
    return abs((a - b + np.pi / 2) % np.pi - np.pi / 2)


def _line_intersect(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> np.ndarray | None:
    A = np.array([d0, -d1], dtype=np.float64).T
    if abs(np.linalg.det(A)) < 1e-8:
        return None
    t, _s = np.linalg.solve(A, p1 - p0)
    return p0 + t * d0


def refine_hough_lines(crop_bgr: np.ndarray) -> np.ndarray | None:
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(gray, 60, 160, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60, minLineLength=min(w, h) * 0.25, maxLineGap=12
    )
    if lines is None or len(lines) < 4:
        return None
    segs = []
    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, ln)
        ang = np.arctan2(y2 - y1, x2 - x1)
        ang = (ang + np.pi / 2) % np.pi - np.pi / 2
        length = float(np.hypot(x2 - x1, y2 - y1))
        segs.append((ang, length, np.array([x1, y1, x2, y2], dtype=np.float64)))
    segs.sort(key=lambda t: -t[1])
    segs = segs[:40]
    angles = np.array([s[0] for s in segs], dtype=np.float64)
    a0 = float(angles[0])
    g0 = [i for i, a in enumerate(angles) if _ang_diff(a, a0) < 0.3]
    rest = [i for i in range(len(angles)) if i not in g0]
    if not rest:
        return None
    a1 = float(np.median(angles[rest]))
    g1 = [i for i in rest if _ang_diff(float(angles[i]), a1) < 0.35]
    if len(g0) < 2 or len(g1) < 2:
        return None

    def _outer_pair(idxs: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
        reps = []
        for i in idxs:
            x1, y1, x2, y2 = segs[i][2]
            p = np.array([x1, y1], dtype=np.float64)
            d = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            n = np.linalg.norm(d)
            if n < 1:
                continue
            d /= n
            nn = np.array([-d[1], d[0]])
            reps.append((float(np.dot(p, nn)), p, d))
        if len(reps) < 2:
            return []
        lo = min(reps, key=lambda t: t[0])
        hi = max(reps, key=lambda t: t[0])
        return [(lo[1], lo[2]), (hi[1], hi[2])]

    L0, L1 = _outer_pair(g0), _outer_pair(g1)
    if len(L0) < 2 or len(L1) < 2:
        return None
    corners = []
    for p0, d0 in L0:
        for p1, d1 in L1:
            pt = _line_intersect(p0, d0, p1, d1)
            if pt is None:
                return None
            corners.append(pt)
    quad = order_quad_tl_tr_br_bl(np.array(corners, dtype=np.float64))
    if not quad_looks_sane(quad, (w, h), min_area=0.12 * w * h, min_edge=min(w, h) * 0.2):
        return None
    return quad


def refine_board_quad(
    frame_bgr: np.ndarray,
    aabb: tuple[int, int, int, int],
    *,
    allow_wood: bool = True,
) -> tuple[np.ndarray, str] | None:
    x0, y0, x1, y1 = aabb
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 40:
        return None
    offset = np.array([x0, y0], dtype=np.float64)

    try:
        cb = detect_chessboard_outer_corners(crop, verbose=False)
    except Exception:
        cb = None
    if cb is not None:
        return cb + offset, "chessboard"

    if not allow_wood:
        return None
    for name, fn in (("wood_contour", refine_wood_contour), ("hough", refine_hough_lines)):
        try:
            local = fn(crop)
        except Exception:
            local = None
        if local is not None:
            return local + offset, name
    return None


def detect_physical_board_quad(
    frame_bgr: np.ndarray,
    *,
    weights_dir: Path | None = None,
    conf: float = 0.05,
    pad_frac: float = 0.12,
    allow_wood: bool = False,
) -> BoardQuadResult | None:
    """ONNX/YOLO AABB → classical refine. Falls back to full-frame chessboard.

    `allow_wood=False` by default — wood_contour flaps badly under projection.
    """
    weights_dir = Path(weights_dir) if weights_dir is not None else default_weights_dir()
    aabb = yolo_board_aabb(frame_bgr, weights_dir=weights_dir, conf=conf, pad_frac=pad_frac)

    if aabb is not None:
        refined = refine_board_quad(frame_bgr, aabb, allow_wood=allow_wood)
        if refined is not None:
            quad, method = refined
            return BoardQuadResult(quad=quad, method=method, aabb=aabb)
        # Don't return raw AABB — it's axis-aligned and looks like a "stuck square".
        # Fall through to full-frame chessboard.

    try:
        full = detect_chessboard_outer_corners(frame_bgr, verbose=False)
    except Exception:
        full = None
    if full is None:
        return None
    h, w = frame_bgr.shape[:2]
    return BoardQuadResult(quad=full, method="chessboard_full", aabb=(0, 0, w, h))
