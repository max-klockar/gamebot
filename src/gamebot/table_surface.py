from __future__ import annotations

"""Physical table detection + virtual projected table (with ChArUco anchors)."""

import cv2
import numpy as np

from gamebot.geometry import TableLayout, warp_points

# Four ArUco ids placed in the table margins (outside the chessboard).
_CORNER_MARKER_IDS = (0, 1, 2, 3)
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def table_corners_mm(layout: TableLayout) -> np.ndarray:
    tw, th = layout.table_size
    return np.array([[0.0, 0.0], [tw, 0.0], [tw, th], [0.0, th]], dtype=np.float64)


def detect_physical_table(frame_bgr: np.ndarray) -> bool:
    """True if a *real* play surface is in view (chessboard or large table rectangle).

    Ignores ArUco alone — those may be our own projected anchors.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for pattern in ((7, 7), (7, 3), (5, 5)):
        found, _ = cv2.findChessboardCorners(gray, pattern, flags)
        if found:
            return True
    # Large bright rectangular region (physical table under light) — not the whole frame.
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    min_area = 0.18 * w * h
    max_area = 0.82 * w * h
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(approx)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect < 2.8:  # table-ish, not a long strip
            return True
    return False


def _wood_fill(poly_px: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warm wood-ish table top clipped to the table polygon."""
    w, h = size
    base = np.zeros((h, w, 3), dtype=np.uint8)
    # Soft walnut tone (BGR)
    cv2.fillPoly(base, [poly_px], (42, 78, 118))
    noise = np.random.default_rng(0).integers(0, 18, (h, w, 1), dtype=np.uint8)
    wood = cv2.add(base, cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR))
    # Gentle grain lines
    for y in range(0, h, 7):
        shade = 8 if (y // 7) % 2 == 0 else -6
        wood[y : y + 2] = np.clip(wood[y : y + 2].astype(np.int16) + shade, 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_px], 255)
    out = np.zeros_like(wood)
    out[mask > 0] = wood[mask > 0]
    return out


def _draw_projected_chessboard(frame: np.ndarray, layout: TableLayout, H: np.ndarray) -> None:
    ox, oy = layout.board_origin
    s = layout.square_mm
    light = (210, 230, 245)
    dark = (70, 110, 40)
    for r in range(8):
        for c in range(8):
            x0, y0 = ox + c * s, oy + r * s
            quad = np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]], dtype=np.float64)
            pts = warp_points(quad, H).astype(np.int32)
            color = light if (r + c) % 2 == 0 else dark
            cv2.fillPoly(frame, [pts], color)
    border = warp_points(layout.board_corners(), H).astype(np.int32)
    cv2.polylines(frame, [border], True, (20, 20, 20), 3)


def _blit_marker(frame: np.ndarray, marker_gray: np.ndarray, dest_quad_px: np.ndarray) -> None:
    """Perspective-paste a square marker into dest_quad (TL,TR,BR,BL)."""
    side = marker_gray.shape[0]
    src = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
    dst = dest_quad_px.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(marker_gray, H, (frame.shape[1], frame.shape[0]))
    color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    mask = warped > 10
    frame[mask] = color[mask]


def _margin_marker_quads_mm(layout: TableLayout, marker_mm: float = 55.0) -> list[np.ndarray]:
    """One square per table corner, in the margin outside the chessboard."""
    tw, th = layout.table_size
    m = marker_mm
    pad = 18.0
    return [
        np.array([[pad, pad], [pad + m, pad], [pad + m, pad + m], [pad, pad + m]], dtype=np.float64),
        np.array([[tw - pad - m, pad], [tw - pad, pad], [tw - pad, pad + m], [tw - pad - m, pad + m]], dtype=np.float64),
        np.array(
            [[tw - pad - m, th - pad - m], [tw - pad, th - pad - m], [tw - pad, th - pad], [tw - pad - m, th - pad]],
            dtype=np.float64,
        ),
        np.array([[pad, th - pad - m], [pad + m, th - pad - m], [pad + m, th - pad], [pad, th - pad]], dtype=np.float64),
    ]


def draw_virtual_table(
    frame: np.ndarray,
    layout: TableLayout,
    H: np.ndarray,
    *,
    with_chessboard: bool = True,
    with_charuco_anchors: bool = True,
) -> np.ndarray:
    """Paint a full virtual table into an existing projector frame (in place)."""
    table_px = warp_points(table_corners_mm(layout), H).astype(np.int32)
    wood = _wood_fill(table_px, (frame.shape[1], frame.shape[0]))
    mask = np.any(wood > 0, axis=2)
    frame[mask] = wood[mask]
    cv2.polylines(frame, [table_px], True, (30, 50, 70), 3)

    if with_chessboard:
        _draw_projected_chessboard(frame, layout, H)

    if with_charuco_anchors:
        for mid, quad_mm in zip(_CORNER_MARKER_IDS, _margin_marker_quads_mm(layout)):
            marker = cv2.aruco.generateImageMarker(_ARUCO_DICT, int(mid), 120)
            # White border so detection works on wood
            bordered = cv2.copyMakeBorder(marker, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            dest = warp_points(quad_mm, H)
            _blit_marker(frame, bordered, dest)

    return frame
