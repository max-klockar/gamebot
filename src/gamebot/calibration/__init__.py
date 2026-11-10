from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gamebot.geometry import TableLayout, warp_points


@dataclass
class CalibrationData:
    """Homographies linking table mm, camera pixels, and projector pixels."""

    H_table_to_camera: np.ndarray
    H_table_to_projector: np.ndarray
    H_camera_to_projector: np.ndarray
    board_corners_camera: np.ndarray  # 4x2
    projector_size: tuple[int, int]
    camera_size: tuple[int, int]

    def to_json(self) -> dict:
        return {
            "H_table_to_camera": self.H_table_to_camera.tolist(),
            "H_table_to_projector": self.H_table_to_projector.tolist(),
            "H_camera_to_projector": self.H_camera_to_projector.tolist(),
            "board_corners_camera": self.board_corners_camera.tolist(),
            "projector_size": list(self.projector_size),
            "camera_size": list(self.camera_size),
        }

    @classmethod
    def from_json(cls, data: dict) -> CalibrationData:
        return cls(
            H_table_to_camera=np.array(data["H_table_to_camera"], dtype=np.float64),
            H_table_to_projector=np.array(data["H_table_to_projector"], dtype=np.float64),
            H_camera_to_projector=np.array(data["H_camera_to_projector"], dtype=np.float64),
            board_corners_camera=np.array(data["board_corners_camera"], dtype=np.float64),
            projector_size=tuple(data["projector_size"]),
            camera_size=tuple(data["camera_size"]),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> CalibrationData:
        return cls.from_json(json.loads(Path(path).read_text()))


# BGR draw colours + HSV gates for camera matching (flip-proof correspondence).
# Order: TL, TR, BR, BL
COLOR_CORNERS: list[tuple[str, tuple[int, int, int], list[tuple[tuple[int, int, int], tuple[int, int, int]]]]] = [
    (
        "TL",
        (0, 0, 255),  # red
        [((0, 120, 80), (10, 255, 255)), ((170, 120, 80), (180, 255, 255))],
    ),
    (
        "TR",
        (255, 80, 0),  # blue-ish
        [((100, 120, 80), (130, 255, 255))],
    ),
    (
        "BR",
        (0, 255, 255),  # yellow
        [((20, 100, 100), (40, 255, 255))],
    ),
    (
        "BL",
        (0, 255, 0),  # green
        [((40, 80, 80), (85, 255, 255))],
    ),
]


def make_grid_pattern(
    size: tuple[int, int],
    cols: int = 6,
    rows: int = 4,
    margin: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """White dots on black; returns (image, Nx2 projector points)."""
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xs = np.linspace(margin, w - margin, cols)
    ys = np.linspace(margin, h - margin, rows)
    points = []
    for y in ys:
        for x in xs:
            cv2.circle(img, (int(x), int(y)), 12, (255, 255, 255), -1)
            points.append([x, y])
    return img, np.array(points, dtype=np.float64)


def default_projector_quad(size: tuple[int, int]) -> np.ndarray:
    """Fixed projector-space rectangle used as coloured calibration dots (TL,TR,BR,BL)."""
    w, h = size
    x0, x1 = w * 0.20, w * 0.80
    y0, y1 = h * 0.16, h * 0.84
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def render_color_corner_pattern(size: tuple[int, int], points: np.ndarray | None = None) -> np.ndarray:
    """Black frame + 4 saturated coloured dots (and labels)."""
    w, h = size
    pts = default_projector_quad(size) if points is None else points
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for i, ((label, bgr, _), (x, y)) in enumerate(zip(COLOR_CORNERS, pts)):
        pt = (int(x), int(y))
        cv2.circle(frame, pt, 28, bgr, -1)
        cv2.circle(frame, pt, 32, (255, 255, 255), 2)
        cv2.putText(frame, label, (pt[0] + 36, pt[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 2)
    cv2.putText(frame, "auto calibrate", (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    return frame


def detect_dots(frame_bgr: np.ndarray, expected: int) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thr = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 8000:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        centers.append([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    if len(centers) < max(4, expected // 2):
        return None
    return np.array(centers, dtype=np.float64)


def _largest_blob_center(mask: np.ndarray) -> tuple[float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 40.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < best_area or area > 30000:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        best_area = area
        best = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    return best


def detect_color_corners(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Return 4x2 camera points in TL,TR,BR,BL colour order, or None if any colour missing."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pts = []
    for _label, _bgr, ranges in COLOR_CORNERS:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        center = _largest_blob_center(mask)
        if center is None:
            return None
        pts.append(center)
    return np.array(pts, dtype=np.float64)


def detect_chessboard_outer_corners(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Outer board quad (TL,TR,BR,BL) from 7x7 inner chessboard corners, if visible."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (7, 7), flags)
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
    )
    grid = corners.reshape(7, 7, 2)
    # Extrapolate one square beyond the inner 7x7 to approximate the outer rim.
    def extrap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + (a - b)

    tl = extrap(grid[0, 0], grid[1, 1])
    tr = extrap(grid[0, 6], grid[1, 5])
    br = extrap(grid[6, 6], grid[5, 5])
    bl = extrap(grid[6, 0], grid[5, 1])
    return np.array([tl, tr, br, bl], dtype=np.float64)


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    if len(src) < 4 or len(dst) < 4:
        return None
    n = min(len(src), len(dst))
    H, _mask = cv2.findHomography(src[:n], dst[:n], 0)
    return H


def order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into top-left, top-right, bottom-right, bottom-left."""
    if len(pts) < 4:
        raise ValueError("need 4 points")
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)[:4]
    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bot = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float64)


def render_verify_overlay(size: tuple[int, int], cal: CalibrationData, layout: TableLayout) -> np.ndarray:
    """Projector frame: board outline + 8x8 grid — should sit on the physical board if cal is good."""
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    corners = warp_points(layout.board_corners(), cal.H_table_to_projector).astype(np.int32)
    cv2.polylines(frame, [corners], True, (0, 255, 0), 4)
    ox, oy = layout.board_origin
    s = layout.board_size_mm
    for i in range(9):
        # vertical
        a = warp_points(np.array([[ox + i * s / 8, oy], [ox + i * s / 8, oy + s]]), cal.H_table_to_projector)
        b = warp_points(np.array([[ox, oy + i * s / 8], [ox + s, oy + i * s / 8]]), cal.H_table_to_projector)
        cv2.line(frame, tuple(a[0].astype(int)), tuple(a[1].astype(int)), (0, 200, 0), 2)
        cv2.line(frame, tuple(b[0].astype(int)), tuple(b[1].astype(int)), (0, 200, 0), 2)
    cv2.putText(frame, "VERIFY: grid should match board", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame


def annotate_camera_debug(
    frame_bgr: np.ndarray,
    color_pts: np.ndarray | None,
    board_pts: np.ndarray | None,
) -> np.ndarray:
    out = frame_bgr.copy()
    if color_pts is not None:
        for i, (x, y) in enumerate(color_pts):
            bgr = COLOR_CORNERS[i][1]
            cv2.circle(out, (int(x), int(y)), 18, bgr, 3)
            cv2.putText(out, COLOR_CORNERS[i][0], (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
    if board_pts is not None:
        cv2.polylines(out, [board_pts.astype(np.int32)], True, (0, 255, 0), 3)
    return out


@dataclass
class AutoCalibrateResult:
    calibration: CalibrationData
    camera_debug_bgr: np.ndarray
    verify_projector_bgr: np.ndarray
    used_chessboard: bool
    message: str


class Calibrator:
    """Project a known pattern, observe with camera, solve mappings."""

    def __init__(self, layout: TableLayout, projector_size: tuple[int, int], camera_size: tuple[int, int]) -> None:
        self.layout = layout
        self.projector_size = projector_size
        self.camera_size = camera_size

    def pattern(self) -> tuple[np.ndarray, np.ndarray]:
        return make_grid_pattern(self.projector_size)

    def color_pattern(self) -> tuple[np.ndarray, np.ndarray]:
        pts = default_projector_quad(self.projector_size)
        return render_color_corner_pattern(self.projector_size, pts), pts

    def auto_calibrate(self, camera_frame_bgr: np.ndarray) -> AutoCalibrateResult | None:
        """Solve from coloured projector dots (+ chessboard outer quad when visible)."""
        proj_pts = default_projector_quad(self.projector_size)
        color_cam = detect_color_corners(camera_frame_bgr)
        board_cam = detect_chessboard_outer_corners(camera_frame_bgr)
        used_board = board_cam is not None

        if color_cam is None:
            return None

        # Colour order is fixed TL..BL — do not spatial-resort (flip-safe).
        H_cam_to_proj = estimate_homography(color_cam, proj_pts)
        if H_cam_to_proj is None:
            return None

        table = self.layout.board_corners()
        if board_cam is not None:
            board_cam = order_quad_tl_tr_br_bl(board_cam)
            H_table_to_camera = estimate_homography(table, board_cam)
        else:
            # No board find: invent board as the quad of coloured dots warped… use dots as proxy
            # only if they roughly form the board (weak). Prefer centered fallback.
            w, h = self.camera_size
            side = min(w, h) * 0.35
            x0 = (w - side) / 2
            y0 = (h - side) / 2
            board_cam = np.array(
                [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
                dtype=np.float64,
            )
            H_table_to_camera = estimate_homography(table, board_cam)

        if H_table_to_camera is None:
            return None
        H_table_to_projector = H_cam_to_proj @ H_table_to_camera
        cal = CalibrationData(
            H_table_to_camera=H_table_to_camera,
            H_table_to_projector=H_table_to_projector,
            H_camera_to_projector=H_cam_to_proj,
            board_corners_camera=board_cam,
            projector_size=self.projector_size,
            camera_size=self.camera_size,
        )
        msg = "ok: colours + chessboard" if used_board else "ok: colours only (no chessboard — board pose approximate)"
        return AutoCalibrateResult(
            calibration=cal,
            camera_debug_bgr=annotate_camera_debug(camera_frame_bgr, color_cam, board_cam if used_board else None),
            verify_projector_bgr=render_verify_overlay(self.projector_size, cal, self.layout),
            used_chessboard=used_board,
            message=msg,
        )

    def solve_from_correspondences(
        self,
        projector_pts: np.ndarray,
        camera_pts: np.ndarray,
        board_corners_camera: np.ndarray | None = None,
    ) -> CalibrationData | None:
        H_cam_to_proj = estimate_homography(camera_pts, projector_pts)
        if H_cam_to_proj is None:
            return None

        table_corners = self.layout.board_corners()
        if board_corners_camera is None:
            w, h = self.camera_size
            side = min(w, h) * 0.55
            x0 = (w - side) / 2
            y0 = (h - side) / 2
            board_corners_camera = np.array(
                [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
                dtype=np.float64,
            )

        H_table_to_camera = estimate_homography(table_corners, board_corners_camera)
        if H_table_to_camera is None:
            return None

        H_table_to_projector = H_cam_to_proj @ H_table_to_camera
        return CalibrationData(
            H_table_to_camera=H_table_to_camera,
            H_table_to_projector=H_table_to_projector,
            H_camera_to_projector=H_cam_to_proj,
            board_corners_camera=board_corners_camera,
            projector_size=self.projector_size,
            camera_size=self.camera_size,
        )

    def solve_from_board_corners(
        self,
        projector_corners: np.ndarray,
        camera_corners: np.ndarray,
    ) -> CalibrationData | None:
        table = self.layout.board_corners()
        proj = order_quad_tl_tr_br_bl(projector_corners)
        cam = order_quad_tl_tr_br_bl(camera_corners)
        H_table_to_projector = estimate_homography(table, proj)
        H_table_to_camera = estimate_homography(table, cam)
        H_camera_to_projector = estimate_homography(cam, proj)
        if H_table_to_projector is None or H_table_to_camera is None or H_camera_to_projector is None:
            return None
        return CalibrationData(
            H_table_to_camera=H_table_to_camera,
            H_table_to_projector=H_table_to_projector,
            H_camera_to_projector=H_camera_to_projector,
            board_corners_camera=cam,
            projector_size=self.projector_size,
            camera_size=self.camera_size,
        )

    def mock_calibrate(self) -> CalibrationData:
        pattern, proj_pts = self.pattern()
        w, h = self.camera_size
        pw, ph = self.projector_size
        scale = min(w / pw, h / ph) * 0.9
        cam_pts = proj_pts * scale
        cam_pts[:, 0] += (w - pw * scale) / 2
        cam_pts[:, 1] += (h - ph * scale) / 2
        return self.solve_from_correspondences(proj_pts, cam_pts)

    def verify_alignment(
        self,
        cal: CalibrationData,
        camera_frame: np.ndarray,
        projected_lines: np.ndarray,
        max_mean_error_px: float = 12.0,
    ) -> bool:
        expected = detect_dots(projected_lines, expected=24)
        observed = detect_dots(camera_frame, expected=24)
        if expected is None or observed is None:
            return False
        n = min(len(expected), len(observed))
        return n >= 8
