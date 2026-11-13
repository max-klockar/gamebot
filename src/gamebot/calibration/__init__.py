from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gamebot.geometry import TableLayout, force_square_quad, warp_points


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
        [((0, 80, 60), (12, 255, 255)), ((168, 80, 60), (180, 255, 255))],
    ),
    (
        "TR",
        (255, 60, 0),  # blue
        [((95, 80, 60), (135, 255, 255))],
    ),
    (
        "BR",
        (0, 255, 255),  # yellow
        [((18, 80, 60), (42, 255, 255))],
    ),
    (
        "BL",
        (0, 255, 0),  # green
        [((40, 60, 60), (90, 255, 255))],
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


def projector_quad(
    size: tuple[int, int],
    cx: float | None = None,
    cy: float | None = None,
    scale: float = 0.32,
) -> np.ndarray:
    """Axis-aligned square TL,TR,BR,BL. scale=1 spans nearly the short frame edge."""
    w, h = size
    cx = w * 0.5 if cx is None else cx
    cy = h * 0.5 if cy is None else cy
    # Always square — using w and h separately made a 16:9 rectangle that looked
    # like a stretched chessboard once mapped onto the table.
    half = min(w, h) * 0.45 * scale
    half = min(half, cx - 20, w - 20 - cx, cy - 20, h - 20 - cy)
    half = max(half, 40)
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )


def default_projector_quad(size: tuple[int, int]) -> np.ndarray:
    """Tighter centered default — full-bleed often misses a downward table."""
    return projector_quad(size, scale=0.32)


def render_color_corner_pattern(
    size: tuple[int, int],
    points: np.ndarray | None = None,
    status: str = "auto calibrate",
) -> np.ndarray:
    """Black frame + 4 saturated coloured dots (and labels)."""
    w, h = size
    pts = default_projector_quad(size) if points is None else points
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for i, ((label, bgr, _), (x, y)) in enumerate(zip(COLOR_CORNERS, pts)):
        pt = (int(x), int(y))
        cv2.circle(frame, pt, 34, bgr, -1)
        cv2.circle(frame, pt, 38, (255, 255, 255), 2)
        cv2.putText(frame, label, (pt[0] + 40, pt[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 2)
    cv2.putText(frame, status, (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
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


def _largest_blob_center(mask: np.ndarray, min_area: float = 40.0) -> tuple[float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = min_area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < best_area or area > 50000:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        best_area = area
        best = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    return best


def detect_bright_dot(frame_bgr: np.ndarray, min_area: float = 80.0) -> tuple[float, float] | None:
    """Find the brightest compact blob — robust when projector colours blow out to white."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    # Top percentile threshold so we track projector, not room lights
    thr_val = float(np.percentile(blur, 99.2))
    thr_val = max(thr_val, 180.0)
    _, mask = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return _largest_blob_center(mask, min_area=min_area)


def detect_dot_by_difference(
    dark_bgr: np.ndarray,
    lit_bgr: np.ndarray,
    min_area: float = 40.0,
    min_peak: float = 25.0,
) -> tuple[float, float] | None:
    """Locate a projected dot as the lit−dark peak (ignores lamps/chairs that stay on)."""
    dark = cv2.cvtColor(dark_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lit = cv2.cvtColor(lit_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = cv2.GaussianBlur(np.clip(lit - dark, 0, 255), (11, 11), 0)
    peak = float(diff.max())
    if peak < min_peak:
        return None
    thr = max(min_peak, peak * 0.45)
    _, mask = cv2.threshold(diff.astype(np.uint8), thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return _largest_blob_center(mask, min_area=min_area)


def quad_looks_sane(
    pts: np.ndarray,
    image_size: tuple[int, int] | None = None,
    *,
    min_area: float = 8000.0,
    min_edge: float = 40.0,
    max_out_frac: float = 0.08,
) -> bool:
    """Reject collapsed / crossed / off-image quads that blow up homographies."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4 or not np.isfinite(pts).all():
        return False
    ordered = order_quad_tl_tr_br_bl(pts)
    area = abs(float(cv2.contourArea(ordered.astype(np.float32))))
    if area < min_area:
        return False
    if not cv2.isContourConvex(ordered.astype(np.float32)):
        return False
    for i in range(4):
        d = float(np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]))
        if d < min_edge:
            return False
    # Diagonals should be somewhat similar (not a dart / spike).
    d1 = float(np.linalg.norm(ordered[0] - ordered[2]))
    d2 = float(np.linalg.norm(ordered[1] - ordered[3]))
    if min(d1, d2) < 1e-3 or max(d1, d2) / min(d1, d2) > 2.8:
        return False
    if image_size is not None:
        w, h = image_size
        margin_x, margin_y = w * max_out_frac, h * max_out_frac
        if np.any(ordered[:, 0] < -margin_x) or np.any(ordered[:, 0] > w + margin_x):
            return False
        if np.any(ordered[:, 1] < -margin_y) or np.any(ordered[:, 1] > h + margin_y):
            return False
    return True


def detect_color_corners_partial(
    frame_bgr: np.ndarray,
) -> list[tuple[float, float] | None]:
    """Per-colour camera centers (TL,TR,BR,BL); None if that colour not found.

    Prefer bright+hue; fall back to hue-only. Still weak if all four are shown at once
    and overexposed — prefer ``capture_corners_sequential``.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bright = cv2.threshold(gray, max(120, float(np.percentile(gray, 96))), 255, cv2.THRESH_BINARY)[1]
    out: list[tuple[float, float] | None] = []
    for _label, _bgr, ranges in COLOR_CORNERS:
        hue = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            hue = cv2.bitwise_or(hue, cv2.inRange(hsv, np.array(lo), np.array(hi)))
        hue = cv2.morphologyEx(hue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # Prefer hue∩bright (rejects laptop/clothes); fall back to hue for dim reds etc.
        center = _largest_blob_center(cv2.bitwise_and(hue, bright), min_area=50.0)
        if center is None:
            center = _largest_blob_center(hue, min_area=50.0)
        out.append(center)
    return out


def detect_color_corners(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Return 4x2 camera points in TL,TR,BR,BL colour order, or None if any colour missing."""
    partial = detect_color_corners_partial(frame_bgr)
    if any(p is None for p in partial):
        return None
    return np.array(partial, dtype=np.float64)


def render_single_color_dot(
    size: tuple[int, int],
    point: tuple[float, float],
    bgr: tuple[int, int, int],
    label: str,
    status: str = "",
) -> np.ndarray:
    """One coloured dot only — for sequential capture (avoids blow-out confusion)."""
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    pt = (int(point[0]), int(point[1]))
    cv2.circle(frame, pt, 40, bgr, -1)
    cv2.circle(frame, pt, 46, (255, 255, 255), 3)
    cv2.putText(frame, label, (pt[0] + 50, pt[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bgr, 2)
    if status:
        cv2.putText(frame, status, (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    return frame


def capture_corners_sequential(
    projector_size: tuple[int, int],
    proj_pts: np.ndarray,
    show_and_grab_frame,
    settle: int = 3,
) -> np.ndarray | None:
    """Show one colour at a time; locate each via lit−dark difference (not room lights).

    ``show_and_grab_frame(bgr_image) -> camera_bgr``.
    """
    w, h = projector_size
    black = np.zeros((h, w, 3), dtype=np.uint8)
    cam_pts: list[tuple[float, float]] = []
    for i, ((label, bgr, _), (x, y)) in enumerate(zip(COLOR_CORNERS, proj_pts)):
        dark = None
        for _ in range(settle):
            dark = show_and_grab_frame(black)
        assert dark is not None
        img = render_single_color_dot(
            projector_size, (float(x), float(y)), bgr, label, status=f"capture {label} ({i + 1}/4)"
        )
        lit = None
        for _ in range(settle):
            lit = show_and_grab_frame(img)
        assert lit is not None
        pt = detect_dot_by_difference(dark, lit)
        if pt is None:
            # Fallback: bright peak on lit frame, but still must move vs dark.
            lit_pt = detect_bright_dot(lit)
            dark_pt = detect_bright_dot(dark)
            if lit_pt is not None and (
                dark_pt is None or float(np.linalg.norm(np.array(lit_pt) - np.array(dark_pt))) > 25.0
            ):
                pt = lit_pt
        if pt is None:
            partial = detect_color_corners_partial(lit)
            pt = partial[i]
        if pt is None:
            print(f"sequential capture missed {label}")
            return None
        # Reject latching onto the same blob twice.
        for prev in cam_pts:
            if float(np.linalg.norm(np.array(pt) - np.array(prev))) < 30.0:
                print(f"sequential {label} too close to previous point — rejecting")
                return None
        print(f"sequential {label} @ camera ({pt[0]:.0f},{pt[1]:.0f})")
        cam_pts.append(pt)
    cam = np.array(cam_pts, dtype=np.float64)
    # Camera size unknown here; geometric checks without image bounds.
    if not quad_looks_sane(cam, min_area=4000.0, min_edge=35.0):
        print("sequential quad looks collapsed/crossed — rejecting")
        return None
    return cam

def _clamp_quad_center(size: tuple[int, int], cx: float, cy: float, scale: float) -> tuple[float, float, float]:
    w, h = size
    scale = float(np.clip(scale, 0.18, 0.95))
    half = min(w, h) * 0.45 * scale
    cx = float(np.clip(cx, 20 + half, w - 20 - half))
    cy = float(np.clip(cy, 20 + half, h - 20 - half))
    return cx, cy, scale


def _count_bright_blobs(frame_bgr: np.ndarray, max_n: int = 6) -> int:
    """How many projector-like bright blobs are in view (overexposure-safe)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    thr_val = max(170.0, float(np.percentile(blur, 98.5)))
    _, mask = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 60 <= area <= 40000:
            n += 1
    return min(n, max_n)


def search_visible_color_quad(
    projector_size: tuple[int, int],
    show_and_grab,
    show_frame,
    max_iters: int = 24,
    settle_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Hill-climb until ~4 bright dots are on-camera, then sequential colour capture.

    ``show_and_grab(points, status) -> frame`` — shows all 4 colours (search).
    ``show_frame(image) -> frame`` — shows an arbitrary projector image (sequential).
    """
    w, h = projector_size
    cx, cy, scale = w * 0.5, h * 0.5, 0.28
    best_count = -1
    best_state = (cx, cy, scale)
    step = min(w, h) * 0.045

    for it in range(max_iters):
        cx, cy, scale = _clamp_quad_center(projector_size, cx, cy, scale)
        pts = projector_quad(projector_size, cx, cy, scale)
        frame = None
        for _ in range(max(1, settle_frames)):
            frame = show_and_grab(pts, f"search {it + 1}/{max_iters} scale={scale:.2f}")
        assert frame is not None
        count = _count_bright_blobs(frame)
        print(f"search {it + 1}/{max_iters}  scale={scale:.2f}  bright_blobs≈{count}")

        if count > best_count:
            best_count = count
            best_state = (cx, cy, scale)

        if count >= 4:
            print("enough blobs on table — sequential colour capture…")
            cam = capture_corners_sequential(projector_size, pts, show_frame)
            if cam is not None:
                frame = show_and_grab(pts, "locked")
                return pts, cam, frame
            print("sequential failed — keep searching")

        # Steer: try grow + 4 shifts; pick max bright blob count
        candidates = [
            (cx, cy, min(0.95, scale * 1.15)),
            (cx + step, cy, scale),
            (cx - step, cy, scale),
            (cx, cy + step, scale),
            (cx, cy - step, scale),
            (cx + step, cy + step, scale),
            (cx - step, cy - step, scale),
        ]
        if count > 0:
            # Also try moving toward / away from bright centroid
            dots = detect_dots(frame, expected=4)
            if dots is not None and len(dots) > 0:
                fh, fw = frame.shape[:2]
                cam_c = dots.mean(axis=0)
                ox = (cam_c[0] - fw / 2) / (fw / 2)
                oy = (cam_c[1] - fh / 2) / (fh / 2)
                candidates.extend(
                    [
                        (cx + ox * step * 2, cy + oy * step * 2, scale),
                        (cx - ox * step * 2, cy - oy * step * 2, scale),
                    ]
                )

        ranked = []
        for ncx, ncy, nsc in candidates:
            ncx, ncy, nsc = _clamp_quad_center(projector_size, ncx, ncy, nsc)
            npts = projector_quad(projector_size, ncx, ncy, nsc)
            nframe = show_and_grab(npts, f"probe blobs")
            ncount = _count_bright_blobs(nframe)
            ranked.append((ncount, nsc, ncx, ncy))
        ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
        ncount, nsc, ncx, ncy = ranked[0]
        print(f"  → best probe bright_blobs≈{ncount} scale={nsc:.2f}")
        cx, cy, scale = ncx, ncy, nsc
        if ncount > best_count:
            best_count = ncount
            best_state = (cx, cy, scale)

    # Last attempt at best state with sequential
    cx, cy, scale = best_state
    pts = projector_quad(projector_size, cx, cy, scale)
    print(f"fallback sequential at scale={scale:.2f}")
    cam = capture_corners_sequential(projector_size, pts, show_frame)
    if cam is None:
        return None
    frame = show_and_grab(pts, "fallback locked")
    return pts, cam, frame


# Full empty board first; then starting-position center bands (ranks 3–6 free).
_CHESSBOARD_PATTERNS: tuple[tuple[int, int], ...] = (
    (7, 7),
    (7, 3),
    (3, 7),
    (5, 3),
    (3, 5),
    (7, 5),
    (5, 7),
    (5, 5),
    (3, 3),
)


def _refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    return cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
    )


def _find_chessboard_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    cols, rows = pattern
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            sb_flags |= cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=sb_flags)
        if found and corners is not None:
            return corners.reshape(-1, 1, 2).astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found or corners is None:
        return None
    return corners


def _outer_from_inner_grid(grid: np.ndarray, full_inner: tuple[int, int] = (7, 7)) -> np.ndarray:
    """Extrapolate outer board quad from a (possibly partial) inner-corner grid.

    `grid` is (rows, cols, 2). Partial detections are assumed centered on the board
    (empty middle ranks/files with pieces in starting position).
    """
    rows, cols = grid.shape[:2]
    full_rows, full_cols = full_inner
    if cols >= 2:
        step_h = np.mean(grid[:, 1:, :] - grid[:, :-1, :], axis=(0, 1))
    else:
        step_h = np.array([1.0, 0.0], dtype=np.float64)
    if rows >= 2:
        step_v = np.mean(grid[1:, :, :] - grid[:-1, :, :], axis=(0, 1))
    else:
        step_v = np.array([0.0, 1.0], dtype=np.float64)

    pad_top = (full_rows - rows) // 2
    pad_left = (full_cols - cols) // 2
    pad_bot = full_rows - rows - pad_top
    pad_right = full_cols - cols - pad_left

    tl = grid[0, 0] - (pad_top + 1) * step_v - (pad_left + 1) * step_h
    tr = grid[0, -1] - (pad_top + 1) * step_v + (pad_right + 1) * step_h
    br = grid[-1, -1] + (pad_bot + 1) * step_v + (pad_right + 1) * step_h
    bl = grid[-1, 0] + (pad_bot + 1) * step_v - (pad_left + 1) * step_h
    return np.array([tl, tr, br, bl], dtype=np.float64)


def detect_chessboard_outer_corners(
    frame_bgr: np.ndarray,
    *,
    prefer_full: bool = True,
) -> np.ndarray | None:
    """Outer board quad (TL,TR,BR,BL). Prefers full 7x7 (empty board), then center bands."""
    gray0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray0.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants = (gray0, clahe.apply(gray0), cv2.bitwise_not(gray0), cv2.bitwise_not(clahe.apply(gray0)))
    patterns = _CHESSBOARD_PATTERNS
    if prefer_full:
        # Empty board: try full / near-full before start-position center bands.
        patterns = ((7, 7), (7, 5), (5, 7), (5, 5)) + tuple(
            p for p in _CHESSBOARD_PATTERNS if p not in {(7, 7), (7, 5), (5, 7), (5, 5)}
        )

    for pattern in patterns:
        cols, rows = pattern
        for gray in variants:
            corners = _find_chessboard_corners(gray, pattern)
            if corners is None:
                continue
            corners = _refine_corners(gray0, corners)
            grid = corners.reshape(rows, cols, 2)
            outer = _outer_from_inner_grid(grid)
            # Partial patterns must land on a plausible board-sized quad in-frame.
            min_area = (w * h) * 0.02
            if quad_looks_sane(outer, (w, h), min_area=min_area, min_edge=min(w, h) * 0.08):
                print(f"chessboard locked pattern={cols}x{rows} inner corners")
                return order_quad_tl_tr_br_bl(outer)
    return None


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
    # Bright crosses on the four corner squares — easy to spot residual skew.
    sq = layout.square_mm
    corner_sq = np.array(
        [
            [ox + 0.5 * sq, oy + 0.5 * sq],
            [ox + 7.5 * sq, oy + 0.5 * sq],
            [ox + 7.5 * sq, oy + 7.5 * sq],
            [ox + 0.5 * sq, oy + 7.5 * sq],
        ],
        dtype=np.float64,
    )
    for p in warp_points(corner_sq, cal.H_table_to_projector):
        x, y = int(p[0]), int(p[1])
        cv2.drawMarker(frame, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=36, thickness=3)
    cv2.putText(frame, "VERIFY: grid + corner crosses on board", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame


def refine_with_board_corner_projections(
    projector_size: tuple[int, int],
    layout: TableLayout,
    cal: CalibrationData,
    board_cam: np.ndarray,
    show_frame,
    settle: int = 3,
) -> CalibrationData | None:
    """Project dots onto estimated board corners, measure lit−dark, re-solve projector map.

    Fixes residual skew when colour-dot H_cam_to_proj was slightly off (wide lens / angle).
    """
    table = layout.board_corners()
    # Sit dots on corner squares, not the outer rim.
    inset = 0.06
    t_center = table.mean(axis=0)
    table_inset = table + (t_center - table) * inset
    b_center = board_cam.mean(axis=0)
    board_inset = board_cam + (b_center - board_cam) * inset
    proj_pts = warp_points(table_inset, cal.H_table_to_projector)

    w, h = projector_size
    black = np.zeros((h, w, 3), dtype=np.uint8)
    cam_seen: list[tuple[float, float]] = []
    print("refine: projecting corner-square dots…")
    for i, ((label, bgr, _), (px, py)) in enumerate(zip(COLOR_CORNERS, proj_pts)):
        dark = None
        for _ in range(settle):
            dark = show_frame(black)
        assert dark is not None
        img = render_single_color_dot(
            projector_size, (float(px), float(py)), bgr, label, status=f"refine {label} ({i + 1}/4)"
        )
        lit = None
        for _ in range(settle):
            lit = show_frame(img)
        assert lit is not None
        pt = detect_dot_by_difference(dark, lit)
        if pt is None:
            print(f"refine missed {label}")
            return None
        err = float(np.linalg.norm(np.array(pt) - board_inset[i]))
        print(
            f"refine {label}: cam_seen=({pt[0]:.0f},{pt[1]:.0f}) "
            f"board=({board_inset[i, 0]:.0f},{board_inset[i, 1]:.0f}) err≈{err:.0f}px"
        )
        cam_seen.append(pt)
    cam_seen_arr = np.array(cam_seen, dtype=np.float64)
    H_cam_to_proj = estimate_homography(cam_seen_arr, proj_pts)
    if H_cam_to_proj is None:
        return None
    # Where to draw so light lands on true board corners:
    proj_for_board = warp_points(board_cam, H_cam_to_proj)
    if not quad_looks_sane(proj_for_board, projector_size, min_area=5000.0, min_edge=40.0):
        print("refine: projector quad not sane")
        return None
    H_table_to_projector = estimate_homography(table, proj_for_board)
    H_table_to_camera = estimate_homography(table, board_cam)
    if H_table_to_projector is None or H_table_to_camera is None:
        return None
    return CalibrationData(
        H_table_to_camera=H_table_to_camera,
        H_table_to_projector=H_table_to_projector,
        H_camera_to_projector=H_cam_to_proj,
        board_corners_camera=board_cam,
        projector_size=cal.projector_size,
        camera_size=cal.camera_size,
    )


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

    def auto_calibrate(
        self,
        camera_frame_bgr: np.ndarray,
        proj_pts: np.ndarray | None = None,
        color_cam: np.ndarray | None = None,
        board_frame_bgr: np.ndarray | None = None,
    ) -> AutoCalibrateResult | None:
        """Solve from coloured projector dots (+ chessboard outer quad when visible)."""
        if proj_pts is None:
            proj_pts = default_projector_quad(self.projector_size)
        if color_cam is None:
            color_cam = detect_color_corners(camera_frame_bgr)
        board_src = board_frame_bgr if board_frame_bgr is not None else camera_frame_bgr
        board_cam = detect_chessboard_outer_corners(board_src, prefer_full=True)
        used_board = board_cam is not None

        if color_cam is None:
            return None
        fh, fw = camera_frame_bgr.shape[:2]
        if not quad_looks_sane(color_cam, (fw, fh), min_area=(fw * fh) * 0.01, min_edge=30.0):
            print("colour quad not sane — refusing solve (would make spoke-grid)")
            return None

        # Colour order is fixed TL..BL — do not spatial-resort (flip-safe).
        H_cam_to_proj = estimate_homography(color_cam, proj_pts)
        if H_cam_to_proj is None:
            return None

        table = self.layout.board_corners()
        if board_cam is not None:
            board_cam = order_quad_tl_tr_br_bl(board_cam)
            if not quad_looks_sane(board_cam, (fw, fh), min_area=(fw * fh) * 0.02, min_edge=min(fw, fh) * 0.08):
                print("chessboard quad failed sanity — falling back to colours only")
                board_cam = None
                used_board = False

        if board_cam is not None:
            H_table_to_camera = estimate_homography(table, board_cam)
            H_table_to_projector = H_cam_to_proj @ H_table_to_camera if H_table_to_camera is not None else None
            msg = "ok: colours + full chessboard"
        else:
            # No chessboard: colour dots define the board. Force a square (shortest
            # edge) so a 16:9 frame never becomes a stretched projected board.
            board_cam = force_square_quad(color_cam)
            proj_pts = force_square_quad(proj_pts)
            H_table_to_camera = estimate_homography(table, board_cam)
            H_table_to_projector = estimate_homography(table, proj_pts)
            msg = "ok: colours only — verify grid on colour square (board contrast too low?)"

        if H_table_to_camera is None or H_table_to_projector is None:
            return None
        # Final check: projected board outline must be a sane quad in projector space.
        proj_board = warp_points(table, H_table_to_projector)
        if not quad_looks_sane(proj_board, self.projector_size, min_area=5000.0, min_edge=40.0):
            print("verify grid would be degenerate — refusing solve")
            return None
        debug_frame = board_src
        cal = CalibrationData(
            H_table_to_camera=H_table_to_camera,
            H_table_to_projector=H_table_to_projector,
            H_camera_to_projector=H_cam_to_proj,
            board_corners_camera=board_cam,
            projector_size=self.projector_size,
            camera_size=self.camera_size,
        )
        return AutoCalibrateResult(
            calibration=cal,
            camera_debug_bgr=annotate_camera_debug(debug_frame, color_cam, board_cam if used_board else color_cam),
            verify_projector_bgr=render_verify_overlay(self.projector_size, cal, self.layout),
            used_chessboard=used_board,
            message=msg,
        )

    def auto_calibrate_with_search(self, show_and_grab, show_frame) -> AutoCalibrateResult | None:
        """Search colour dots, full chessboard, then corner-projection refine."""
        found = search_visible_color_quad(self.projector_size, show_and_grab, show_frame)
        if found is None:
            return None
        proj_pts, color_cam, _frame = found
        w, h = self.projector_size
        black = np.zeros((h, w, 3), dtype=np.uint8)
        print("projector black — detecting full chessboard (pieces off helps)…")
        board_frame = None
        for _ in range(5):
            board_frame = show_frame(black)
        assert board_frame is not None
        result = self.auto_calibrate(
            board_frame, proj_pts=proj_pts, color_cam=color_cam, board_frame_bgr=board_frame
        )
        if result is None or not result.used_chessboard:
            return result
        refined = refine_with_board_corner_projections(
            self.projector_size,
            self.layout,
            result.calibration,
            result.calibration.board_corners_camera,
            show_frame,
        )
        if refined is None:
            print("corner refine skipped — keeping initial solve")
            return result
        print("corner refine applied")
        return AutoCalibrateResult(
            calibration=refined,
            camera_debug_bgr=annotate_camera_debug(
                board_frame, color_cam, refined.board_corners_camera
            ),
            verify_projector_bgr=render_verify_overlay(self.projector_size, refined, self.layout),
            used_chessboard=True,
            message=result.message + " + corner refine",
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
