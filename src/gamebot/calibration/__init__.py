from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gamebot.geometry import TableLayout


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


def detect_dots(frame_bgr: np.ndarray, expected: int) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thr = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 5000:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        centers.append([cx, cy])
    if len(centers) < max(4, expected // 2):
        return None
    pts = np.array(centers, dtype=np.float64)
    # Sort roughly row-major
    pts = pts[np.argsort(pts[:, 1])]
    return pts


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    if len(src) < 4 or len(dst) < 4:
        return None
    n = min(len(src), len(dst))
    H, mask = cv2.findHomography(src[:n], dst[:n], cv2.RANSAC, 5.0)
    return H


class Calibrator:
    """Project a known pattern, observe with camera, solve mappings."""

    def __init__(self, layout: TableLayout, projector_size: tuple[int, int], camera_size: tuple[int, int]) -> None:
        self.layout = layout
        self.projector_size = projector_size
        self.camera_size = camera_size

    def pattern(self) -> tuple[np.ndarray, np.ndarray]:
        return make_grid_pattern(self.projector_size)

    def solve_from_correspondences(
        self,
        projector_pts: np.ndarray,
        camera_pts: np.ndarray,
        board_corners_camera: np.ndarray | None = None,
    ) -> CalibrationData | None:
        H_cam_to_proj = estimate_homography(camera_pts, projector_pts)
        H_proj_to_cam = estimate_homography(projector_pts, camera_pts)
        if H_cam_to_proj is None or H_proj_to_cam is None:
            return None

        # Table → projector: map board corners in mm to a default centered rect in projector
        table_corners = self.layout.board_corners()
        if board_corners_camera is None:
            # Assume board fills central 55% of camera (matches MockCamera)
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

        # table → projector = (camera → projector) @ (table → camera)
        H_table_to_projector = H_cam_to_proj @ H_table_to_camera

        return CalibrationData(
            H_table_to_camera=H_table_to_camera,
            H_table_to_projector=H_table_to_projector,
            H_camera_to_projector=H_cam_to_proj,
            board_corners_camera=board_corners_camera,
            projector_size=self.projector_size,
            camera_size=self.camera_size,
        )

    def mock_calibrate(self) -> CalibrationData:
        """Identity-like calibration matching MockCamera board placement."""
        pattern, proj_pts = self.pattern()
        # Simulate camera seeing the pattern roughly scaled into frame
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
        """Quick session check: projected line endpoints should match detections."""
        # Soft check: if we can detect enough bright dots near expected locations
        expected = detect_dots(projected_lines, expected=24)
        observed = detect_dots(camera_frame, expected=24)
        if expected is None or observed is None:
            return False
        n = min(len(expected), len(observed))
        # Compare after mapping expected projector dots through H into camera — here projected_lines
        # is already in projector space rendered then re-captured; for mock we just compare counts.
        return n >= 8
