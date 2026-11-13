from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np


@dataclass(frozen=True)
class TableLayout:
    """Physical layout in millimetres, origin at outer table top-left.

    Board sits inset by `margin_mm` on left/top; menu strip is to the right of the board.
    """

    board_size_mm: float = 400.0
    margin_mm: float = 120.0
    menu_width_mm: float = 120.0

    @property
    def board_origin(self) -> tuple[float, float]:
        return (self.margin_mm, self.margin_mm)

    @property
    def square_mm(self) -> float:
        return self.board_size_mm / 8

    @property
    def table_size(self) -> tuple[float, float]:
        w = self.margin_mm + self.board_size_mm + self.menu_width_mm + self.margin_mm * 0.25
        h = self.margin_mm + self.board_size_mm + self.margin_mm
        return (w, h)

    def square_center(self, square: chess.Square) -> tuple[float, float]:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        # a1 at bottom-left of board for white
        ox, oy = self.board_origin
        s = self.square_mm
        x = ox + (file + 0.5) * s
        y = oy + (7 - rank + 0.5) * s
        return (x, y)

    def square_rect(self, square: chess.Square) -> tuple[float, float, float, float]:
        cx, cy = self.square_center(square)
        h = self.square_mm / 2
        return (cx - h, cy - h, cx + h, cy + h)

    def board_corners(self) -> np.ndarray:
        ox, oy = self.board_origin
        s = self.board_size_mm
        return np.array(
            [[ox, oy], [ox + s, oy], [ox + s, oy + s], [ox, oy + s]],
            dtype=np.float64,
        )


def warp_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2_perspective(pts, H)
    return out.reshape(-1, 2)


def cv2_perspective(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.perspectiveTransform(pts, H)


def force_square_quad(corners: np.ndarray) -> np.ndarray:
    """Force a TL,TR,BR,BL quad to a square using the shortest edge.

    Keeps the same centre and approximate orientation (from the longest edge).
    Useful for projected boards: humans place two corner pieces and the result
    otherwise drifts into a rectangle that's hard to notice until play starts.
    Do **not** use on a physical chessboard under strong perspective (that's a
    trapezoid in the image on purpose).
    """
    p = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edges = np.array([p[(i + 1) % 4] - p[i] for i in range(4)], dtype=np.float64)
    lengths = np.linalg.norm(edges, axis=1)
    side = float(np.min(lengths))
    if side < 1.0 or not np.isfinite(side):
        return p.copy()

    i = int(np.argmax(lengths))
    u = edges[i] / lengths[i]
    v = np.array([-u[1], u[0]], dtype=np.float64)
    if float(np.dot(v, edges[(i + 1) % 4])) < 0:
        v = -v

    c = p.mean(axis=0)
    h = side / 2.0
    raw = np.array(
        [
            c - u * h - v * h,
            c + u * h - v * h,
            c + u * h + v * h,
            c - u * h + v * h,
        ],
        dtype=np.float64,
    )
    by_y = raw[np.argsort(raw[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bot = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float64)
