from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import cv2
import numpy as np

from gamebot.geometry import TableLayout, warp_points


@dataclass
class HitZone:
    id: str
    label: str
    polygon_mm: np.ndarray
    meta: dict = field(default_factory=dict)

    def contains(self, xy_mm: tuple[float, float]) -> bool:
        contour = self.polygon_mm.astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, xy_mm, False) >= 0


@dataclass
class DwellState:
    zone_id: str | None = None
    started_at: float | None = None
    last_xy: tuple[float, float] | None = None
    fired: bool = False


class DwellTapTracker:
    def __init__(self, dwell_ms: int = 600, motion_mm: float = 6.0) -> None:
        self.dwell_ms = dwell_ms
        self.motion_mm = motion_mm
        self._state = DwellState()

    def reset(self) -> None:
        self._state = DwellState()

    def update(
        self,
        zones: list[HitZone],
        token_xy_mm: tuple[float, float] | None,
        now: float | None = None,
    ) -> str | None:
        now = monotonic() if now is None else now
        if token_xy_mm is None:
            self.reset()
            return None
        hit = next((z for z in zones if z.contains(token_xy_mm)), None)
        if hit is None:
            self.reset()
            return None
        st = self._state
        if st.zone_id != hit.id:
            self._state = DwellState(zone_id=hit.id, started_at=now, last_xy=token_xy_mm)
            return None
        assert st.last_xy is not None and st.started_at is not None
        dx = token_xy_mm[0] - st.last_xy[0]
        dy = token_xy_mm[1] - st.last_xy[1]
        if (dx * dx + dy * dy) ** 0.5 > self.motion_mm:
            self._state = DwellState(zone_id=hit.id, started_at=now, last_xy=token_xy_mm)
            return None
        st.last_xy = token_xy_mm
        elapsed_ms = (now - st.started_at) * 1000
        if elapsed_ms >= self.dwell_ms and not st.fired:
            st.fired = True
            return hit.id
        return None

    def progress(self, now: float | None = None) -> float:
        st = self._state
        if st.started_at is None or st.fired:
            return 0.0
        now = monotonic() if now is None else now
        return min(1.0, (now - st.started_at) * 1000 / self.dwell_ms)


@dataclass
class DrawPrim:
    kind: str
    data: dict


class Scene:
    def __init__(self, layout: TableLayout) -> None:
        self.layout = layout
        self.zones: list[HitZone] = []
        self.prims: list[DrawPrim] = []
        self.title: str = ""

    def clear(self) -> None:
        self.zones.clear()
        self.prims.clear()

    def add_button(self, zone_id: str, label: str, x: float, y: float, w: float, h: float, **meta: object) -> None:
        poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
        self.zones.append(HitZone(zone_id, label, poly, meta))
        self.prims.append(DrawPrim("button", {"id": zone_id, "label": label, "rect": (x, y, w, h)}))

    def add_arrow(self, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int] = (0, 200, 255)) -> None:
        self.prims.append(DrawPrim("arrow", {"p0": (x0, y0), "p1": (x1, y1), "color": color}))

    def add_text(self, text: str, x: float, y: float, color: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.prims.append(DrawPrim("text", {"text": text, "xy": (x, y), "color": color}))

    def add_icon(self, label: str, x: float, y: float, color: tuple[int, int, int] = (255, 220, 0)) -> None:
        self.prims.append(DrawPrim("icon", {"label": label, "xy": (x, y), "color": color}))

    def add_image_button(
        self,
        zone_id: str,
        label: str,
        x: float,
        y: float,
        w: float,
        h: float,
        image_bgr: np.ndarray | None = None,
        **meta: object,
    ) -> None:
        poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
        self.zones.append(HitZone(zone_id, label, poly, meta))
        self.prims.append(
            DrawPrim("image_button", {"id": zone_id, "label": label, "rect": (x, y, w, h), "image": image_bgr})
        )


class UIRenderer:
    def __init__(
        self,
        size: tuple[int, int],
        layout: TableLayout,
        H_table_to_proj: np.ndarray,
        debug: bool = True,
    ) -> None:
        self.size = size
        self.layout = layout
        self.H = H_table_to_proj
        self.debug = debug

    def set_homography(self, H: np.ndarray) -> None:
        self.H = H

    def render(self, scene: Scene, dwell_progress: float = 0.0, active_zone: str | None = None) -> np.ndarray:
        w, h = self.size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        corners = warp_points(self.layout.board_corners(), self.H).astype(np.int32)
        cv2.polylines(frame, [corners], True, (60, 60, 60), 2)

        for prim in scene.prims:
            if prim.kind == "button":
                self._draw_button(frame, prim.data, dwell_progress if prim.data["id"] == active_zone else 0.0)
            elif prim.kind == "image_button":
                self._draw_image_button(frame, prim.data, dwell_progress if prim.data["id"] == active_zone else 0.0)
            elif prim.kind == "arrow":
                self._draw_arrow(frame, prim.data)
            elif prim.kind == "text":
                self._draw_text(frame, prim.data)
            elif prim.kind == "icon":
                self._draw_icon(frame, prim.data)

        if scene.title:
            cv2.putText(frame, scene.title, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)

        if self.debug:
            for z in scene.zones:
                pts = warp_points(z.polygon_mm, self.H).astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 128), 1)
        return frame

    def _to_px(self, xy_mm: tuple[float, float]) -> tuple[int, int]:
        pts = warp_points(np.array([xy_mm]), self.H)[0]
        return int(pts[0]), int(pts[1])

    def _draw_button(self, frame: np.ndarray, data: dict, progress: float) -> None:
        x, y, w, h = data["rect"]
        poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
        pts = warp_points(poly, self.H).astype(np.int32)
        cv2.fillPoly(frame, [pts], (40, 80, 140))
        cv2.polylines(frame, [pts], True, (200, 220, 255), 2)
        if progress > 0:
            x0, y0 = self._to_px((x, y))
            x1, _ = self._to_px((x + w * progress, y))
            cv2.line(frame, (x0, y0 - 6), (x1, y0 - 6), (0, 255, 255), 4)
        cx, cy = self._to_px((x + w / 2, y + h / 2))
        label = data["label"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_image_button(self, frame: np.ndarray, data: dict, progress: float) -> None:
        x, y, w, h = data["rect"]
        poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
        pts = warp_points(poly, self.H).astype(np.int32)
        cv2.fillPoly(frame, [pts], (30, 30, 40))
        cv2.polylines(frame, [pts], True, (200, 220, 255), 2)
        img = data.get("image")
        if img is not None and img.size:
            x0, y0 = self._to_px((x + 8, y + 8))
            x1, y1 = self._to_px((x + w - 8, y + h - 28))
            bw, bh = max(8, x1 - x0), max(8, y1 - y0)
            thumb = cv2.resize(img, (bw, bh))
            y0c, y1c = max(0, y0), min(frame.shape[0], y0 + bh)
            x0c, x1c = max(0, x0), min(frame.shape[1], x0 + bw)
            frame[y0c:y1c, x0c:x1c] = thumb[: y1c - y0c, : x1c - x0c]
        cx, cy = self._to_px((x + w / 2, y + h - 12))
        (tw, th), _ = cv2.getTextSize(data["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, data["label"], (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if progress > 0:
            px0, py0 = self._to_px((x, y))
            px1, _ = self._to_px((x + w * progress, y))
            cv2.line(frame, (px0, py0 - 6), (px1, py0 - 6), (0, 255, 255), 4)

    def _draw_arrow(self, frame: np.ndarray, data: dict) -> None:
        cv2.arrowedLine(frame, self._to_px(data["p0"]), self._to_px(data["p1"]), data["color"], 4, tipLength=0.25)

    def _draw_text(self, frame: np.ndarray, data: dict) -> None:
        cv2.putText(frame, data["text"], self._to_px(data["xy"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, data["color"], 2)

    def _draw_icon(self, frame: np.ndarray, data: dict) -> None:
        xy = self._to_px(data["xy"])
        cv2.circle(frame, xy, 28, data["color"], -1)
        cv2.putText(frame, data["label"][:1], (xy[0] - 10, xy[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)


def build_game_chooser_scene(layout: TableLayout, games: list[tuple[str, str, np.ndarray | None]]) -> Scene:
    """games: list of (plugin_id, name, icon_bgr). Laid out on empty table / board area."""
    scene = Scene(layout)
    scene.title = "Which game?"
    scene.add_text("Place token on a game", layout.margin_mm, layout.margin_mm - 25)
    n = max(1, len(games))
    ox, oy = layout.board_origin
    gap = 16.0
    cell = min(layout.board_size_mm / 2 - gap, 140.0)
    cols = 2 if n > 1 else 1
    for i, (pid, name, icon) in enumerate(games):
        row, col = divmod(i, cols)
        x = ox + col * (cell + gap) + 20
        y = oy + row * (cell + gap + 20) + 20
        scene.add_image_button(f"game:{pid}", name, x, y, cell, cell, image_bgr=icon, plugin_id=pid)
    return scene
