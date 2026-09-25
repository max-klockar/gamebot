from __future__ import annotations

import argparse
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from gamebot.adapters import create_camera, create_projector
from gamebot.adapters.camera import MockCamera
from gamebot.adapters.led import LedRing, create_led_backend
from gamebot.calibration import CalibrationData, Calibrator
from gamebot.config import AppConfig
from gamebot.geometry import TableLayout, warp_points
from gamebot.plugins import IdentifyContext, RuntimeHandles, discover_plugins, identify_matches
from gamebot.plugins.protocol import GamePlugin, GameSession
from gamebot.ui import DwellTapTracker, Scene, UIRenderer, build_game_chooser_scene
from gamebot.vision import TokenTapDetector


class HostPhase(Enum):
    SPLASH = auto()
    CALIBRATE = auto()
    IDENTIFY = auto()
    CHOOSE = auto()
    PLAYING = auto()


class App:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.layout = TableLayout(board_size_mm=config.board_size_mm, margin_mm=config.table_margin_mm)
        self.camera = create_camera(config.camera, config.camera_size, config.camera_file)
        self.projector = create_projector(config.projector, config.projector_size)
        led_kind = config.led_backend
        if led_kind == "auto" and config.camera == "mock" and config.projector == "mock":
            led_kind = "mock"
        self.led = LedRing(backend=create_led_backend(led_kind), enabled=True)
        self.led.thermal.hot_celsius = config.hot_celsius
        self.calibrator = Calibrator(self.layout, config.projector_size, config.camera_size)
        self.calibration: CalibrationData | None = None
        self.renderer = UIRenderer(
            config.projector_size, self.layout, config.identity_H(), debug=config.debug_overlay
        )
        self.dwell = DwellTapTracker(dwell_ms=config.dwell_ms)
        self.token_detector = TokenTapDetector()
        self.plugins = discover_plugins()
        self.phase = HostPhase.SPLASH
        self.candidates: list[GamePlugin] = []
        self.session: GameSession | None = None
        self._thermal_check_counter = 0

    def load_or_mock_calibration(self) -> None:
        path = self.config.calibration_path
        assert path is not None
        if path.exists():
            self.calibration = CalibrationData.load(path)
        else:
            self.calibration = self.calibrator.mock_calibrate()
            self.calibration.save(path)
        self.renderer.set_homography(self.calibration.H_table_to_projector)

    def run(self) -> int:
        self.camera.open()
        self.projector.open()
        self.led.open()
        self.led.on()
        self.load_or_mock_calibration()
        self.phase = HostPhase.SPLASH
        try:
            return self._loop()
        finally:
            self.led.close()
            self.projector.close()
            self.camera.close()

    def _loop(self) -> int:
        splash_frames = 0
        while True:
            frame = self.camera.read()
            scene = self._build_scene()
            active = self.dwell._state.zone_id
            progress = self.dwell.progress()
            proj = self.renderer.render(scene, progress, active)
            self._draw_led_debug(proj)
            self.projector.show(proj)

            token_mm = self._token_table_mm(frame)
            clicked = self.dwell.update(scene.zones, token_mm)
            if clicked:
                self._handle_zone(clicked)

            key = self.projector.poll_key(30)
            if key in (ord("q"), 27):
                return 0
            if key != 255 and key != -1:
                self._handle_key(key, scene)

            self._thermal_check_counter += 1
            if self._thermal_check_counter % 15 == 0:
                self.led.tick_thermal()

            if self.phase == HostPhase.SPLASH:
                splash_frames += 1
                if splash_frames > 15:
                    self.phase = HostPhase.CALIBRATE
                    splash_frames = 0
            elif self.phase == HostPhase.IDENTIFY:
                self._run_identify(frame)
            elif self.phase == HostPhase.PLAYING and self.session is not None:
                self.session.tick(frame)
                if self.session.wants_exit():
                    self.session = None
                    self.phase = HostPhase.IDENTIFY

    def _run_identify(self, frame: np.ndarray) -> None:
        ctx = IdentifyContext(
            layout=self.layout,
            board_corners_camera=None if self.calibration is None else self.calibration.board_corners_camera,
        )
        matches = identify_matches(self.plugins, frame, ctx)
        if len(matches) == 1:
            self._start_plugin(matches[0])
        else:
            # 0 or many → chooser over empty table (all plugins if none matched)
            self.candidates = matches if matches else list(self.plugins)
            self.phase = HostPhase.CHOOSE

    def _start_plugin(self, plugin: GamePlugin) -> None:
        runtime = RuntimeHandles(
            data_dir=self.config.data_dir,
            layout=self.layout,
            thinking=self.led.thinking,
            config=self.config,
        )
        # Attach host handles plugins may need
        runtime.camera = self.camera  # type: ignore[attr-defined]
        runtime.calibration = self.calibration  # type: ignore[attr-defined]
        self.session = plugin.create_session(runtime)
        self.session.on_enter()
        self.phase = HostPhase.PLAYING
        self.dwell.reset()

    def _build_scene(self) -> Scene:
        if self.phase == HostPhase.SPLASH:
            sc = Scene(self.layout)
            sc.title = "Gamebot"
            sc.add_text("Place a game on the table", self.layout.margin_mm, self.layout.margin_mm + 40)
            return sc
        if self.phase == HostPhase.CALIBRATE:
            sc = Scene(self.layout)
            sc.title = "Calibrate"
            sc.add_button(
                "cal:done",
                "Done",
                self.layout.margin_mm + self.layout.board_size_mm + 10,
                self.layout.margin_mm,
                100,
                40,
            )
            return sc
        if self.phase == HostPhase.CHOOSE:
            games = [(p.id, p.name, p.icon()) for p in self.candidates]
            if not games:
                sc = Scene(self.layout)
                sc.title = "No games installed"
                sc.add_text("Install a plugin package", self.layout.margin_mm, self.layout.margin_mm + 40)
                return sc
            return build_game_chooser_scene(self.layout, games)
        if self.phase == HostPhase.PLAYING and self.session is not None:
            return self.session.build_scene()
        sc = Scene(self.layout)
        sc.title = "…"
        return sc

    def _handle_zone(self, zone_id: str) -> None:
        if zone_id == "cal:done":
            self._run_calibration()
            self.phase = HostPhase.IDENTIFY
            return
        if zone_id.startswith("game:"):
            pid = zone_id.split(":", 1)[1]
            plugin = next((p for p in self.candidates if p.id == pid), None)
            if plugin is None:
                plugin = next((p for p in self.plugins if p.id == pid), None)
            if plugin is not None:
                self._start_plugin(plugin)
            return
        if self.phase == HostPhase.PLAYING and self.session is not None:
            self.session.handle_zone(zone_id)

    def _run_calibration(self) -> None:
        pattern, proj_pts = self.calibrator.pattern()
        self.projector.show(pattern)
        self.projector.poll_key(200)
        frame = self.camera.read()
        from gamebot.calibration import detect_dots

        cam_pts = detect_dots(frame, expected=len(proj_pts))
        if cam_pts is None or len(cam_pts) < 8:
            cal = self.calibrator.mock_calibrate()
        else:
            order_p = np.lexsort((proj_pts[:, 0], proj_pts[:, 1]))
            order_c = np.lexsort((cam_pts[:, 0], cam_pts[:, 1]))
            n = min(len(order_p), len(order_c))
            cal = self.calibrator.solve_from_correspondences(proj_pts[order_p[:n]], cam_pts[order_c[:n]])
            if cal is None:
                cal = self.calibrator.mock_calibrate()
        assert cal is not None
        self.calibration = cal
        cal.save(self.config.calibration_path)  # type: ignore[arg-type]
        self.renderer.set_homography(cal.H_table_to_projector)

    def _token_table_mm(self, frame: np.ndarray) -> tuple[float, float] | None:
        det = self.token_detector.detect(frame)
        if det is None or self.calibration is None:
            return None
        H_inv = np.linalg.inv(self.calibration.H_table_to_camera)
        pts = warp_points(np.array([det], dtype=np.float64), H_inv)[0]
        return float(pts[0]), float(pts[1])

    def _handle_key(self, key: int, scene: Scene) -> None:
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(scene.zones):
                self._handle_zone(scene.zones[idx].id)
                self.dwell.reset()
            return
        if key == ord("c"):
            if self.phase == HostPhase.CALIBRATE:
                self._handle_zone("cal:done")
            else:
                self.phase = HostPhase.CALIBRATE
            return
        if key == ord("h"):
            hot = (
                self.led.thermal._mock_celsius is None
                or self.led.thermal._mock_celsius < self.led.thermal.hot_celsius
            )
            self.led.thermal.set_mock(self.led.thermal.hot_celsius + 5 if hot else None)
            self.led.tick_thermal()
            return
        if key == ord("t") and isinstance(self.camera, MockCamera):
            if scene.zones and self.calibration is not None:
                z = scene.zones[0]
                cx = float(z.polygon_mm[:, 0].mean())
                cy = float(z.polygon_mm[:, 1].mean())
                pts = warp_points(np.array([[cx, cy]]), self.calibration.H_table_to_camera)[0]
                self.camera.set_token((int(pts[0]), int(pts[1])))
            return
        if self.phase == HostPhase.PLAYING and self.session is not None:
            self.session.handle_key(key)

    def _draw_led_debug(self, frame: np.ndarray) -> None:
        if not self.config.debug_overlay:
            return
        r, g, b = self.led.current_rgb
        cv2.rectangle(frame, (frame.shape[1] - 70, 20), (frame.shape[1] - 20, 70), (b, g, r), -1)
        cv2.rectangle(frame, (frame.shape[1] - 70, 20), (frame.shape[1] - 20, 70), (255, 255, 255), 1)
        cv2.putText(
            frame,
            self.led._effective_mode().value,
            (frame.shape[1] - 120, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gamebot", description="Projected tabletop game machine")
    p.add_argument("--mock", action="store_true", help="Force mock camera + projector")
    p.add_argument("--camera", choices=["mock", "picam", "file"], default=None)
    p.add_argument("--projector", choices=["mock", "hdmi"], default=None)
    p.add_argument("--camera-file", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--engine", type=str, default=None, help="Path to stockfish (chess plugin)")
    p.add_argument("--no-debug", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    camera = args.camera or "mock"
    projector = args.projector or "mock"
    if args.mock:
        camera, projector = "mock", "mock"
    kwargs = dict(
        camera=camera,
        projector=projector,
        camera_file=args.camera_file,
        engine_path=args.engine,
        debug_overlay=not args.no_debug,
    )
    if args.data_dir is not None:
        cfg = AppConfig(data_dir=args.data_dir, **kwargs)
    else:
        cfg = AppConfig(**kwargs)
    raise SystemExit(App(cfg).run())


if __name__ == "__main__":
    main()
