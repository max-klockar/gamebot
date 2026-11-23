from __future__ import annotations

import json
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from chess_plugin.db import Database
from chess_plugin.game import ChessSession, Phase, SkillBand, skill_to_stockfish
from chess_plugin.plugin import ChessPlugin, looks_like_chessboard
from chess_plugin.ui import load_icon
from chess_plugin.vision import (
    OccupancyKind,
    classify_occupancy_vs_board,
    hypothesize_moves,
    occupancy_frame_is_noise,
    occupancy_from_board,
)
from gamebot.adapters.camera import MockCamera
from gamebot.adapters.led import LedMode, LedRing, MockLedBackend, ThermalSensor, pulse_brightness
from gamebot.calibration import (
    Calibrator,
    CalibrationData,
    detect_chessboard_outer_corners,
    detect_color_corners,
    detect_dot_by_difference,
    make_grid_pattern,
    order_quad_tl_tr_br_bl,
    outside_board_table_mm,
    quad_bbox_aspect,
    quad_looks_sane,
    render_color_corner_pattern,
)
from gamebot.config import AppConfig
from gamebot.geometry import TableLayout
from gamebot.plugins import IdentifyContext, discover_plugins, identify_matches
from gamebot.ui import DwellTapTracker, HitZone, build_game_chooser_scene
from gamebot.vision import TokenTapDetector


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


def test_discover_chess_plugin() -> None:
    plugins = discover_plugins()
    assert any(p.id == "chess" for p in plugins)


def test_chess_icon() -> None:
    icon = load_icon()
    assert icon.shape[0] >= 32 and icon.shape[2] == 3


def test_identify_mock_board() -> None:
    cam = MockCamera()
    cam.open()
    frame = cam.read()
    plugin = ChessPlugin()
    ctx = IdentifyContext(layout=TableLayout())
    assert plugin.identify(frame, ctx) is True


def test_identify_blank_is_false() -> None:
    blank = np.full((720, 1280, 3), 40, dtype=np.uint8)
    assert looks_like_chessboard(blank) is False


def test_chooser_when_zero_or_many() -> None:
    plugins = discover_plugins()
    blank = np.full((720, 1280, 3), 40, dtype=np.uint8)
    ctx = IdentifyContext(layout=TableLayout())
    matches = identify_matches(plugins, blank, ctx)
    assert matches == []
    scene = build_game_chooser_scene(
        TableLayout(), [(p.id, p.name, p.icon()) for p in plugins]
    )
    assert any(z.id.startswith("game:") for z in scene.zones)


def test_skill_mapping() -> None:
    assert skill_to_stockfish(1)[0] == 0
    assert skill_to_stockfish(10)[0] == 20
    from chess_plugin.game import elo_to_skill, SkillBand

    band, level = elo_to_skill(500)
    assert band == SkillBand.BEGINNER and level == 1


def test_profiles_and_game(db: Database) -> None:
    p = db.create_profile("Alice", "beginner", 2)
    assert any(x.name == "Guest" for x in db.list_profiles())
    g = db.start_game(p.id, 2)
    db.add_move(g.id, 1, "e2e4", "e4", False)
    moves = db.moves_for_game(g.id)
    assert moves[0]["uci"] == "e2e4"


def test_illegal_and_promotion(db: Database) -> None:
    s = ChessSession(db)
    guest = next(p for p in db.list_profiles() if p.name == "Guest")
    s.select_profile(guest)
    s.select_skill_band(SkillBand.BEGINNER)
    s.select_skill_level(1)
    s.confirm_setup()
    assert not s.try_human_move(chess.Move.from_uci("e2e5"))
    assert s.revert_hint is not None

    s.board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    s.phase = Phase.PLAY
    s.game = db.start_game(guest.id, 1)
    assert s.try_human_move(chess.Move.from_uci("a7a8"))
    assert s.phase == Phase.PROMOTE
    assert s.confirm_promotion(chess.QUEEN)


def test_dwell_and_token() -> None:
    poly = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float64)
    tracker = DwellTapTracker(dwell_ms=100, motion_mm=5)
    t0 = 1000.0
    assert tracker.update([HitZone("a", "A", poly)], (25, 25), now=t0) is None
    assert tracker.update([HitZone("a", "A", poly)], (25, 25), now=t0 + 0.12) == "a"
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 15, (0, 0, 255), -1)
    assert TokenTapDetector().detect(frame) is not None


def test_hypothesize_and_calibration(tmp_path: Path) -> None:
    board = chess.Board()
    before = occupancy_from_board(board)
    board.push_uci("e2e4")
    after = occupancy_from_board(board)
    board.pop()
    hyps = hypothesize_moves(before, after, board)
    assert hyps[0].move.uci() == "e2e4"
    cal = Calibrator(TableLayout(), (1280, 720), (1280, 720)).mock_calibrate()
    cal.save(tmp_path / "cal.json")
    assert CalibrationData.load(tmp_path / "cal.json").projector_size == (1280, 720)
    img, pts = make_grid_pattern((640, 480), cols=4, rows=3)
    assert len(pts) == 12
    messy = np.array([[200.0, 10.0], [10.0, 10.0], [10.0, 200.0], [200.0, 200.0]])
    ordered = order_quad_tl_tr_br_bl(messy)
    assert ordered[0].tolist() == [10.0, 10.0]
    assert ordered[2].tolist() == [200.0, 200.0]
    # Synthetic camera view of the colour pattern should detect 4 blobs.
    pattern = render_color_corner_pattern((640, 360))
    found = detect_color_corners(pattern)
    assert found is not None and len(found) == 4
    cal = Calibrator(TableLayout(), (640, 360), (640, 360)).auto_calibrate(pattern)
    assert cal is not None


def test_corner_pieces_define_board() -> None:
    from chess_plugin.vision import board_quad_from_opposite_corners, setup_from_corner_pieces
    from gamebot.geometry import force_square_quad

    img = np.full((400, 400, 3), 90, dtype=np.uint8)
    # White only, near bottom → bot white
    cv2.circle(img, (200, 340), 22, (230, 230, 230), -1)
    existing = np.array([[40, 40], [360, 40], [360, 360], [40, 360]], dtype=np.float64)
    setup = setup_from_corner_pieces(img, existing_board=existing)
    assert setup is not None
    assert setup.board_corners_px.shape == (4, 2)
    assert setup.bot_is_white is True
    assert setup.human_is_white is False
    quad = board_quad_from_opposite_corners((0.0, 0.0), (100.0, 100.0))
    assert quad.shape == (4, 2)
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    assert max(edges) - min(edges) < 1e-6

    rect = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float64)
    sq = force_square_quad(rect)
    sq_edges = [np.linalg.norm(sq[(i + 1) % 4] - sq[i]) for i in range(4)]
    assert abs(min(sq_edges) - 100.0) < 1e-6
    assert max(sq_edges) - min(sq_edges) < 1e-6


def test_projector_quad_is_square() -> None:
    from gamebot.calibration import projector_quad

    q = projector_quad((1280, 720), scale=0.32)
    edges = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    assert max(edges) - min(edges) < 1e-6


def test_center_zoom_keeps_size() -> None:
    from gamebot.adapters.camera import center_zoom_bgr

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[300:420, 540:740] = 255
    out = center_zoom_bgr(frame, 1.2)
    assert out.shape == frame.shape
    assert center_zoom_bgr(frame, 1.0) is frame


def test_virtual_table_draw_and_detect() -> None:
    from gamebot.table_surface import detect_physical_table, draw_virtual_table
    from gamebot.geometry import TableLayout

    blank = np.full((360, 640, 3), 30, dtype=np.uint8)
    assert detect_physical_table(blank) is False
    layout = TableLayout()
    H = np.array([[1.2, 0, 40], [0, 1.2, 30], [0, 0, 1]], dtype=np.float64)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_virtual_table(frame, layout, H)
    assert frame.sum() > 0


def test_difference_dot_ignores_static_lamp() -> None:
    dark = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(dark, (40, 40), 12, (255, 255, 255), -1)  # static lamp
    lit = dark.copy()
    cv2.circle(lit, (200, 150), 18, (0, 0, 255), -1)  # projected dot
    pt = detect_dot_by_difference(dark, lit)
    assert pt is not None
    assert abs(pt[0] - 200) < 8 and abs(pt[1] - 150) < 8
    assert not quad_looks_sane(np.array([[10, 10], [12, 10], [11, 11], [10, 40]], dtype=np.float64))


def test_difference_dot_roi_picks_near_expected() -> None:
    """Refine knows the board corner — ROI must ignore a brighter blob elsewhere."""
    dark = np.zeros((240, 320, 3), dtype=np.uint8)
    lit = dark.copy()
    cv2.circle(lit, (40, 40), 22, (255, 255, 255), -1)  # bright distractor
    cv2.circle(lit, (200, 150), 10, (255, 60, 0), -1)  # dim blue near board corner
    global_pt = detect_dot_by_difference(dark, lit)
    assert global_pt is not None
    assert abs(global_pt[0] - 40) < 15  # distractor wins globally
    roi_pt = detect_dot_by_difference(dark, lit, roi_center=(200, 150), roi_radius=50)
    assert roi_pt is not None
    assert abs(roi_pt[0] - 200) < 8 and abs(roi_pt[1] - 150) < 8


def test_sword_quad_rejected_and_outside_dots() -> None:
    """Bad H made a ~96×642 sword — bbox aspect must fail. Outside dots leave the board."""
    sword = np.array([[220, 580], [115, 514], [600, 425], [855, 407]], dtype=np.float64)
    assert quad_bbox_aspect(sword) > 2.4
    assert not quad_looks_sane(sword, (1280, 720), min_area=1000.0, min_edge=20.0)
    layout = TableLayout()
    board = layout.board_corners()
    outside = outside_board_table_mm(layout)
    c = board.mean(axis=0)
    for b, o in zip(board, outside):
        assert np.linalg.norm(o - c) > np.linalg.norm(b - c)


def test_chessboard_center_with_start_pieces() -> None:
    """Empty middle ranks (start position) should still yield an outer board quad."""
    size = 400
    sq = size // 8
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 220
            else:
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 40
    # Cover ranks 1–2 and 7–8 (top/bottom two rows) like starting pieces.
    img[0 : 2 * sq, :] = 30
    img[6 * sq : 8 * sq, :] = 200
    outer = detect_chessboard_outer_corners(img)
    assert outer is not None
    ordered = order_quad_tl_tr_br_bl(outer)
    assert ordered[0, 0] < 30 and ordered[0, 1] < 30
    assert ordered[2, 0] > size - 30 and ordered[2, 1] > size - 30


def test_refine_board_quad_inside_aabb() -> None:
    """YOLO-style padded AABB + classical refine recovers outer corners."""
    from gamebot.board_detect import refine_board_quad

    size = 500
    margin = 50
    board = 400
    img = np.full((size, size, 3), (200, 180, 40), dtype=np.uint8)  # bluish table
    sq = board // 8
    for r in range(8):
        for c in range(8):
            color = 220 if (r + c) % 2 == 0 else 40
            y0, x0 = margin + r * sq, margin + c * sq
            img[y0 : y0 + sq, x0 : x0 + sq] = color
    aabb = (margin - 20, margin - 20, margin + board + 20, margin + board + 20)
    refined = refine_board_quad(img, aabb)
    assert refined is not None
    quad, method = refined
    assert method == "chessboard"
    assert quad[0, 0] < margin + 15 and quad[0, 1] < margin + 15
    assert quad[2, 0] > margin + board - 15 and quad[2, 1] > margin + board - 15


def test_led_modes() -> None:
    led = LedRing(backend=MockLedBackend(), enabled=True)
    led.set_mode(LedMode.IDLE)
    r, g, b = led.sample(0.0)
    assert b > r
    led.set_mode(LedMode.THINKING)
    r, g, b = led.sample(0.0)
    assert r > 50 and b > 50
    led.set_mode(LedMode.CALIBRATE)
    a = led.sample(0.0)
    bcol = led.sample(2.0)
    assert a != bcol  # rainbow advances with time
    pixels = led.sample_pixels(0.1)
    assert len(pixels) >= 1
    # Neighbours should be close in colour (smooth gradient, not hard bands).
    from gamebot.adapters.led import rainbow_rgb

    p0 = rainbow_rgb(0.0, 0, 90)
    p1 = rainbow_rgb(0.0, 1, 90)
    assert sum(abs(p0[i] - p1[i]) for i in range(3)) < 40
    led.thermal = ThermalSensor(hot_celsius=75)
    led.thermal.set_mock(90)
    led.tick_thermal()
    assert led._effective_mode() == LedMode.OVERHEAT
    assert pulse_brightness(0.0, 0.12, 1.0) <= 1.0


def test_config(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=tmp_path / "data")
    assert cfg.data_dir.exists()
    tw, th = cfg.identity_H()[0, 0], cfg.identity_H()[1, 1]
    layout_w, layout_h = TableLayout().table_size
    assert abs(cfg.projector_size[0] / layout_w - tw) < 1e-9
    assert abs(cfg.projector_size[1] / layout_h - th) < 1e-9


def test_extra_piece_is_noise_not_a_move() -> None:
    """33 detections on a full board must not look like a move or a fault."""
    board = chess.Board()
    seen = occupancy_from_board(board).copy()
    seen[3, 3] = True
    verdict = classify_occupancy_vs_board(seen, board)
    assert verdict.kind == OccupancyKind.ILLEGAL_OR_MULTI
    assert occupancy_frame_is_noise(seen, board, verdict)


def test_half_detected_board_is_ignored() -> None:
    """A 13-piece reading of a full board is a failed detect, not a position."""
    board = chess.Board()
    seen = occupancy_from_board(board).copy()
    seen[2:6, :] = False
    seen[0, :] = False
    verdict = classify_occupancy_vs_board(seen, board)
    assert occupancy_frame_is_noise(seen, board, verdict)


def test_wrong_color_is_not_a_human_move() -> None:
    board = chess.Board()
    seen = occupancy_from_board(board).copy()
    seen[6, 4] = False  # e2
    seen[4, 4] = True  # e4
    colors = np.full((8, 8), -1, dtype=np.int8)
    colors[4, 4] = 1  # black piece, white to move
    verdict = classify_occupancy_vs_board(seen, board, colors=colors)
    assert verdict.kind != OccupancyKind.ONE_LEGAL
    colors[4, 4] = 0
    verdict = classify_occupancy_vs_board(seen, board, colors=colors)
    assert verdict.move is not None and verdict.move.uci() == "e2e4"


def test_arrow_pulse_scores_only_in_the_dark_gap() -> None:
    from chess_plugin.plugin import ArrowPulse, step_arrow_pulse

    pulse = ArrowPulse(lit=True, on_since=0.0, dark_skip=0)
    assert step_arrow_pulse(pulse, active=True, now=0.2, on_s=0.6) is False
    assert pulse.lit is True
    assert step_arrow_pulse(pulse, active=True, now=0.7, on_s=0.6) is False
    assert pulse.lit is False and pulse.dark_skip == 1
    # First dark frame can still contain the previous light.
    assert step_arrow_pulse(pulse, active=True, now=0.8, on_s=0.6) is False
    assert pulse.lit is False and pulse.dark_skip == 0
    # The next dark frame is the one that may be scored, then the arrow returns.
    assert step_arrow_pulse(pulse, active=True, now=0.9, on_s=0.6) is True
    assert pulse.lit is True and pulse.on_since == 0.9
    assert step_arrow_pulse(pulse, active=False, now=1.2, on_s=0.6) is True
    assert pulse.lit is False


def test_portal_page_starts_a_game() -> None:
    import urllib.error
    import urllib.request

    from gamebot.portal import OperatorPortal, annotate_pieces

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    marked = annotate_pieces(
        frame,
        [{"x1": 2, "y1": 2, "x2": 20, "y2": 30, "x": 8, "y": 16, "cls": 0, "conf": 0.9}],
    )
    assert int(marked[16, 8, 1]) > 150

    portal = OperatorPortal(0)
    portal.start()
    try:
        base = f"http://127.0.0.1:{portal.port}"
        html = urllib.request.urlopen(base + "/", timeout=2).read().decode()
        assert "Start chess" in html and "Capture still" in html
        req = urllib.request.Request(
            base + "/api/start",
            data=b'{"elo":500,"skip_cal":true,"capture_moves":true}',
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as res:
            assert res.status == 204
        cmd = portal.poll_start()
        assert cmd is not None and cmd["elo"] == 500 and cmd["capture_moves"] is True
        state = json.loads(urllib.request.urlopen(base + "/api/state", timeout=2).read())
        assert state["host"] == "STARTING"
        again = urllib.request.Request(base + "/api/start", data=b'{"elo":500}', method="POST")
        try:
            urllib.request.urlopen(again, timeout=2)
            raised = False
        except urllib.error.HTTPError as exc:
            raised = exc.code == 409
        assert raised
    finally:
        portal.close()


def test_save_move_still_writes_sidecar(tmp_path: Path) -> None:
    from gamebot.capture_frames import save_move_still

    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    path = save_move_still(
        frame,
        tmp_path,
        uci="e2e4",
        who="human",
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        camera_zoom=1.2,
        board_corners_camera=[[0, 0], [10, 0], [10, 10], [0, 10]],
    )
    assert path.name.endswith("_e2e4_human.jpg")
    assert path.is_file()
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["uci"] == "e2e4" and meta["who"] == "human" and meta["set"] == "04_play"
    assert meta["camera_zoom"] == 1.2
    assert meta["fen"].split()[1] == "b"


def test_standing_point_moves_toward_image_center() -> None:
    from gamebot.piece_detect import FOOT_BIAS, standing_xy

    frame = (1280, 720)
    # Under the lens the head has nowhere to lean.
    assert standing_xy((600, 320, 680, 400), frame) == (640.0, 360.0)

    def shift(xyxy: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = xyxy
        sx, sy = standing_xy(xyxy, frame)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5

    # Same center, right of the image middle. A radially longer box (a king)
    # walks farther inward than a short one (a pawn).
    pawn = (800.0, 350.0, 860.0, 390.0)
    king = (760.0, 350.0, 900.0, 390.0)
    assert standing_xy(pawn, frame)[0] < 830.0
    assert shift(king) > shift(pawn) * 1.5
    # The step stays on the ray from the image center through the box center.
    x1, y1, x2, y2 = 900.0, 300.0, 980.0, 460.0
    sx, sy = standing_xy((x1, y1, x2, y2), frame)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    assert sx < cx
    assert abs((sy - 360) * (cx - 640) - (cy - 360) * (sx - 640)) < 1e-4
    assert (sx - 640) ** 2 + (sy - 360) ** 2 < (cx - 640) ** 2 + (cy - 360) ** 2
    assert 0.0 < FOOT_BIAS < 1.0


def test_nested_piece_box_is_one_detection() -> None:
    from gamebot.piece_detect import PieceDet, suppress_duplicate_pieces

    def box(xyxy, conf, cls=0):
        x1, y1, x2, y2 = xyxy
        return PieceDet(xy=((x1 + x2) / 2, (y1 + y2) / 2), cls=cls, conf=conf, xyxy=xyxy)

    nested, dropped = suppress_duplicate_pieces(
        [
            box((100, 100, 160, 180), 0.9),
            box((110, 120, 150, 170), 0.4),  # same piece, tight box inside
        ]
    )
    assert dropped == 1 and len(nested) == 1 and nested[0].conf == 0.9

    neighbours, kept_drop = suppress_duplicate_pieces(
        [
            box((0, 0, 40, 50), 0.8),
            box((70, 0, 110, 50), 0.7),
        ]
    )
    assert kept_drop == 0 and len(neighbours) == 2


def test_turn_leds_travel() -> None:
    from gamebot.adapters.led import ring_wave

    tide = ring_wave(16, 0.0, (0, 20, 30), (80, 255, 220), period=3.4, laps=1.0)
    assert tide[0][1] > tide[8][1]
    comets = ring_wave(16, 0.2, (18, 0, 32), (210, 40, 255), period=1.05, laps=2.0)
    assert len({px[0] for px in comets}) > 2
    led = LedRing(backend=MockLedBackend(), enabled=True)
    led.set_mode(LedMode.PLAYER_TURN)
    r, g, b = led.sample(0.0)
    assert g > 200 and b > 180 and g > r
