"""Operator page served by the game process.

The Pi camera can only be open in one process. This server runs beside the
game loop: the loop publishes the latest frame, and the page posts start /
capture / shutdown back onto a queue the loop already checks.
"""

from __future__ import annotations

import json
import socket
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gamebot</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 15px/1.4 ui-sans-serif, system-ui, sans-serif;
         background: #121418; color: #e7e4dc; }
  header { padding: 14px 18px 0; }
  h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  .sub { color: #9a958a; font-size: 13px; }
  main { display: grid; grid-template-columns: minmax(280px, 1.4fr) minmax(240px, 0.8fr);
         gap: 16px; padding: 16px 18px 28px; }
  @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
  img.cam { width: 100%; background: #000; border-radius: 8px; display: block; }
  .card { background: #1c2026; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; }
  label { display: block; margin: 8px 0; }
  input[type="number"] { width: 6em; background: #0e1014; color: inherit;
         border: 1px solid #3a4038; border-radius: 4px; padding: 4px 6px; }
  button { background: #d8d2c4; color: #161616; border: 0; border-radius: 6px;
           padding: 8px 12px; font: inherit; cursor: pointer; margin-right: 8px; }
  button.danger { background: #8d2a2a; color: #fff; }
  button:disabled { opacity: 0.45; cursor: default; }
  table.board { border-collapse: collapse; }
  table.board td { width: 32px; height: 32px; text-align: center; padding: 0;
                   font-size: 22px; line-height: 32px; }
  td.l { background: #d8c7a1; } td.d { background: #7d5a3a; }
  span.w { color: #f7f3ea; text-shadow: 0 0 1px #111, 0 1px 0 #111; }
  span.b { color: #1a120c; text-shadow: 0 0 1px #f4efe4; }
  span.bad { outline: 2px solid #e0b030; outline-offset: -2px; }
  span.unk { color: #e0b030; font-weight: 700; }
  pre { white-space: pre-wrap; font: 12px/1.35 ui-monospace, monospace;
        max-height: 280px; overflow: auto; margin: 0; color: #c8c2b6; }
  .err { color: #ff8d8d; min-height: 1.2em; }
</style>
</head>
<body>
<header>
  <h1>Gamebot</h1>
  <div class="sub" id="phase">connecting…</div>
</header>
<main>
  <section>
    <img class="cam" id="cam" alt="camera" src="/stream">
    <div class="card">
      <pre id="log"></pre>
    </div>
  </section>
  <section>
    <div class="card">
      <div id="board"></div>
      <div class="sub">Symbols are the game's pieces. ? is a camera piece the game cannot name. A gold ring means the camera disagrees.</div>
      <div class="sub" id="fen"></div>
    </div>
    <div class="card">
      <form id="start">
        <label>Elo <input id="elo" type="number" min="100" max="3200" value="500"></label>
        <label><input id="skip" type="checkbox"> Skip calibration</label>
        <label><input id="caps" type="checkbox" checked> Save a still after each move</label>
        <p><button type="submit" id="go">Start chess</button></p>
      </form>
      <p>
        <button type="button" id="accept">Save these boxes</button>
        <button type="button" id="shot">Capture still</button>
        <button type="button" id="off" class="danger">Shut down Pi</button>
      </p>
      <div class="sub">Save these boxes keeps the piece rectangles on screen, and the board outline, as training labels.</div>
      <div class="err" id="err"></div>
      <div class="sub" id="saved"></div>
    </div>
  </section>
</main>
<script>
const phase = document.getElementById("phase");
const fen = document.getElementById("fen");
const board = document.getElementById("board");
const log = document.getElementById("log");
const err = document.getElementById("err");
const saved = document.getElementById("saved");
const go = document.getElementById("go");
let armOff = false;

const GLYPH = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

function fenGrid(fen) {
  const grid = [];
  const rows = (fen || "").split(" ")[0].split("/");
  for (let r = 0; r < 8; r++) {
    const row = [];
    let c = 0;
    for (const ch of (rows[r] || "")) {
      if (c >= 8) break;
      if (ch >= "1" && ch <= "8") {
        for (let i = 0; i < Number(ch) && c < 8; i++) { row.push(""); c++; }
      } else {
        row.push(ch);
        c++;
      }
    }
    while (row.length < 8) row.push("");
    grid.push(row);
  }
  return grid;
}

function drawBoard(colors, fenStr) {
  const pieces = fenGrid(fenStr);
  let html = '<table class="board">';
  for (let r = 0; r < 8; r++) {
    html += "<tr>";
    for (let c = 0; c < 8; c++) {
      const v = colors && colors[r] ? colors[r][c] : -1;
      const piece = pieces[r][c];
      const light = (r + c) % 2 ? "d" : "l";
      let mark = "";
      if (piece) {
        const white = piece === piece.toUpperCase();
        const clash = v < 0 || v === 2 || (white && v === 1) || (!white && v === 0);
        mark = `<span class="${white ? "w" : "b"}${clash ? " bad" : ""}">${GLYPH[piece] || piece}</span>`;
      } else if (v === 0 || v === 1 || v === 2) {
        const tone = v === 0 ? "w" : v === 1 ? "b" : "unk";
        mark = `<span class="${tone}">?</span>`;
      }
      html += `<td class="${light}">${mark}</td>`;
    }
    html += "</tr>";
  }
  board.innerHTML = html + "</table>";
}

async function tick() {
  try {
    const res = await fetch("/api/state");
    const s = await res.json();
    const game = s.game ? " · " + s.game : "";
    phase.textContent = s.host + game;
    fen.textContent = s.fen || "No position yet. Rank 8 is the top.";
    drawBoard(s.colors, s.fen);
    log.textContent = (s.log || []).join("\\n");
    log.scrollTop = log.scrollHeight;
    err.textContent = s.error || "";
    saved.textContent = s.last_capture ? "Saved " + s.last_capture : "";
    go.disabled = s.host !== "PORTAL";
  } catch (e) {
    err.textContent = "Page lost the game process.";
  }
}

document.getElementById("start").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  armOff = false;
  document.getElementById("off").textContent = "Shut down Pi";
  const body = {
    elo: Number(document.getElementById("elo").value),
    skip_cal: document.getElementById("skip").checked,
    capture_moves: document.getElementById("caps").checked,
  };
  const res = await fetch("/api/start", {method: "POST", body: JSON.stringify(body)});
  if (!res.ok) err.textContent = await res.text();
});

document.getElementById("shot").addEventListener("click", () => {
  fetch("/api/capture", {method: "POST"});
});

document.getElementById("accept").addEventListener("click", () => {
  fetch("/api/accept", {method: "POST"});
});

document.getElementById("off").addEventListener("click", async () => {
  if (!armOff) {
    armOff = true;
    document.getElementById("off").textContent = "Confirm shut down";
    return;
  }
  await fetch("/api/shutdown", {method: "POST"});
});

drawBoard(null, "");
setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""


def lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


def annotate_pieces(frame_bgr: np.ndarray, pieces: list[dict]) -> np.ndarray:
    """Draw each detection and the standing point the square is scored from."""
    if not pieces:
        return frame_bgr
    out = frame_bgr.copy()
    for p in pieces:
        x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        white = int(p.get("cls", 0)) == 0
        color = (235, 235, 235) if white else (40, 40, 220)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (int(p["x"]), int(p["y"])), 4, (40, 220, 220), -1)
    return out


def stream_jpeg(frame_bgr: np.ndarray, pieces: list[dict] | None, *, max_width: int = 960) -> bytes:
    view = annotate_pieces(frame_bgr, pieces or [])
    h, w = view.shape[:2]
    if w > max_width:
        view = cv2.resize(view, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return b""
    return buf.tobytes()


class _Tee:
    """Copy complete lines into the portal log. The real stdout still prints."""

    def __init__(self, underlying, portal: OperatorPortal) -> None:
        self._u = underlying
        self._p = portal
        self._buf = ""

    def write(self, s: str) -> int:
        self._u.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._p.log(line)
        return len(s)

    def flush(self) -> None:
        self._u.flush()

    def isatty(self) -> bool:
        return False


class OperatorPortal:
    def __init__(self, port: int = 8080) -> None:
        self.port = port
        self.url = f"http://{lan_ip()}:{port}"
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._jpg = b""
        self._jpg_seq = 0
        self.host = "PORTAL"
        self.game = ""
        self.fen = ""
        self.colors: list | None = None
        self.lines: deque[str] = deque(maxlen=200)
        self.error = ""
        self.last_capture = ""
        self._start: dict[str, Any] | None = None
        self._capture = False
        self._accept = False
        self._shutdown = False
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "waiting for camera", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        self._jpg = stream_jpeg(blank, None)
        handler = _handler(self)
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        self._httpd.daemon_threads = True
        self.port = int(self._httpd.server_address[1])
        self.url = f"http://{lan_ip()}:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="portal", daemon=True)
        self._thread.start()
        print(f"portal: {self.url}")
        self.log(f"portal: {self.url}")

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def log(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)

    def publish(
        self,
        jpg: bytes,
        *,
        host: str,
        game: str = "",
        fen: str = "",
        colors: list | None = None,
    ) -> None:
        with self._cv:
            if jpg:
                self._jpg = jpg
                self._jpg_seq += 1
            self.host = host
            self.game = game
            self.fen = fen
            self.colors = colors
            self._cv.notify_all()

    def wait_jpeg(self, after_seq: int, timeout: float) -> tuple[int, bytes]:
        with self._cv:
            if self._jpg_seq == after_seq:
                self._cv.wait(timeout)
            return self._jpg_seq, self._jpg

    def poll_start(self) -> dict[str, Any] | None:
        with self._lock:
            cmd = self._start
            self._start = None
            return cmd

    def take_capture(self) -> bool:
        with self._lock:
            flag = self._capture
            self._capture = False
            return flag

    def take_shutdown(self) -> bool:
        with self._lock:
            flag = self._shutdown
            self._shutdown = False
            return flag

    def note_capture_path(self, path: str) -> None:
        with self._lock:
            self.last_capture = path
            self.error = ""

    def set_error(self, message: str) -> None:
        with self._lock:
            self.error = message

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "host": self.host,
                "game": self.game,
                "fen": self.fen,
                "colors": self.colors,
                "log": list(self.lines)[-80:],
                "error": self.error,
                "last_capture": self.last_capture,
            }

    def request_start(self, elo: int, skip_cal: bool, capture_moves: bool) -> str | None:
        if not 100 <= elo <= 3200:
            return "Elo must be between 100 and 3200."
        with self._lock:
            if self.host != "PORTAL":
                return "A game is already running."
            self._start = {"elo": elo, "skip_cal": skip_cal, "capture_moves": capture_moves}
            self.host = "STARTING"
            self.error = ""
        return None

    def take_accept(self) -> bool:
        with self._lock:
            flag = self._accept
            self._accept = False
            return flag

    def request_capture(self) -> None:
        with self._lock:
            self._capture = True

    def request_accept(self) -> None:
        with self._lock:
            self._accept = True

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutdown = True


def _handler(portal: OperatorPortal):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                body = json.dumps(portal.snapshot()).encode()
                self._send(200, body, "application/json")
                return
            if path == "/stream":
                self._stream()
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(min(length, 8192)) if length else b""
            if path == "/api/start":
                try:
                    data = json.loads(raw.decode() or "{}")
                    elo = int(data.get("elo") or 500)
                except (json.JSONDecodeError, TypeError, ValueError):
                    self._send(400, b"bad json", "text/plain")
                    return
                err = portal.request_start(
                    elo,
                    bool(data.get("skip_cal", False)),
                    bool(data.get("capture_moves", False)),
                )
                if err:
                    self._send(409, err.encode(), "text/plain")
                    return
                self._send(204, b"", "text/plain")
                return
            if path == "/api/capture":
                portal.request_capture()
                self._send(204, b"", "text/plain")
                return
            if path == "/api/accept":
                portal.request_accept()
                self._send(204, b"", "text/plain")
                return
            if path == "/api/shutdown":
                portal.request_shutdown()
                self._send(204, b"", "text/plain")
                return
            self._send(404, b"not found", "text/plain")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seq = -1
            try:
                while True:
                    seq, jpg = portal.wait_jpeg(seq, 1.0)
                    if not jpg:
                        continue
                    chunk = (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode()
                        + b"\r\n\r\n"
                        + jpg
                        + b"\r\n"
                    )
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return

    return Handler
