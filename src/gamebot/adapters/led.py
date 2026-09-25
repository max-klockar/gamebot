from __future__ import annotations

"""RGB LED ring status light.

Modes:
  idle     — soft vague blue (gentle breathe)
  thinking — purple pulse while the engine calculates
  overheat — violent red pulse if the Pi is too hot (highest priority)

Hardware backends:
  mock     — tracks RGB in memory (laptop / tests)
  rgb_pwm  — three GPIO PWM channels (R/G/B)
  ws2812   — addressable ring via rpi_ws281x (optional)

Wire NeoPixels to BCM 18 (PWM) or use separate R/G/B MOSFET pins.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class LedMode(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    OVERHEAT = "overheat"


# (base RGB 0-255, pulse amplitude 0-1, period seconds)
_MODE_STYLE: dict[LedMode, tuple[tuple[int, int, int], float, float]] = {
    LedMode.IDLE: ((28, 55, 140), 0.22, 3.2),       # vague blue, slow
    LedMode.THINKING: ((150, 20, 200), 0.55, 0.75),  # purple
    LedMode.OVERHEAT: ((255, 0, 0), 1.0, 0.12),      # violent red
}


def pulse_brightness(t: float, period: float, amplitude: float) -> float:
    """Return 0..1 brightness multiplier. amplitude scales how deep the pulse goes."""
    if period <= 0:
        return 1.0
    # Cosine breathe: (1 - amp/2) + (amp/2)*cos → stays in [1-amp, 1]
    phase = (2.0 * math.pi * t) / period
    return max(0.0, min(1.0, (1.0 - amplitude) + amplitude * (0.5 + 0.5 * math.cos(phase))))


def apply_pulse(rgb: tuple[int, int, int], brightness: float) -> tuple[int, int, int]:
    b = max(0.0, min(1.0, brightness))
    return (int(rgb[0] * b), int(rgb[1] * b), int(rgb[2] * b))


@dataclass
class ThermalSensor:
    """Read SoC temperature from sysfs; mockable for tests."""

    path: str = "/sys/class/thermal/thermal_zone0/temp"
    hot_celsius: float = 75.0
    _mock_celsius: float | None = None

    def read_celsius(self) -> float | None:
        if self._mock_celsius is not None:
            return self._mock_celsius
        try:
            raw = open(self.path, encoding="utf-8").read().strip()
            return int(raw) / 1000.0
        except (OSError, ValueError):
            return None

    def is_hot(self) -> bool:
        t = self.read_celsius()
        return t is not None and t >= self.hot_celsius

    def set_mock(self, celsius: float | None) -> None:
        self._mock_celsius = celsius


class LedBackend:
    def open(self) -> None: ...
    def show(self, rgb: tuple[int, int, int]) -> None: ...
    def close(self) -> None: ...


class MockLedBackend(LedBackend):
    def __init__(self) -> None:
        self.rgb: tuple[int, int, int] = (0, 0, 0)
        self.history: list[tuple[int, int, int]] = []

    def open(self) -> None:
        self.rgb = (0, 0, 0)

    def show(self, rgb: tuple[int, int, int]) -> None:
        self.rgb = rgb
        self.history.append(rgb)

    def close(self) -> None:
        self.rgb = (0, 0, 0)


class RgbPwmBackend(LedBackend):
    """Common-anode/cathode RGB strip via three PWM GPIOs."""

    def __init__(self, pin_r: int = 17, pin_g: int = 27, pin_b: int = 22) -> None:
        self.pin_r = pin_r
        self.pin_g = pin_g
        self.pin_b = pin_b
        self._pwms: list[object] = []

    def open(self) -> None:
        import RPi.GPIO as GPIO

        GPIO.setmode(GPIO.BCM)
        self._pwms = []
        for pin in (self.pin_r, self.pin_g, self.pin_b):
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, 1000)
            pwm.start(0)
            self._pwms.append(pwm)

    def show(self, rgb: tuple[int, int, int]) -> None:
        if len(self._pwms) != 3:
            return
        for pwm, channel in zip(self._pwms, rgb):
            pwm.ChangeDutyCycle(max(0.0, min(100.0, channel * 100.0 / 255.0)))

    def close(self) -> None:
        for pwm in self._pwms:
            try:
                pwm.stop()
            except Exception:
                pass
        self._pwms = []
        try:
            import RPi.GPIO as GPIO

            GPIO.cleanup((self.pin_r, self.pin_g, self.pin_b))
        except Exception:
            pass


class Ws2812Backend(LedBackend):
    """Addressable LED ring (NeoPixel) via rpi_ws281x."""

    def __init__(self, count: int = 16, pin_bcm: int = 18, brightness: int = 80) -> None:
        self.count = count
        self.pin_bcm = pin_bcm
        self.brightness = brightness
        self._strip = None

    def open(self) -> None:
        from rpi_ws281x import PixelStrip, Color  # type: ignore

        self._Color = Color
        strip = PixelStrip(self.count, self.pin_bcm, brightness=self.brightness)
        strip.begin()
        self._strip = strip

    def show(self, rgb: tuple[int, int, int]) -> None:
        if self._strip is None:
            return
        color = self._Color(rgb[0], rgb[1], rgb[2])
        for i in range(self.count):
            self._strip.setPixelColor(i, color)
        self._strip.show()

    def close(self) -> None:
        if self._strip is not None:
            self.show((0, 0, 0))
            self._strip = None


def create_led_backend(kind: str = "auto") -> LedBackend:
    kind = kind.lower()
    if kind == "mock":
        return MockLedBackend()
    if kind == "rgb_pwm":
        return RgbPwmBackend()
    if kind == "ws2812":
        return Ws2812Backend()
    if kind == "auto":
        try:
            import RPi.GPIO  # noqa: F401

            try:
                import rpi_ws281x  # noqa: F401

                return Ws2812Backend()
            except ImportError:
                return RgbPwmBackend()
        except ImportError:
            return MockLedBackend()
    raise ValueError(f"Unknown LED backend: {kind}")


@dataclass
class LedRing:
    """Animated RGB status ring. Safe to call set_mode from the app thread."""

    backend: LedBackend = field(default_factory=MockLedBackend)
    thermal: ThermalSensor = field(default_factory=ThermalSensor)
    enabled: bool = True
    fps: float = 30.0
    _mode: LedMode = LedMode.IDLE
    _app_mode: LedMode = LedMode.IDLE  # requested mode before thermal override
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _t0: float = field(default_factory=time.monotonic)
    current_rgb: tuple[int, int, int] = (0, 0, 0)

    def open(self) -> None:
        if not self.enabled:
            return
        try:
            self.backend.open()
        except Exception:
            self.backend = MockLedBackend()
            self.backend.open()
        self._stop.clear()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="schack-led", daemon=True)
        self._thread.start()
        self.set_mode(LedMode.IDLE)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self.backend.show((0, 0, 0))
            self.backend.close()
        except Exception:
            pass
        self.current_rgb = (0, 0, 0)

    def set_mode(self, mode: LedMode) -> None:
        with self._lock:
            self._app_mode = mode

    def thinking(self, active: bool) -> None:
        self.set_mode(LedMode.THINKING if active else LedMode.IDLE)

    def tick_thermal(self) -> None:
        """Call periodically from the main loop to apply overheat override."""
        # Animation thread also checks thermal; this keeps tests deterministic
        # when mock temperature is set without waiting for the animator.
        with self._lock:
            if self.thermal.is_hot():
                self._mode = LedMode.OVERHEAT
            else:
                self._mode = self._app_mode

    def _effective_mode(self) -> LedMode:
        with self._lock:
            if self.thermal.is_hot():
                return LedMode.OVERHEAT
            return self._app_mode

    def sample(self, t: float | None = None) -> tuple[int, int, int]:
        """Compute RGB for time t (seconds since open). Useful for tests."""
        mode = self._effective_mode()
        base, amp, period = _MODE_STYLE[mode]
        now = time.monotonic() - self._t0 if t is None else t
        return apply_pulse(base, pulse_brightness(now, period, amp))

    def _run(self) -> None:
        interval = 1.0 / max(1.0, self.fps)
        while not self._stop.is_set():
            rgb = self.sample()
            self.current_rgb = rgb
            try:
                if self.enabled:
                    self.backend.show(rgb)
            except Exception:
                pass
            self._stop.wait(interval)

    # Back-compat shims
    def on(self) -> None:
        self.set_mode(LedMode.IDLE)

    def off(self) -> None:
        self.set_mode(LedMode.IDLE)
        self.current_rgb = (0, 0, 0)
        try:
            self.backend.show((0, 0, 0))
        except Exception:
            pass

    def set_brightness(self, value: float) -> None:
        # Not used by RGB modes; kept for old call sites
        pass
