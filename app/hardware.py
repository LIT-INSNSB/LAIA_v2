from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
import threading
import time

import cv2

from .cameras import v4l2_device
from .state_machine import AppState


LOGGER = logging.getLogger(__name__)


class LedController:
    def __init__(self, *, success_pin: int, error_pin: int, enabled: bool = True, blink_count: int = 3, blink_seconds: float = 0.20, hold_seconds: float = 3.0) -> None:
        self.available = False
        self.detail = "deshabilitado"
        self._success = None
        self._error = None
        self._blink_count = max(1, blink_count)
        self._blink_seconds = max(0.05, blink_seconds)
        self._hold_seconds = max(0.0, hold_seconds)
        self._generation = 0
        self._lock = threading.Lock()
        if not enabled:
            return
        try:
            from gpiozero import LED

            self._success = LED(success_pin, active_high=True, initial_value=False)
            self._error = LED(error_pin, active_high=True, initial_value=False)
            self.available = True
            self.detail = f"acierto=BCM{success_pin}, error=BCM{error_pin}"
            self.off()
        except Exception as exc:  # hardware opcional durante desarrollo
            self.detail = str(exc)
            LOGGER.warning("GPIO no disponible: %s", exc)

    def apply(self, state: AppState) -> None:
        if not self.available:
            return
        if state is AppState.SUCCESS:
            self._signal(self._success)
        elif state is AppState.CORRECTION:
            self._signal(self._error)
        else:
            self.off()

    def _signal(self, led) -> None:
        if not self.available or led is None:
            return
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._success.off()
            self._error.off()

        def run() -> None:
            try:
                for _ in range(self._blink_count):
                    with self._lock:
                        if generation != self._generation:
                            return
                        self._success.off()
                        self._error.off()
                        led.on()
                    time.sleep(self._blink_seconds)
                    with self._lock:
                        if generation != self._generation:
                            return
                        led.off()
                    time.sleep(self._blink_seconds)
                with self._lock:
                    if generation != self._generation:
                        return
                    led.on()
                time.sleep(self._hold_seconds)
            finally:
                with self._lock:
                    if generation == self._generation:
                        led.off()

        threading.Thread(target=run, name="laia-gpio-signal", daemon=True).start()

    def off(self) -> None:
        if self.available:
            with self._lock:
                self._generation += 1
                self._success.off()
                self._error.off()

    def close(self) -> None:
        if not self.available:
            return
        self.off()
        self._success.close()
        self._error.close()


class AudioPlayer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.player = shutil.which("mpg123") or shutil.which("ffplay")
        self.detail = self.player or "reproductor no encontrado"

    def play(self, path: Path) -> None:
        if not self.enabled or self.player is None or not path.is_file():
            return
        with self._lock:
            self.stop()
            command = (
                [self.player, "-q", str(path)]
                if Path(self.player).name == "mpg123"
                else [self.player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
            )
            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                LOGGER.warning("No se pudo reproducir %s: %s", path, exc)

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        self._process = None

    def close(self) -> None:
        with self._lock:
            self.stop()


class CameraSource:
    """Abre la fuente solicitada y entrega siempre frames BGR uint8.

    En la Raspberry, el stream de Picamera2 anunciado como ``RGB888`` llega
    a ``capture_array`` con orden BGR en sus tres bytes intercalados, igual
    que los frames de OpenCV/V4L2. Por eso el contrato de esta clase es BGR y
    no se hace ninguna conversión al leer Picamera2. Los consumidores que
    necesitan RGB (MediaPipe Tasks y la GUI) hacen una única conversión en
    su propio borde.
    """

    def __init__(self, width: int, height: int, fps: float, camera_index: int = 0, source: str = "auto") -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.source = source
        self.backend = ""
        self._picam = None
        self._capture = None
        self._open()

    def _open(self) -> None:
        errors: list[str] = []
        if self.source.startswith("v4l2:"):
            self._open_v4l2(errors)
            return
        try:
            from picamera2 import Picamera2

            self._picam = Picamera2()
            config = self._picam.create_preview_configuration(
                main={"format": "RGB888", "size": (self.width, self.height)},
                controls={"FrameRate": self.fps},
            )
            self._picam.configure(config)
            self._picam.start()
            time.sleep(0.25)
            self.backend = "Picamera2"
            return
        except Exception as exc:
            errors.append(f"Picamera2: {exc}")
            if self._picam is not None:
                try:
                    self._picam.close()
                except Exception:
                    pass
                self._picam = None

        if self.source == "picamera2":
            raise RuntimeError("; ".join(errors))
        self._open_v4l2(errors)

    def _open_v4l2(self, errors: list[str]) -> None:
        device = v4l2_device(self.source, self.camera_index)
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                self._capture = capture
                self.backend = f"V4L2 {device}"
                return
        capture.release()
        errors.append(f"V4L2: {device} no disponible")
        raise RuntimeError("; ".join(errors))

    def read(self):
        if self._picam is not None:
            return self._picam.capture_array("main")
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
            finally:
                self._picam.close()
        if self._capture is not None:
            self._capture.release()
