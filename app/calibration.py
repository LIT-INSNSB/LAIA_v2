"""Aplicación independiente para calibrar cámara, manos y LEDs de LAIA.

No importa ``app.runtime`` ni ningún módulo del clasificador. La única pieza
de MediaPipe utilizada aquí es Hand Landmarker para obtener 21 landmarks por
mano; la geometría y la GUI son independientes del modelo WHO.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
import queue
from pathlib import Path
import threading
import time
import tkinter as tk

import cv2
import numpy as np

from .calibration_config import CalibrationConfig, CalibrationState, CalibrationStateMachine, HandPlacementValidator, PlacementResult
from .calibration_ui import CalibrationView
from .cameras import CameraOption, discover_camera_options
from .config import LOGS
from .display import DisplayOutput, DisplayRotationController
from .hardware import CameraSource
from .ui import CORAL, FOREST, MINT, TEAL, WHITE


LOGGER = logging.getLogger("laia.calibration")


@dataclass
class CalibrationFrame:
    frame_bgr: np.ndarray | None = None
    hands: tuple[tuple[tuple[float, float], ...], ...] = ()
    camera_backend: str = ""
    error: str | None = None


class CalibrationCameraWorker:
    """Captura BGR con el backend productivo y ejecuta solo MediaPipe Tasks."""

    def __init__(self, config: CalibrationConfig, camera_index: int = 0, camera_source: str = "auto") -> None:
        self.config = config
        self.camera_index = camera_index
        self.camera_source = camera_source
        self.updates: queue.Queue[CalibrationFrame] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="laia-calibration-camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def latest(self) -> CalibrationFrame | None:
        result = None
        while True:
            try:
                result = self.updates.get_nowait()
            except queue.Empty:
                return result

    def _publish(self, update: CalibrationFrame) -> None:
        try:
            self.updates.put_nowait(update)
        except queue.Full:
            try:
                self.updates.get_nowait()
            except queue.Empty:
                pass
            self.updates.put_nowait(update)

    def _create_landmarker(self):
        import mediapipe as mp

        if not self.config.hand_landmarker_path.is_file():
            raise FileNotFoundError(self.config.hand_landmarker_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.config.hand_landmarker_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.config.max_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        return mp, mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _run(self) -> None:
        camera = None
        landmarker = None
        try:
            camera = CameraSource(
                self.config.camera_width,
                self.config.camera_height,
                self.config.camera_fps,
                self.camera_index,
                source=self.camera_source,
            )
            mp, landmarker = self._create_landmarker()
            started = time.monotonic()
            last_timestamp_ms = -1
            while not self._stop.is_set():
                frame_bgr = camera.read()
                if frame_bgr is None:
                    raise RuntimeError("la cámara dejó de entregar imágenes")

                # CameraSource siempre entrega BGR, igual que la app principal.
                # MediaPipe Tasks recibe SRGB; esta es la única conversión.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((time.monotonic() - started) * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frame_rgb),
                )
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                hands = tuple(
                    tuple((float(landmark.x), float(landmark.y)) for landmark in hand)
                    for hand in result.hand_landmarks[: self.config.max_hands]
                )
                self._publish(CalibrationFrame(frame_bgr=frame_bgr, hands=hands, camera_backend=camera.backend))
        except Exception as exc:
            LOGGER.exception("Cámara/calibración detenida")
            self._publish(CalibrationFrame(error=str(exc)))
        finally:
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    LOGGER.debug("No se pudo cerrar MediaPipe", exc_info=True)
            if camera is not None:
                camera.close()


class CalibrationLedController:
    """Control sólido de tres LEDs; el amarillo se fuerza siempre a OFF."""

    def __init__(self, config: CalibrationConfig, enabled: bool = True, led_factory=None) -> None:
        self.config = config
        self.available = False
        self.detail = "deshabilitado"
        self._leds: dict[str, object] = {}
        if not enabled:
            return
        if led_factory is None:
            from gpiozero import LED

            led_factory = LED
        for name, pin in (
            ("red", config.red_gpio_bcm),
            ("yellow", config.yellow_gpio_bcm),
            ("green", config.green_gpio_bcm),
        ):
            try:
                self._leds[name] = led_factory(pin, active_high=True, initial_value=False)
            except Exception as exc:
                self.detail = f"GPIO{pin}: {exc}"
                LOGGER.warning("GPIO de calibración no disponible en BCM%s: %s", pin, exc)
        self.available = bool(self._leds)
        if self.available:
            self.detail = "rojo=BCM17, amarillo=BCM27, verde=BCM22"
            self.off()

    def apply(self, state: CalibrationState) -> None:
        if not self.available:
            return
        red = state in {CalibrationState.POSITIONING, CalibrationState.READY}
        green = state in {CalibrationState.COUNTDOWN, CalibrationState.CALIBRATION_OK}
        self._set("red", red)
        self._set("green", green)
        # Nunca se utiliza el amarillo en esta primera versión.
        self._set("yellow", False)

    def _set(self, name: str, value: bool) -> None:
        led = self._leds.get(name)
        if led is None:
            return
        (led.on if value else led.off)()

    def off(self) -> None:
        for name in ("red", "yellow", "green"):
            self._set(name, False)

    def close(self) -> None:
        self.off()
        for led in self._leds.values():
            close = getattr(led, "close", None)
            if close is not None:
                close()


def configure_logging() -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / "calibration-app.log"
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    return path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="LAIA: calibración de cámara, manos y LEDs")
    result.add_argument("--windowed", action="store_true", help="ventana redimensionable en vez de pantalla completa")
    result.add_argument("--width", type=int, default=480)
    result.add_argument("--height", type=int, default=800)
    result.add_argument("--camera-index", type=int, help="fuerza /dev/videoN como fallback V4L2")
    result.add_argument("--camera-source", default="auto", help="auto, picamera2 o v4l2:/dev/videoN")
    result.add_argument("--no-gpio", action="store_true", help="no inicializa LEDs durante una prueba de escritorio")
    result.add_argument("--display-output", help="output Wayland que rotará el botón")
    return result


class CalibrationApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = CalibrationConfig()
        self.validator = HandPlacementValidator(self.config)
        self.machine = CalibrationStateMachine(self.config)
        self.camera_source = f"v4l2:/dev/video{args.camera_index}" if args.camera_index is not None else args.camera_source
        self.worker: CalibrationCameraWorker | None = None
        self._ui_actions: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self._closing = False
        self._camera_error = ""
        self._last_placement = self.validator.evaluate(())

        self.root = tk.Tk()
        self.root.title("LAIA · Calibración")
        self.root.configure(background="#FFFFFF")
        if args.windowed:
            self.root.geometry(f"{args.width}x{args.height}")
        else:
            self.root.overrideredirect(True)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{screen_width}x{screen_height}+0+0")
            self.root.attributes("-topmost", True)

        self.view = CalibrationView(self.root)
        self.view.pack(fill="both", expand=True)
        self.rotation = DisplayRotationController(args.display_output)
        self._build_controls()
        self.leds = CalibrationLedController(self.config, enabled=not args.no_gpio)
        self.leds.apply(self.machine.state)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("<r>", lambda _event: self.reset_calibration())

        self.refresh_camera_menu()
        self._start_worker(self.camera_source)
        self.root.after(30, self._poll)

    def _build_controls(self) -> None:
        button_kwargs = {
            "font": ("DejaVu Sans", 13, "bold"),
            "foreground": WHITE,
            "background": TEAL,
            "activeforeground": WHITE,
            "activebackground": FOREST,
            "relief": "flat",
            "borderwidth": 0,
            "highlightthickness": 0,
            "padx": 12,
            "pady": 8,
            "cursor": "hand2",
            "takefocus": False,
        }
        self.rotate_button = tk.Button(self.root, text="↻ Girar", command=self.rotate_interface, **button_kwargs)
        self.rotate_button.place(relx=0.975, rely=0.018, anchor="ne")

        self.camera_button_text = tk.StringVar(value="Cámara")
        self.camera_choice = tk.StringVar(value=self.camera_source)
        self.camera_button = tk.Menubutton(self.root, textvariable=self.camera_button_text, **button_kwargs)
        self.camera_menu = tk.Menu(self.camera_button, tearoff=False, font=("DejaVu Sans", 15), background=WHITE, foreground=FOREST, activebackground=MINT, activeforeground=FOREST)
        self.camera_button.configure(menu=self.camera_menu)
        self.camera_menu.configure(postcommand=self.refresh_camera_menu)
        self.camera_button.place(relx=0.025, rely=0.018, anchor="nw")

        self.exit_button = tk.Button(self.root, text="Salir", command=self.request_exit, background=CORAL, activebackground="#B42318", **{key: value for key, value in button_kwargs.items() if key not in {"background", "activebackground"}})
        self.exit_button.place(relx=0.975, rely=0.105, anchor="ne")

        self.repeat_button = tk.Button(self.root, text="Repetir calibración", command=self.reset_calibration, background=TEAL, activebackground=FOREST, **{key: value for key, value in button_kwargs.items() if key not in {"background", "activebackground"}})
        self.repeat_button.place(relx=0.5, rely=0.945, anchor="center")
        self.repeat_button.place_forget()

    @staticmethod
    def _camera_caption(source: str) -> str:
        if source == "auto":
            return "Cámara: Auto"
        if source == "picamera2":
            return "Cámara: Pi"
        return "Cámara: USB"

    def refresh_camera_menu(self) -> None:
        options = discover_camera_options()
        selected = next((option for option in options if option.source == self.camera_source), None)
        if selected is None:
            selected = CameraOption(self.camera_source, f"No disponible · {self.camera_source}")
            options.append(selected)
        self.camera_menu.delete(0, "end")
        for option in options:
            self.camera_menu.add_radiobutton(label=option.label, variable=self.camera_choice, value=option.source, command=lambda source=option.source: self.select_camera(source))
        self.camera_menu.add_separator()
        self.camera_menu.add_command(label="Actualizar cámaras", command=self.refresh_camera_menu)
        self.camera_menu.add_command(label="Reconectar seleccionada", command=lambda: self.select_camera(self.camera_source, force=True))
        self.camera_button_text.set(self._camera_caption(selected.source))
        self.camera_choice.set(self.camera_source)

    def _start_worker(self, source: str) -> None:
        worker = CalibrationCameraWorker(self.config, self.args.camera_index or 0, source)
        self.worker = worker
        self._camera_error = ""
        worker.start()

    def select_camera(self, source: str, force: bool = False) -> None:
        if source == self.camera_source and not force:
            return
        self.camera_source = source
        self.camera_choice.set(source)
        self.camera_button.configure(state="disabled")
        self.camera_button_text.set("Cámara…")
        previous = self.worker
        self.worker = None
        thread = threading.Thread(target=self._restart_camera_worker, args=(previous, source), daemon=True)
        thread.start()

    def _restart_camera_worker(self, previous: CalibrationCameraWorker | None, source: str) -> None:
        try:
            if previous is not None:
                previous.stop()
            replacement = CalibrationCameraWorker(self.config, self.args.camera_index or 0, source)
            replacement.start()
        except Exception as exc:
            self._ui_actions.put(("camera_failed", str(exc)))
            return
        self._ui_actions.put(("camera_ready", replacement))

    def _drain_ui_actions(self) -> None:
        while True:
            try:
                action, payload = self._ui_actions.get_nowait()
            except queue.Empty:
                return
            if action == "camera_ready" and isinstance(payload, CalibrationCameraWorker):
                self.worker = payload
                self.camera_button.configure(state="normal")
                self.refresh_camera_menu()
            elif action == "camera_failed" and isinstance(payload, str):
                self.worker = None
                self._camera_error = payload
                self.camera_button.configure(state="normal")
                self.refresh_camera_menu()
            elif action == "rotation_ready" and isinstance(payload, DisplayOutput):
                width, height = payload.logical_size
                if width > 0 and height > 0 and not self.args.windowed:
                    self.root.geometry(f"{width}x{height}+0+0")
                self.rotate_button.configure(state="normal", text="↻ Girar")
            elif action == "rotation_failed" and isinstance(payload, str):
                LOGGER.error("No se pudo girar la calibración: %s", payload)
                self.rotate_button.configure(state="normal", text="↻ Girar")

    def _poll(self) -> None:
        if self._closing:
            return
        self._drain_ui_actions()
        if self.worker is not None:
            update = self.worker.latest()
            if update is not None:
                if update.frame_bgr is not None:
                    self.view.set_frame(update.frame_bgr)
                if update.camera_backend:
                    self._camera_error = ""
                if update.error:
                    self._camera_error = update.error
                    self._last_placement = self.validator.evaluate(())
                else:
                    self._last_placement = self.validator.evaluate(update.hands)

                now = time.monotonic()
                self.machine.update(self._last_placement, now)
                self.view.set_landmarks(self._last_placement.hands, self._last_placement.hand_valid)

        now = time.monotonic()
        countdown = self.machine.countdown_number(now)
        if self.machine.state is CalibrationState.CALIBRATION_OK:
            message = "¡Perfecto!"
        elif self.machine.state is CalibrationState.COUNTDOWN:
            message = "Mantén tus manos aquí"
        elif self.machine.state is CalibrationState.READY:
            message = "Mantén tus manos aquí"
        else:
            message = self._last_placement.message
        camera_message = "No puedo abrir la cámara" if self._camera_error else ""
        self.leds.apply(self.machine.state)
        self.view.set_feedback(self.machine.state, message, countdown, camera_message)
        if self.machine.state is CalibrationState.CALIBRATION_OK:
            self.repeat_button.place(relx=0.5, rely=0.945, anchor="center")
        else:
            self.repeat_button.place_forget()
        self.root.after(30, self._poll)

    def reset_calibration(self) -> None:
        self.machine.reset()
        self._last_placement = self.validator.evaluate(())
        self.leds.apply(self.machine.state)
        self.repeat_button.place_forget()

    def rotate_interface(self) -> None:
        self.rotate_button.configure(state="disabled", text="Girando…")
        threading.Thread(target=self._rotate_worker, daemon=True).start()

    def _rotate_worker(self) -> None:
        try:
            output = self.rotation.rotate_next()
            self._ui_actions.put(("rotation_ready", output))
        except Exception as exc:
            self._ui_actions.put(("rotation_failed", str(exc)))

    def request_exit(self) -> None:
        if not self._closing:
            self.exit_button.configure(state="disabled", text="Saliendo…")
            self.root.after(10, self.close)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.worker is not None:
            self.worker.stop()
        self.leds.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parser().parse_args()
    configure_logging()
    LOGGER.info("Inicio de calibración; fuente=%s", args.camera_source)
    application = CalibrationApplication(args)
    try:
        application.run()
    except KeyboardInterrupt:
        application.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
