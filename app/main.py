from __future__ import annotations

import argparse
import logging
import math
import re
import queue
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading
import time
import tkinter as tk

from .config import LOGS, RuntimeConfig, audio_path, required_paths
from .cameras import CameraOption, discover_camera_options
from .display import DisplayOutput, DisplayRotationController
from .hardware import AudioPlayer, LedController
from .runtime import InferenceRuntime
from .state_machine import AppEvent, AppState, SessionStateMachine
from .ui import LaiaView


LOGGER = logging.getLogger("laia")


_AUDIO_EVENT_KEYS = {
    "session_started": "start",
    "correction": "retry",
    "recovered": "recover",
    "hands_lost": "hands_lost",
    "attempt_incomplete": "almost",
    "success": "success",
    "reset": "welcome",
}


def audio_key_for_event(event_name: str) -> str | None:
    return _AUDIO_EVENT_KEYS.get(event_name)


def configure_logging() -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / "laia-app.log"
    LOGGER.setLevel(logging.INFO)
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="LAIA: asistente de lavado de manos")
    result.add_argument("--simulate", action="store_true", help="no abre cámara; usa teclas para simular")
    result.add_argument("--windowed", action="store_true", help="ventana redimensionable en lugar de fullscreen")
    result.add_argument("--width", type=int, default=1024)
    result.add_argument("--height", type=int, default=600)
    result.add_argument("--camera-index", type=int, help="compatibilidad: fuerza /dev/videoN")
    result.add_argument("--camera-source", default="auto", help="auto, picamera2 o v4l2:/dev/videoN")
    result.add_argument("--no-gpio", action="store_true")
    result.add_argument("--no-audio", action="store_true")
    result.add_argument("--preview-state", choices=("waiting", "washing", "correction", "success", "incomplete"))
    result.add_argument("--preview-step", type=int, default=3)
    result.add_argument("--display-output", help="output Wayland que girará el botón; autodetectado por defecto")
    return result


class LaiaApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        missing = [path for path in required_paths() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Faltan archivos requeridos:\n" + "\n".join(map(str, missing)))

        self.args = args
        self.config = RuntimeConfig()
        self.machine = SessionStateMachine(self.config)
        self.camera_source = (
            f"v4l2:/dev/video{args.camera_index}" if args.camera_index is not None else args.camera_source
        )
        self._target_display_size: tuple[int, int] | None = None
        self._ui_actions: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.root = tk.Tk()
        self.root.title("LAIA")
        self.root.configure(background="#FFFFFF")
        if args.windowed:
            self.root.geometry(f"{args.width}x{args.height}")
        else:
            # Kiosco robusto también bajo labwc/Xwayland: sin bordes y usando
            # la geometría real del output, incluida la salida headless.
            self.root.overrideredirect(True)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{screen_width}x{screen_height}+0+0")
            self.root.attributes("-topmost", True)

        self.view = LaiaView(self.root, self.machine)
        self.view.pack(fill="both", expand=True)
        self.rotation = DisplayRotationController(args.display_output)
        self.rotate_button = tk.Button(
            self.root,
            text="↻ Girar",
            command=self.rotate_interface,
            font=("DejaVu Sans", 13, "bold"),
            foreground="#FFFFFF",
            background="#10B58B",
            activeforeground="#FFFFFF",
            activebackground="#064E3B",
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=8,
            cursor="hand2",
            takefocus=False,
        )
        self.rotate_button.place(relx=0.975, rely=0.018, anchor="ne")
        self.camera_button_text = tk.StringVar(value="Cámara")
        self.camera_choice = tk.StringVar(value=self.camera_source)
        self.camera_button = tk.Menubutton(
            self.root,
            textvariable=self.camera_button_text,
            font=("DejaVu Sans", 13, "bold"),
            foreground="#FFFFFF",
            background="#10B58B",
            activeforeground="#FFFFFF",
            activebackground="#064E3B",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=8,
            cursor="hand2",
            takefocus=False,
        )
        self.camera_menu = tk.Menu(
            self.camera_button,
            tearoff=False,
            font=("DejaVu Sans", 15),
            background="#FFFFFF",
            foreground="#064E3B",
            activebackground="#DDF7EC",
            activeforeground="#064E3B",
        )
        self.camera_button.configure(menu=self.camera_menu)
        self.camera_menu.configure(postcommand=self.refresh_camera_menu)
        self.camera_button.place(relx=0.025, rely=0.018, anchor="nw")
        self.refresh_camera_menu()
        self.exit_button = tk.Button(
            self.root,
            text="Salir",
            command=self.request_exit,
            font=("DejaVu Sans", 13, "bold"),
            foreground="#FFFFFF",
            background="#FF6B5F",
            activeforeground="#FFFFFF",
            activebackground="#B42318",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=18,
            pady=8,
            cursor="hand2",
            takefocus=False,
        )
        self.exit_button.place(relx=0.975, rely=0.105, anchor="ne")
        self.leds = LedController(
            success_pin=self.config.success_gpio_bcm,
            error_pin=self.config.error_gpio_bcm,
            blink_count=self.config.gpio_blink_count,
            blink_seconds=self.config.gpio_blink_seconds,
            hold_seconds=self.config.gpio_hold_seconds,
            enabled=not args.no_gpio and args.preview_state is None,
        )
        self.audio = AudioPlayer(enabled=not args.no_audio and args.preview_state is None)
        self.runtime: InferenceRuntime | None = None
        self._last_state = self.machine.state
        self._last_frame = None
        self._camera_error = ""
        self._inference_generation = 0
        self._last_render_key = None
        self._closing = False

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("<Right>", lambda _event: self._simulate_correct())
        self.root.bind("<space>", lambda _event: self._simulate_correct())
        self.root.bind("<x>", lambda _event: self._simulate_incorrect())
        self.root.bind("<r>", lambda _event: self._reset_session())

        if args.preview_state:
            state = {
                "waiting": AppState.WAITING_FOR_HANDS,
                "washing": AppState.WASHING,
                "correction": AppState.CORRECTION,
                "success": AppState.SUCCESS,
                "incomplete": AppState.INCOMPLETE,
            }[args.preview_state]
            self.machine.force_preview(state, args.preview_step)
        elif args.simulate:
            self.audio.play(audio_path("welcome"))
        else:
            self.runtime = InferenceRuntime(
                self.config,
                camera_index=args.camera_index or 0,
                camera_source=self.camera_source,
            )
            self.runtime.start()
            self.audio.play(audio_path("welcome"))

        self.leds.apply(self.machine.state)
        self.root.after(20, self._poll)


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
            self.camera_menu.add_radiobutton(label=option.label, variable=self.camera_choice, value=option.source,
                                             command=lambda source=option.source: self.select_camera(source))
        self.camera_menu.add_separator()
        self.camera_menu.add_command(label="Actualizar cámaras", command=self.refresh_camera_menu)
        self.camera_menu.add_command(label="Reconectar seleccionada", command=lambda: self.select_camera(self.camera_source, force=True))
        self.camera_button_text.set(self._camera_caption(selected.source))
        self.camera_choice.set(self.camera_source)

    def select_camera(self, source: str, force: bool = False) -> None:
        if source == self.camera_source and not force:
            return
        self.camera_source = source
        self.camera_choice.set(source)
        self.camera_button.configure(state="disabled")
        self.camera_button_text.set("Cámara…")
        if self.args.preview_state or self.args.simulate:
            self.camera_button.configure(state="normal")
            self.refresh_camera_menu()
            return
        previous = self.runtime
        self.runtime = None
        self._last_frame = None
        self.view.frame_bgr = None
        self._camera_error = ""
        self.view.set_camera_detail("Conectando cámara…")
        worker = threading.Thread(target=self._restart_camera_worker, args=(previous, source), daemon=True)
        worker.start()

    def _restart_camera_worker(self, previous: InferenceRuntime | None, source: str) -> None:
        try:
            if previous is not None:
                previous.stop()
            replacement = InferenceRuntime(
                self.config,
                camera_index=self.args.camera_index or 0,
                camera_source=source,
            )
            replacement.start()
        except Exception as exc:
            message = str(exc)
            self._ui_actions.put(("camera_failed", message))
            return
        self._ui_actions.put(("camera_ready", replacement))

    def _camera_restart_done(self, replacement: InferenceRuntime) -> None:
        if self._closing:
            replacement.stop()
            return
        self.runtime = replacement
        self._inference_generation = 0
        self.camera_button.configure(state="normal")
        self.refresh_camera_menu()
        LOGGER.info("Cámara seleccionada: %s", self.camera_source)

    def _camera_restart_failed(self, message: str) -> None:
        self.camera_button.configure(state="normal")
        self.camera_button_text.set("Cámara: error")
        self.view.set_camera_detail(message)
        self._camera_error = message
        LOGGER.error("No se pudo cambiar de cámara: %s", message)
    def _lift_controls(self) -> None:
        self.camera_button.lift()
        self.rotate_button.lift()
        self.exit_button.lift()

    def _fit_to_screen(self, width: int | None = None, height: int | None = None) -> None:
        if self.args.windowed:
            return
        width = width or self.root.winfo_screenwidth()
        height = height or self.root.winfo_screenheight()
        self.root.geometry(f"{width}x{height}+0+0")
        self.view.configure(width=width, height=height)
        self.root.update_idletasks()
        self._last_render_key = None
        self.view.render()
        self._lift_controls()

    def rotate_interface(self) -> None:
        self.rotate_button.configure(state="disabled", text="Girando…")
        worker = threading.Thread(target=self._rotate_worker, name="laia-display-rotation", daemon=True)
        worker.start()

    def _rotate_worker(self) -> None:
        try:
            output = self.rotation.rotate_next()
        except Exception as exc:
            message = str(exc)
            self._ui_actions.put(("rotation_failed", message))
            return
        self._ui_actions.put(("rotation_ready", output))

    def _rotation_succeeded(self, output: DisplayOutput) -> None:
        width, height = output.logical_size
        if width <= 0 or height <= 0:
            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
        self._target_display_size = (width, height)
        self.rotate_button.configure(text="↻ Girar")
        LOGGER.info("Orientación solicitada: %s=%s (%sx%s)", output.name, output.transform, width, height)
        for delay in (100, 350, 800):
            self.root.after(
                delay,
                lambda width=width, height=height: self._fit_to_screen(width, height),
            )
        self.root.after(1100, self._finish_rotation)

    def _rotation_failed(self, message: str) -> None:
        LOGGER.error("No se pudo girar la interfaz: %s", message)
        self.rotate_button.configure(text="No se pudo girar")
        self.root.after(1800, self._finish_rotation)

    def _finish_rotation(self) -> None:
        if self._target_display_size is not None:
            self._fit_to_screen(*self._target_display_size)
        else:
            self._fit_to_screen()
        self.rotate_button.configure(state="normal", text="↻ Girar")
        self._lift_controls()

    @staticmethod
    def _prediction_step_ids(prediction: dict) -> set[int]:
        result: set[int] = set()
        for key in ("recognized_steps", "steps_identified", "identified_steps", "step_ids"):
            raw = prediction.get(key)
            if raw is None:
                continue
            if isinstance(raw, dict):
                values = raw.keys()
            elif isinstance(raw, (list, tuple, set)):
                values = raw
            else:
                values = (raw,)
            for value in values:
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    step = int(value)
                else:
                    match = re.fullmatch(r"(?:step|paso)?[ _-]*([1-6])", str(value), re.IGNORECASE)
                    if match is None:
                        continue
                    step = int(match.group(1))
                if step in range(1, 7):
                    result.add(step)
        return result

    @staticmethod
    def _prediction_step_coverage(prediction: dict) -> float | None:
        for key in (
            "step_coverage",
            "steps_coverage",
            "recognized_step_coverage",
            "identified_steps_coverage",
        ):
            raw = prediction.get(key)
            if raw is None:
                continue
            try:
                coverage = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(coverage):
                continue
            if coverage > 1.0:
                coverage /= 100.0
            return min(1.0, max(0.0, coverage))
        return None

    def _prediction_to_observation(
        self,
        prediction: dict,
    ) -> tuple[int | None, bool, set[int], float | None] | None:
        status = prediction.get("status")
        if status == "prediction":
            try:
                class_id = int(prediction.get("class_id"))
            except (TypeError, ValueError):
                class_id = None
            present = float(prediction.get("pose_coverage_ge1", 0.0)) > 0.15
            return (
                class_id,
                present,
                self._prediction_step_ids(prediction),
                self._prediction_step_coverage(prediction),
            )
        if status == "insufficient_pose":
            return None, False, set(), None
        return None

    def _drain_ui_actions(self) -> None:
        while True:
            try:
                action, payload = self._ui_actions.get_nowait()
            except queue.Empty:
                return
            if action == "camera_ready" and isinstance(payload, InferenceRuntime):
                self._camera_restart_done(payload)
            elif action == "camera_failed" and isinstance(payload, str):
                self._camera_restart_failed(payload)
            elif action == "rotation_ready" and isinstance(payload, DisplayOutput):
                self._rotation_succeeded(payload)
            elif action == "rotation_failed" and isinstance(payload, str):
                self._rotation_failed(payload)

    def _poll(self) -> None:
        self._drain_ui_actions()
        if self.runtime is not None:
            update = self.runtime.latest()
            if update is not None and update.inference_generation < self._inference_generation:
                # A reset is applied on the inference worker. Packets already
                # queued from the previous temporal window must not reach the
                # state machine.
                update = None
            if update is not None:
                self._inference_generation = max(
                    self._inference_generation,
                    update.inference_generation,
                )
                if update.frame_bgr is not None:
                    self._last_frame = update.frame_bgr
                    self.view.set_frame(update.frame_bgr)
                if update.camera_backend:
                    self.view.set_camera_detail(update.camera_backend)
                if update.error:
                    self._camera_error = update.error
                    self.view.set_camera_detail(update.error)
                    LOGGER.error("Cámara/inferencia: %s", update.error)
                if update.prediction:
                    observation = self._prediction_to_observation(update.prediction)
                    if observation is not None:
                        class_id, present, recognized_steps, step_coverage = observation
                        events = self.machine.observe(
                            class_id=class_id,
                            pose_present=present,
                            recognized_steps=recognized_steps,
                            step_coverage=step_coverage,
                        )
                        self._handle_events(events)

        if self.args.preview_state is None:
            self._handle_events(self.machine.tick())
        if self.machine.state is not self._last_state:
            LOGGER.info("Estado: %s; paso esperado: %s", self.machine.state.value, self.machine.expected_step)
            self.leds.apply(self.machine.state)
            self._last_state = self.machine.state
        render_key = (
            self.machine.state,
            self.machine.expected_step,
            self._last_frame is not None,
            self._camera_error,
        )
        if render_key != self._last_render_key:
            self.view.render()
            self._lift_controls()
            self._last_render_key = render_key
        self.root.after(50, self._poll)

    def _handle_events(self, events: list[AppEvent]) -> None:
        for event in events:
            LOGGER.info("Evento: %s value=%s", event.name, event.value)
            sound = audio_key_for_event(event.name)
            if sound:
                self.audio.play(audio_path(sound))
            if event.name in {"step_accepted", "hands_returned", "reset"}:
                self._reset_temporal_inference()

    def _reset_temporal_inference(self) -> None:
        """Reset MediaPipe/tracking/buffer while keeping Picamera2 running."""

        if self.runtime is None:
            return
        self._inference_generation = self.runtime.reset_temporal_state()

    def _reset_session(self) -> None:
        self._handle_events(self.machine.reset())

    def _simulate_correct(self) -> None:
        if not (self.args.simulate or self.args.preview_state):
            return
        if self.machine.state is AppState.WAITING_FOR_HANDS:
            self._handle_events(self.machine.observe(class_id=None, pose_present=True))
            return
        for _ in range(self.config.correct_predictions_required):
            self._handle_events(
                self.machine.observe(class_id=self.machine.expected_step, pose_present=True)
            )

    def _simulate_incorrect(self) -> None:
        if not (self.args.simulate or self.args.preview_state):
            return
        wrong = 6 if self.machine.expected_step != 6 else 5
        for _ in range(self.config.incorrect_predictions_required):
            self._handle_events(self.machine.observe(class_id=wrong, pose_present=True))

    def request_exit(self) -> None:
        if self._closing:
            return
        self.exit_button.configure(state="disabled", text="Saliendo…")
        self.root.after(10, self.close)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.runtime is not None:
            self.runtime.stop()
        self.audio.close()
        self.leds.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parser().parse_args()
    log_path = configure_logging()
    LOGGER.info("Inicio LAIA; log=%s", log_path)
    try:
        application = LaiaApplication(args)
        try:
            application.run()
        except KeyboardInterrupt:
            LOGGER.info("Interrupción solicitada")
            application.close()
    except Exception:
        LOGGER.exception("LAIA no pudo iniciar")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
