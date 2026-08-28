from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

from .config import DEPLOYMENT, RuntimeConfig
from .hardware import CameraSource


LOGGER = logging.getLogger(__name__)


@dataclass
class RuntimeUpdate:
    frame_bgr: np.ndarray | None = None
    prediction: dict | None = None
    error: str | None = None
    camera_backend: str | None = None


class InferenceRuntime:
    def __init__(self, config: RuntimeConfig, camera_index: int = 0, camera_source: str = "auto") -> None:
        self.config = config
        self.camera_index = camera_index
        self.camera_source = camera_source
        self.updates: queue.Queue[RuntimeUpdate] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="laia-inference", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def latest(self) -> RuntimeUpdate | None:
        result = None
        while True:
            try:
                result = self.updates.get_nowait()
            except queue.Empty:
                return result

    def _publish(self, update: RuntimeUpdate) -> None:
        try:
            self.updates.put_nowait(update)
        except queue.Full:
            try:
                self.updates.get_nowait()
            except queue.Empty:
                pass
            self.updates.put_nowait(update)

    def _run(self) -> None:
        camera = None
        recognizer = None
        try:
            if str(DEPLOYMENT) not in sys.path:
                sys.path.insert(0, str(DEPLOYMENT))
            from laia_inference.api import StreamingHandwashingRecognizer

            camera = CameraSource(
                self.config.camera_width,
                self.config.camera_height,
                self.config.camera_fps,
                self.camera_index,
                source=self.camera_source,
            )
            recognizer = StreamingHandwashingRecognizer(
                DEPLOYMENT / "model" / "laia_lightstgcnv2_fp32.onnx",
                fps=self.config.camera_fps,
                width=self.config.camera_width,
                height=self.config.camera_height,
                pose_api="tasks",
                task_model_path=DEPLOYMENT / "model" / "hand_landmarker.task",
                intra_op_threads=self.config.inference_threads,
            )
            started = time.monotonic()
            while not self._stop.is_set():
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("la cámara dejó de entregar frames")
                result = recognizer.update_frame(frame, timestamp_s=time.monotonic() - started)
                self._publish(
                    RuntimeUpdate(frame_bgr=frame, prediction=result, camera_backend=camera.backend)
                )
        except Exception as exc:
            LOGGER.exception("Runtime de cámara/inferencia detenido")
            self._publish(RuntimeUpdate(error=str(exc)))
        finally:
            if recognizer is not None:
                recognizer.close()
            if camera is not None:
                camera.close()

