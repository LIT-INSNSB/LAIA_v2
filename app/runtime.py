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
    frame_timestamp_s: float | None = None
    pose_snapshot: object | None = None
    prediction: dict | None = None
    error: str | None = None
    camera_backend: str | None = None
    inference_generation: int = 0
    runtime_frame_index: int = 0
    update_frame_ms: float | None = None


class InferenceRuntime:
    def __init__(self, config: RuntimeConfig, camera_index: int = 0, camera_source: str = "auto") -> None:
        self.config = config
        self.camera_index = camera_index
        self.camera_source = camera_source
        self.updates: queue.Queue[RuntimeUpdate] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._temporal_reset_requested = threading.Event()
        self._generation_lock = threading.Lock()
        self._requested_inference_generation = 0
        self._thread = threading.Thread(target=self._run, name="laia-inference", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def reset_temporal_state(self) -> int:
        """Request a pose/tracking/buffer reset without reopening the camera.

        The reset is applied on the inference worker thread, where the
        recognizer is owned.  The generation lets the UI discard packets
        produced by the previous temporal window while that request is being
        applied.
        """

        with self._generation_lock:
            self._requested_inference_generation += 1
            generation = self._requested_inference_generation
        self._temporal_reset_requested.set()
        return generation

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
        inference_generation = 0
        runtime_frame_index = 0
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
                if self._temporal_reset_requested.is_set():
                    self._temporal_reset_requested.clear()
                    recognizer.reset_temporal_state()
                    with self._generation_lock:
                        inference_generation = max(
                            inference_generation,
                            self._requested_inference_generation,
                        )
                    continue
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("la cámara dejó de entregar frames")
                frame_timestamp_s = time.monotonic() - started
                update_started = time.perf_counter()
                result = recognizer.update_frame(frame, timestamp_s=frame_timestamp_s)
                update_frame_ms = (time.perf_counter() - update_started) * 1000.0
                runtime_frame_index += 1
                self._publish(
                    RuntimeUpdate(
                        frame_bgr=frame,
                        frame_timestamp_s=frame_timestamp_s,
                        pose_snapshot=getattr(recognizer, "latest_pose_snapshot", None),
                        prediction=result,
                        camera_backend=camera.backend,
                        inference_generation=inference_generation,
                        runtime_frame_index=runtime_frame_index,
                        update_frame_ms=update_frame_ms,
                    )
                )
        except Exception as exc:
            LOGGER.exception("Runtime de cámara/inferencia detenido")
            self._publish(RuntimeUpdate(error=str(exc)))
        finally:
            if recognizer is not None:
                recognizer.close()
            if camera is not None:
                camera.close()
