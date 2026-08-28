"""Configuración y máquina de estados de la calibración de LAIA.

Este módulo no conoce el clasificador de LAIA. Solo describe la geometría
necesaria para verificar que dos manos estén listas para una futura sesión.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]


class CalibrationState(str, Enum):
    POSITIONING = "POSITIONING"
    READY = "READY"
    COUNTDOWN = "COUNTDOWN"
    CALIBRATION_OK = "CALIBRATION_OK"


@dataclass(frozen=True)
class CalibrationConfig:
    """Parámetros calibrables, todos reunidos en un solo lugar."""

    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 20.0

    # ROI matemática invisible para la persona usuaria.
    roi: tuple[float, float, float, float] = (0.16, 0.14, 0.84, 0.86)
    inside_ratio: float = 0.90
    min_hand_width: float = 0.08
    min_hand_height: float = 0.10
    max_hand_width: float = 0.60
    max_hand_height: float = 0.75
    landmark_count: int = 21
    max_hands: int = 2

    stable_seconds: float = 1.0
    countdown_seconds: float = 3.0

    min_detection_confidence: float = 0.50
    min_presence_confidence: float = 0.50
    min_tracking_confidence: float = 0.50

    red_gpio_bcm: int = 17
    yellow_gpio_bcm: int = 27
    green_gpio_bcm: int = 22

    # Es el modelo de landmarks de MediaPipe, no el clasificador WHO.
    hand_landmarker_path: Path = ROOT / "deployment" / "laia_model" / "model" / "hand_landmarker.task"


@dataclass(frozen=True)
class HandMeasurement:
    """Medidas internas de una mano; nunca se muestran en la UI."""

    valid: bool
    inside_ratio: float = 0.0
    width: float = 0.0
    height: float = 0.0
    reason: str = "invalid"


@dataclass(frozen=True)
class PlacementResult:
    """Resultado de validar una captura de landmarks."""

    hands: tuple[tuple[tuple[float, float], ...], ...]
    measurements: tuple[HandMeasurement, ...]
    valid: bool
    message: str

    @property
    def hand_count(self) -> int:
        return len(self.hands)

    @property
    def hand_valid(self) -> tuple[bool, ...]:
        return tuple(measurement.valid for measurement in self.measurements)


def _point_xy(point: object) -> tuple[float, float]:
    """Acepta landmarks de MediaPipe, tuplas o secuencias numéricas."""

    if hasattr(point, "x") and hasattr(point, "y"):
        x = float(getattr(point, "x"))
        y = float(getattr(point, "y"))
    else:
        x = float(point[0])  # type: ignore[index]
        y = float(point[1])  # type: ignore[index]
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("landmark no finito")
    return x, y


class HandPlacementValidator:
    """Valida dos manos sin depender de cámara, GUI o GPIO reales."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()

    def evaluate(self, hands: Iterable[Sequence[object]]) -> PlacementResult:
        normalized = tuple(self._normalize_hand(hand) for hand in hands)
        if not normalized:
            return PlacementResult((), (), False, "No veo tus manos")
        if len(normalized) == 1:
            return PlacementResult(normalized, (), False, "Muéstrame las dos manos")
        if len(normalized) > self.config.max_hands:
            return PlacementResult(normalized, (), False, "Muéstrame las dos manos")

        measurements = tuple(self._measure(hand) for hand in normalized)
        if all(measurement.valid for measurement in measurements):
            return PlacementResult(normalized, measurements, True, "Mantén tus manos aquí")

        reasons = {measurement.reason for measurement in measurements}
        if "too_small" in reasons:
            message = "Acércalas un poquito"
        elif "too_large" in reasons:
            message = "Aléjalas un poquito"
        else:
            message = "Coloca tus manos aquí"
        return PlacementResult(normalized, measurements, False, message)

    def _normalize_hand(self, hand: Sequence[object]) -> tuple[tuple[float, float], ...]:
        try:
            points = tuple(_point_xy(point) for point in hand)
        except (IndexError, TypeError, ValueError):
            return ()
        return points

    def _measure(self, hand: tuple[tuple[float, float], ...]) -> HandMeasurement:
        if len(hand) != self.config.landmark_count:
            return HandMeasurement(False, reason="invalid_landmarks")

        x1, y1, x2, y2 = self.config.roi
        inside = sum(x1 <= x <= x2 and y1 <= y <= y2 for x, y in hand)
        ratio = inside / self.config.landmark_count
        xs = [point[0] for point in hand]
        ys = [point[1] for point in hand]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        if ratio < self.config.inside_ratio:
            return HandMeasurement(False, ratio, width, height, "outside")
        if width < self.config.min_hand_width or height < self.config.min_hand_height:
            return HandMeasurement(False, ratio, width, height, "too_small")
        if width > self.config.max_hand_width or height > self.config.max_hand_height:
            return HandMeasurement(False, ratio, width, height, "too_large")
        return HandMeasurement(True, ratio, width, height, "ok")


@dataclass
class CalibrationStateMachine:
    """Estados no bloqueantes para posicionamiento y cuenta regresiva."""

    config: CalibrationConfig = CalibrationConfig()
    state: CalibrationState = CalibrationState.POSITIONING
    ready_since: float | None = None
    countdown_started: float | None = None

    def reset(self) -> None:
        self.state = CalibrationState.POSITIONING
        self.ready_since = None
        self.countdown_started = None

    def update(self, placement: PlacementResult, now: float) -> tuple[str, ...]:
        now = float(now)
        if self.state is CalibrationState.CALIBRATION_OK:
            return ()

        if not placement.valid:
            was_countdown = self.state is CalibrationState.COUNTDOWN
            changed = self.state is not CalibrationState.POSITIONING
            self.state = CalibrationState.POSITIONING
            self.ready_since = None
            self.countdown_started = None
            if was_countdown:
                return ("countdown_cancelled",)
            return ("positioning",) if changed else ()

        if self.state is CalibrationState.POSITIONING:
            self.state = CalibrationState.READY
            self.ready_since = now
            return ("ready",)

        if self.state is CalibrationState.READY:
            if self.ready_since is None:
                self.ready_since = now
            if now - self.ready_since >= max(0.0, self.config.stable_seconds):
                self.state = CalibrationState.COUNTDOWN
                self.countdown_started = now
                return ("countdown_started",)
            return ()

        if self.state is CalibrationState.COUNTDOWN:
            if self.countdown_started is None:
                self.countdown_started = now
            if now - self.countdown_started >= max(0.0, self.config.countdown_seconds):
                self.state = CalibrationState.CALIBRATION_OK
                return ("calibration_ok",)
        return ()

    def countdown_number(self, now: float) -> int | None:
        if self.state is not CalibrationState.COUNTDOWN or self.countdown_started is None:
            return None
        remaining = self.config.countdown_seconds - (float(now) - self.countdown_started)
        if remaining <= 0:
            return None
        return max(1, math.ceil(remaining))

    @property
    def red_on(self) -> bool:
        return self.state in {CalibrationState.POSITIONING, CalibrationState.READY}

    @property
    def green_on(self) -> bool:
        return self.state in {CalibrationState.COUNTDOWN, CalibrationState.CALIBRATION_OK}
