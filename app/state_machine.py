from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
import math
import time

from .config import RuntimeConfig, expected_asset


class AppState(str, Enum):
    WAITING_FOR_HANDS = "WAITING_FOR_HANDS"
    WASHING = "WASHING"
    CORRECTION = "CORRECTION"
    SUCCESS = "SUCCESS"


@dataclass(frozen=True)
class AppEvent:
    name: str
    value: int | None = None


@dataclass
class SessionStateMachine:
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    state: AppState = AppState.WAITING_FOR_HANDS
    expected_step: int = 1
    correct_streak: int = 0
    incorrect_streak: int = 0
    last_pose_at: float | None = None
    hands_lost_announced: bool = False
    success_since: float | None = None
    washing_since: float | None = None
    recognized_steps: set[int] = field(default_factory=set)
    reported_step_coverage: float = 0.0

    @property
    def asset(self):
        return expected_asset(self.state.value, self.expected_step)

    def reset(self, now: float | None = None) -> list[AppEvent]:
        self.state = AppState.WAITING_FOR_HANDS
        self.expected_step = 1
        self.correct_streak = 0
        self.incorrect_streak = 0
        self.last_pose_at = now
        self.hands_lost_announced = False
        self.success_since = None
        self.washing_since = None
        self.recognized_steps.clear()
        self.reported_step_coverage = 0.0
        return [AppEvent("reset")]

    def force_preview(self, state: AppState, step: int = 1) -> None:
        self.state = state
        self.expected_step = min(6, max(1, int(step)))
        self.correct_streak = 0
        self.incorrect_streak = 0
        self.success_since = time.monotonic() if state is AppState.SUCCESS else None
        self.last_pose_at = time.monotonic() if state in {AppState.WASHING, AppState.CORRECTION} else None
        self.hands_lost_announced = False
        self.washing_since = self.last_pose_at
        self.recognized_steps.clear()
        self.reported_step_coverage = 0.0

    @property
    def recognized_step_count(self) -> int:
        return len(self.recognized_steps)

    @property
    def recognized_step_coverage(self) -> float:
        return max(self.recognized_step_count / 6.0, self.reported_step_coverage)

    def _record_step_evidence(
        self,
        class_id: int | None,
        recognized_steps: Iterable[int] | None,
        step_coverage: float | None,
    ) -> None:
        candidates: list[object] = []
        if class_id is not None:
            candidates.append(class_id)
        if recognized_steps is not None:
            try:
                candidates.extend(recognized_steps)
            except TypeError:
                candidates.append(recognized_steps)
        for candidate in candidates:
            try:
                step = int(candidate)
            except (TypeError, ValueError):
                continue
            if step in range(1, 7):
                self.recognized_steps.add(step)
        if step_coverage is not None:
            try:
                coverage = float(step_coverage)
            except (TypeError, ValueError):
                coverage = 0.0
            if math.isfinite(coverage):
                if coverage > 1.0:
                    coverage /= 100.0
                self.reported_step_coverage = max(
                    self.reported_step_coverage,
                    min(1.0, max(0.0, coverage)),
                )

    def _success_ready(self, now: float) -> bool:
        if self.state not in {AppState.WASHING, AppState.CORRECTION}:
            return False
        if self.washing_since is None:
            return False
        if now - self.washing_since < max(0.0, self.config.minimum_washing_seconds):
            return False
        if self.last_pose_at is None or now - self.last_pose_at >= self.config.hands_lost_seconds:
            return False
        enough_steps = self.recognized_step_count >= max(1, self.config.minimum_steps_for_success)
        enough_coverage = self.recognized_step_coverage >= min(
            1.0, max(0.0, self.config.minimum_step_coverage)
        )
        return enough_steps or enough_coverage

    def _success_event(self, now: float) -> AppEvent:
        self.state = AppState.SUCCESS
        self.success_since = now
        self.correct_streak = 0
        self.incorrect_streak = 0
        return AppEvent("success", self.recognized_step_count)
    def tick(self, now: float | None = None) -> list[AppEvent]:
        now = time.monotonic() if now is None else float(now)
        if self._success_ready(now):
            return [self._success_event(now)]
        if (
            self.state is AppState.SUCCESS
            and self.success_since is not None
            and now - self.success_since >= self.config.success_hold_seconds
        ):
            return self.reset(now)
        return []

    def observe(
        self,
        *,
        class_id: int | None,
        pose_present: bool,
        now: float | None = None,
        recognized_steps: Iterable[int] | None = None,
        step_coverage: float | None = None,
    ) -> list[AppEvent]:
        now = time.monotonic() if now is None else float(now)
        events: list[AppEvent] = []

        if pose_present:
            self.last_pose_at = now
            self.hands_lost_announced = False
            if self.state in {AppState.WASHING, AppState.CORRECTION} and self.washing_since is None:
                self.washing_since = now
            self._record_step_evidence(class_id, recognized_steps, step_coverage)
        elif self.state in {AppState.WASHING, AppState.CORRECTION}:
            last = self.last_pose_at if self.last_pose_at is not None else now
            if now - last >= self.config.hands_lost_seconds and not self.hands_lost_announced:
                self.hands_lost_announced = True
                self.correct_streak = 0
                self.incorrect_streak = 0
                self.washing_since = None
                events.append(AppEvent("hands_lost"))
            return events

        if self.state is AppState.WAITING_FOR_HANDS:
            if pose_present:
                self.state = AppState.WASHING
                self.washing_since = now
                events.append(AppEvent("session_started", self.expected_step))
            return events
        if self._success_ready(now):
            events.append(self._success_event(now))
            return events

        if self.state is AppState.SUCCESS or class_id is None:
            return events

        # La clase 0 es otro movimiento de lavado, no una clase negativa.
        if class_id == 0:
            self.correct_streak = 0
            return events

        if class_id == self.expected_step:
            self.correct_streak += 1
            self.incorrect_streak = 0
            if self.correct_streak < self.config.correct_predictions_required:
                return events

            recovered = self.state is AppState.CORRECTION
            completed_step = self.expected_step
            self.correct_streak = 0
            if completed_step == 6:
                self.state = AppState.WASHING
                return events

            self.expected_step += 1
            self.state = AppState.WASHING
            if recovered:
                events.append(AppEvent("recovered", completed_step))
            events.append(AppEvent("step_advanced", self.expected_step))
            return events

        if class_id in range(1, 7):
            self.correct_streak = 0
            self.incorrect_streak += 1
            if self.incorrect_streak >= self.config.incorrect_predictions_required:
                first_correction = self.state is not AppState.CORRECTION
                self.state = AppState.CORRECTION
                self.incorrect_streak = 0
                events.append(AppEvent("correction" if first_correction else "retry", self.expected_step))
        return events

