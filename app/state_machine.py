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
    INCOMPLETE = "INCOMPLETE"


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
    hands_lost_grace_started_at: float | None = None
    success_since: float | None = None
    washing_since: float | None = None
    incomplete_since: float | None = None
    incomplete_wall_since: float | None = None
    # Observed classes are evidence only. Accepted steps are confirmations
    # made by the state machine and are the only source used by SUCCESS.
    accepted_steps: set[int] = field(default_factory=set)
    observed_classes: set[int] = field(default_factory=set)
    observed_step_coverage: float = 0.0
    # Time spent in a temporary hand-loss grace period is excluded from the
    # active washing timer and accumulated here for deterministic resumption.
    washing_paused_seconds: float = 0.0
    last_hands_lost_duration: float | None = field(default=None, init=False)
    last_active_washing_time: float | None = field(default=None, init=False)
    last_success_diagnostics: dict[str, object] | None = field(default=None, init=False)

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
        self.hands_lost_grace_started_at = None
        self.success_since = None
        self.washing_since = None
        self.incomplete_since = None
        self.incomplete_wall_since = None
        self.accepted_steps.clear()
        self.observed_classes.clear()
        self.observed_step_coverage = 0.0
        self.washing_paused_seconds = 0.0
        self.last_hands_lost_duration = None
        self.last_active_washing_time = None
        self.last_success_diagnostics = None
        return [AppEvent("reset")]

    def force_preview(self, state: AppState, step: int = 1) -> None:
        clock = time.monotonic()
        self.state = state
        self.expected_step = min(6, max(1, int(step)))
        self.correct_streak = 0
        self.incorrect_streak = 0
        self.success_since = clock if state is AppState.SUCCESS else None
        self.last_pose_at = clock if state in {AppState.WASHING, AppState.CORRECTION} else None
        self.hands_lost_announced = False
        self.hands_lost_grace_started_at = None
        self.washing_since = self.last_pose_at
        self.incomplete_since = clock if state is AppState.INCOMPLETE else None
        self.incomplete_wall_since = self.incomplete_since
        self.accepted_steps.clear()
        self.observed_classes.clear()
        self.observed_step_coverage = 0.0
        self.washing_paused_seconds = 0.0
        self.last_hands_lost_duration = None
        self.last_active_washing_time = None
        self.last_success_diagnostics = None

    @property
    def accepted_step_count(self) -> int:
        return len(self.accepted_steps)

    @property
    def accepted_step_coverage(self) -> float:
        return self.accepted_step_count / 6.0

    @property
    def observed_class_count(self) -> int:
        return len(self.observed_classes)

    def washing_time(self, now: float | None = None) -> float | None:
        """Return active washing time, excluding temporary hand-loss pauses."""

        if self.washing_since is None:
            return None
        now = time.monotonic() if now is None else float(now)
        paused_seconds = max(0.0, self.washing_paused_seconds)
        if self.hands_lost_grace_started_at is not None:
            paused_seconds += max(0.0, now - self.hands_lost_grace_started_at)
        return max(0.0, now - self.washing_since - paused_seconds)

    def success_diagnostics(self, now: float | None = None) -> dict[str, object]:
        """Expose success predicates without changing their behavior."""

        now = time.monotonic() if now is None else float(now)
        state_ok = self.state in {AppState.WASHING, AppState.CORRECTION}
        washing_time = self.washing_time(now)
        time_ok = bool(
            state_ok
            and washing_time is not None
            and washing_time >= max(0.0, self.config.minimum_washing_seconds)
        )
        hands_ok = bool(
            state_ok
            and self.hands_lost_grace_started_at is None
            and self.last_pose_at is not None
            and now - self.last_pose_at < max(0.0, self.config.hands_lost_seconds)
        )
        steps_ok = bool(
            state_ok
            and sorted(self.accepted_steps) == list(range(1, 7))
        )
        coverage_ok = bool(
            state_ok
            and self.accepted_step_coverage
            >= min(1.0, max(0.0, self.config.minimum_step_coverage))
        )
        return {
            "time_ok": time_ok,
            "steps_ok": steps_ok,
            "coverage_ok": coverage_ok,
            "hands_ok": hands_ok,
            "final_success": bool(state_ok and time_ok and hands_ok and steps_ok),
            "washing_time": washing_time,
            "washing_paused_seconds": self.washing_paused_seconds,
            "accepted_steps": sorted(self.accepted_steps),
            "accepted_step_count": self.accepted_step_count,
            "accepted_step_coverage": self.accepted_step_coverage,
            "observed_classes": sorted(self.observed_classes),
            "observed_class_count": self.observed_class_count,
            "observed_step_coverage": self.observed_step_coverage,
        }

    def _record_observation_evidence(
        self,
        class_id: int | None,
        observed_classes: Iterable[int] | None,
        recognized_steps: Iterable[int] | None,
        step_coverage: float | None,
    ) -> None:
        candidates: list[object] = []
        if class_id is not None:
            candidates.append(class_id)
        for values in (observed_classes, recognized_steps):
            if values is None:
                continue
            try:
                candidates.extend(values)
            except TypeError:
                candidates.append(values)
        for candidate in candidates:
            try:
                step = int(candidate)
            except (TypeError, ValueError):
                continue
            if step in range(0, 7):
                self.observed_classes.add(step)
        if step_coverage is not None:
            try:
                coverage = float(step_coverage)
            except (TypeError, ValueError):
                coverage = 0.0
            if math.isfinite(coverage):
                if coverage > 1.0:
                    coverage /= 100.0
                self.observed_step_coverage = max(
                    self.observed_step_coverage,
                    min(1.0, max(0.0, coverage)),
                )

    def _success_ready(self, now: float) -> bool:
        return bool(self.success_diagnostics(now)["final_success"])

    def _success_event(self, now: float) -> AppEvent:
        # Keep the predicates available to callers before changing the state.
        self.last_success_diagnostics = self.success_diagnostics(now)
        self.state = AppState.SUCCESS
        self.success_since = now
        self.correct_streak = 0
        self.incorrect_streak = 0
        return AppEvent("success", self.accepted_step_count)

    def _enter_incomplete(self, now: float) -> list[AppEvent]:
        # washing_time() excludes the current grace interval before it is
        # closed, so this captures the actual active duration for diagnostics.
        self.last_active_washing_time = self.washing_time(now)
        if self.hands_lost_grace_started_at is not None:
            self.last_hands_lost_duration = max(
                0.0, now - self.hands_lost_grace_started_at
            )
            self.washing_paused_seconds += self.last_hands_lost_duration
        self.hands_lost_announced = True
        self.correct_streak = 0
        self.incorrect_streak = 0
        self.washing_since = None
        self.hands_lost_grace_started_at = None
        self.incomplete_since = now
        self.incomplete_wall_since = time.monotonic()
        self.state = AppState.INCOMPLETE
        return [AppEvent("hands_lost"), AppEvent("attempt_incomplete", self.expected_step)]

    def tick(self, now: float | None = None) -> list[AppEvent]:
        now = time.monotonic() if now is None else float(now)
        if self.state is AppState.INCOMPLETE:
            wall_since = self.incomplete_wall_since
            if (
                wall_since is not None
                and time.monotonic() - wall_since
                >= max(0.0, self.config.incomplete_hold_seconds)
            ):
                return self.reset(now)
            return []
        if self.state in {AppState.WASHING, AppState.CORRECTION}:
            if self.hands_lost_grace_started_at is not None:
                if now - self.hands_lost_grace_started_at >= max(
                    0.0, self.config.hands_lost_seconds
                ):
                    return self._enter_incomplete(now)
                return []
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
        observed_classes: Iterable[int] | None = None,
        recognized_steps: Iterable[int] | None = None,
        step_coverage: float | None = None,
    ) -> list[AppEvent]:
        now = time.monotonic() if now is None else float(now)
        events: list[AppEvent] = []

        if self.state is AppState.INCOMPLETE:
            return events

        if pose_present:
            grace_started_at = self.hands_lost_grace_started_at
            returned_from_grace = (
                self.state in {AppState.WASHING, AppState.CORRECTION}
                and grace_started_at is not None
            )
            self.last_pose_at = now
            self.hands_lost_announced = False
            self.hands_lost_grace_started_at = None
            if self.state in {AppState.WASHING, AppState.CORRECTION} and self.washing_since is None:
                self.washing_since = now
            if returned_from_grace:
                hands_lost_duration = max(0.0, now - grace_started_at)
                self.last_hands_lost_duration = hands_lost_duration
                self.washing_paused_seconds += hands_lost_duration
            self._record_observation_evidence(
                class_id,
                observed_classes,
                recognized_steps,
                step_coverage,
            )
            if returned_from_grace:
                self.correct_streak = 0
                self.incorrect_streak = 0
                events.append(AppEvent("hands_returned", self.expected_step))
                return events
        elif self.state in {AppState.WASHING, AppState.CORRECTION}:
            if self.hands_lost_grace_started_at is None:
                self.hands_lost_grace_started_at = now
                events.append(AppEvent("hands_lost_grace_started", self.expected_step))
            if (
                now - self.hands_lost_grace_started_at
                >= max(0.0, self.config.hands_lost_seconds)
                and not self.hands_lost_announced
            ):
                events.extend(self._enter_incomplete(now))
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
            self.accepted_steps.add(completed_step)
            self.correct_streak = 0
            events.append(AppEvent("step_accepted", completed_step))
            if completed_step == 6:
                self.state = AppState.WASHING
                if self._success_ready(now):
                    events.append(self._success_event(now))
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
