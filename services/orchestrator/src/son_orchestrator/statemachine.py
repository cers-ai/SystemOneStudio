"""The 7-step workflow state machine (需求方案.txt 4.1 / 5).

This is the orchestrator's spine. Per 技术方案.md section 8 it is the single
definition of which steps exist and which may follow which, so a wizard screen
and a database ``status`` column cannot drift apart -- both read from
``son_contracts.StepState``.

Three product rules are encoded here rather than left to callers:

* Every step may be skipped with its defaults (需求方案.txt 4.1: "每步右上角有
  '跳过，用默认值'按钮"). A skipped step is recorded distinctly from a configured
  one, otherwise evaluation data cannot tell the difference.
* Rapid mode ends after step 2 (需求方案.txt 4.1 / AGENTS.md). Beyond that the
  orchestrator drives, and a manual user action past the boundary is rejected
  so the "30 minutes, two steps" promise stays true.
* Steps do not have to be visited in order once complete; re-running an earlier
  step is allowed because 需求方案.txt principle 5 requires every step to be
  reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from son_contracts import (
    RAPID_MODE_LAST_MANUAL_STEP,
    WIZARD_STEP_ORDER,
    RunState,
    StepState,
    UiMode,
)


class TransitionError(RuntimeError):
    """An illegal workflow move."""


class ManualStepError(TransitionError):
    """A manual action attempted past the rapid-mode boundary."""


#: The seven steps a user walks through. EVALUATED and DEPLOYED follow from step
#: 7 and are not separately configured.
USER_STEPS: tuple[StepState, ...] = WIZARD_STEP_ORDER[:7]

#: Steps the orchestrator drives after the user-configured ones.
AUTOMATED_STEPS: tuple[StepState, ...] = WIZARD_STEP_ORDER[7:]


@dataclass(frozen=True)
class StepRecord:
    """One step's entry in the workflow log."""

    step: StepState
    state: RunState = RunState.PENDING
    used_defaults: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    @property
    def is_terminal_failure(self) -> bool:
        return self.state is RunState.FAILED


@dataclass
class Workflow:
    """A run's position in the pipeline."""

    workflow_id: str
    mode: UiMode = UiMode.WIZARD
    records: dict[StepState, StepRecord] = field(default_factory=dict)

    def _record(self, step: StepState) -> StepRecord:
        return self.records.get(step) or StepRecord(step=step)

    def position(self, step: StepState) -> int:
        return WIZARD_STEP_ORDER.index(step)

    def is_complete(self, step: StepState) -> bool:
        return self._record(step).state is RunState.SUCCEEDED

    def completed_steps(self) -> tuple[StepState, ...]:
        return tuple(s for s in WIZARD_STEP_ORDER if self.is_complete(s))

    def next_step(self) -> StepState | None:
        """The first step that has not succeeded."""
        for step in WIZARD_STEP_ORDER:
            if not self.is_complete(step):
                return step
        return None

    def failed_step(self) -> StepState | None:
        for step in WIZARD_STEP_ORDER:
            if self._record(step).is_terminal_failure:
                return step
        return None

    def manual_boundary(self) -> StepState:
        """The last step a human is allowed to drive in this mode.

        Wizard and expert modes let the user configure all seven; rapid mode
        stops after step 2 (需求方案.txt 4.1).
        """
        if self.mode is UiMode.RAPID:
            return RAPID_MODE_LAST_MANUAL_STEP
        return USER_STEPS[-1]

    def _assert_manual_allowed(self, step: StepState) -> None:
        if self.mode is UiMode.RAPID and self.position(step) > self.position(
            RAPID_MODE_LAST_MANUAL_STEP
        ):
            raise ManualStepError(
                f"极速模式下第 {self.position(step) + 1} 步由系统自动完成，"
                f"人工配置只到第 {self.position(RAPID_MODE_LAST_MANUAL_STEP) + 1} 步"
            )

    def complete_step(
        self,
        step: StepState,
        *,
        used_defaults: bool = False,
        payload: dict[str, Any] | None = None,
        reason: str | None = None,
        by_user: bool = True,
    ) -> Workflow:
        """Mark a step done.

        Marking a step the user has not completed with a human decision is
        exactly what rapid mode does automatically, hence ``by_user``.
        """
        if by_user:
            self._assert_manual_allowed(step)

        self.records[step] = StepRecord(
            step=step,
            state=RunState.SUCCEEDED,
            used_defaults=used_defaults,
            payload=payload or {},
            reason=reason,
        )
        return self

    def fail_step(self, step: StepState, reason: str) -> Workflow:
        self.records[step] = StepRecord(step=step, state=RunState.FAILED, reason=reason)
        return self

    def skip_step(self, step: StepState, reason: str = "使用默认值") -> Workflow:
        return self.complete_step(step, used_defaults=True, reason=reason)

    def reset_step(self, step: StepState) -> Workflow:
        """Clear a step so it can be revisited.

        需求方案.txt principle 5 requires every step to be reversible without
        losing data, so this only moves the marker; the recorded payload stays
        available to the caller.
        """
        self.records[step] = StepRecord(step=step)
        return self

    def reset_from(self, step: StepState) -> Workflow:
        """Clear `step` and everything downstream of it.

        Downstream steps were derived from the earlier inputs, so leaving them
        marked complete would let an evaluation report describe a model built
        from data that no longer exists.
        """
        start = self.position(step)
        for candidate in WIZARD_STEP_ORDER[start:]:
            self.records[candidate] = StepRecord(step=candidate)
        return self

    def assert_ready_for(self, step: StepState) -> None:
        """Every earlier user step must have succeeded before `step` runs."""
        for earlier in USER_STEPS:
            if self.position(earlier) >= self.position(step):
                break
            if not self.is_complete(earlier):
                raise TransitionError(f"步骤 {earlier.value} 未完成，无法进入 {step.value}")

    def describe(self) -> str:
        parts = []
        for step in WIZARD_STEP_ORDER:
            record = self._record(step)
            mark = {
                RunState.SUCCEEDED: "✓",
                RunState.FAILED: "✗",
                RunState.PENDING: "·",
            }.get(record.state, "?")
            suffix = "（默认）" if record.used_defaults else ""
            parts.append(f"{mark} {step.value}{suffix}")
        return " | ".join(parts)


def start_workflow(workflow_id: str, mode: UiMode = UiMode.WIZARD) -> Workflow:
    return Workflow(workflow_id=workflow_id, mode=mode)


def advance_through_defaults(workflow: Workflow, up_to: StepState) -> Workflow:
    """Fill in every step up to and including `up_to` with its defaults.

    This is the rapid-mode driver: after the user finishes step 2, the
    orchestrator marks the remaining steps with their computed defaults rather
    than stopping and asking. Steps the user already configured are left alone.
    """
    target = workflow.position(up_to)
    for step in WIZARD_STEP_ORDER:
        if workflow.position(step) > target:
            break
        if not workflow.is_complete(step):
            workflow.complete_step(
                step, used_defaults=True, reason="系统自动采用推荐值", by_user=False
            )
    return workflow


def clone_workflow(source: Workflow, workflow_id: str) -> Workflow:
    """Deep-ish copy for the 克隆任务 control (需求方案.txt 5.6).

    Carries the configuration but resets completion state, so the clone is a new
    run rather than a second name for the same one.
    """
    return replace(
        start_workflow(workflow_id, mode=source.mode),
        records={
            step: replace(record, state=RunState.PENDING) for step, record in source.records.items()
        },
    )
