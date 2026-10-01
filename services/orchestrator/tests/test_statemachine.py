"""State machine tests.

The rules under test are product rules from 需求方案.txt 4.1 and principle 5,
not implementation details.
"""

from __future__ import annotations

import pytest
from son_orchestrator import (
    AUTOMATED_STEPS,
    USER_STEPS,
    ManualStepError,
    TransitionError,
    Workflow,
    advance_through_defaults,
    clone_workflow,
    start_workflow,
)

from son_contracts import (
    RAPID_MODE_LAST_MANUAL_STEP,
    WIZARD_STEP_ORDER,
    RunState,
    StepState,
    UiMode,
)

S1, S2, S3, S4, S5, S6, S7 = USER_STEPS


def _wizard(*done: StepState) -> Workflow:
    flow = start_workflow("wf-1")
    for step in done:
        flow.complete_step(step)
    return flow


class TestStepCount:
    def test_seven_user_facing_steps(self) -> None:
        assert len(USER_STEPS) == 7

    def test_evaluate_and_deploy_are_automated_follow_ons(self) -> None:
        assert AUTOMATED_STEPS == (StepState.EVALUATED, StepState.DEPLOYED)

    def test_steps_match_the_contract_enum(self) -> None:
        assert WIZARD_STEP_ORDER[:7] == USER_STEPS


class TestOrdering:
    def test_next_step_starts_at_step_one(self) -> None:
        assert start_workflow("wf").next_step() is S1

    def test_next_step_advances(self) -> None:
        assert _wizard(S1).next_step() is S2

    def test_next_step_is_none_when_all_done(self) -> None:
        assert _wizard(*WIZARD_STEP_ORDER).next_step() is None

    def test_cannot_enter_a_step_before_its_predecessor(self) -> None:
        with pytest.raises(TransitionError, match="scene_selected"):
            _wizard().assert_ready_for(S3)

    def test_ready_when_predecessors_are_done(self) -> None:
        _wizard(S1, S2).assert_ready_for(S3)

    def test_a_failed_step_is_reported(self) -> None:
        flow = _wizard(S1)
        flow.fail_step(S2, "列数不匹配")
        assert flow.failed_step() is S2
        assert flow.next_step() is S2

    def test_failure_carries_its_reason(self) -> None:
        flow = start_workflow("wf")
        flow.fail_step(S2, "列数不匹配")
        assert flow.records[S2].reason == "列数不匹配"


class TestSkipWithDefaults:
    """需求方案.txt 4.1: every step offers "跳过，用默认值"."""

    def test_skipped_step_counts_as_complete(self) -> None:
        flow = start_workflow("wf")
        flow.skip_step(S1)
        assert flow.is_complete(S1)

    def test_defaults_are_recorded_distinctly(self) -> None:
        """Evaluation cannot otherwise tell a default run from a configured one."""
        flow = start_workflow("wf")
        flow.skip_step(S1)
        assert flow.records[S1].used_defaults is True

    def test_configured_step_is_not_marked_as_defaults(self) -> None:
        flow = start_workflow("wf")
        flow.complete_step(S1, payload={"template": "fraud"})
        assert flow.records[S1].used_defaults is False

    def test_payload_is_kept(self) -> None:
        flow = start_workflow("wf")
        flow.complete_step(S1, payload={"template": "fraud"})
        assert flow.records[S1].payload == {"template": "fraud"}

    def test_skip_reason_is_recorded(self) -> None:
        flow = start_workflow("wf")
        flow.skip_step(S1)
        assert flow.records[S1].reason == "使用默认值"


class TestReversibility:
    """需求方案.txt principle 5: every step can be rolled back without data loss."""

    def test_reset_clears_a_step(self) -> None:
        flow = _wizard(S1, S2)
        flow.reset_step(S2)
        assert not flow.is_complete(S2)

    def test_reset_from_clears_downstream(self) -> None:
        flow = _wizard(*USER_STEPS)
        flow.reset_from(S3)
        assert flow.is_complete(S1) and flow.is_complete(S2)
        assert not flow.is_complete(S3)
        assert not flow.is_complete(S7)

    def test_reset_from_leaves_earlier_steps_alone(self) -> None:
        flow = _wizard(S1, S2, S3)
        flow.reset_from(S3)
        assert flow.records[S3].state is RunState.PENDING

    def test_revisiting_an_earlier_step_is_allowed(self) -> None:
        flow = _wizard(S1, S2, S3)
        flow.complete_step(S1, payload={"template": "other"})
        assert flow.records[S1].payload["template"] == "other"


class TestRapidMode:
    """需求方案.txt 4.1: rapid mode requires only steps 1 and 2 from the user."""

    def test_boundary_is_step_two(self) -> None:
        assert start_workflow("wf", UiMode.RAPID).manual_boundary() is S2
        assert RAPID_MODE_LAST_MANUAL_STEP is S2

    def test_step_one_and_two_are_allowed(self) -> None:
        flow = start_workflow("wf", UiMode.RAPID)
        flow.complete_step(S1)
        flow.complete_step(S2)

    def test_manually_configuring_step_three_is_refused(self) -> None:
        flow = start_workflow("wf", UiMode.RAPID)
        with pytest.raises(ManualStepError, match="极速模式"):
            flow.complete_step(S3)

    def test_refusal_explains_the_boundary(self) -> None:
        flow = start_workflow("wf", UiMode.RAPID)
        with pytest.raises(ManualStepError, match="第 2 步"):
            flow.complete_step(S4)

    def test_automated_completion_bypasses_the_boundary(self) -> None:
        flow = start_workflow("wf", UiMode.RAPID)
        flow.complete_step(S3, used_defaults=True, by_user=False)
        assert flow.is_complete(S3)

    def test_wizard_mode_allows_all_seven(self) -> None:
        flow = start_workflow("wf", UiMode.WIZARD)
        for step in USER_STEPS:
            flow.complete_step(step)
        assert len(flow.completed_steps()) == 7

    def test_manual_step_error_is_a_transition_error(self) -> None:
        assert issubclass(ManualStepError, TransitionError)


class TestAdvanceThroughDefaults:
    def test_rapid_driver_fills_the_rest(self) -> None:
        """After step 2 the orchestrator drives through training and evaluation."""
        flow = start_workflow("wf", UiMode.RAPID)
        flow.complete_step(S1)
        flow.complete_step(S2)
        advance_through_defaults(flow, StepState.EVALUATED)
        assert flow.next_step() is StepState.DEPLOYED

    def test_deployment_stays_with_the_user(self) -> None:
        """一键部署 is an explicit user action, not something to auto-complete."""
        flow = start_workflow("wf", UiMode.RAPID)
        flow.complete_step(S2)
        advance_through_defaults(flow, StepState.EVALUATED)
        assert not flow.is_complete(StepState.DEPLOYED)

    def test_driver_can_complete_the_whole_run(self) -> None:
        flow = start_workflow("wf", UiMode.RAPID)
        flow.complete_step(S2)
        advance_through_defaults(flow, StepState.DEPLOYED)
        assert flow.next_step() is None

    def test_filled_steps_are_marked_as_defaults(self) -> None:
        flow = start_workflow("wf", UiMode.RAPID)
        flow.complete_step(S2)
        advance_through_defaults(flow, StepState.EVALUATED)
        assert flow.records[S4].used_defaults is True

    def test_already_completed_steps_are_not_overwritten(self) -> None:
        flow = start_workflow("wf", UiMode.RAPID)
        flow.complete_step(S1, payload={"template": "fraud"})
        advance_through_defaults(flow, StepState.EVALUATED)
        assert flow.records[S1].payload == {"template": "fraud"}

    def test_driver_marks_the_system_as_the_source(self) -> None:
        flow = start_workflow("wf", UiMode.RAPID)
        advance_through_defaults(flow, StepState.EVALUATED)
        assert "系统自动" in (flow.records[S5].reason or "")


class TestDescribe:
    def test_shows_every_step(self) -> None:
        text = start_workflow("wf").describe()
        for step in WIZARD_STEP_ORDER:
            assert step.value in text

    def test_marks_default_steps(self) -> None:
        flow = start_workflow("wf")
        flow.skip_step(S1)
        assert "（默认）" in flow.describe()

    def test_marks_failure(self) -> None:
        flow = start_workflow("wf")
        flow.fail_step(S2, "boom")
        assert "✗" in flow.describe()


class TestClone:
    def test_clone_keeps_configuration(self) -> None:
        source = start_workflow("wf-1")
        source.complete_step(S1, payload={"template": "fraud"})
        cloned = clone_workflow(source, "wf-2")
        assert cloned.records[S1].payload == {"template": "fraud"}

    def test_clone_resets_completion(self) -> None:
        """A clone is a new run, not a second name for the same one."""
        source = start_workflow("wf-1")
        source.complete_step(S1)
        cloned = clone_workflow(source, "wf-2")
        assert not cloned.is_complete(S1)

    def test_clone_has_a_new_id(self) -> None:
        source = start_workflow("wf-1")
        assert clone_workflow(source, "wf-2").workflow_id == "wf-2"

    def test_clone_keeps_the_mode(self) -> None:
        source = start_workflow("wf-1", UiMode.RAPID)
        assert clone_workflow(source, "wf-2").mode is UiMode.RAPID

    def test_completion_order_matches_the_enum(self) -> None:
        flow = start_workflow("wf")
        for step in reversed(USER_STEPS):
            flow.complete_step(step)
        assert flow.completed_steps() == USER_STEPS
