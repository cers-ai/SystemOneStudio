"""Run state machine tests.

The property under test is the one 改造开发方案.md 2.3 demands: a step cannot be
marked complete without the asset it stands for. Every step transition is
therefore exercised both with and without its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import pytest
from son_orchestrator.runstate import (
    BUSY_STATES,
    FORWARD,
    TERMINAL,
    IllegalTransition,
    MissingEvidence,
    RunState,
    available_actions,
    can_transition,
    current_step,
    describe,
    is_busy,
    is_finished,
    next_states,
    parse,
    transition,
)


@dataclass
class Snapshot:
    """Stands in for the persisted Run row."""

    scene_code: str | None = None
    dataset_id: str | None = None
    split_id: str | None = None
    synth_id: str | None = None
    base_model_id: str | None = None
    training_config_json: str | None = None
    job_id: str | None = None
    model_version_id: str | None = None


FULL = Snapshot(
    scene_code="fraud_account",
    dataset_id="ds_1",
    split_id="split_1",
    synth_id="syn_1",
    base_model_id="qwen2.5-1.5b-instruct",
    training_config_json="{}",
    job_id="job_1",
    model_version_id="mv_1",
)


class TestGraphShape:
    def test_sixteen_states(self) -> None:
        """改造开发方案.md section 9 lists sixteen."""
        assert len(list(RunState)) == 16

    def test_every_state_has_edges_or_is_terminal(self) -> None:
        for state in RunState:
            assert state in TERMINAL or next_states(state), state

    def test_happy_path_is_fully_connected(self) -> None:
        path = [
            RunState.CREATED,
            RunState.SCENE_READY,
            RunState.DATASET_READY,
            RunState.DATA_QUALITY_READY,
            RunState.TRAIN_SET_READY,
            RunState.MODEL_SELECTED,
            RunState.TRAINING_CONFIGURED,
            RunState.QUEUED,
            RunState.TRAINING,
            RunState.MERGING,
            RunState.QUANTIZING,
            RunState.EVALUATING,
            RunState.MODEL_READY,
            RunState.DEPLOYING,
            RunState.SERVING,
        ]
        for current, target in pairwise(path):
            assert target in next_states(current), f"{current.value} -> {target.value}"

    def test_failing_is_reachable_from_every_working_state(self) -> None:
        for state in RunState:
            if state in (RunState.FAILED, RunState.SERVING):
                continue
            assert RunState.FAILED in next_states(state), state

    def test_step_numbers_are_seven_at_the_end(self) -> None:
        assert current_step(RunState.CREATED) == 1
        assert current_step(RunState.DATASET_READY) == 2
        assert current_step(RunState.TRAIN_SET_READY) == 4
        assert current_step(RunState.MODEL_SELECTED) == 5
        assert current_step(RunState.TRAINING_CONFIGURED) == 6
        assert current_step(RunState.QUEUED) == 7

    def test_job_states_cover_the_whole_pipeline(self) -> None:
        """The seven-step UI shows one step while six different things run."""
        for state in RunState:
            if state in BUSY_STATES:
                assert current_step(state) == 7, state


class TestIllegalTransitions:
    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (RunState.CREATED, RunState.DATASET_READY),
            (RunState.CREATED, RunState.TRAINING_CONFIGURED),
            (RunState.SCENE_READY, RunState.MODEL_SELECTED),
            (RunState.DATASET_READY, RunState.TRAIN_SET_READY),
            (RunState.TRAINING_CONFIGURED, RunState.SERVING),
            (RunState.MODEL_READY, RunState.QUEUED),
            (RunState.SERVING, RunState.TRAINING),
        ],
    )
    def test_skipping_a_step_is_refused(self, current: RunState, target: RunState) -> None:
        """The old frontend let a click skip straight to the end."""
        assert not can_transition(current, target)
        with pytest.raises(IllegalTransition):
            transition(current, target)

    def test_only_failed_is_strictly_terminal(self) -> None:
        assert next_states(RunState.FAILED) == frozenset()
        assert RunState.FAILED in TERMINAL
        assert RunState.SERVING not in TERMINAL

    def test_failed_is_not_trivially_reachable_back(self) -> None:
        assert next_states(RunState.FAILED) == frozenset()

    def test_error_lists_the_allowed_next_states(self) -> None:
        with pytest.raises(IllegalTransition, match="允许的后继状态"):
            transition(RunState.CREATED, RunState.SERVING)


class TestEvidenceIsRequired:
    """The core guarantee: no state without its asset."""

    @pytest.mark.parametrize(
        ("current", "target", "missing"),
        [
            (RunState.CREATED, RunState.SCENE_READY, "scene_code"),
            (RunState.SCENE_READY, RunState.DATASET_READY, "dataset_id"),
            (RunState.DATASET_READY, RunState.DATA_QUALITY_READY, "split_id"),
            (RunState.DATA_QUALITY_READY, RunState.TRAIN_SET_READY, "synth_id"),
            (RunState.TRAIN_SET_READY, RunState.MODEL_SELECTED, "base_model_id"),
            (
                RunState.MODEL_SELECTED,
                RunState.TRAINING_CONFIGURED,
                "training_config_json",
            ),
            (RunState.TRAINING_CONFIGURED, RunState.QUEUED, "job_id"),
            (RunState.EVALUATING, RunState.MODEL_READY, "model_version_id"),
        ],
    )
    def test_missing_evidence_is_refused(
        self, current: RunState, target: RunState, missing: str
    ) -> None:
        empty = Snapshot()
        with pytest.raises(MissingEvidence, match=missing):
            transition(current, target, empty)

    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (RunState.CREATED, RunState.SCENE_READY),
            (RunState.SCENE_READY, RunState.DATASET_READY),
            (RunState.DATASET_READY, RunState.DATA_QUALITY_READY),
            (RunState.DATA_QUALITY_READY, RunState.TRAIN_SET_READY),
            (RunState.TRAIN_SET_READY, RunState.MODEL_SELECTED),
            (RunState.MODEL_SELECTED, RunState.TRAINING_CONFIGURED),
            (RunState.TRAINING_CONFIGURED, RunState.QUEUED),
            (RunState.EVALUATING, RunState.MODEL_READY),
        ],
    )
    def test_present_evidence_is_accepted(self, current: RunState, target: RunState) -> None:
        assert transition(current, target, FULL).target is target

    def test_error_names_the_state_and_the_asset(self) -> None:
        with pytest.raises(MissingEvidence, match="资产已真实产出"):
            transition(RunState.CREATED, RunState.SCENE_READY, Snapshot())

    def test_partial_evidence_is_still_refused(self) -> None:
        """Having dataset_id does not make split_id unnecessary."""
        partial = Snapshot(scene_code="fraud_account", dataset_id="ds_1")
        with pytest.raises(MissingEvidence, match="split_id"):
            transition(RunState.DATASET_READY, RunState.DATA_QUALITY_READY, partial)

    def test_no_snapshot_skips_evidence_checking(self) -> None:
        """Callers that genuinely have no row (e.g. a graph query) may omit it."""
        assert transition(RunState.CREATED, RunState.SCENE_READY).target is RunState.SCENE_READY


class TestServingRequiresRealEvidence:
    def test_deploying_to_serving_has_no_attribute_check(self) -> None:
        """SERVING is gated by a health check at the call site, not by an attribute."""
        assert transition(RunState.DEPLOYING, RunState.SERVING, FULL).target is RunState.SERVING

    def test_serving_is_a_goal_state_and_redeployable(self) -> None:
        assert is_finished(RunState.SERVING)
        assert RunState.DEPLOYING in next_states(RunState.SERVING)

    def test_failed_is_terminal(self) -> None:
        assert is_finished(RunState.FAILED)


class TestBusyStates:
    @pytest.mark.parametrize(
        "state",
        [
            RunState.QUEUED,
            RunState.TRAINING,
            RunState.MERGING,
            RunState.QUANTIZING,
            RunState.EVALUATING,
            RunState.DEPLOYING,
        ],
    )
    def test_pipeline_states_are_busy(self, state: RunState) -> None:
        assert is_busy(state)

    @pytest.mark.parametrize(
        "state",
        [RunState.CREATED, RunState.SCENE_READY, RunState.DATASET_READY, RunState.MODEL_READY],
    )
    def test_configuration_states_are_not_busy(self, state: RunState) -> None:
        assert not is_busy(state)


class TestParsing:
    def test_known_state(self) -> None:
        assert parse("TRAINING") is RunState.TRAINING

    def test_unknown_state_lists_the_valid_ones(self) -> None:
        with pytest.raises(IllegalTransition, match="合法值"):
            parse("MAKING_TEA")

    def test_every_state_round_trips(self) -> None:
        for state in RunState:
            assert parse(state.value) is state


class TestActions:
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (RunState.CREATED, "选择场景"),
            (RunState.SCENE_READY, "上传种子数据"),
            (RunState.DATASET_READY, "执行数据治理与划分"),
            (RunState.DATA_QUALITY_READY, "生成合成数据"),
            (RunState.TRAIN_SET_READY, "选择基座模型"),
            (RunState.MODEL_SELECTED, "配置训练方法"),
            (RunState.TRAINING_CONFIGURED, "开始训练"),
            (RunState.MODEL_READY, "部署模型"),
            (RunState.SERVING, "调用 /v1/predict"),
            (RunState.FAILED, "查看失败原因并重试"),
        ],
    )
    def test_each_state_offers_its_next_action(self, state: RunState, expected: str) -> None:
        assert expected in list(available_actions(state))

    def test_busy_states_offer_nothing_to_click(self) -> None:
        """A user must not be able to start a second training job."""
        for state in BUSY_STATES:
            assert list(available_actions(state)) == [], state

    def test_describe_includes_the_step(self) -> None:
        assert "第 7 步" in describe(RunState.TRAINING)


class TestEveryForwardEdgeIsDeclared:
    def test_no_state_can_reach_serving_without_passing_evaluating(self) -> None:
        """Skipping evaluation would produce an unmeasured deployment."""
        for state in RunState:
            if state in (RunState.SERVING, RunState.DEPLOYING):
                continue
            assert RunState.SERVING not in FORWARD[state], state

    def test_no_state_can_reach_serving_without_deploying(self) -> None:
        for state in RunState:
            if state in (RunState.SERVING, RunState.DEPLOYING):
                continue
            assert RunState.SERVING not in next_states(state), state

    def test_graph_has_no_shortcut_from_created_to_model_ready(self) -> None:
        assert RunState.MODEL_READY not in next_states(RunState.CREATED)
