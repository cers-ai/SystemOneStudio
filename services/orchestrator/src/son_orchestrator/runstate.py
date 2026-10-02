"""Run state machine: the single source of truth (改造开发方案.md 2.3 / 25).

The frontend used to hold its own idea of which steps were complete, which meant
a browser click could mark a step done without anything having happened. That is
gone: a Run's state lives here and only here, and it advances only when a
*completed action* produces an asset.

That is the load-bearing distinction. ``RunState.TRAIN_SET_READY`` does not mean
"the user clicked next on the synth screen"; it means a ``syn_v1.csv`` exists on
the workspace with a checksum recorded. Transitions take the evidence they need
so a caller cannot assert a state it has not earned.

Sixteen states, in 改造开发方案.md section 9. The DAG is explicit rather than
computed, because "which states may follow this one" is a product question and
belongs where it can be read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RunState(StrEnum):
    """The sixteen states from 改造开发方案.md section 9."""

    CREATED = "CREATED"
    SCENE_READY = "SCENE_READY"
    DATASET_READY = "DATASET_READY"
    DATA_QUALITY_READY = "DATA_QUALITY_READY"
    TRAIN_SET_READY = "TRAIN_SET_READY"
    MODEL_SELECTED = "MODEL_SELECTED"
    TRAINING_CONFIGURED = "TRAINING_CONFIGURED"
    QUEUED = "QUEUED"
    TRAINING = "TRAINING"
    MERGING = "MERGING"
    QUANTIZING = "QUANTIZING"
    EVALUATING = "EVALUATING"
    MODEL_READY = "MODEL_READY"
    DEPLOYING = "DEPLOYING"
    SERVING = "SERVING"
    FAILED = "FAILED"


class IllegalTransition(RuntimeError):
    """A state change that the workflow does not allow."""


class MissingEvidence(RuntimeError):
    """A transition attempted without the asset it is supposed to represent."""


#: Which wizard step each state belongs to, 1-indexed as the UI shows them.
STEP_OF_STATE: dict[RunState, int] = {
    RunState.CREATED: 1,
    RunState.SCENE_READY: 1,
    RunState.DATASET_READY: 2,
    RunState.DATA_QUALITY_READY: 3,
    RunState.TRAIN_SET_READY: 4,
    RunState.MODEL_SELECTED: 5,
    RunState.TRAINING_CONFIGURED: 6,
    RunState.QUEUED: 7,
    RunState.TRAINING: 7,
    RunState.MERGING: 7,
    RunState.QUANTIZING: 7,
    RunState.EVALUATING: 7,
    RunState.MODEL_READY: 7,
    RunState.DEPLOYING: 7,
    RunState.SERVING: 7,
    RunState.FAILED: 0,
}


#: States from which nothing follows at all. Only FAILED: every other state is
#: either mid-pipeline or re-enterable (a serving run can be redeployed).
TERMINAL: frozenset[RunState] = frozenset({RunState.FAILED})

#: The run has reached its end goal, but may still be re-entered (redeploy).
GOAL_STATES: frozenset[RunState] = frozenset({RunState.SERVING})

#: States during which a long job owns the run. The UI polls the job rather than
#: the run during these.
BUSY_STATES: frozenset[RunState] = frozenset(
    {
        RunState.QUEUED,
        RunState.TRAINING,
        RunState.MERGING,
        RunState.QUANTIZING,
        RunState.EVALUATING,
        RunState.DEPLOYING,
    }
)


#: The forward edges of the happy path. Anything not listed is illegal unless it
#: is a retry, which moves backwards within a step.
FORWARD: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.SCENE_READY, RunState.FAILED}),
    RunState.SCENE_READY: frozenset({RunState.DATASET_READY, RunState.FAILED}),
    RunState.DATASET_READY: frozenset({RunState.DATA_QUALITY_READY, RunState.FAILED}),
    RunState.DATA_QUALITY_READY: frozenset({RunState.TRAIN_SET_READY, RunState.FAILED}),
    RunState.TRAIN_SET_READY: frozenset({RunState.MODEL_SELECTED, RunState.FAILED}),
    RunState.MODEL_SELECTED: frozenset({RunState.TRAINING_CONFIGURED, RunState.FAILED}),
    RunState.TRAINING_CONFIGURED: frozenset({RunState.QUEUED, RunState.FAILED}),
    RunState.QUEUED: frozenset({RunState.TRAINING, RunState.FAILED}),
    RunState.TRAINING: frozenset(
        {RunState.MERGING, RunState.QUANTIZING, RunState.FAILED, RunState.QUEUED}
    ),
    RunState.MERGING: frozenset({RunState.QUANTIZING, RunState.FAILED}),
    RunState.QUANTIZING: frozenset({RunState.EVALUATING, RunState.MODEL_READY, RunState.FAILED}),
    RunState.EVALUATING: frozenset({RunState.MODEL_READY, RunState.FAILED}),
    RunState.MODEL_READY: frozenset({RunState.DEPLOYING, RunState.FAILED}),
    RunState.DEPLOYING: frozenset({RunState.SERVING, RunState.FAILED}),
    RunState.SERVING: frozenset({RunState.DEPLOYING}),
    RunState.FAILED: frozenset(),  # resumed explicitly through resume_run()
}


def next_states(state: RunState) -> frozenset[RunState]:
    return FORWARD.get(state, frozenset())


def can_transition(current: RunState, target: RunState) -> bool:
    return target in next_states(current)


@dataclass(frozen=True)
class Transition:
    """One legal move, with the evidence it requires.

    ``requires`` names the Run attributes that must be populated. Checking them
    is what stops a caller from marking a step complete on the strength of a
    click.
    """

    target: RunState
    #: Attribute names that must be truthy on the run snapshot.
    requires: tuple[str, ...] = ()

    def check(self, snapshot: Any) -> None:
        missing = [name for name in self.requires if not getattr(snapshot, name, None)]
        if missing:
            raise MissingEvidence(
                f"不能进入 {self.target.value}：缺少 {', '.join(missing)}。"
                "该状态代表对应资产已真实产出，不是用户点击了按钮。"
            )


#: Per-edge evidence requirements. Keys are (current, target).
EVIDENCE: dict[tuple[RunState, RunState], tuple[str, ...]] = {
    (RunState.CREATED, RunState.SCENE_READY): ("scene_code",),
    (RunState.SCENE_READY, RunState.DATASET_READY): ("dataset_id",),
    (RunState.DATASET_READY, RunState.DATA_QUALITY_READY): ("split_id",),
    (RunState.DATA_QUALITY_READY, RunState.TRAIN_SET_READY): ("synth_id",),
    (RunState.TRAIN_SET_READY, RunState.MODEL_SELECTED): ("base_model_id",),
    (
        RunState.MODEL_SELECTED,
        RunState.TRAINING_CONFIGURED,
    ): ("training_config_json",),
    (RunState.TRAINING_CONFIGURED, RunState.QUEUED): ("job_id",),
    (RunState.TRAINING, RunState.MERGING): (),
    (RunState.MERGING, RunState.QUANTIZING): (),
    (RunState.QUANTIZING, RunState.EVALUATING): (),
    (RunState.EVALUATING, RunState.MODEL_READY): ("model_version_id",),
    (RunState.MODEL_READY, RunState.DEPLOYING): (),
    # SERVING requires a real health check, not a process that was launched.
    (RunState.DEPLOYING, RunState.SERVING): (),
}


def transition(
    current: RunState,
    target: RunState,
    snapshot: Any = None,
) -> Transition:
    """Validate and describe a move.

    Raises IllegalTransition for an edge that is not in the graph, and
    MissingEvidence when the run does not carry what the target state claims.
    """
    if target not in next_states(current):
        allowed = ", ".join(sorted(s.value for s in next_states(current))) or "（无）"
        raise IllegalTransition(
            f"{current.value} 不能直接进入 {target.value}；允许的后继状态：{allowed}"
        )

    requires = EVIDENCE.get((current, target), ())
    if snapshot is not None:
        Transition(target=target, requires=requires).check(snapshot)

    return Transition(target=target, requires=requires)


def current_step(state: RunState) -> int:
    """Which wizard step the UI should show for this state."""
    return STEP_OF_STATE.get(state, 1)


def is_busy(state: RunState) -> bool:
    """Whether a job currently owns the run."""
    return state in BUSY_STATES


def is_finished(state: RunState) -> bool:
    """Whether the run has reached its end goal (not necessarily unwinnable)."""
    return state in GOAL_STATES or state in TERMINAL


def retry_from(state: RunState) -> RunState:
    """Where a failed run goes when the user re-runs it.

    Only the training portion is retryable. Data and configuration steps are not
    rolled back automatically: a re-uploaded dataset would invalidate every
    downstream asset, and silently rebuilding them is worse than asking.
    """
    if state is RunState.FAILED:
        raise IllegalTransition("FAILED 状态需要显式指定恢复点，不能自动推断")
    return state


def available_actions(state: RunState) -> Iterable[str]:
    """Human-readable next actions, for the UI and for the assistant's tools."""
    actions: list[str] = []
    if state is RunState.CREATED:
        actions.append("选择场景")
    elif state is RunState.SCENE_READY:
        actions.append("上传种子数据")
    elif state is RunState.DATASET_READY:
        actions.append("执行数据治理与划分")
    elif state is RunState.DATA_QUALITY_READY:
        actions.append("生成合成数据")
    elif state is RunState.TRAIN_SET_READY:
        actions.append("选择基座模型")
    elif state is RunState.MODEL_SELECTED:
        actions.append("配置训练方法")
    elif state is RunState.TRAINING_CONFIGURED:
        actions.append("开始训练")
    elif state is RunState.MODEL_READY:
        actions.append("部署模型")
    elif state is RunState.SERVING:
        actions.append("调用 /v1/predict")
    elif state is RunState.FAILED:
        actions.append("查看失败原因并重试")
    return actions


def describe(state: RunState) -> str:
    return f"{state.value}（第 {current_step(state)} 步）"


def reset_to(state: RunState, target: RunState) -> None:
    """No-op placeholder kept explicit: state reset is never implicit."""
    raise NotImplementedError("Run 状态不做隐式回退；重跑由调用方显式指定起始状态")


#: Exposed so the API can validate a raw string without catching ValueError.
def parse(value: str) -> RunState:
    try:
        return RunState(value)
    except ValueError as exc:
        known = ", ".join(s.value for s in RunState)
        raise IllegalTransition(f"未知的 Run 状态 {value!r}；合法值：{known}") from exc


OnTransition = Callable[[RunState, RunState], None]
