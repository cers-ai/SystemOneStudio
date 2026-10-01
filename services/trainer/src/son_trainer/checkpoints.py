"""Checkpoint store and resume semantics.

需求方案.txt principle 5: "训练失败自动保存断点，可续训" and 5.6 shows
[暂停] [断点续训] [克隆任务] controls.

A local-filesystem store is enough for one node and keeps the resume logic
testable without MinIO. The interface is narrow on purpose so an object-store
implementation is a drop-in.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path

from son_contracts import RunState


@dataclass(frozen=True)
class Checkpoint:
    """An anchor a run can resume from."""

    job_id: str
    stage: str
    step: int
    uri: str
    metrics: dict[str, float] = field(default_factory=dict)
    adapter_path: str | None = None
    is_final: bool = False

    def describe(self) -> str:
        kind = "final" if self.is_final else "partial"
        return f"{self.job_id}/{self.stage} step={self.step} ({kind})"


@dataclass(frozen=True)
class JobState:
    """Persisted state for one training job."""

    job_id: str
    state: RunState = RunState.PENDING
    completed_stages: tuple[str, ...] = ()
    last_checkpoint: Checkpoint | None = None
    failure_reason: str | None = None

    def with_state(self, state: RunState, reason: str | None = None) -> JobState:
        return replace(self, state=state, failure_reason=reason)

    def advance(self, stage: str) -> JobState:
        if stage in self.completed_stages:
            return self
        return replace(self, completed_stages=(*self.completed_stages, stage))


class CheckpointStore(ABC):
    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> None: ...

    @abstractmethod
    def load(self, job_id: str) -> JobState | None: ...

    @abstractmethod
    def latest(self, job_id: str, stage: str | None = None) -> Checkpoint | None: ...


class LocalCheckpointStore(CheckpointStore):
    """JSON-per-job store under a root directory."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _job_file(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        state = self.load(checkpoint.job_id) or JobState(job_id=checkpoint.job_id)
        state = state.advance(checkpoint.stage)
        state = replace(
            state,
            state=RunState.SUCCEEDED if checkpoint.is_final else RunState.RUNNING,
            last_checkpoint=checkpoint,
        )
        self._job_file(checkpoint.job_id).write_text(
            json.dumps(_serialize(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, job_id: str) -> JobState | None:
        path = self._job_file(job_id)
        if not path.exists():
            return None
        return _deserialize(json.loads(path.read_text(encoding="utf-8")))

    def latest(self, job_id: str, stage: str | None = None) -> Checkpoint | None:
        state = self.load(job_id)
        if state is None or state.last_checkpoint is None:
            return None
        if stage is not None and state.last_checkpoint.stage != stage:
            return None
        return state.last_checkpoint


def _serialize(state: JobState) -> dict[str, object]:
    cp = state.last_checkpoint
    return {
        "job_id": state.job_id,
        "state": state.state.value,
        "completed_stages": list(state.completed_stages),
        "failure_reason": state.failure_reason,
        "last_checkpoint": (
            None
            if cp is None
            else {
                "job_id": cp.job_id,
                "stage": cp.stage,
                "step": cp.step,
                "uri": cp.uri,
                "metrics": cp.metrics,
                "adapter_path": cp.adapter_path,
                "is_final": cp.is_final,
            }
        ),
    }


def _deserialize(raw: dict[str, object]) -> JobState:
    cp_raw = raw.get("last_checkpoint")
    checkpoint = None
    if isinstance(cp_raw, dict):
        metrics_raw = cp_raw.get("metrics")
        checkpoint = Checkpoint(
            job_id=str(cp_raw["job_id"]),
            stage=str(cp_raw["stage"]),
            step=int(str(cp_raw["step"])),
            uri=str(cp_raw["uri"]),
            metrics={str(k): float(v) for k, v in metrics_raw.items()}
            if isinstance(metrics_raw, dict)
            else {},
            adapter_path=str(cp_raw["adapter_path"]) if cp_raw.get("adapter_path") else None,
            is_final=bool(cp_raw.get("is_final", False)),
        )

    stages_raw = raw.get("completed_stages")
    reason = raw.get("failure_reason")

    return JobState(
        job_id=str(raw["job_id"]),
        state=RunState(str(raw.get("state", RunState.PENDING.value))),
        completed_stages=tuple(str(s) for s in stages_raw) if isinstance(stages_raw, list) else (),
        last_checkpoint=checkpoint,
        failure_reason=str(reason) if reason is not None else None,
    )


def resume_point(state: JobState | None, stage: str) -> tuple[Checkpoint | None, tuple[str, ...]]:
    """What to resume from when (re)entering `stage`.

    Returns the checkpoint to resume from and the stages already done. A
    checkpoint for a *later* stage is ignored: resuming stage SFT from a DPO
    checkpoint would silently skip the cold start that DPO depends on.
    """
    if state is None or state.last_checkpoint is None:
        return None, ()
    order = state.completed_stages
    if stage in order:
        # Already finished; nothing to redo, so do not resume at all.
        return None, order
    if state.last_checkpoint.stage != _previous_stage(order, stage):
        return None, order
    return state.last_checkpoint, order


def _previous_stage(completed: tuple[str, ...], current: str) -> str | None:
    if not completed:
        return None
    return completed[-1]
