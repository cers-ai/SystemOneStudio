"""Run-state advancement from inside the worker.

The state machine lives in ``son_orchestrator`` and the routes that produce
assets live in ``apps/api``. The worker sits between them: it performs the work
that turns a queued job into a model version, so it is the worker -- not a test
and not the browser -- that moves the run through TRAINING, MERGING,
QUANTIZING, EVALUATING and MODEL_READY.

Kept out of the API router deliberately. The worker depends on libraries; the
reverse would mean a training process importing FastAPI to advance one row.
"""

from __future__ import annotations

from typing import Any

from son_db import Database
from son_db.models import RunRow
from son_orchestrator.runstate import IllegalTransition, MissingEvidence, RunState, transition
from son_orchestrator.runstate import parse as parse_state


class RunAdvanceError(RuntimeError):
    """The run could not be moved to the next state."""


def advance(database: Database, run_id: str, target: RunState, *, strict: bool = True) -> str:
    """Move a run forward, recording the state it came from.

    ``strict=False`` is used while the worker is still attaching evidence: the
    intermediate states (TRAINING, MERGING, ...) legitimately have no new asset
    to point at, because the asset appears at the end of the stage.
    """
    with database.session() as session:
        run = session.get(RunRow, run_id)
        if run is None:
            raise RunAdvanceError(f"Run 不存在：{run_id}")

        previous = parse_state(run.state)
        try:
            transition(previous, target, run if strict else None)
        except MissingEvidence as exc:
            raise RunAdvanceError(str(exc)) from exc
        except IllegalTransition as exc:
            raise RunAdvanceError(
                f"Run {run_id} 无法从 {previous.value} 进入 {target.value}：{exc}"
            ) from exc

        run.state = target.value
        return previous.value


def attach(
    database: Database,
    run_id: str,
    **fields: Any,
) -> None:
    """Record ids on the run so the next transition has its evidence."""
    allowed = {
        "dataset_id",
        "split_id",
        "synth_id",
        "base_model_id",
        "training_config_json",
        "job_id",
        "model_version_id",
        "evaluation_id",
        "deployment_id",
        "error_message",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"未知的 Run 字段：{sorted(unknown)}")

    with database.session() as session:
        run = session.get(RunRow, run_id)
        if run is None:
            raise RunAdvanceError(f"Run 不存在：{run_id}")
        for key, value in fields.items():
            setattr(run, key, value)


def mark_failed(database: Database, run_id: str, reason: str) -> None:
    """Record a failure on the run, in addition to the job row.

    A job that failed while the run still reads TRAINING leaves the operator
    staring at a spinner with no explanation.
    """
    with database.session() as session:
        run = session.get(RunRow, run_id)
        if run is None:
            return
        run.state = RunState.FAILED.value
        run.error_message = reason
