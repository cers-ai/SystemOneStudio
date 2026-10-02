"""Job queue backed by SQLite (改造开发方案.md 11).

No Redis. A worker polls for QUEUED jobs and claims one inside a
``BEGIN IMMEDIATE`` transaction, so two workers cannot both read a row as
available and both take it.

The design assumes exactly one worker, which is the documented intent -- one GPU,
one job at a time. The claim protocol still uses an immediate transaction so the
assumption is enforced rather than merely documented: if a second worker ever
appears, it loses the race cleanly instead of double-claiming.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from son_db.database import Database, now_iso
from son_db.models import JobRow
from sqlalchemy.orm import Session


class JobType(StrEnum):
    SYNTH = "SYNTH"
    TRAIN = "TRAIN"
    MERGE = "MERGE"
    QUANTIZE = "QUANTIZE"
    EVALUATE = "EVALUATE"
    DEPLOY = "DEPLOY"
    PIPELINE = "PIPELINE"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    #: Claimed by a worker that then died. Distinct from FAILED because the
    #: difference is whether a human needs to look at it.
    INTERRUPTED = "INTERRUPTED"

    @property
    def is_final(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


#: States a worker may move a job into from RUNNING.
TERMINAL_FROM_RUNNING = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})


class JobNotFound(LookupError):
    pass


def _fifo_order(*, descending: bool = False) -> Any:
    """Insertion order.

    Prefers the monotonic ``seq`` and falls back to created_at so a database
    migrated before the column existed still works.
    """
    seq = getattr(JobRow, "seq", None)
    primary = JobRow.created_at if seq is None else sa.func.coalesce(JobRow.seq, 0)
    return (
        [primary.desc(), JobRow.created_at.desc()]
        if descending
        else [
            primary.asc(),
            JobRow.created_at.asc(),
        ]
    )


class IllegalJobTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class ClaimResult:
    """Outcome of a claim attempt."""

    claimed: bool
    job: JobRow | None = None
    reason: str = ""


def enqueue(
    session: Session,
    run_id: str,
    job_type: JobType,
    *,
    payload: dict[str, Any] | None = None,
    log_path: str | None = None,
) -> JobRow:
    """Create a QUEUED job.

    ``seq`` is assigned here rather than by the database: a plain INTEGER column
    is not auto-populated, and created_at alone ties when two jobs land in the
    same millisecond, which would make FIFO order arbitrary.
    """
    next_seq = session.execute(sa.select(sa.func.max(JobRow.seq))).scalar_one() or 0
    job = JobRow(
        run_id=run_id,
        type=job_type.value,
        status=JobStatus.QUEUED.value,
        payload_json=_dump(payload),
        log_path=log_path,
        message="等待执行",
        seq=int(next_seq) + 1,
    )
    session.add(job)
    session.flush()
    return job


def claim_next(database: Database) -> ClaimResult:
    """Atomically take the oldest QUEUED job.

    ``BEGIN IMMEDIATE`` takes the write lock before the SELECT, so the
    read-then-update cannot interleave with another worker's claim.
    """
    with database.transaction() as session:
        row = session.execute(
            sa.select(JobRow)
            .where(JobRow.status == JobStatus.QUEUED.value)
            .order_by(*_fifo_order())
            .limit(1)
        ).scalar_one_or_none()

        if row is None:
            return ClaimResult(claimed=False, reason="队列为空")

        row.status = JobStatus.RUNNING.value
        row.started_at = now_iso()
        row.claimed_at = now_iso()
        row.message = "已开始"
        session.flush()
        return ClaimResult(claimed=True, job=row)


def get_job(session: Session, job_id: str) -> JobRow:
    job = session.get(JobRow, job_id)
    if job is None:
        raise JobNotFound(f"任务 {job_id} 不存在")
    return job


def get_job_for_run(session: Session, run_id: str) -> JobRow | None:
    return (
        session.execute(
            sa.select(JobRow).where(JobRow.run_id == run_id).order_by(*_fifo_order(descending=True))
        )
        .scalars()
        .first()
    )


def set_progress(
    session: Session,
    job_id: str,
    *,
    progress: int,
    stage: str | None = None,
    message: str | None = None,
    metrics: dict[str, float] | None = None,
) -> JobRow:
    """Record progress. The UI polls this (改造开发方案.md 16: no WebSocket yet)."""
    job = get_job(session, job_id)
    if job.status != JobStatus.RUNNING.value:
        raise IllegalJobTransition(f"任务处于 {job.status}，不能更新进度")
    job.progress = max(0, min(100, int(progress)))
    if stage:
        job.stage = stage
    if message:
        job.message = message
    if metrics:
        merged = _load(job.result_json) or {}
        merged["metrics"] = metrics
        job.result_json = _dump(merged)
    session.flush()
    return job


def finish(
    session: Session,
    job_id: str,
    *,
    status: JobStatus = JobStatus.SUCCEEDED,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> JobRow:
    """Move a RUNNING job to a terminal state.

    A FAILED job always records an error message. Rule 8: no swallowed
    exceptions, no silent failure.
    """
    if status not in TERMINAL_FROM_RUNNING:
        raise IllegalJobTransition(f"{status.value} 不是可结束的状态")

    job = get_job(session, job_id)
    if job.status not in (JobStatus.RUNNING.value, JobStatus.QUEUED.value):
        raise IllegalJobTransition(f"任务已处于 {job.status}，不能重复结束")

    job.status = status.value
    job.finished_at = now_iso()
    job.progress = 100 if status is JobStatus.SUCCEEDED else job.progress
    if result is not None:
        job.result_json = _dump(result)
    if status is JobStatus.FAILED and not error_message:
        raise ValueError("失败任务必须记录 error_message")
    job.error_message = error_message
    job.message = message_for(status, error_message)
    session.flush()
    return job


def message_for(status: JobStatus, error: str | None) -> str:
    if status is JobStatus.SUCCEEDED:
        return "已完成"
    if status is JobStatus.CANCELLED:
        return "已取消"
    return f"失败：{error}" if error else "失败"


def cancel(session: Session, job_id: str) -> JobRow:
    """Cancel a queued or running job.

    A RUNNING job is marked cancelled; the worker observes the status and stops.
    It is not killed here -- this process does not own the worker's pid on every
    platform, and signalling the wrong pid is worse than a cooperative stop.
    """
    job = get_job(session, job_id)
    if JobStatus(job.status).is_final:
        raise IllegalJobTransition(f"任务已处于 {job.status}，无法取消")
    return finish(session, job_id, status=JobStatus.CANCELLED)


def mark_stale_running_as_interrupted(database: Database) -> list[str]:
    """Reclaim jobs a dead worker left behind.

    Called at worker startup. A job claimed by a process that no longer exists
    would otherwise sit at RUNNING forever, which looks identical to a very slow
    training run.
    """
    recovered: list[str] = []
    with database.session() as session:
        rows = session.execute(
            sa.select(JobRow).where(JobRow.status == JobStatus.RUNNING.value)
        ).scalars()
        for job in rows:
            if _worker_is_gone(job.pid):
                job.status = JobStatus.INTERRUPTED.value
                job.finished_at = now_iso()
                job.error_message = "执行进程已不存在，任务被中断"
                job.message = "中断"
                recovered.append(job.id)
        session.flush()
    return recovered


def _worker_is_gone(pid: int | None) -> bool:
    if pid is None:
        return True
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False
    return False


def pending_count(session: Session) -> int:
    return int(
        session.execute(
            sa.select(sa.func.count())
            .select_from(JobRow)
            .where(JobRow.status == JobStatus.QUEUED.value)
        ).scalar_one()
    )


def _dump(payload: Any) -> str | None:
    import json

    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, default=str)


def _load(raw: str | None) -> Any:
    import json

    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def serialize(job: JobRow) -> dict[str, Any]:
    """Shape the UI polls (改造开发方案.md 16)."""
    result = _load(job.result_json) or {}
    return {
        "id": job.id,
        "run_id": job.run_id,
        "type": job.type,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "message": job.message,
        "metrics": result.get("metrics") or {},
        "result": {k: v for k, v in result.items() if k != "metrics"},
        "error_message": job.error_message,
        "log_path": job.log_path,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }
