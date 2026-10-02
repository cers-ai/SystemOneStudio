"""Repository helpers.

Deliberately thin: each function is a query or a commit, and nothing wraps
SQLAlchemy in a framework (改造开发方案.md 29). A repository that hides the
session is harder to reason about than the session itself, and at this size the
session is small.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

import sqlalchemy as sa
from sqlalchemy.orm import Session

from son_db.models import (
    ArtifactRow,
    AuditRow,
    DatasetRow,
    DeploymentRow,
    EvaluationRow,
    JobRow,
    ModelVersionRow,
    ProjectRow,
    RunRow,
    SplitRow,
    SynthRow,
    next_code,
)

T = TypeVar("T")


def get(session: Session, model: type[T], row_id: str) -> T | None:
    return session.get(model, row_id)


def require(session: Session, model: type[T], row_id: str, label: str) -> T:
    """Fetch or raise a domain error the API turns into 404."""
    row = session.get(model, row_id)
    if row is None:
        raise LookupError(f"{label} {row_id} 不存在")
    return row


def rows_for_run(session: Session, model: type[T], run_id: str) -> list[T]:
    """Rows belonging to one run.

    ``RunRow`` is keyed by run id rather than owning a ``run_id`` column, so the
    attribute is resolved dynamically and typed as Any.
    """
    column: Any = getattr(model, "run_id", None)
    if column is None:
        raise AttributeError(f"{model.__name__} 没有 run_id 列")
    return list(session.execute(sa.select(model).where(column == run_id)).scalars())


def count(session: Session, model: type[T], *where: Any) -> int:
    statement = sa.select(sa.func.count()).select_from(model)
    for clause in where:
        statement = statement.where(clause)
    return int(session.execute(statement).scalar_one())


def dump(payload: Any) -> str | None:
    """JSON column helper. None in, None out -- so a caller can store "unset"
    without writing the four characters 'null' into the column."""
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, default=str)


def load(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def record_audit(
    session: Session,
    action: str,
    target: str,
    detail: str = "",
    *,
    actor: str = "system",
) -> AuditRow:
    entry = AuditRow(actor=actor, action=action, target=target, detail=detail)
    session.add(entry)
    return entry


# --------------------------------------------------------------------------
# Code allocation
# --------------------------------------------------------------------------


def next_dataset_code(session: Session, run_id: str) -> str:
    return next_code("ds", rows_for_run(session, DatasetRow, run_id))


def next_split_code(session: Session, run_id: str) -> str:
    return next_code("split", rows_for_run(session, SplitRow, run_id))


def next_synth_code(session: Session, run_id: str) -> str:
    return next_code("syn", rows_for_run(session, SynthRow, run_id))


def next_model_version_code(session: Session, run_id: str) -> str:
    return next_code("mv", rows_for_run(session, ModelVersionRow, run_id))


__all__ = [
    "ArtifactRow",
    "AuditRow",
    "DatasetRow",
    "DeploymentRow",
    "EvaluationRow",
    "JobRow",
    "ModelVersionRow",
    "ProjectRow",
    "RunRow",
    "SplitRow",
    "SynthRow",
    "count",
    "dump",
    "get",
    "load",
    "next_dataset_code",
    "next_model_version_code",
    "next_split_code",
    "next_synth_code",
    "record_audit",
    "require",
    "rows_for_run",
]
