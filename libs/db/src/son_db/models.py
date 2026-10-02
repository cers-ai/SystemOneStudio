"""ORM models for the eleven metadata tables.

Deliberately plain: declarative classes, columns, and a handful of helpers. No
repository framework, no unit of work, no events (改造开发方案.md 29).

Two conventions worth knowing:

* Timestamps are stored as ISO-8601 strings, not datetimes. SQLite has no
  native datetime type and SQLAlchemy's DateTime on SQLite round-trips as a
  string anyway; being explicit keeps the raw SQL readable when debugging.
* ``bytes`` is used for file sizes rather than a nullable column with a sentinel.
  A missing size and a zero-byte file are different facts.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from son_db.database import Base, now_iso


def _new_id(prefix: str) -> str:
    """Short, sortable, readable id.

    A prefix makes a raw id self-describing in a log line, which matters when a
    traceback spans repository calls.
    """
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Enumerations kept as strings; the check belongs in the state machine, not the
# database, so a new state does not require a migration.
# --------------------------------------------------------------------------


class ProjectRow(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("prj"))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)
    updated_at: Mapped[str] = mapped_column(String(32), default=now_iso, onupdate=now_iso)
    deleted_at: Mapped[str | None] = mapped_column(String(32), nullable=True)

    @staticmethod
    def new(name: str, description: str = "") -> ProjectRow:
        return ProjectRow(id=_new_id("prj"), name=name, description=description)


class RunRow(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("idx_runs_project", "project_id"),
        Index("idx_runs_state", "state"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("run"))
    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    scene_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mode: Mapped[str] = mapped_column(String(32), default="wizard")
    state: Mapped[str] = mapped_column(String(32), default="CREATED")

    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    split_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    synth_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    base_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    training_config_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    deployment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)
    updated_at: Mapped[str] = mapped_column(String(32), default=now_iso, onupdate=now_iso)


class DatasetRow(Base):
    __tablename__ = "datasets"
    __table_args__ = (Index("idx_datasets_run", "run_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("ds"))
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )

    code: Mapped[str] = mapped_column(String(64), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(400), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)

    rows: Mapped[int] = mapped_column(Integer, default=0)
    cols: Mapped[int] = mapped_column(Integer, default=0)
    label_column: Mapped[str | None] = mapped_column(String(200), nullable=True)
    label_mapping_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="READY")
    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)


class SplitRow(Base):
    __tablename__ = "dataset_splits"
    __table_args__ = (Index("idx_splits_run", "run_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("spl"))
    dataset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )

    code: Mapped[str] = mapped_column(String(64), nullable=False)

    train_path: Mapped[str] = mapped_column(Text, nullable=False)
    valid_path: Mapped[str] = mapped_column(Text, nullable=False)
    test_path: Mapped[str] = mapped_column(Text, nullable=False)

    train_rows: Mapped[int] = mapped_column(Integer, default=0)
    valid_rows: Mapped[int] = mapped_column(Integer, default=0)
    test_rows: Mapped[int] = mapped_column(Integer, default=0)

    test_synth_rows: Mapped[int] = mapped_column(Integer, default=0)
    quality_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)


class SynthRow(Base):
    __tablename__ = "synth_runs"
    __table_args__ = (Index("idx_synth_run", "run_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("syn"))
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )

    code: Mapped[str] = mapped_column(String(64), nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    config_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    output_path: Mapped[str] = mapped_column(Text, nullable=False)
    rows: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    fidelity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    privacy_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="READY")
    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)


class JobRow(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("job"))
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )

    type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")

    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    claimed_at: Mapped[str | None] = mapped_column(String(32), nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)
    started_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(32), nullable=True)


class ModelVersionRow(Base):
    __tablename__ = "model_versions"
    __table_args__ = (Index("idx_mv_run", "run_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("mv"))
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )

    code: Mapped[str] = mapped_column(String(64), nullable=False)
    base_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    training_method: Mapped[str] = mapped_column(String(32), nullable=False)

    adapter_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    merged_model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    lineage_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    __table_args__ = (Index("idx_artifacts_mv", "model_version_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("art"))
    model_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False
    )

    type: Mapped[str] = mapped_column(String(32), nullable=False)
    format: Mapped[str] = mapped_column(String(32), nullable=False)
    quant: Mapped[str | None] = mapped_column(String(32), nullable=True)

    path: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)


class EvaluationRow(Base):
    __tablename__ = "evaluations"
    __table_args__ = (Index("idx_evaluations_mv", "model_version_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("ev"))
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    model_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False
    )

    test_dataset_path: Mapped[str] = mapped_column(Text, nullable=False)
    predictions_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_cases_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    performance_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)


class DeploymentRow(Base):
    __tablename__ = "deployments"
    __table_args__ = (Index("idx_deployments_mv", "model_version_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("dep"))
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    model_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False
    )
    artifact_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False
    )

    runtime: Mapped[str] = mapped_column(String(32), default="llama_cpp")
    host: Mapped[str] = mapped_column(String(64), default="127.0.0.1")
    port: Mapped[int] = mapped_column(Integer, default=8081)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stopped_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), default=now_iso)


class AuditRow(Base):
    __tablename__ = "audit_log"
    __table_args__ = (Index("idx_audit_ts", "ts"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("aud"))
    actor: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target: Mapped[str] = mapped_column(String(128), default="")
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    ts: Mapped[str] = mapped_column(String(32), default=now_iso)


ALL_MODELS = (
    ProjectRow,
    RunRow,
    DatasetRow,
    SplitRow,
    SynthRow,
    JobRow,
    ModelVersionRow,
    ArtifactRow,
    EvaluationRow,
    DeploymentRow,
    AuditRow,
)


def next_code(prefix: str, table_rows: list[Any], width: int = 1) -> str:
    """Human-readable sequential code, e.g. ds_v1 / mv_v3.

    Counts existing rows rather than tracking a sequence, which keeps the
    numbering reproducible after a deletion and matches the naming style in
    需求方案.txt 5.7.1.
    """
    return f"{prefix}_v{len(table_rows) + width}"
