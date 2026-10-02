"""Local job worker and SQLite-backed job queue.

见 改造开发计划.md WP5. Not a microservice: it is a process that calls the
existing capability packages as libraries.
"""

from son_worker.queue import (
    ClaimResult,
    JobNotFound,
    JobStatus,
    JobType,
    cancel,
    claim_next,
    enqueue,
    finish,
    get_job,
    get_job_for_run,
    mark_stale_running_as_interrupted,
    pending_count,
    serialize,
    set_progress,
)

__all__ = [
    "ClaimResult",
    "JobNotFound",
    "JobStatus",
    "JobType",
    "cancel",
    "claim_next",
    "enqueue",
    "finish",
    "get_job",
    "get_job_for_run",
    "mark_stale_running_as_interrupted",
    "pending_count",
    "serialize",
    "set_progress",
]
