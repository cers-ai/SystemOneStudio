"""Job queue and worker tests.

The properties 改造开发方案.md requires:
- the API submits; it never executes training itself
- a stopped worker never reports a fake success
- two workers cannot both claim the same job
- every failure leaves status + message behind (Rule 8)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from son_db import Database, migrate
from son_db.models import ProjectRow, RunRow
from son_worker import (
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
from son_worker.__main__ import Worker


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    monkeypatch.setenv("SON_DB_PATH", str(tmp_path / "runtime" / "son.db"))
    monkeypatch.setenv("SON_WORKSPACE", str(tmp_path / "runtime" / "workspace"))
    database = Database.open()
    migrate(database)
    yield database
    database.dispose()


@pytest.fixture
def run_id(db: Database) -> str:
    with db.session() as session:
        project = ProjectRow.new("queue-test")
        session.add(project)
        session.flush()
        run = RunRow(project_id=project.id, state="CREATED")
        session.add(run)
        session.flush()
        return run.id


class TestEnqueueAndClaim:
    def test_enqueue_creates_queued(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job = enqueue(session, run_id, JobType.TRAIN, payload={"lr": 1e-4})
            job_id = job.id
        with db.read() as session:
            assert get_job(session, job_id).status == "QUEUED"

    def test_claim_marks_running(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            enqueue(session, run_id, JobType.TRAIN)
        claim = claim_next(db)
        assert claim.claimed is True
        assert claim.job is not None
        assert claim.job.status == "RUNNING"

    def test_empty_queue_is_not_an_error(self, db: Database) -> None:
        claim = claim_next(db)
        assert claim.claimed is False
        assert claim.job is None
        assert claim.reason

    def test_oldest_is_claimed_first(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            first = enqueue(session, run_id, JobType.TRAIN)
            second = enqueue(session, run_id, JobType.SYNTH)
            first_id, second_id = first.id, second.id
        claim = claim_next(db)
        assert claim.job is not None
        assert claim.job.id == first_id
        with db.read() as session:
            assert get_job(session, second_id).status == "QUEUED"

    def test_a_claimed_job_cannot_be_claimed_again(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            enqueue(session, run_id, JobType.TRAIN)
        assert claim_next(db).claimed is True
        assert claim_next(db).claimed is False

    def test_two_sequential_claims_never_overlap(self, db: Database, run_id: str) -> None:
        """BEGIN IMMEDIATE makes the read-then-update atomic for one writer."""
        with db.session() as session:
            enqueue(session, run_id, JobType.TRAIN)
            enqueue(session, run_id, JobType.SYNTH)
        first = claim_next(db)
        second = claim_next(db)
        assert first.claimed and second.claimed
        assert first.job is not None and second.job is not None
        assert first.job.id != second.job.id


class TestProgress:
    def test_progress_is_recorded(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim = claim_next(db)
        assert claim.job is not None
        with db.session() as session:
            set_progress(session, job_id, progress=40, stage="TRAIN", message="训练中")
        with db.read() as session:
            job = get_job(session, job_id)
        assert job.progress == 40
        assert job.stage == "TRAIN"
        assert job.message == "训练中"

    def test_progress_is_clamped(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim_next(db)
        with db.session() as session:
            set_progress(session, job_id, progress=500)
        with db.read() as session:
            assert get_job(session, job_id).progress == 100

    def test_progress_on_a_finished_job_is_refused(self, db: Database, run_id: str) -> None:
        from son_worker.queue import IllegalJobTransition

        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim_next(db)
        with db.session() as session:
            finish(session, job_id)
        with db.session() as session, pytest.raises(IllegalJobTransition):
            set_progress(session, job_id, progress=10)


class TestFinish:
    def test_success_sets_finished_and_full_progress(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim_next(db)
        with db.session() as session:
            job = finish(session, job_id, result={"gguf_path": "/x.gguf"})
        assert job.status == "SUCCEEDED"
        assert job.progress == 100
        assert job.finished_at is not None

    def test_failure_must_carry_a_message(self, db: Database, run_id: str) -> None:
        """Rule 8: no swallowed exceptions."""
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim_next(db)
        with db.session() as session, pytest.raises(ValueError, match="error_message"):
            finish(session, job_id, status=JobStatus.FAILED)

    def test_failure_records_the_message(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim_next(db)
        with db.session() as session:
            job = finish(session, job_id, status=JobStatus.FAILED, error_message="CUDA OOM")
        assert job.status == "FAILED"
        assert job.error_message == "CUDA OOM"
        assert "CUDA OOM" in (job.message or "")

    def test_finishing_twice_is_refused(self, db: Database, run_id: str) -> None:
        from son_worker.queue import IllegalJobTransition

        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim_next(db)
        with db.session() as session:
            finish(session, job_id)
        with db.session() as session, pytest.raises(IllegalJobTransition):
            finish(session, job_id)

    def test_result_is_stored(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.QUANTIZE).id
        claim_next(db)
        with db.session() as session:
            finish(session, job_id, result={"gguf_path": "/m.gguf", "bytes": 123})
        with db.read() as session:
            payload = serialize(get_job(session, job_id))
        assert payload["result"]["gguf_path"] == "/m.gguf"


class TestCancel:
    def test_queued_job_can_be_cancelled(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        with db.session() as session:
            job = cancel(session, job_id)
        assert job.status == "CANCELLED"

    def test_finished_job_cannot_be_cancelled(self, db: Database, run_id: str) -> None:
        from son_worker.queue import IllegalJobTransition

        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim_next(db)
        with db.session() as session:
            finish(session, job_id)
        with db.session() as session, pytest.raises(IllegalJobTransition):
            cancel(session, job_id)

    def test_cancelled_job_is_not_picked_up(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        with db.session() as session:
            cancel(session, job_id)
        assert claim_next(db).claimed is False


class TestInterruptedRecovery:
    def test_job_with_no_live_pid_is_recovered(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id
        claim = claim_next(db)
        assert claim.job is not None
        # No pid recorded, so the worker cannot be alive.
        recovered = mark_stale_running_as_interrupted(db)
        assert recovered == [job_id]
        with db.read() as session:
            job = get_job(session, job_id)
        assert job.status == "INTERRUPTED"
        assert "中断" in (job.error_message or "")

    def test_interrupted_is_distinct_from_failed(self) -> None:
        """The difference is whether a human needs to look at it."""
        assert {s.value for s in JobStatus} >= {"INTERRUPTED", "FAILED"}
        assert JobStatus.INTERRUPTED.is_final is False

    def test_a_dead_pid_is_detected(self, db: Database, run_id: str) -> None:
        """The recovery path is what stops a dead worker looking like a slow one."""
        from son_worker import queue

        with db.session() as session:
            job = enqueue(session, run_id, JobType.TRAIN)
            job_id = job.id
        claim_next(db)

        original = queue._worker_is_gone
        queue._worker_is_gone = lambda pid: True
        try:
            assert mark_stale_running_as_interrupted(db) == [job_id]
        finally:
            queue._worker_is_gone = original

    def test_a_live_pid_is_left_alone(self, db: Database, run_id: str) -> None:
        from son_worker import queue

        with db.session() as session:
            job = enqueue(session, run_id, JobType.TRAIN)
            job.pid = 1234
            session.flush()
        claim_next(db)

        original = queue._worker_is_gone
        queue._worker_is_gone = lambda pid: False
        try:
            assert mark_stale_running_as_interrupted(db) == []
        finally:
            queue._worker_is_gone = original

    def test_a_null_pid_counts_as_gone(self) -> None:
        from son_worker.queue import _worker_is_gone

        assert _worker_is_gone(None) is True


class TestLookups:
    def test_missing_job_raises(self, db: Database) -> None:
        with db.read() as session, pytest.raises(JobNotFound):
            get_job(session, "job_nope")

    def test_latest_job_for_run(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            enqueue(session, run_id, JobType.TRAIN)
            enqueue(session, run_id, JobType.SYNTH)
        with db.read() as session:
            latest = get_job_for_run(session, run_id)
        assert latest is not None
        assert latest.type == "SYNTH"

    def test_pending_count(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            enqueue(session, run_id, JobType.TRAIN)
            enqueue(session, run_id, JobType.SYNTH)
        with db.read() as session:
            assert pending_count(session) == 2


class TestWorker:
    def test_runs_a_registered_skill(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.SYNTH).id

        worker = Worker(db)
        worker.register(JobType.SYNTH, lambda ctx: {"ok": True})
        claim = worker.run_once()

        assert claim.claimed is True
        with db.read() as session:
            job = get_job(session, job_id)
        assert job.status == "SUCCEEDED"

    def test_unregistered_skill_fails_loudly(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.TRAIN).id

        Worker(db).run_once()

        with db.read() as session:
            job = get_job(session, job_id)
        assert job.status == "FAILED"
        assert "没有为任务类型" in (job.error_message or "")

    def test_a_raising_skill_is_recorded_not_swallowed(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.SYNTH).id

        def boom(_ctx: Any) -> dict[str, Any]:
            raise RuntimeError("CUDA out of memory")

        worker = Worker(db)
        worker.register(JobType.SYNTH, boom)
        worker.run_once()

        with db.read() as session:
            job = get_job(session, job_id)
        assert job.status == "FAILED"
        assert "CUDA out of memory" in (job.error_message or "")

    def test_api_does_not_execute_the_job(self, db: Database, run_id: str) -> None:
        """Enqueuing must not run anything; that is the worker's job only."""
        with db.session() as session:
            job = enqueue(session, run_id, JobType.TRAIN)
            assert job.status == "QUEUED"
            assert job.started_at is None

    def test_stop_ends_the_loop(self, db: Database) -> None:
        worker = Worker(db, idle_sleep_s=0.01)
        worker.stop()
        assert worker.run_forever(max_idle_polls=1) == 0

    def test_idle_poll_limit_exits(self, db: Database) -> None:
        worker = Worker(db, idle_sleep_s=0.01)
        assert worker.run_forever(max_idle_polls=2) == 0

    def test_context_progress_writes_through(self, db: Database, run_id: str) -> None:
        with db.session() as session:
            job_id = enqueue(session, run_id, JobType.SYNTH).id

        seen: list[int] = []

        def skill(ctx: Any) -> dict[str, Any]:
            ctx.progress(50, stage="SYNTH", message="生成中")
            with db.read() as session:
                seen.append(get_job(session, job_id).progress)
            return {}

        worker = Worker(db)
        worker.register(JobType.SYNTH, skill)
        # No pre-claim here: the worker claims the job itself.
        worker.run_once()
        assert seen == [50]
