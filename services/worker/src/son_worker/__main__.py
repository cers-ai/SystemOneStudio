"""Local job worker (改造开发方案.md 12).

Polls the SQLite queue, claims one job, and drives the existing capability
packages. It does not reimplement anything: `son_data_pipeline`,
`son_synth`, `son_trainer`, `son_quantizer`, `son_evaluator` and
`son_inference` are libraries it calls, not services it owns.

Entry point::

    python -m son_worker

One worker, one GPU job at a time -- which is exactly the documented design
(改造开发方案.md 11). The claim protocol uses an immediate transaction anyway, so
a second worker appearing later loses the race cleanly instead of double-claiming.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from son_db import Database, migrate
from son_db.models import JobRow

from son_worker.queue import (
    ClaimResult,
    JobStatus,
    JobType,
    claim_next,
    finish,
    mark_stale_running_as_interrupted,
    set_progress,
)

#: A skill receives the job context and returns its result payload.
Skill = Callable[["JobContext"], dict[str, Any]]

logger = logging.getLogger("son_worker")

#: How long to wait between polls when the queue is empty.
IDLE_SLEEP_S = 1.0


@dataclass
class JobContext:
    """Everything a skill needs, plus a progress callback."""

    database: Database
    job: JobRow
    payload: dict[str, Any]

    def progress(
        self, progress: int, *, stage: str | None = None, message: str | None = None
    ) -> None:
        with self.database.session() as session:
            set_progress(session, self.job.id, progress=progress, stage=stage, message=message)


class SkillNotImplemented(RuntimeError):
    """A job type has no skill registered yet."""


class Worker:
    """Claims and runs one job at a time."""

    def __init__(
        self,
        database: Database,
        *,
        skills: dict[str, Skill] | None = None,
        idle_sleep_s: float = IDLE_SLEEP_S,
    ) -> None:
        self.database = database
        self.skills: dict[str, Skill] = skills or {}
        self.idle_sleep_s = idle_sleep_s
        self._stop = False

    def register(self, job_type: JobType, skill: Skill) -> None:
        self.skills[job_type.value] = skill

    def stop(self) -> None:
        """Ask the loop to exit after the current job.

        Cooperative: the run in progress is allowed to finish so its
        checkpoints and artifacts are written. Killing it mid-stage would leave
        exactly the interrupted-job case this design tries to avoid.
        """
        self._stop = True

    def run_once(self) -> ClaimResult:
        """Claim and execute a single job. Returns the claim outcome."""
        claim = claim_next(self.database)
        if not claim.claimed or claim.job is None:
            return claim

        job = claim.job
        skill = self.skills.get(job.type)
        logger.info("claimed job %s type=%s run=%s", job.id, job.type, job.run_id)

        if skill is None:
            with self.database.session() as session:
                finish(
                    session,
                    job.id,
                    status=JobStatus.FAILED,
                    error_message=(
                        f"没有为任务类型 {job.type} 注册执行能力；已注册：{sorted(self.skills)}"
                    ),
                )
            return claim

        context = JobContext(database=self.database, job=job, payload=_payload(job))

        try:
            result = skill(context) or {}
        except Exception as exc:
            # Rule 8: every failure leaves status + message + log_path behind.
            logger.exception("job %s failed", job.id)
            with self.database.session() as session:
                finish(
                    session,
                    job.id,
                    status=JobStatus.FAILED,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            return claim

        with self.database.session() as session:
            finish(session, job.id, status=JobStatus.SUCCEEDED, result=result)
        logger.info("job %s finished", job.id)
        return claim

    def run_forever(self, *, max_idle_polls: int | None = None) -> int:
        """Loop until stopped. Returns the number of jobs executed."""
        recovered = mark_stale_running_as_interrupted(self.database)
        if recovered:
            logger.warning("recovered %d interrupted job(s): %s", len(recovered), recovered)

        executed = 0
        idle_polls = 0
        while not self._stop:
            claim = self.run_once()
            if claim.claimed:
                executed += 1
                idle_polls = 0
                continue
            idle_polls += 1
            if max_idle_polls is not None and idle_polls >= max_idle_polls:
                break
            time.sleep(self.idle_sleep_s)
        return executed


def _payload(job: JobRow) -> dict[str, Any]:
    import json

    if not job.payload_json:
        return {}
    try:
        value = json.loads(job.payload_json)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="son_worker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="领取并执行任务")
    run.add_argument("--idle-sleep", type=float, default=IDLE_SLEEP_S)
    run.add_argument("--once", action="store_true", help="只尝试一次后退出")

    doctor = sub.add_parser("doctor", help="检查队列与 GPU 环境")
    doctor.set_defaults(command="doctor")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)

    database = Database.open()
    migrate(database)

    if args.command == "doctor":
        return _doctor(database)

    worker = Worker(database, idle_sleep_s=args.idle_sleep)
    _install_skills(worker)

    def _on_signal(_signum: int, _frame: Any) -> None:
        logger.info("signal received; finishing the current job then exiting")
        worker.stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if args.once:
        # Exit 0 either way: an empty queue is not an error for a one-shot poll.
        worker.run_once()
        return 0

    executed = worker.run_forever()
    logger.info("worker exiting after %d job(s)", executed)
    return 0


def _install_skills(worker: Worker) -> None:
    """Register every capability the worker can run today.

    Importing the GPU skills is lazy: a CPU-only machine can start the worker and
    run the data-prep jobs, and only a training job reports that its GPU skill is
    unavailable. That keeps the failure at the boundary where it belongs instead
    of at worker startup.
    """
    from son_worker.skills import synth_skill, train_skill

    worker.register(JobType.SYNTH, synth_skill)
    worker.register(JobType.PIPELINE, train_skill)
    worker.register(JobType.TRAIN, train_skill)


def _doctor(database: Database) -> int:
    """Report what this node can actually run."""
    import json

    from son_inference.llamacpp_client import gpu_report

    with database.read() as session:
        from son_db.models import JobRow as JobModel
        from sqlalchemy import func, select

        counts = dict(
            session.execute(select(JobModel.status, func.count()).group_by(JobModel.status)).all()
        )

    report: dict[str, Any] = {
        "queued": counts.get(JobStatus.QUEUED.value, 0),
        "running": counts.get(JobStatus.RUNNING.value, 0),
        "interrupted": counts.get(JobStatus.INTERRUPTED.value, 0),
        "gpu": dict(gpu_report()),
    }

    import shutil

    report["llama_server"] = shutil.which("llama-server")

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report["gpu"].get("cuda_available"):
        print(
            "本节点无 GPU：可执行数据准备与合成，训练/量化/推理不可执行",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
