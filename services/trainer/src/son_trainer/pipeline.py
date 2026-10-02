"""Training pipeline orchestration (需求方案.txt 5.6 / 8).

Stage order follows the requirement:

    SFT cold start -> sample -> build preference pairs -> DPO -> (QAT/distill)

Everything here is control flow and bookkeeping, so it is fully exercised by
tests against a scripted backend. Actual optimization is behind
:class:`~son_trainer.backends.TrainingBackend` and needs a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from son_contracts import Decision, RunState, TrainingMethod
from son_trainer.backends import StageRequest, StageResult, TrainingBackend, TrainSample
from son_trainer.checkpoints import Checkpoint, CheckpointStore, JobState, LocalCheckpointStore
from son_trainer.hyperparams import DatasetProfile, Hyperparams, recommend_hyperparams
from son_trainer.preferences import (
    PairBuildReport,
    build_preference_pairs,
    parse_decision,
    to_train_samples,
)

#: Stages in execution order. DPO only runs if it is requested and pairs exist.
BASE_STAGES: tuple[TrainingMethod, ...] = (TrainingMethod.SFT,)


@dataclass
class TrainRequest:
    """Everything needed to run a training job."""

    job_id: str
    base_model: str
    methods: tuple[TrainingMethod, ...]
    samples: tuple[TrainSample, ...]
    profile: DatasetProfile
    difficulty: str = "normal"
    seed: int = 42
    jev_training_compat: bool = True
    n_per_prompt: int = 4
    #: Optional hook so a caller can supply pre-sampled outputs instead of
    #: having the backend sample (used by tests and by replay).
    sampled_outputs: tuple[str, ...] | None = None


@dataclass
class TrainRun:
    """Result of a pipeline run."""

    job_id: str
    state: RunState
    stages: list[StageResult] = field(default_factory=list)
    hyperparams: Hyperparams | None = None
    pair_report: PairBuildReport | None = None
    final_adapter_path: str | None = None
    failure_reason: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.state is RunState.SUCCEEDED

    def stage(self, method: TrainingMethod) -> StageResult | None:
        return next((s for s in self.stages if s.stage == method.value), None)

    def describe(self) -> str:
        parts = [f"{s.stage}={s.status}(step {s.step})" for s in self.stages]
        return f"{self.job_id}: {self.state.value} | " + ", ".join(parts)


def resolve_stages(methods: tuple[TrainingMethod, ...]) -> tuple[TrainingMethod, ...]:
    """Order the requested methods into a runnable pipeline.

    SFT always runs first: DPO on top of an un-tuned base model produces
    preference signal with no decision style to anchor it, which is the
    cold-start dependency the requirement spells out in 8.2. Requesting any
    post-SFT method therefore pulls SFT in, rather than running DPO alone and
    silently producing a worse model.
    """
    post_sft = (
        TrainingMethod.DPO,
        TrainingMethod.QAT,
        TrainingMethod.DISTILLATION,
        TrainingMethod.FEWSHOT,
    )
    needs_sft = TrainingMethod.SFT in methods or TrainingMethod.LORA in methods
    needs_sft = needs_sft or any(m in methods for m in post_sft)

    ordered: list[TrainingMethod] = []
    if needs_sft:
        ordered.append(TrainingMethod.SFT)
    for optional in post_sft:
        if optional in methods:
            ordered.append(optional)
    return tuple(ordered)


def _checkpoint_uri(job_id: str, stage: str, step: int) -> str:
    return f"checkpoints/{job_id}/{stage}/step-{step}"


def run_training(
    request: TrainRequest,
    backend: TrainingBackend,
    *,
    store: CheckpointStore | None = None,
) -> TrainRun:
    """Execute the pipeline, resuming from any checkpoint already on record."""
    store = store or LocalCheckpointStore(".son-checkpoints")
    stages = resolve_stages(request.methods)
    hyperparams = recommend_hyperparams(
        request.profile, request.methods, difficulty=request.difficulty, seed=request.seed
    )

    run = TrainRun(job_id=request.job_id, state=RunState.RUNNING, hyperparams=hyperparams)
    notes: list[str] = []

    state: JobState | None = store.load(request.job_id)
    if state is not None and state.last_checkpoint is not None:
        notes.append(f"从断点续训：{state.last_checkpoint.describe()}")

    adapter_path: str | None = None
    pair_report: PairBuildReport | None = None

    for method in stages:
        checkpoint = state.last_checkpoint if state is not None else None
        resume_from: str | None = None

        if checkpoint is not None:
            if checkpoint.stage == method.value:
                # Re-entering the stage we were interrupted in: that is exactly
                # what 断点续训 means, so use it rather than restarting.
                if checkpoint.is_final:
                    # Already done on a previous attempt. Report the recorded
                    # result so the run reflects reality instead of looking
                    # like it did nothing.
                    notes.append(f"{method.value} 已完成，跳过")
                    run.stages.append(
                        StageResult(
                            stage=method.value,
                            status="succeeded",
                            step=checkpoint.step,
                            metrics=checkpoint.metrics,
                            checkpoint_uri=checkpoint.uri,
                            notes=("复用已有断点",),
                        )
                    )
                    adapter_path = checkpoint.adapter_path or adapter_path
                    continue
                resume_from = checkpoint.uri
            elif method.value in (state.completed_stages if state else ()):
                notes.append(f"{method.value} 已完成，跳过")
                continue

        stage_samples: tuple[TrainSample, ...]
        if method is TrainingMethod.DPO:
            try:
                if pair_report is None:
                    pair_report = _build_pairs(request, backend, adapter_path, state=state)
            except Exception as exc:
                run.state = RunState.FAILED
                run.failure_reason = f"{method.value}: {exc}"
                run.notes = tuple(notes)
                return run
            if pair_report.size == 0:
                notes.extend(pair_report.notes)
                # Requested but impossible to run. Recorded as a failed stage so
                # the run cannot finish SUCCEEDED while silently omitting a
                # method the user explicitly asked for (需求方案.txt 8.2).
                notes.append("无可用偏好对，决策优化训练未能执行")
                run.stages.append(
                    StageResult(
                        stage=TrainingMethod.DPO.value,
                        status="failed",
                        step=0,
                        notes=tuple(pair_report.notes),
                    )
                )
                continue
            stage_samples = tuple(to_train_samples(pair_report.pairs))
        else:
            stage_samples = request.samples

        if not stage_samples:
            notes.append(f"{method.value} 缺少训练样本，已跳过")
            continue

        stage_request = StageRequest(
            stage=method,
            samples=stage_samples,
            base_model=request.base_model,
            hyperparams=hyperparams,
            adapter_path=adapter_path,
            resume_from=resume_from,
            seed=request.seed,
        )

        try:
            result = backend.run_stage(stage_request)
        except Exception as exc:
            run.state = RunState.FAILED
            run.failure_reason = f"{method.value}: {exc}"
            store.save(
                Checkpoint(
                    job_id=request.job_id,
                    stage=method.value,
                    step=checkpoint.step if checkpoint else 0,
                    uri=checkpoint.uri
                    if checkpoint
                    else _checkpoint_uri(request.job_id, method.value, 0),
                )
            )
            run.notes = tuple(notes)
            return run

        run.stages.append(result)
        adapter_path = result.checkpoint_uri or adapter_path

        store.save(
            Checkpoint(
                job_id=request.job_id,
                stage=method.value,
                step=result.step,
                uri=result.checkpoint_uri
                or _checkpoint_uri(request.job_id, method.value, result.step),
                metrics=result.metrics,
                adapter_path=adapter_path,
                is_final=result.succeeded,
            )
        )
        state = store.load(request.job_id)

    run.pair_report = pair_report
    run.final_adapter_path = adapter_path
    run.notes = tuple(notes)

    if any(not s.succeeded for s in run.stages):
        run.state = RunState.FAILED
        run.failure_reason = run.failure_reason or "存在未成功的阶段"
    elif run.stages:
        run.state = RunState.SUCCEEDED
    else:
        run.state = RunState.CANCELLED
        run.failure_reason = "没有可执行的阶段"

    return run


def _build_pairs(
    request: TrainRequest,
    backend: TrainingBackend,
    adapter_path: str | None,
    *,
    state: JobState | None = None,
) -> PairBuildReport:
    """Sample the SFT model and turn the outputs into preference pairs.

    Raises when DPO is requested but there is nothing to sample from. Silently
    producing zero pairs and skipping the stage would report a successful run
    that is missing a method the user explicitly asked for (需求方案.txt 8.2
    lists DPO as a core JEV-paradigm method).
    """
    labelled = [(s.prompt, _label_of(s), s.origin) for s in request.samples]
    unlabelled = sum(1 for _, label, _ in labelled if label is None)
    prompts = tuple((p, label, origin) for p, label, origin in labelled if label is not None)

    if not prompts:
        raise RuntimeError(
            f"全部 {unlabelled} 条训练样本都无法解析出标签，无法判断采样输出是否正确；"
            "请检查训练目标的输出格式"
        )

    if request.sampled_outputs is not None:
        samples = request.sampled_outputs
    elif adapter_path is not None:
        samples = backend.sample_predictions(
            adapter_path, tuple(p for p, _, _ in prompts), n_per_prompt=request.n_per_prompt
        )
    else:
        # The SFT stage finished without producing a resumable adapter, so there
        # is no model to sample. That is a pipeline fault, not an empty result.
        resumed = state.last_checkpoint.adapter_path if state and state.last_checkpoint else None
        if resumed is None:
            raise RuntimeError(
                "决策优化训练需要 SFT 产出的适配器权重，但上一阶段没有产出；无法生成偏好对"
            )
        samples = backend.sample_predictions(
            resumed, tuple(p for p, _, _ in prompts), n_per_prompt=request.n_per_prompt
        )

    return build_preference_pairs(prompts, samples, n_per_prompt=request.n_per_prompt)


def _label_of(sample: TrainSample) -> Decision | None:
    """Read the ground-truth label a supervised sample was built from.

    Returns None when the response carries no recognizable decision. The caller
    must then drop the sample: defaulting to BLACK classified every unparseable
    sample as ground-truth fraud, so genuinely-correct white or gray outputs were
    fed to DPO as `rejected`.
    """
    return parse_decision(sample.response)
