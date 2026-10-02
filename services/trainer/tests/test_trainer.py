"""Training pipeline tests.

These cover orchestration, hyperparameters, resume and preference bookkeeping --
the parts that are verifiable without a GPU. They prove nothing about whether
optimization converges; that needs a GPU node (see AGENTS.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from son_trainer import (
    Checkpoint,
    DatasetProfile,
    LocalCheckpointStore,
    StageRequest,
    StageResult,
    TrainingBackend,
    TrainRequest,
    TrainSample,
    build_preference_pairs,
    estimate_duration_minutes,
    parse_decision,
    recommend_batch_size,
    recommend_hyperparams,
    recommend_learning_rate,
    recommend_max_steps,
    resolve_stages,
    resume_point,
    run_training,
)

from son_contracts import DataOrigin, Decision, RunState, TrainingMethod


class ScriptedBackend(TrainingBackend):
    """Deterministic backend that records what it was asked to do."""

    name = "scripted"

    def __init__(
        self,
        *,
        steps: int = 500,
        fail_on: TrainingMethod | None = None,
        samples: tuple[str, ...] = (),
    ) -> None:
        self.steps = steps
        self.fail_on = fail_on
        self._samples = samples
        self.calls: list[StageRequest] = []

    def run_stage(self, request: StageRequest) -> StageResult:
        self.calls.append(request)
        if self.fail_on is request.stage:
            raise RuntimeError(f"{request.stage.value} exploded")
        return StageResult(
            stage=request.stage.value,
            status="succeeded",
            step=self.steps,
            metrics={"loss": 0.234},
            checkpoint_uri=f"adapters/{request.stage.value}",
        )

    def sample_predictions(
        self,
        adapter_path: str,
        prompts: tuple[str, ...],
        *,
        n_per_prompt: int = 4,
    ) -> tuple[str, ...]:
        return self._samples


def _samples(n: int = 12, origin: str = DataOrigin.SEED.value) -> tuple[TrainSample, ...]:
    return tuple(
        TrainSample(
            prompt=f"sample-{i}",
            response='{"decision": "black", "reason": "金额异常"}',
            origin=origin,
        )
        for i in range(n)
    )


def _request(**overrides: object) -> TrainRequest:
    base: dict[str, object] = {
        "job_id": "job-1",
        "base_model": "qwen2.5-3b-instruct",
        "methods": (TrainingMethod.SFT,),
        "samples": _samples(),
        "profile": DatasetProfile(rows=10_000, params_b=3.0),
    }
    base.update(overrides)
    return TrainRequest(**base)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> LocalCheckpointStore:
    return LocalCheckpointStore(tmp_path / "ckpt")


class TestHyperparametersAreComputed:
    """需求方案.txt principle 2: defaults are derived, not hardcoded."""

    def test_smaller_model_gets_a_higher_rate(self) -> None:
        small = recommend_learning_rate(DatasetProfile(rows=1000, params_b=1.5))
        large = recommend_learning_rate(DatasetProfile(rows=1000, params_b=3.0))
        assert small > large

    def test_rate_is_bounded(self) -> None:
        for params in (0.5, 1.5, 3.0, 7.0):
            rate = recommend_learning_rate(DatasetProfile(rows=1000, params_b=params))
            assert 5e-5 <= rate <= 5e-4

    def test_bigger_model_gets_a_smaller_batch(self) -> None:
        assert recommend_batch_size(DatasetProfile(rows=1000, params_b=3.0)) < recommend_batch_size(
            DatasetProfile(rows=1000, params_b=1.5)
        )

    def test_qlora_allows_a_larger_batch(self) -> None:
        plain = recommend_batch_size(DatasetProfile(rows=1000, params_b=3.0))
        quantized = recommend_batch_size(DatasetProfile(rows=1000, params_b=3.0, quantized=True))
        assert quantized > plain

    def test_batch_never_exceeds_the_vram_envelope(self) -> None:
        for quantized in (True, False):
            assert (
                recommend_batch_size(DatasetProfile(rows=1000, params_b=3.0, quantized=quantized))
                <= 16
            )

    def test_more_data_needs_no_more_steps(self) -> None:
        small = recommend_max_steps(DatasetProfile(rows=100, params_b=1.5))
        large = recommend_max_steps(DatasetProfile(rows=100_000, params_b=1.5))
        assert small <= large

    def test_steps_have_a_floor(self) -> None:
        assert recommend_max_steps(DatasetProfile(rows=1, params_b=1.5)) >= 200

    def test_dpo_halves_the_learning_rate(self) -> None:
        profile = DatasetProfile(rows=1000, params_b=1.5)
        sft_only = recommend_hyperparams(profile, (TrainingMethod.SFT,))
        with_dpo = recommend_hyperparams(profile, (TrainingMethod.SFT, TrainingMethod.DPO))
        assert with_dpo["learning_rate"] < sft_only["learning_rate"]

    def test_gradient_accumulation_keeps_the_effective_batch(self) -> None:
        hp = recommend_hyperparams(DatasetProfile(rows=1000, params_b=3.0), (TrainingMethod.SFT,))
        assert int(hp["batch_size"]) * int(hp["gradient_accumulation_steps"]) == 16

    def test_duration_estimate_is_positive(self) -> None:
        profile = DatasetProfile(rows=1000, params_b=1.5)
        hp = recommend_hyperparams(profile, (TrainingMethod.SFT,))
        assert estimate_duration_minutes(profile, hp) > 0


class TestStageResolution:
    def test_sft_is_forced_first(self) -> None:
        assert resolve_stages((TrainingMethod.DPO,))[0] is TrainingMethod.SFT

    def test_lora_implies_sft(self) -> None:
        assert TrainingMethod.SFT in resolve_stages((TrainingMethod.LORA,))

    def test_dpo_follows_sft(self) -> None:
        stages = resolve_stages((TrainingMethod.DPO, TrainingMethod.SFT))
        assert stages.index(TrainingMethod.SFT) < stages.index(TrainingMethod.DPO)

    def test_qat_comes_last(self) -> None:
        stages = resolve_stages((TrainingMethod.QAT, TrainingMethod.DPO, TrainingMethod.SFT))
        assert stages[-1] is TrainingMethod.QAT


class TestPipeline:
    def test_sft_only_run_succeeds(self, store: LocalCheckpointStore) -> None:
        run = run_training(_request(), ScriptedBackend(), store=store)
        assert run.state is RunState.SUCCEEDED
        assert run.succeeded
        assert run.stage(TrainingMethod.SFT) is not None

    def test_final_adapter_path_is_recorded(self, store: LocalCheckpointStore) -> None:
        run = run_training(_request(), ScriptedBackend(), store=store)
        assert run.final_adapter_path == "adapters/sft"

    def test_failure_is_recorded_not_swallowed(self, store: LocalCheckpointStore) -> None:
        backend = ScriptedBackend(fail_on=TrainingMethod.SFT)
        run = run_training(_request(), backend, store=store)
        assert run.state is RunState.FAILED
        assert "exploded" in (run.failure_reason or "")

    def test_failure_still_persists_a_checkpoint(self, store: LocalCheckpointStore) -> None:
        run_training(_request(), ScriptedBackend(fail_on=TrainingMethod.SFT), store=store)
        assert store.load("job-1") is not None

    def test_empty_samples_cancels_rather_than_succeeds(self, store: LocalCheckpointStore) -> None:
        run = run_training(_request(samples=()), ScriptedBackend(), store=store)
        assert run.state is RunState.CANCELLED

    def test_hyperparams_reach_the_backend(self, store: LocalCheckpointStore) -> None:
        backend = ScriptedBackend()
        run_training(_request(), backend, store=store)
        assert backend.calls[0].hyperparams["learning_rate"] > 0

    def test_no_samples_leaves_the_backend_untouched(self, store: LocalCheckpointStore) -> None:
        backend = ScriptedBackend()
        run_training(_request(samples=()), backend, store=store)
        assert backend.calls == []


class TestResume:
    def test_checkpoint_survives_a_reload(self, store: LocalCheckpointStore) -> None:
        run_training(_request(), ScriptedBackend(steps=321), store=store)
        state = store.load("job-1")
        assert state is not None and state.last_checkpoint is not None
        assert state.last_checkpoint.step == 321

    def test_interrupted_stage_resumes_from_its_checkpoint(
        self, store: LocalCheckpointStore
    ) -> None:
        """A partial checkpoint is what 断点续训 is for; it must actually be used."""
        store.save(
            Checkpoint(
                job_id="job-1",
                stage=TrainingMethod.SFT.value,
                step=137,
                uri="checkpoints/job-1/sft/step-137",
                metrics={"loss": 0.4},
                is_final=False,
            )
        )
        backend = ScriptedBackend()
        run_training(_request(), backend, store=store)
        assert backend.calls[0].resume_from == "checkpoints/job-1/sft/step-137"

    def test_persisted_json_is_readable(self, store: LocalCheckpointStore) -> None:
        run_training(_request(), ScriptedBackend(), store=store)
        files = list(Path(store.root).glob("*.json"))
        assert files
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["job_id"] == "job-1"

    def test_fresh_job_has_no_resume_point(self, store: LocalCheckpointStore) -> None:
        assert resume_point(store.load("nope"), "sft") == (None, ())

    def test_completed_stage_is_skipped_on_re_run(self, store: LocalCheckpointStore) -> None:
        run_training(_request(), ScriptedBackend(), store=store)
        backend = ScriptedBackend()
        run = run_training(_request(), backend, store=store)
        assert backend.calls == []
        assert run.state is RunState.SUCCEEDED
        assert run.stage(TrainingMethod.SFT) is not None
        assert any("已完成" in note for note in run.notes)

    def test_a_later_stage_checkpoint_is_ignored_for_sft(self, store: LocalCheckpointStore) -> None:
        """Resuming SFT from a DPO checkpoint would skip the cold start DPO needs."""
        contrasts = tuple(
            '{"decision":"black","reason":"ok"}'
            if i % 2 == 0
            else '{"decision":"white","reason":"no"}'
            for i in range(24)
        )
        run_training(
            _request(
                methods=(TrainingMethod.SFT, TrainingMethod.DPO),
                sampled_outputs=contrasts,
                n_per_prompt=2,
            ),
            ScriptedBackend(),
            store=store,
        )
        state = store.load("job-1")
        assert state is not None and state.last_checkpoint is not None
        assert state.last_checkpoint.stage == TrainingMethod.DPO.value
        checkpoint, _ = resume_point(state, "sft")
        assert checkpoint is None


class TestPreferencePairs:
    def test_pairs_come_from_contrasting_samples(self) -> None:
        prompts = (("p1", Decision.BLACK, DataOrigin.SEED.value),)
        samples = (
            '{"decision":"black","reason":"ok"}',
            '{"decision":"white","reason":"no"}',
        )
        report = build_preference_pairs(prompts, samples, n_per_prompt=2)
        assert report.size == 1
        assert "black" in report.pairs[0].chosen
        assert "white" in report.pairs[0].rejected

    def test_chosen_is_always_a_real_model_output(self) -> None:
        """Building `chosen` from the label template would make DPO a no-op."""
        prompts = (("p1", Decision.BLACK, DataOrigin.SEED.value),)
        samples = ('{"decision":"black","reason":"模型自己的简短理由"}',)
        report = build_preference_pairs(prompts, samples, n_per_prompt=1)
        assert report.pairs == []

    def test_all_correct_samples_produce_no_pair(self) -> None:
        prompts = (("p1", Decision.BLACK, DataOrigin.SEED.value),)
        samples = ('{"decision":"black"}', '{"decision":"black"}')
        report = build_preference_pairs(prompts, samples, n_per_prompt=2)
        assert report.size == 0
        assert report.prompts_without_a_valid_pair == 1

    def test_all_wrong_samples_produce_no_pair(self) -> None:
        prompts = (("p1", Decision.BLACK, DataOrigin.SEED.value),)
        samples = ('{"decision":"white"}', '{"decision":"gray"}')
        assert build_preference_pairs(prompts, samples, n_per_prompt=2).size == 0

    def test_unparseable_output_produces_no_pair(self) -> None:
        prompts = (("p1", Decision.BLACK, DataOrigin.SEED.value),)
        assert build_preference_pairs(prompts, ("total nonsense",), n_per_prompt=1).size == 0

    def test_synthetic_rows_are_skipped(self) -> None:
        """Sampling synthetic rows feeds synthesis artefacts back into training."""
        prompts = (("p1", Decision.BLACK, DataOrigin.SYNTH.value),)
        report = build_preference_pairs(
            prompts, ('{"decision":"black"}', '{"decision":"white"}'), n_per_prompt=2
        )
        assert report.size == 0
        assert report.skipped_non_seed == 1
        assert any("种子" in note for note in report.notes)

    def test_chosen_is_shorter_and_rejected_longer(self) -> None:
        prompts = (("p1", Decision.BLACK, DataOrigin.SEED.value),)
        samples = (
            '{"decision":"black","reason":"短"}',
            '{"decision":"black","reason":"稍长的正确理由"}',
            '{"decision":"white","reason":"错误但简短"}',
            '{"decision":"white","reason":"错误而且理由很长很长很长"}',
        )
        pair = build_preference_pairs(prompts, samples, n_per_prompt=4).pairs[0]
        assert len(pair.chosen) <= len(pair.rejected)

    def test_zero_n_per_prompt_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            build_preference_pairs((), (), n_per_prompt=0)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"decision":"black"}', Decision.BLACK),
            ('{"decision":"gray"}', Decision.GRAY),
            ("decision: white", Decision.WHITE),
            ("判定为黑样本", Decision.BLACK),
            ("nothing useful", None),
        ],
    )
    def test_parse_decision_handles_real_model_output(
        self, raw: str, expected: Decision | None
    ) -> None:
        assert parse_decision(raw) is expected


class TestDpoStage:
    def test_dpo_runs_when_pairs_exist(self, store: LocalCheckpointStore) -> None:
        # Half correct, half wrong, so every prompt yields a contrast.
        samples = tuple(
            '{"decision":"black","reason":"ok"}'
            if i % 2 == 0
            else '{"decision":"white","reason":"no"}'
            for i in range(24)
        )
        backend = ScriptedBackend()
        run = run_training(
            _request(
                methods=(TrainingMethod.SFT, TrainingMethod.DPO),
                sampled_outputs=samples,
                n_per_prompt=2,
            ),
            backend,
            store=store,
        )
        assert run.stage(TrainingMethod.DPO) is not None

    def test_dpo_fails_rather_than_vanishing_when_no_pairs(
        self, store: LocalCheckpointStore
    ) -> None:
        """A requested method that cannot run must not disappear.

        It used to be skipped with no StageResult, leaving the run reporting
        SUCCEEDED with a stage the user explicitly asked for missing.
        """
        run = run_training(
            _request(
                methods=(TrainingMethod.SFT, TrainingMethod.DPO),
                sampled_outputs=('{"decision":"black"}',) * 24,
                n_per_prompt=2,
            ),
            ScriptedBackend(),
            store=store,
        )
        dpo = run.stage(TrainingMethod.DPO)
        assert dpo is not None
        assert dpo.status == "failed"

    def test_run_cannot_succeed_with_a_failed_stage(self, store: LocalCheckpointStore) -> None:
        run = run_training(
            _request(
                methods=(TrainingMethod.SFT, TrainingMethod.DPO),
                sampled_outputs=('{"decision":"black"}',) * 24,
                n_per_prompt=2,
            ),
            ScriptedBackend(),
            store=store,
        )
        assert run.state is RunState.FAILED
        assert run.succeeded is False
        assert any("未能执行" in note for note in run.notes)

    def test_backend_sampling_is_refused_by_default(self, store: LocalCheckpointStore) -> None:
        class NoSampling(TrainingBackend):
            name = "no-sampling"

            def run_stage(self, request: StageRequest) -> StageResult:
                return StageResult(stage=request.stage.value, status="succeeded", step=1)

        run = run_training(
            _request(methods=(TrainingMethod.SFT, TrainingMethod.DPO), n_per_prompt=2),
            NoSampling(),
            store=store,
        )
        # NotImplementedError is recorded as a failure, never silently ignored.
        assert run.state is RunState.FAILED
