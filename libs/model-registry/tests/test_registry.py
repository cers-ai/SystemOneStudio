"""Registry seed and adapter contract tests."""

import pytest
from pydantic import ValidationError

from son_contracts import JevCompatLevel
from son_model_registry import (
    REGISTRY_SEED,
    BaseModelAdapter,
    RegistryModel,
    TrainConfig,
    get_adapter,
    mvp_models,
    register_adapter,
    registered_adapters,
)
from son_model_registry.seed import DECISION_TASKS


class TestAdapterInterface:
    def test_cannot_instantiate_without_all_methods(self) -> None:
        class Partial(BaseModelAdapter):
            adapter_type = "partial"

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_requires_concrete_adapter_type(self) -> None:
        class Unkeyed(BaseModelAdapter):
            def load_tokenizer(self, model_path: str) -> object:
                return None

            def format_prompt(self, sample: dict[str, object]) -> str:
                return ""

            def format_jev_prompt(self, sample: dict[str, object]) -> str:
                return ""

            def build_trainer(self, train_config: object, dataset: object) -> object:
                return None

            def merge_lora(self, base_path: str, lora_path: str, output_path: str) -> str:
                return output_path

            def export_gguf(
                self, model_path: str, output_path: str, quant_level: object = None
            ) -> str:
                return output_path

            def infer(self, model_path: str, input_data: dict[str, object]) -> dict[str, object]:
                return {}

        with pytest.raises(ValueError, match="adapter_type"):
            register_adapter(Unkeyed)

    def test_registration_roundtrip(self) -> None:
        @register_adapter
        class Dummy(BaseModelAdapter):
            adapter_type = "dummy_m0"

            def load_tokenizer(self, model_path: str) -> object:
                return None

            def format_prompt(self, sample: dict[str, object]) -> str:
                return ""

            def format_jev_prompt(self, sample: dict[str, object]) -> str:
                return ""

            def build_trainer(self, train_config: object, dataset: object) -> object:
                return None

            def merge_lora(self, base_path: str, lora_path: str, output_path: str) -> str:
                return output_path

            def export_gguf(
                self, model_path: str, output_path: str, quant_level: object = None
            ) -> str:
                return output_path

            def infer(self, model_path: str, input_data: dict[str, object]) -> dict[str, object]:
                return {}

        assert "dummy_m0" in registered_adapters()
        assert get_adapter("dummy_m0") is Dummy

        with pytest.raises(ValueError, match="already registered"):
            register_adapter(Dummy)

    def test_unknown_adapter_reports_known_keys(self) -> None:
        with pytest.raises(KeyError, match="unknown adapter_type"):
            get_adapter("nope")


class TestL3RequiresMeasurement:
    """需求方案.txt 6.2: L3 is verified by comparison test, not by assertion."""

    def _model(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "model_id": "m",
            "display_name": "M",
            "family": "F",
            "params": "3B",
            "license": "Apache-2.0",
            "min_gpu_memory": "12GB",
            "adapter_type": "f_adapter",
            "supported_tasks": DECISION_TASKS,
            "jev_compatible": True,
        }
        base.update(overrides)
        return base

    def test_l3_with_declared_evidence_rejected(self) -> None:
        with pytest.raises(ValidationError, match="measured"):
            RegistryModel.model_validate(
                self._model(jev_compat_level=JevCompatLevel.L3, jev_level_evidence="declared")
            )

    def test_l3_with_no_evidence_rejected(self) -> None:
        with pytest.raises(ValidationError, match="measured"):
            RegistryModel.model_validate(self._model(jev_compat_level=JevCompatLevel.L3))

    def test_l3_with_measurement_accepted(self) -> None:
        model = RegistryModel.model_validate(
            self._model(jev_compat_level=JevCompatLevel.L3, jev_level_evidence="measured")
        )
        assert model.jev_compat_level is JevCompatLevel.L3

    def test_l1_may_be_declared(self) -> None:
        model = RegistryModel.model_validate(
            self._model(jev_compat_level=JevCompatLevel.L1, jev_level_evidence="declared")
        )
        assert model.jev_compat_level is JevCompatLevel.L1

    def test_level_without_compatible_flag_rejected(self) -> None:
        with pytest.raises(ValidationError, match="jev_compatible"):
            RegistryModel.model_validate(
                self._model(
                    jev_compatible=False,
                    jev_compat_level=JevCompatLevel.L1,
                    jev_level_evidence="measured",
                )
            )


class TestRegistrySeed:
    def test_mvp_covers_the_four_required_families(self) -> None:
        families = {m.family for m in mvp_models()}
        assert families == {"Qwen", "Gemma", "Llama", "自研"}

    def test_qwen_variants_present(self) -> None:
        ids = {m.model_id for m in mvp_models()}
        assert {"qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct"} <= ids

    def test_no_seed_entry_claims_l3(self) -> None:
        """No comparison test has been run, so nothing may claim L3."""
        assert not [m.model_id for m in REGISTRY_SEED if m.jev_compat_level is JevCompatLevel.L3]

    def test_mvp_entries_are_within_the_size_cap(self) -> None:
        for model in mvp_models():
            assert model.in_mvp_scope
            assert not model.params.endswith("B") or model.params in {
                "1B",
                "1.5B",
                "2B",
                "3B",
                "<1B",
            }

    def test_large_models_flagged_out_of_scope(self) -> None:
        out = {m.model_id for m in REGISTRY_SEED if not m.in_mvp_scope}
        assert out == {"qwen2.5-7b-instruct", "gemma-2-9b-it"}

    def test_every_seed_model_supports_gguf(self) -> None:
        assert all("gguf" in [f.value for f in m.available_formats] for m in REGISTRY_SEED)

    def test_licenses_are_recorded_for_scenario_filtering(self) -> None:
        """需求方案.txt 13: licenses must be recorded so scenarios can filter."""
        assert all(m.license for m in REGISTRY_SEED)
        assert {m.license for m in REGISTRY_SEED} != {"Apache-2.0"}


class TestTrainConfigDefaults:
    def test_defaults_match_documented_examples(self) -> None:
        config = TrainConfig(methods=[])
        assert config.learning_rate == 2e-4
        assert config.batch_size == 4
        assert config.max_steps == 1000
