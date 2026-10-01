"""Inference tests.

Enforcement and command assembly are verified. Latency is not: this machine has
no GPU and no llama.cpp, so no timing assertion here is based on a real request.
"""

from __future__ import annotations

from typing import Any

import pytest
from son_inference import (
    InferenceEngine,
    PredictionService,
    SchemaViolation,
    ServerConfig,
    ServerError,
    deployment_summary,
    start_server,
)

from son_contracts import PredictRequest

GOOD = (
    '{"decision": "black", "score": 0.87, "confidence": 0.92, '
    '"reason": "交易金额异常度高，设备风险评分高，账号年龄短"}'
)


class StubEngine(InferenceEngine):
    """Returns canned completions and records what it was asked for."""

    def __init__(self, completion: str = GOOD) -> None:
        self.completion = completion
        self.prompts: list[str] = []
        self.grammars: list[str | None] = []

    def complete(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        max_tokens: int = 256,
        stop: tuple[str, ...] = (),
    ) -> str:
        self.prompts.append(prompt)
        self.grammars.append(grammar)
        return self.completion

    def is_ready(self) -> bool:
        return True


class TestSchemaEnforcement:
    def test_valid_completion_is_accepted(self) -> None:
        result = PredictionService(engine=StubEngine()).predict(PredictRequest(account="A1"))
        assert result.response.decision.value == "black"
        assert result.response.score == pytest.approx(0.87)

    def test_binary_decision_is_rejected(self) -> None:
        engine = StubEngine(GOOD.replace('"black"', '"fraud"'))
        with pytest.raises(SchemaViolation, match="decision"):
            PredictionService(engine=engine).predict(PredictRequest(account="A1"))

    def test_missing_reason_is_rejected(self) -> None:
        engine = StubEngine('{"decision": "black", "score": 0.5, "confidence": 0.5}')
        with pytest.raises(SchemaViolation, match="reason"):
            PredictionService(engine=engine).predict(PredictRequest(account="A1"))

    def test_blank_reason_is_rejected(self) -> None:
        engine = StubEngine(GOOD.replace("交易金额异常度高，设备风险评分高，账号年龄短", "   "))
        with pytest.raises(SchemaViolation, match="reason"):
            PredictionService(engine=engine).predict(PredictRequest(account="A1"))

    def test_out_of_range_score_is_rejected(self) -> None:
        engine = StubEngine(GOOD.replace('"score": 0.87', '"score": 1.9'))
        with pytest.raises(SchemaViolation):
            PredictionService(engine=engine).predict(PredictRequest(account="A1"))

    def test_unparseable_output_is_rejected(self) -> None:
        with pytest.raises(SchemaViolation, match="输出格式"):
            PredictionService(engine=StubEngine("I refuse.")).predict(PredictRequest(account="A1"))

    def test_over_long_reason_is_rejected_not_truncated(self) -> None:
        """Truncating would falsify the format-compliance metric."""
        engine = StubEngine(GOOD.replace("账号年龄短", "长" * 300))
        with pytest.raises(SchemaViolation, match="maxLength"):
            PredictionService(engine=engine).predict(PredictRequest(account="A1"))

    def test_violation_lists_every_reason(self) -> None:
        engine = StubEngine('{"decision": "nope", "score": 5}')
        with pytest.raises(SchemaViolation) as exc:
            PredictionService(engine=engine).predict(PredictRequest(account="A1"))
        assert len(str(exc.value).split("；")) >= 2


class TestFormatCompatSwitch:
    def test_grammar_is_attached_when_format_compat_on(self) -> None:
        engine = StubEngine()
        PredictionService(engine=engine).predict(PredictRequest(account="A1"))
        assert engine.grammars[0] is not None
        assert "reason_field" in engine.grammars[0]

    def test_no_grammar_when_format_compat_off(self) -> None:
        """A non-JEV scene must not be silently constrained to JSON."""
        engine = StubEngine()
        PredictionService(engine=engine, jev_format_compat=False).predict(
            PredictRequest(account="A1")
        )
        assert engine.grammars[0] is None

    def test_three_way_enum_holds_even_with_switch_off(self) -> None:
        """格式开关控制输出 schema，不改变产品是三分类这一事实。

        Letting decision be an arbitrary string would make accuracy and recall
        uncomputable, so the enum is enforced regardless of the switch.
        """
        engine = StubEngine(GOOD.replace('"black"', '"unsure"'))
        with pytest.raises(SchemaViolation, match="decision"):
            PredictionService(engine=engine, jev_format_compat=False).predict(
                PredictRequest(account="A1")
            )

    def test_reason_cap_relaxes_when_switch_is_off(self) -> None:
        """The 200-char cap is a JEV schema limit, so it does not apply off-JEV."""
        long_reason = "很长的理由" * 60
        engine = StubEngine(
            '{"decision": "gray", "score": 0.5, "confidence": 0.5, "reason": "' + long_reason + '"}'
        )
        on = PredictionService(engine=engine, jev_format_compat=True)
        with pytest.raises(SchemaViolation):
            on.predict(PredictRequest(account="A1"))

        off = PredictionService(engine=engine, jev_format_compat=False)
        result = off.predict(PredictRequest(account="A1"))
        assert len(result.response.reason) > 200

    def test_non_jev_payload_still_needs_a_reason(self) -> None:
        """The mandatory reason is a compliance floor, not a JEV feature."""
        engine = StubEngine('{"decision": "unsure", "score": 0.4, "confidence": 0.3}')
        with pytest.raises(SchemaViolation, match="reason"):
            PredictionService(engine=engine, jev_format_compat=False).predict(
                PredictRequest(account="A1")
            )

    def test_non_jev_payload_missing_a_required_field_is_rejected(self) -> None:
        engine = StubEngine('{"verdict": "unsure", "note": "证据不足"}')
        with pytest.raises(SchemaViolation, match="reason"):
            PredictionService(engine=engine, jev_format_compat=False).predict(
                PredictRequest(account="A1")
            )

    def test_response_is_flagged_non_jev(self) -> None:
        service = PredictionService(engine=StubEngine(), jev_format_compat=False)
        result = service.predict(PredictRequest(account="A1"))
        assert result.response.jev_compatible is False

    def test_response_is_flagged_jev_by_default(self) -> None:
        result = PredictionService(engine=StubEngine()).predict(PredictRequest(account="A1"))
        assert result.response.jev_compatible is True

    def test_prompt_switches_with_the_flag(self) -> None:
        engine = StubEngine()
        PredictionService(engine=engine, jev_format_compat=False).predict(
            PredictRequest(account="A1")
        )
        assert engine.prompts[0].startswith("<|im_start|>system")


class TestPromptAssembly:
    def test_scene_fields_reach_the_prompt(self) -> None:
        engine = StubEngine()
        PredictionService(engine=engine).predict(PredictRequest(account="A1", amount=50000))
        assert "account: A1" in engine.prompts[0]
        assert "amount: 50000" in engine.prompts[0]

    def test_platform_fields_are_excluded(self) -> None:
        engine = StubEngine()
        PredictionService(engine=engine).predict(
            PredictRequest(model_version_id="mv_1", account="A1")
        )
        assert "model_version_id" not in engine.prompts[0]


class TestTiming:
    def test_a_sample_is_recorded_per_request(self) -> None:
        service = PredictionService(engine=StubEngine())
        service.predict(PredictRequest(account="A1"))
        service.predict(PredictRequest(account="A2"))
        assert len(service.measured_samples()) == 2

    def test_ttft_is_not_faked_from_total(self) -> None:
        """Equating TTFT with total would double-count one number as two metrics."""
        result = PredictionService(engine=StubEngine()).predict(PredictRequest(account="A1"))
        assert result.latency.ttft_ms is None


class TestServerCommand:
    def test_argv_carries_model_port_and_layers(self) -> None:
        argv = ServerConfig(model_path="/models/mv_0017.gguf", port=8080, gpu_layers=20).argv()
        assert "/models/mv_0017.gguf" in argv
        assert "8080" in argv
        assert "20" in argv

    def test_continuous_batching_is_enabled(self) -> None:
        """Without it concurrent requests serialise and QPS collapses."""
        assert "--cont-batching" in ServerConfig(model_path="m").argv()

    def test_concurrency_maps_to_parallel(self) -> None:
        argv = ServerConfig(model_path="m", concurrency=8).argv()
        assert argv[argv.index("--parallel") + 1] == "8"

    def test_concurrency_is_at_least_one(self) -> None:
        argv = ServerConfig(model_path="m", concurrency=0).argv()
        assert argv[argv.index("--parallel") + 1] == "1"

    def test_dry_run_does_not_spawn(self) -> None:
        """Lets the command be inspected on a machine without a GPU."""
        from son_inference import llama_cpp_available

        if llama_cpp_available():
            pytest.skip("llama-server is installed; dry_run path not exercised")
        with pytest.raises(ServerError, match="llama-server"):
            start_server(ServerConfig(model_path="m"))


class TestDeploymentSummary:
    def _summary(self, **kwargs: Any) -> dict[str, Any]:
        return deployment_summary(ServerConfig(model_path="m.gguf", port=8080), **kwargs)

    def test_exposes_api_and_docs_urls(self) -> None:
        summary = self._summary(jev_format_compat=True)
        assert summary["api_url"] == "http://127.0.0.1:8080/v1/predict"
        assert summary["docs_url"] == "http://127.0.0.1:8080/docs"

    def test_curl_example_is_copyable(self) -> None:
        curl = self._summary(jev_format_compat=True)["curl"]
        assert curl.startswith("curl -X POST")
        assert "/v1/predict" in curl

    def test_disabling_format_compat_is_disclosed(self) -> None:
        summary = self._summary(jev_format_compat=False)
        assert summary["jev_format_compat"] is False
        assert any("jev_compatible=false" in note for note in summary["notes"])

    def test_no_note_when_format_compat_on(self) -> None:
        assert self._summary(jev_format_compat=True)["notes"] == ()
