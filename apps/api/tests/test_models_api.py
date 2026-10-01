"""Model shelf, evaluation, deployment and JEV-claim endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from son_api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


class TestModelShelf:
    def test_lists_three_recommendations(self) -> None:
        body = _client().get("/api/models").json()
        assert len(body["recommended"]) == 3

    def test_recommendations_are_in_mvp_scope(self) -> None:
        for card in _client().get("/api/models").json()["recommended"]:
            assert card["in_mvp_scope"] is True

    def test_full_shelf_includes_out_of_scope_models(self) -> None:
        """Shown but flagged, rather than pretending a 7B model does not exist."""
        body = _client().get("/api/models").json()
        assert any(card["in_mvp_scope"] is False for card in body["all"])

    def test_no_recommendation_exceeds_3b(self) -> None:
        """需求方案.txt 13: the MVP cap at 1.5B-3B is deliberate, not an omission."""
        for card in _client().get("/api/models").json()["recommended"]:
            assert card["params"] in {"1B", "1.5B", "2B", "3B", "<1B"}

    def test_every_card_records_a_license(self) -> None:
        """需求方案.txt 13: licenses are recorded so scenarios can filter."""
        assert all(card["license"] for card in _client().get("/api/models").json()["all"])

    def test_shelf_discloses_unverified_numbers(self) -> None:
        notes = _client().get("/api/models").json()["notes"]
        assert any("GPU" in n for n in notes)

    def test_adapter_status_reports_not_implemented(self) -> None:
        body = _client().get("/api/models/qwen2.5-3b-instruct/adapter").json()
        assert body["registered"] is False
        assert "尚未实现" in body["note"]

    def test_unknown_model_is_404(self) -> None:
        assert _client().get("/api/models/nope/adapter").status_code == 404


class TestModelVersionRegistration:
    def _lineage(self) -> dict[str, str]:
        return {
            "scene": "sc_v3",
            "dataset": "ds_v2",
            "synth": "syn_v5",
            "base_model": "qwen2.5-3b-instruct",
            "method": "jev_lora_dpo",
            "model_version": "mv_0017",
        }

    def test_accepts_documented_lineage(self) -> None:
        response = _client().post(
            "/api/model-versions",
            json={
                "lineage": self._lineage(),
                "artifacts": [{"format": "gguf", "quant": "Q4_K_M", "is_recommended": True}],
            },
        )
        assert response.status_code == 200
        assert response.json()["model_version"] == "mv_0017"

    def test_malformed_code_is_422(self) -> None:
        """A bad code is caught here, not when the artifact goes missing later."""
        response = _client().post(
            "/api/model-versions",
            json={"lineage": {**self._lineage(), "scene": "scene_v3"}, "artifacts": []},
        )
        assert response.status_code == 422

    def test_missing_q4_k_m_is_disclosed(self) -> None:
        response = _client().post(
            "/api/model-versions",
            json={"lineage": self._lineage(), "artifacts": [{"format": "gguf", "quant": "Q8_0"}]},
        )
        assert any("JEV 基线" in n for n in response.json()["notes"])

    def test_missing_native_weights_is_disclosed(self) -> None:
        response = _client().post(
            "/api/model-versions",
            json={"lineage": self._lineage(), "artifacts": [{"format": "gguf", "quant": "Q4_K_M"}]},
        )
        assert any("原生权重" in n for n in response.json()["notes"])


class TestEvaluationEndpoint:
    def _body(self, **overrides: object) -> dict[str, object]:
        """Build an evaluation payload.

        Overriding y_true/y_pred without also overriding scores would produce a
        length mismatch, so scores are derived from the label count.
        """
        y_true = overrides.get("y_true") or ["black", "white", "gray", "black"]
        base: dict[str, object] = {
            "y_true": y_true,
            "y_pred": ["black", "white", "black", "black"],
            "scores": [0.9] * len(y_true),  # type: ignore[arg-type]
            "contributions": {"amount": 0.23, "device": 0.19},
        }
        base.update(overrides)
        return base

    def test_returns_both_blocks(self) -> None:
        body = _client().post("/api/evaluations", json=self._body()).json()
        assert "performance" in body and "effect" in body

    def test_performance_comes_first_in_the_contract(self) -> None:
        """需求方案.txt 9.1/9.2: the ordering is architectural."""
        schema = _client().get("/openapi.json").json()
        props = list(schema["components"]["schemas"]["EvaluationResponse"]["properties"])
        assert props.index("performance") < props.index("effect")

    def test_unmeasured_latency_is_stated(self) -> None:
        notes = _client().post("/api/evaluations", json=self._body()).json()["notes"]
        assert any("无 GPU" in n for n in notes)

    def test_p95_target_is_left_undecided(self) -> None:
        body = (
            _client()
            .post(
                "/api/evaluations",
                json=self._body(
                    latency_samples=[{"total_ms": 250.0, "ttft_ms": 20.0} for _ in range(20)],
                    wall_clock_s=5.0,
                    peak_memory_mb=900.0,
                ),
            )
            .json()
        )
        assert body["performance"]["p95_ms"] == 250.0
        assert body["performance"]["targets"]["p95_ms"]["passed"] is None

    def test_effect_targets_are_judged(self) -> None:
        """A clean run clears the documented thresholds, so the check is real."""
        labels = ["black", "white", "gray"] * 10
        body = (
            _client()
            .post(
                "/api/evaluations",
                json=self._body(y_true=labels, y_pred=labels),
            )
            .json()
        )
        targets = body["effect"]["targets"]
        assert targets["accuracy"]["passed"] is True
        assert targets["macro_f1"]["passed"] is True
        assert targets["false_kill_rate"]["passed"] is True

    def test_a_poor_run_fails_the_target_check(self) -> None:
        body = (
            _client()
            .post(
                "/api/evaluations",
                json=self._body(y_true=["white"] * 10, y_pred=["black"] * 10),
            )
            .json()
        )
        assert body["effect"]["targets"]["false_kill_rate"]["passed"] is False

    def test_gray_appears_in_per_label(self) -> None:
        body = _client().post("/api/evaluations", json=self._body()).json()
        assert set(body["effect"]["per_label"]) == {"black", "white", "gray"}

    def test_top_factors_are_ranked(self) -> None:
        body = _client().post("/api/evaluations", json=self._body()).json()
        assert body["top_factors"][0][0] == "amount"

    def test_y_pred_length_mismatch_is_422_not_500(self) -> None:
        """A 500 would tell the caller nothing about what to fix."""
        response = _client().post(
            "/api/evaluations", json=self._body(y_true=["black"], y_pred=["black", "white"])
        )
        assert response.status_code == 422
        assert "length mismatch" in response.json()["detail"]

    def test_scores_must_match_the_label_count(self) -> None:
        response = _client().post("/api/evaluations", json=self._body(scores=[0.5, 0.5, 0.5]))
        assert response.status_code == 422


class TestDeployment:
    def _body(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "model_version": "mv_0017",
            "artifact_path": "/models/mv_0017.Q4_K_M.gguf",
            "port": 8080,
            "concurrency": 8,
        }
        base.update(overrides)
        return base

    def test_returns_pending_not_a_running_service(self) -> None:
        """Claiming a running server from a GPU-less process would be false."""
        assert _client().post("/api/deployments", json=self._body()).json()["status"] == "pending"

    def test_launch_plan_names_llama_server(self) -> None:
        plan = _client().post("/api/deployments", json=self._body()).json()["launch_plan"]
        assert "llama-server" in plan[0]
        assert "/models/mv_0017.Q4_K_M.gguf" in plan[0]

    def test_concurrency_reaches_the_command(self) -> None:
        plan = _client().post("/api/deployments", json=self._body(concurrency=16)).json()
        assert "--parallel 16" in plan["launch_plan"][0]

    def test_curl_example_is_returned(self) -> None:
        body = _client().post("/api/deployments", json=self._body()).json()
        assert body["curl"].startswith("curl -X POST")
        assert "/v1/predict" in body["curl"]

    def test_disabling_format_compat_is_disclosed(self) -> None:
        body = _client().post("/api/deployments", json=self._body(jev_format_compat=False)).json()
        assert body["jev_format_compat"] is False
        assert any("jev_compatible=false" in n for n in body["notes"])

    def test_gpu_requirement_is_stated(self) -> None:
        notes = _client().post("/api/deployments", json=self._body()).json()["notes"]
        assert any("GPU" in n for n in notes)


class TestJevClaimEndpoint:
    def test_l1_is_accepted(self) -> None:
        response = _client().post(
            "/api/jev/level-claim", json={"model_id": "gemma-2-2b-it", "level": "L1"}
        )
        assert response.status_code == 200

    def test_l3_without_evidence_is_refused(self) -> None:
        response = _client().post(
            "/api/jev/level-claim", json={"model_id": "qwen2.5-3b-instruct", "level": "L3"}
        )
        assert response.status_code == 409
        assert "对比测试" in response.json()["detail"]

    def test_l3_with_declared_evidence_is_still_refused(self) -> None:
        response = _client().post(
            "/api/jev/level-claim",
            json={
                "model_id": "qwen2.5-3b-instruct",
                "level": "L3",
                "evidence": "declared",
            },
        )
        assert response.status_code == 409

    def test_l2_is_refused_on_the_missing_benchmark(self) -> None:
        """技术方案.md Q3: the JEV benchmark dataset is not in this repo."""
        response = _client().post(
            "/api/jev/level-claim",
            json={"model_id": "qwen2.5-3b-instruct", "level": "L2", "evidence": "measured"},
        )
        assert response.status_code == 409
        assert "Q3" in response.json()["detail"]

    def test_unknown_model_is_404(self) -> None:
        response = _client().post("/api/jev/level-claim", json={"model_id": "nope", "level": "L1"})
        assert response.status_code == 404

    @pytest.mark.parametrize("level", ["L1", "L2", "L3"])
    def test_every_level_has_a_documented_outcome(self, level: str) -> None:
        response = _client().post(
            "/api/jev/level-claim", json={"model_id": "gemma-2-2b-it", "level": level}
        )
        assert response.status_code in (200, 409)
