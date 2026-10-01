"""Control-plane API smoke tests.

M0 scope is the contract surface: health, the machine-readable JEV spec, and
validation of a prediction-shaped payload. Feature endpoints arrive with their
milestones in 开发计划.md.
"""

from fastapi.testclient import TestClient

from son_api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


class TestHealth:
    def test_health_reports_service_and_jev_spec_version(self) -> None:
        response = _client().get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "son-api"
        assert body["jev_spec_version"] == "jev-1.0"


class TestJevSpecEndpoint:
    def test_exposes_four_required_output_fields(self) -> None:
        schema = _client().get("/meta/jev-spec").json()["output_schema"]
        assert set(schema["required"]) == {"decision", "score", "confidence", "reason"}
        assert schema["properties"]["decision"]["enum"] == ["black", "white", "gray"]
        assert schema["properties"]["reason"]["maxLength"] == 200

    def test_both_switches_default_on(self) -> None:
        flags = _client().get("/meta/jev-spec").json()["default_flags"]
        assert flags == {"jev_format_compat": True, "jev_training_compat": True}

    def test_flags_out_the_paraphrase_provenance(self) -> None:
        """The served spec is a paraphrase; callers must not treat it as upstream."""
        note = _client().get("/meta/jev-spec").json()["note"]
        assert "not the upstream JEV" in note


class TestEchoPrediction:
    def test_accepts_arbitrary_scene_fields(self) -> None:
        response = _client().post(
            "/meta/echo-prediction",
            json={"account": "A12345", "amount": 50000, "device": "iOS 15.0"},
        )
        assert response.status_code == 200
        assert response.json()["decision"] == "gray"
        assert "3 field(s)" in response.json()["reason"]

    def test_response_is_jev_compatible_by_default(self) -> None:
        response = _client().post("/meta/echo-prediction", json={"account": "A1"})
        body = response.json()
        assert body["jev_compatible"] is True
        assert set(body) == {"decision", "score", "confidence", "reason", "jev_compatible"}

    def test_accepts_unknown_scene_field_names(self) -> None:
        """Scene schemas are user-defined, so unknown fields must not 422."""
        response = _client().post(
            "/meta/echo-prediction",
            json={"some_finance_specific_column": 1, "另一个中文列": "值"},
        )
        assert response.status_code == 200
        assert "2 field(s)" in response.json()["reason"]


class TestOpenApi:
    def test_schema_is_servable_for_type_generation(self) -> None:
        """apps/web generates its TypeScript types from this document."""
        schema = _client().get("/openapi.json").json()
        assert schema["info"]["title"] == "SystemOneStudio Control Plane"
        assert "/meta/jev-spec" in schema["paths"]
        assert "/meta/echo-prediction" in schema["paths"]
