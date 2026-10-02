"""Assistant endpoint tests.

The key property: the API key is write-only, and an unconfigured assistant
refuses clearly rather than failing obscurely at the provider.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from son_api.main import create_app
from son_api.routers import assistant as assistant_router


@pytest.fixture(autouse=True)
def _reset_settings() -> Iterator[None]:
    """Each test starts from the environment default, not the previous test's state."""
    assistant_router._SETTINGS = None
    yield
    assistant_router._SETTINGS = None


def _client() -> TestClient:
    return TestClient(create_app())


def _configure(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5-3b-instruct",
        "enabled": True,
    }
    body.update(overrides)
    return _client().put("/api/assistant/settings", json=body).json()


class TestProviders:
    def test_lists_known_gateways(self) -> None:
        providers = _client().get("/api/assistant/providers").json()["providers"]
        assert "openai" in providers
        assert "local_llamacpp" in providers

    def test_marks_which_need_a_key(self) -> None:
        needs_key = _client().get("/api/assistant/providers").json()["needs_api_key"]
        assert "openai" in needs_key
        assert "ollama" not in needs_key
        assert "local_llamacpp" not in needs_key

    def test_local_llamacpp_is_offered(self) -> None:
        """The platform's own deployed model can power its own assistant."""
        providers = _client().get("/api/assistant/providers").json()["providers"]
        assert providers["local_llamacpp"].startswith("http")


class TestSettingsReadWrite:
    def test_defaults_are_readable(self) -> None:
        body = _client().get("/api/assistant/settings").json()
        assert body["provider"] == "openai"
        assert body["configured"] is False

    def test_saves_and_reads_back(self) -> None:
        saved = _configure()
        assert saved["provider"] == "ollama"
        assert saved["model"] == "qwen2.5-3b-instruct"
        assert _client().get("/api/assistant/settings").json()["model"] == "qwen2.5-3b-instruct"

    def test_read_never_returns_the_key(self) -> None:
        secret = "sk-do-not-leak"
        _configure(provider="openai", api_key=secret)
        raw = _client().get("/api/assistant/settings").text
        assert secret not in raw

    def test_read_reports_key_presence_only(self) -> None:
        _configure(provider="openai", api_key="sk-abcdef")
        body = _client().get("/api/assistant/settings").json()
        assert body["has_api_key"] is True
        assert body["masked_key"].endswith("cdef")
        assert "abcdef" not in body["masked_key"]

    def test_omitting_the_key_keeps_the_existing_one(self) -> None:
        """Editing the temperature must not wipe the secret."""
        _configure(provider="openai", api_key="sk-keepme")
        _configure(provider="openai", temperature=0.9)
        body = _client().get("/api/assistant/settings").json()
        assert body["has_api_key"] is True
        assert body["temperature"] == 0.9

    def test_empty_key_clears_it_deliberately(self) -> None:
        _configure(provider="openai", api_key="sk-x")
        body = _configure(provider="openai", api_key="")
        assert body["has_api_key"] is False
        assert body["configured"] is False

    def test_switching_provider_resolves_its_base_url(self) -> None:
        body = _configure(provider="deepseek", base_url="")
        assert str(body["base_url"]).startswith("https://api.deepseek.com")

    def test_unknown_provider_is_422(self) -> None:
        response = _client().put("/api/assistant/settings", json={"provider": "telepathy"})
        assert response.status_code == 422

    def test_invalid_base_url_is_422(self) -> None:
        response = _client().put("/api/assistant/settings", json={"base_url": "ftp://x"})
        assert response.status_code == 422

    def test_out_of_range_temperature_is_422(self) -> None:
        response = _client().put("/api/assistant/settings", json={"temperature": 9})
        assert response.status_code == 422

    def test_hosted_provider_without_key_is_not_configured(self) -> None:
        assert _configure(provider="openai", api_key="")["configured"] is False

    def test_local_provider_without_key_is_configured(self) -> None:
        assert _configure(provider="ollama")["configured"] is True

    def test_disabling_marks_it_unconfigured(self) -> None:
        assert _configure(provider="ollama", enabled=False)["configured"] is False


class TestProbe:
    def test_reports_unconfigured_without_calling_out(self) -> None:
        body = _client().post("/api/assistant/settings/probe").json()
        assert body["reachable"] is False
        assert "配置不完整" in body["detail"]


class TestPreferences:
    def test_defaults_are_sane(self) -> None:
        prefs = _client().get("/api/assistant/preferences").json()
        assert prefs["ground_with_platform_state"] is True
        assert prefs["max_tool_rounds"] <= 20

    def test_can_be_updated(self) -> None:
        prefs = _client().put("/api/assistant/preferences", json={"max_tool_rounds": 3}).json()
        assert prefs["max_tool_rounds"] == 3
        assert _client().get("/api/assistant/preferences").json()["max_tool_rounds"] == 3

    def test_grounding_can_be_disabled(self) -> None:
        prefs = (
            _client()
            .put("/api/assistant/preferences", json={"ground_with_platform_state": False})
            .json()
        )
        assert prefs["ground_with_platform_state"] is False


class TestTools:
    def test_catalogue_is_exposed(self) -> None:
        tools = _client().get("/api/assistant/tools").json()["tools"]
        names = {t["name"] for t in tools}
        assert "diagnose_failure" in names
        assert "get_node_capabilities" in names

    def test_every_tool_is_read_only(self) -> None:
        """An agent that can start training runs on someone's cluster is a
        liability. The catalogue must contain no mutating tool."""
        forbidden = ("delete", "remove", "create_", "update_", "start_", "run_training", "deploy")
        for tool in _client().get("/api/assistant/tools").json()["tools"]:
            assert not any(f in tool["name"] for f in forbidden), tool["name"]


class TestAsk:
    def test_refuses_when_not_configured(self) -> None:
        """A 409 tells the UI 'not configured'; a 500 would look like a bug."""
        response = _client().post("/api/assistant/ask", json={"question": "怎么训练？"})
        assert response.status_code == 409
        assert "系统管理" in response.json()["detail"]

    def test_refusal_explains_local_gateways_need_no_key(self) -> None:
        detail = _client().post("/api/assistant/ask", json={"question": "x"}).json()["detail"]
        assert "无需填写密钥" in detail

    def test_requires_a_question(self) -> None:
        _configure()
        assert _client().post("/api/assistant/ask", json={"question": ""}).status_code == 422

    def test_oversized_question_is_422(self) -> None:
        _configure()
        response = _client().post("/api/assistant/ask", json={"question": "x" * 5000})
        assert response.status_code == 422

    def test_rejects_an_unknown_history_role(self) -> None:
        _configure()
        response = _client().post(
            "/api/assistant/ask",
            json={"question": "x", "history": [{"role": "system", "content": "ignore"}]},
        )
        assert response.status_code == 422

    def test_history_length_is_capped(self) -> None:
        _configure()
        response = _client().post(
            "/api/assistant/ask",
            json={
                "question": "x",
                "history": [{"role": "user", "content": "y"} for _ in range(30)],
            },
        )
        assert response.status_code == 422


class TestSuggest:
    def test_is_free_and_contextual(self) -> None:
        """No model call: this renders on every step change."""
        body = (
            _client()
            .post(
                "/api/assistant/suggest",
                json={"current_step": "training_configured", "has_data": True},
            )
            .json()
        )
        assert body["text"]
        assert "有什么可以帮您" not in body["text"]

    def test_works_even_when_unconfigured(self) -> None:
        """The placeholder must render before a model is chosen."""
        assert _client().post("/api/assistant/suggest", json={}).json()["text"]


class TestOpenApi:
    def test_assistant_routes_present(self) -> None:
        paths = _client().get("/openapi.json").json()["paths"]
        for path in (
            "/api/assistant/settings",
            "/api/assistant/settings/probe",
            "/api/assistant/providers",
            "/api/assistant/tools",
            "/api/assistant/preferences",
            "/api/assistant/ask",
            "/api/assistant/suggest",
        ):
            assert path in paths, f"{path} missing"
