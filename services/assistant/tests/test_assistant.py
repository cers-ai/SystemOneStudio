"""Assistant tests.

The agent's whole value is that it is grounded and honest, so the tests target
those two properties rather than the prose it happens to produce.

The LLM is stubbed. What is verified here is the loop, the tool dispatch, the
prompt assembly, and the refusal paths -- all of which run identically on a
machine with no model configured.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from son_assistant import (
    DEFAULT_PREFERENCES,
    KNOWN_PROVIDERS,
    AssistantMessage,
    AssistantPreferences,
    LlmNotConfigured,
    LlmSettings,
    ToolContext,
    TrainingAssistant,
    build_system_prompt,
    describe_tools,
    run_tool,
    tool_names,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class TestSettings:
    def test_known_provider_fills_its_base_url(self) -> None:
        settings = LlmSettings(provider="deepseek", model="deepseek-chat")
        assert settings.base_url == KNOWN_PROVIDERS["deepseek"]

    def test_custom_provider_keeps_its_url(self) -> None:
        settings = LlmSettings(provider="custom", base_url="https://llm.internal/v1", model="mine")
        assert settings.base_url == "https://llm.internal/v1"

    def test_base_url_must_be_http(self) -> None:
        with pytest.raises(ValueError, match="http"):
            LlmSettings(base_url="llm.internal")

    def test_trailing_slash_is_normalized(self) -> None:
        assert LlmSettings(base_url="https://x.test/v1/").base_url == "https://x.test/v1"

    def test_blank_key_becomes_none(self) -> None:
        assert LlmSettings(api_key="   ").api_key is None

    def test_local_gateways_need_no_key(self) -> None:
        for provider in ("ollama", "vllm", "local_llamacpp"):
            assert LlmSettings(provider=provider).configured is True, provider

    def test_hosted_providers_need_a_key(self) -> None:
        assert LlmSettings(provider="openai").configured is False
        assert LlmSettings(provider="openai", api_key="k").configured is True

    def test_disabled_is_never_configured(self) -> None:
        assert LlmSettings(provider="ollama", enabled=False).configured is False

    def test_public_view_never_exposes_the_key(self) -> None:
        secret = "sk-super-secret-value"
        public = LlmSettings(api_key=secret).public()
        assert secret not in json.dumps(public)
        assert public["has_api_key"] is True

    def test_public_view_reports_absence_as_false(self) -> None:
        assert LlmSettings().public()["has_api_key"] is False

    def test_masked_key_shows_only_the_tail(self) -> None:
        masked = LlmSettings(api_key="sk-abcdef123456").masked_key()
        assert masked.endswith("3456")
        assert "abcdef" not in masked

    def test_api_key_is_absent_from_repr(self) -> None:
        """A stray log line or traceback must not print the key."""
        assert "sk-leak-me" not in repr(LlmSettings(api_key="sk-leak-me"))


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class TestTools:
    def test_catalogue_is_non_empty_and_unique(self) -> None:
        names = tool_names()
        assert len(names) >= 5
        assert len(set(names)) == len(names)

    def test_every_tool_declares_a_schema(self) -> None:
        for tool in describe_tools():
            assert tool["parameters"].get("type") == "object"
            assert len(tool["description"]) > 20

    def test_unknown_tool_returns_an_error_instead_of_raising(self) -> None:
        """A model may invent a tool; the turn must survive it."""
        result = run_tool("teleport", {}, ToolContext())
        assert "error" in result
        assert "available" in result

    def test_node_capabilities_reflect_the_gpu_flag(self) -> None:
        with_gpu = run_tool("get_node_capabilities", {}, ToolContext(gpu_available=True))
        without = run_tool("get_node_capabilities", {}, ToolContext(gpu_available=False))
        assert "均可在本节点执行" in with_gpu["implication"]
        assert "没有图形处理器" in without["implication"]

    def test_workflow_state_reports_the_step(self) -> None:
        result = run_tool("get_workflow_state", {}, ToolContext(current_step="synth_done"))
        assert result["current_step"] == "synth_done"

    def test_workflow_state_without_a_step_guides_to_the_gallery(self) -> None:
        result = run_tool("get_workflow_state", {}, ToolContext())
        assert "场景展廊" in result["hint"]

    def test_dataset_quality_without_a_report_refuses_to_guess(self) -> None:
        result = run_tool("get_dataset_quality", {}, ToolContext())
        assert result["available"] is False
        assert "不要凭空猜测" in result["guidance"]

    def test_dataset_quality_passes_through_a_report(self) -> None:
        result = run_tool(
            "get_dataset_quality", {"report": {"score": 87, "anomalies": ["x"]}}, ToolContext()
        )
        assert result["available"] is True
        assert result["score"] == 87

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            ("CUDA out of memory", "显存不足"),
            ("no kernel image is available", "CUDA 版本"),
            ("llama-quantize: not found", "推理与压缩工具"),
            ("libcuda.so.1: cannot open", "容器内未挂载 GPU"),
        ],
    )
    def test_diagnose_failure_maps_known_causes(self, error: str, expected: str) -> None:
        result = run_tool("diagnose_failure", {"error": error}, ToolContext())
        assert expected in result["likely_cause"]
        assert result["actions"]

    def test_diagnose_failure_on_unknown_input_asks_for_more(self) -> None:
        result = run_tool("diagnose_failure", {"error": "weird thing"}, ToolContext())
        assert result["likely_cause"] == "无法从错误信息判断"
        assert any("son_cli doctor" in a for a in result["actions"])

    @pytest.mark.parametrize("topic", ["latency", "label", "reason", "test", "quant", "size"])
    def test_product_constraints_are_retrievable(self, topic: str) -> None:
        result = run_tool("explain_product_constraint", {"topic": topic}, ToolContext())
        assert result["constraints"]

    def test_unknown_constraint_topic_refuses_to_invent(self) -> None:
        result = run_tool("explain_product_constraint", {"topic": "branding"}, ToolContext())
        assert result["constraints"] == []
        assert "不要编造" in result["note"]


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------


class TestSystemPrompt:
    def _prompt(self, **kwargs: object) -> str:
        return build_system_prompt(
            LlmSettings(),
            AssistantPreferences(),
            ToolContext(**kwargs),  # type: ignore[arg-type]
        )

    def test_states_the_business_language_rule(self) -> None:
        prompt = self._prompt()
        assert "业务语言" in prompt
        assert "快速风格适配" in prompt
        assert "标准压缩" in prompt

    def test_forbids_inventing_numbers(self) -> None:
        assert "绝不编造指标数字" in self._prompt()

    def test_names_the_three_way_constraint(self) -> None:
        assert "三分类" in self._prompt()

    def test_names_the_synth_test_constraint(self) -> None:
        assert "合成数据不得进入测试集" in self._prompt()

    def test_discloses_the_undecided_latency_conflict(self) -> None:
        """The assistant must not resolve a conflict the product has not."""
        assert "待确认" in self._prompt() or "待定案" in self._prompt()

    def test_injects_gpu_state(self) -> None:
        assert "图形处理器：不可用" in self._prompt(gpu_available=False)

    def test_injects_current_step(self) -> None:
        assert "用户当前步骤：synth_done" in self._prompt(current_step="synth_done")

    def test_grounding_can_be_disabled_for_debugging(self) -> None:
        prompt = build_system_prompt(
            LlmSettings(),
            AssistantPreferences(ground_with_platform_state=False),
            ToolContext(gpu_available=True, current_step="trained"),
        )
        assert "用户当前步骤" not in prompt

    def test_lists_available_tools(self) -> None:
        for name in tool_names():
            assert name in self._prompt()


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------


class ScriptedClient:
    """Stands in for the model. Returns a canned sequence of responses."""

    def __init__(self, responses: list[tuple[str, list[dict[str, Any]]]]) -> None:
        self.responses = list(responses)
        self.calls: list[list[AssistantMessage]] = []
        self.tools_offered: list[Any] = []

    def chat(
        self, messages: list[AssistantMessage], *, tools: list[dict[str, Any]] | None = None
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        self.calls.append(list(messages))
        self.tools_offered.append(tools)
        if not self.responses:
            return "（无更多脚本响应）", [], {}
        content, tool_calls = self.responses.pop(0)
        return content, tool_calls, {"total_tokens": 100}


def _client(responses: list[tuple[str, list[dict[str, Any]]]]) -> ScriptedClient:
    return ScriptedClient(responses)


def _assistant(
    responses: list[tuple[str, list[dict[str, Any]]]],
    *,
    preferences: AssistantPreferences | None = None,
) -> tuple[TrainingAssistant, ScriptedClient]:
    client = _client(responses)
    assistant = TrainingAssistant(
        LlmSettings(provider="ollama", model="qwen2.5-3b-instruct"),
        preferences or AssistantPreferences(),
        client=client,  # type: ignore[arg-type]
    )
    return assistant, client


def _tool_call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": f"call_{name}",
        "name": name,
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}, ensure_ascii=False),
        },
    }


class TestAgentLoop:
    def test_answers_without_tools_when_not_needed(self) -> None:
        assistant, client = _assistant([("直接回答。", [])])
        reply = assistant.ask("这一步做什么？")
        assert reply.text == "直接回答。"
        assert reply.trace.rounds == 1
        assert len(client.calls) == 1

    def test_calls_a_tool_then_answers(self) -> None:
        assistant, _ = _assistant(
            [
                ("", [_tool_call("get_node_capabilities")]),
                ("本节点可以训练。", []),
            ]
        )
        reply = assistant.ask("能训练吗？")
        assert reply.text == "本节点可以训练。"
        assert reply.trace.rounds == 2
        assert [c["name"] for c in reply.trace.tool_calls] == ["get_node_capabilities"]

    def test_tool_result_is_fed_back_as_a_tool_message(self) -> None:
        assistant, client = _assistant(
            [
                ("", [_tool_call("get_node_capabilities")]),
                ("可以。", []),
            ]
        )
        assistant.ask("能训练吗？", ctx=ToolContext(gpu_available=True))
        last = client.calls[1][-1]
        assert last.role == "tool"
        assert last.name == "get_node_capabilities"
        assert "gpu_available" in last.content

    def test_offered_tools_come_from_the_catalogue(self) -> None:
        assistant, client = _assistant([("好。", [])])
        assistant.ask("x")
        offered = cast(list[dict[str, Any]], client.tools_offered[0])
        names = {t["function"]["name"] for t in offered}
        assert "diagnose_failure" in names

    def test_tools_are_withheld_when_unsupported(self) -> None:
        client = _client([("好。", [])])
        assistant = TrainingAssistant(
            LlmSettings(provider="ollama", supports_tools=False),
            client=client,  # type: ignore[arg-type]
        )
        assistant.ask("x")
        assert client.tools_offered[0] is None

    def test_round_cap_prevents_an_endless_loop(self) -> None:
        """A model that keeps calling tools must not loop until the window fills."""
        assistant, _ = _assistant(
            [("", [_tool_call("get_node_capabilities")])] * 10,
            preferences=AssistantPreferences(max_tool_rounds=3),
        )
        reply = assistant.ask("卡住了")
        assert reply.trace.truncated is True
        assert reply.trace.rounds == 3
        assert "拆小" in reply.text

    def test_usage_is_accumulated(self) -> None:
        assistant, _ = _assistant([("", [_tool_call("get_node_capabilities")]), ("好。", [])])
        reply = assistant.ask("x")
        assert reply.trace.usage["total_tokens"] == 200

    def test_extra_context_is_attached_as_user_context(self) -> None:
        """Client-side facts are labelled, so the model does not treat stale UI
        numbers as tool-fetched truth."""
        assistant, client = _assistant([("好。", [])])
        assistant.ask("质量如何？", extra_context={"quality": {"score": 91}})
        contents = " ".join(m.content for m in client.calls[0])
        assert "平台已附带的上下文" in contents
        assert "91" in contents

    def test_history_is_carried_into_the_turn(self) -> None:
        assistant, client = _assistant([("好。", [])])
        assistant.ask("继续", history=[AssistantMessage(role="user", content="之前的问题")])
        contents = " ".join(m.content for m in client.calls[0])
        assert "之前的问题" in contents

    def test_empty_model_output_is_reported_not_returned_blank(self) -> None:
        assistant, _ = _assistant([("", [])])
        assert "没有返回内容" in assistant.ask("x").text


class TestUnconfiguredAssistant:
    def test_refuses_before_calling_the_provider(self) -> None:
        """No model configured must say so, not return a generic 401 to the user."""
        assistant = TrainingAssistant(LlmSettings(provider="openai"))
        with pytest.raises(LlmNotConfigured, match="系统管理"):
            assistant.ask("x")

    def test_local_provider_message_explains_no_key_needed(self) -> None:
        assistant = TrainingAssistant(LlmSettings(provider="ollama", enabled=False))
        with pytest.raises(LlmNotConfigured, match="本地网关"):
            assistant.ask("x")


class TestSuggestion:
    def test_suggests_gpu_guidance_on_a_cpu_only_node(self) -> None:
        assistant, _ = _assistant([("", [])])
        text = assistant.suggest(
            ToolContext(gpu_available=False, current_step="training_configured"),
            has_data=True,
            has_quality=True,
        )
        assert "没有图形处理器" in text

    def test_suggests_upload_before_data(self) -> None:
        assistant, _ = _assistant([("", [])])
        text = assistant.suggest(ToolContext(gpu_available=True), has_data=False, has_quality=False)
        assert "上传" in text

    def test_suggests_quality_report_next(self) -> None:
        assistant, _ = _assistant([("", [])])
        text = assistant.suggest(ToolContext(gpu_available=True), has_data=True, has_quality=False)
        assert "质量报告" in text

    def test_suggestion_is_never_generic(self) -> None:
        assistant, _ = _assistant([("", [])])
        for step in ("scene_selected", "synth_done", "evaluated", None):
            text = assistant.suggest(
                ToolContext(gpu_available=True, current_step=step),
                has_data=True,
                has_quality=True,
            )
            assert "有什么可以帮您" not in text


class TestPreferences:
    def test_default_round_cap_is_bounded(self) -> None:
        assert DEFAULT_PREFERENCES.max_tool_rounds <= 20

    def test_actions_require_confirmation_by_default(self) -> None:
        assert DEFAULT_PREFERENCES.require_confirmation_for_actions is True

    def test_round_cap_bounds_are_enforced(self) -> None:
        with pytest.raises(ValueError):
            AssistantPreferences(max_tool_rounds=0)
