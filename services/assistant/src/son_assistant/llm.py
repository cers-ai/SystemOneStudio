"""OpenAI-compatible chat client.

One client covers every provider in :mod:`son_assistant.config`, because they
all speak the same wire format. That includes the platform's own llama.cpp
server, which is the interesting case: the assistant can be powered by exactly
the model the customer trained, with no extra provider and no data leaving the
deployment.

Two things handled explicitly rather than left to the provider:

* **Tool calling is required.** The agent's only route to platform state is a
  tool call, so a gateway without it cannot run the agent. That is reported as
  such rather than degraded into guesses.
* **Streaming is optional and off by default.** Tool-call arguments can arrive
  split across chunks, and assembling them incorrectly yields a silently wrong
  call. Correctness over perceived latency for an operator console.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from son_assistant.config import LlmSettings


class LlmError(RuntimeError):
    """The model provider could not be reached or refused the request."""


class LlmNotConfigured(LlmError):
    """No usable model is configured."""


@dataclass
class AssistantMessage:
    role: str
    content: str = ""
    #: Tool calls requested by the model on this turn.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: Identifier of the tool call this message answers.
    tool_call_id: str | None = None
    name: str | None = None

    def as_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


class OpenAiCompatClient:
    """Chat completions against an OpenAI-compatible endpoint."""

    def __init__(self, settings: LlmSettings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_s)

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def chat(
        self,
        messages: list[AssistantMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """One round trip. Returns (content, tool_calls, raw_usage).

        ``tool_calls`` is normalized to ``{id, name, arguments}`` with arguments
        already parsed, because every provider spells the field differently and
        some return a string rather than an object.
        """
        if not self.settings.configured:
            raise LlmNotConfigured(
                "智能训练助手尚未配置可用的大模型。"
                "请到「系统管理 → 智能训练助手」选择服务商并填写配置"
                + ("，本地网关则无需填写密钥。" if self._is_local() else "与访问密钥。")
            )

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [m.as_wire() for m in messages],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        if tools:
            if not self.settings.supports_tools:
                raise LlmError(
                    f"服务商 {self.settings.provider} 未启用工具调用支持。"
                    "助手只能通过工具读取平台真实状态，因此无法在该服务商下运行。"
                )
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = self._client.post(
                f"{self.settings.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise LlmError(
                f"无法连接模型服务 {self.settings.base_url}：{exc}。"
                + ("本地网关需要确认推理服务已启动。" if self._is_local() else "")
            ) from exc

        if response.status_code == 401:
            raise LlmError(
                f"模型服务拒绝了访问密钥（{self.settings.provider}）。请在系统管理中重新填写。"
            )
        if response.status_code >= 400:
            body = response.text[:400]
            raise LlmError(f"模型服务返回 {response.status_code}：{body}")

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LlmError(f"模型服务返回的不是合法 JSON：{exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LlmError(f"模型服务未返回任何结果：{json.dumps(data)[:300]}")

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        tool_calls = [normalize_tool_call(c) for c in (message.get("tool_calls") or [])]
        return content, tool_calls, data.get("usage") or {}

    def stream_chat(
        self,
        messages: list[AssistantMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Yield content deltas.

        Tool-call arguments can be split across chunks; this yields content only
        and the caller re-issues a non-streaming call if it needs tools, which is
        why the console defaults to non-streaming.
        """
        if not self.settings.configured:
            raise LlmNotConfigured("智能训练助手尚未配置可用的大模型。")

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [m.as_wire() for m in messages],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            with self._client.stream(
                "POST",
                f"{self.settings.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    raise LlmError(f"模型服务返回 {response.status_code}")
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        return
                    try:
                        parsed = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    for choice in parsed.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            yield piece
        except httpx.HTTPError as exc:
            raise LlmError(f"流式请求失败：{exc}") from exc

    def probe(self) -> dict[str, Any]:
        """Cheapest possible reachability check, for the settings screen.

        A save button that reports success for a broken endpoint is worse than no
        save button, so this exists and the settings screen calls it.
        """
        result: dict[str, Any] = {"reachable": False, "detail": ""}
        try:
            response = self._client.get(
                f"{self.settings.base_url}/models",
                headers=self._headers(),
                timeout=min(self.settings.timeout_s, 10.0),
            )
        except httpx.HTTPError as exc:
            result["detail"] = f"无法连接：{exc}"
            return result

        if response.status_code == 401:
            result["detail"] = "可连接但密钥被拒绝"
            return result
        if response.status_code >= 400:
            result["detail"] = f"可连接但返回 {response.status_code}（该网关可能未实现 /models）"
            result["reachable"] = True
            return result

        result["reachable"] = True
        try:
            models = [m.get("id") for m in (response.json().get("data") or [])]
            result["models"] = [m for m in models if m]
            if self.settings.model not in result["models"]:
                result["detail"] = (
                    f"可连接，但列出的模型中没有 {self.settings.model}。请确认模型名是否正确。"
                )
        except json.JSONDecodeError:
            result["detail"] = "可连接"
        return result

    def _is_local(self) -> bool:
        return self.settings.provider in {"ollama", "vllm", "local_llamacpp", "custom"}


def normalize_tool_call(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a provider tool call to {id, name, arguments}.

    Providers disagree on whether ``arguments`` arrives as a JSON string or an
    object, and on whether ``id`` is present. Both are handled here so the agent
    loop does not have to care.

    Idempotent: a call that is already normalized is returned unchanged. That
    matters because the agent loop also normalizes defensively -- it should not
    have to trust that a particular client implementation did so.
    """
    function = raw.get("function")
    if not isinstance(function, dict) and "name" in raw:
        return {
            "id": raw.get("id") or f"call_{raw.get('name', 'unknown')}",
            "name": str(raw.get("name") or ""),
            "arguments": raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {},
        }

    function = function if isinstance(function, dict) else {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            arguments = {"__unparsed__": arguments}
    return {
        "id": raw.get("id") or f"call_{function.get('name', 'unknown')}",
        "name": function.get("name") or "",
        "arguments": arguments if isinstance(arguments, dict) else {},
    }
