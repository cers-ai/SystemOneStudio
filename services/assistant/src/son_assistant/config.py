"""Assistant configuration: which model answers, and how.

The LLM is operator-configurable from 系统管理 rather than hardcoded, because
the platform itself ships a llama.cpp server that exposes an OpenAI-compatible
endpoint. The same client therefore covers:

* a hosted provider (any OpenAI-compatible gateway)
* the platform's own deployed model via llama.cpp
* a local vLLM or Ollama gateway

Three rules this module enforces:

* **The API key is never returned by any read path.** ``LlmSettings.public()``
  is the only shape that leaves the process; the key is reduced to a boolean.
* **A base URL is required for anything that is not the obvious default.**
  Silently pointing at an unreachable endpoint and returning a generic error is
  worse than refusing to start.
* **No provider key is committed.** Defaults are placeholders; the operator sets
  them in 系统管理 or via environment.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

#: Well-known OpenAI-compatible gateways, so the operator picks from a list
#: instead of remembering base URLs.
KNOWN_PROVIDERS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    #: The platform's own deployed model. A llama.cpp server exposes
    #: /v1/chat/completions, so the assistant can be powered by exactly the model
    #: the customer trained -- no extra provider, no data leaving the deployment.
    "local_llamacpp": "http://127.0.0.1:8080/v1",
}

ProviderId = Literal[
    "openai",
    "deepseek",
    "dashscope",
    "moonshot",
    "ollama",
    "vllm",
    "local_llamacpp",
    "custom",
]


class LlmSettings(BaseModel):
    """How the assistant reaches its model."""

    model_config = {"extra": "forbid"}

    provider: ProviderId = "openai"
    base_url: str = KNOWN_PROVIDERS["openai"]
    model: str = "gpt-4o-mini"
    api_key: str | None = Field(
        default=None,
        repr=False,
        description="Never returned by read paths; see public().",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, ge=64, le=32000)
    timeout_s: float = Field(default=60.0, ge=1.0, le=600.0)
    #: Tool calling is the agent's only path to platform state. If a gateway
    #: lacks it, the assistant says so rather than degrading to guesses.
    supports_tools: bool = True
    enabled: bool = True

    @field_validator("base_url")
    @classmethod
    def _must_be_http(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"base_url 必须以 http:// 或 https:// 开头，收到 {value!r}")
        return value.rstrip("/")

    @field_validator("api_key")
    @classmethod
    def _blank_means_absent(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()

    @model_validator(mode="after")
    def _known_provider_fills_base_url(self) -> LlmSettings:
        """A provider whose base_url is still the default gets the known one."""
        known = KNOWN_PROVIDERS.get(self.provider)
        if known and self.base_url == KNOWN_PROVIDERS["openai"] and self.provider != "custom":
            object.__setattr__(self, "base_url", known)
        return self

    @property
    def configured(self) -> bool:
        """Whether a call could plausibly succeed.

        Local gateways need no key; hosted ones do. Checking this up front lets
        the assistant say "not configured" instead of returning a raw HTTP 401
        to the user.
        """
        if not self.enabled:
            return False
        needs_key = self.provider not in {"ollama", "vllm", "local_llamacpp", "custom"}
        return not needs_key or bool(self.api_key)

    def public(self) -> dict[str, Any]:
        """The only shape allowed to leave the process."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "supports_tools": self.supports_tools,
            "enabled": self.enabled,
            "configured": self.configured,
            # Presence only. The value never leaves.
            "has_api_key": bool(self.api_key),
        }

    def masked_key(self) -> str:
        if not self.api_key:
            return ""
        tail = self.api_key[-4:]
        return f"••••••••{tail}"


class AssistantPreferences(BaseModel):
    """Operator-level behaviour toggles for the assistant."""

    model_config = {"extra": "forbid"}

    #: Inject node capabilities and workflow state into every turn. Off only for
    #: debugging; with it off the assistant cannot know what this deployment can
    #: actually run and will happily advise the user to train on a CPU-only node.
    ground_with_platform_state: bool = True
    #: Cap how many tool round trips per user turn. Prevents a confused model
    #: from looping, and bounds the cost of one question.
    max_tool_rounds: int = Field(default=6, ge=1, le=20)
    #: Ask before any action that would write. The assistant currently only
    #: reads, but the flag is here so turning on write tools is a decision
    #: rather than a code change.
    require_confirmation_for_actions: bool = True


def default_settings() -> LlmSettings:
    """Settings from the environment, so a deployment can configure without the UI."""
    raw_provider = os.environ.get("SON_ASSISTANT_PROVIDER", "openai")
    # An unrecognised provider string would be a startup crash otherwise; fall
    # back and let the operator notice it in 系统管理 rather than in a log.
    provider: ProviderId = (
        raw_provider if raw_provider in KNOWN_PROVIDERS else "custom"  # type: ignore[assignment]
    )
    return LlmSettings(
        provider=provider,
        base_url=os.environ.get("SON_ASSISTANT_BASE_URL", KNOWN_PROVIDERS.get(provider, "")),
        model=os.environ.get("SON_ASSISTANT_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("SON_ASSISTANT_API_KEY"),
    )


DEFAULT_PREFERENCES = AssistantPreferences()
