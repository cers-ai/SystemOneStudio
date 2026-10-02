"""Intelligent training assistant.

An LLM-backed helper for using the platform: explains each step, diagnoses
failures, and grounds its answers in the deployment's real state rather than
generically. It closes the 智能辅助层 in 需求方案.txt chapter 3.

Two boundaries it does not cross:

* **It only reads.** Every tool inspects platform state; none of them mutate. An
  agent that can silently start training runs on someone's cluster is a
  liability, not a feature.
* **It never invents a number.** Metrics, VRAM and durations come from tools or
  from the client. The system prompt requires it to say "unmeasured" instead.

The model is operator-configurable and the API key never leaves the process; see
:mod:`son_assistant.config`.
"""

from son_assistant.agent import AssistantReply, TrainingAssistant, TurnTrace
from son_assistant.config import (
    DEFAULT_PREFERENCES,
    KNOWN_PROVIDERS,
    AssistantPreferences,
    LlmSettings,
    default_settings,
)
from son_assistant.llm import (
    AssistantMessage,
    LlmError,
    LlmNotConfigured,
    OpenAiCompatClient,
)
from son_assistant.prompt import build_system_prompt, suggestion_prompt
from son_assistant.tools import Tool, ToolContext, describe_tools, run_tool, tool_names

__all__ = [
    "DEFAULT_PREFERENCES",
    "KNOWN_PROVIDERS",
    "AssistantMessage",
    "AssistantPreferences",
    "AssistantReply",
    "LlmError",
    "LlmNotConfigured",
    "LlmSettings",
    "OpenAiCompatClient",
    "Tool",
    "ToolContext",
    "TrainingAssistant",
    "TurnTrace",
    "build_system_prompt",
    "default_settings",
    "describe_tools",
    "run_tool",
    "suggestion_prompt",
    "tool_names",
]
