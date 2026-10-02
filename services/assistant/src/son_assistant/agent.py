"""The assistant turn loop.

A user turn is a sequence of: model call -> tool calls -> tool results -> model
call again, until the model answers with text or the round cap is hit.

The loop is bounded by ``max_tool_rounds`` rather than trusted to terminate. A
model that keeps calling the same tool would otherwise loop until the context
window filled, which on a paid endpoint is a real bill.

Usage is recorded per turn so the console can show what a question cost. Without
that, an assistant wired into an operator console is an invisible spend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import AssistantPreferences, LlmSettings
from .llm import AssistantMessage, LlmError, OpenAiCompatClient, normalize_tool_call
from .prompt import build_system_prompt, suggestion_prompt
from .tools import ToolContext, run_tool, tool_schemas


@dataclass
class TurnTrace:
    """What actually happened during one turn."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    elapsed_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    def describe(self) -> str:
        names = ", ".join(c["name"] for c in self.tool_calls) or "无"
        return f"{self.rounds} 轮 / 调用 {names} / {self.elapsed_ms:.0f}ms"


@dataclass
class AssistantReply:
    """One answer plus its provenance."""

    text: str
    trace: TurnTrace
    round_trips: int = 0


class TrainingAssistant:
    """Answers questions about the platform, grounded in tools and node state."""

    def __init__(
        self,
        settings: LlmSettings,
        preferences: AssistantPreferences | None = None,
        client: OpenAiCompatClient | None = None,
    ) -> None:
        self.settings = settings
        self.preferences = preferences or AssistantPreferences()
        self.client = client or OpenAiCompatClient(settings)

    def ask(
        self,
        question: str,
        *,
        ctx: ToolContext | None = None,
        history: list[AssistantMessage] | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> AssistantReply:
        """Answer one question.

        ``extra_context`` carries client-side facts the API cannot look up, such
        as the quality report the browser already holds. They are offered to the
        model as a tool result rather than pasted into the prompt, so it cannot
        treat stale UI numbers as authoritative.
        """
        ctx = ctx or ToolContext()
        trace = TurnTrace()
        started = time.perf_counter()

        messages: list[AssistantMessage] = [
            AssistantMessage(
                role="system", content=build_system_prompt(self.settings, self.preferences, ctx)
            ),
            *(history or []),
            AssistantMessage(role="user", content=question),
        ]

        if extra_context:
            # Seeded as a tool result so the provenance is explicit and the
            # model can tell user-reported data from tool-fetched data.
            messages.append(
                AssistantMessage(
                    role="user",
                    content="[平台已附带的上下文]",
                )
            )
            messages.append(
                AssistantMessage(
                    role="user",
                    content=_render_extra_context(extra_context),
                )
            )

        schemas = tool_schemas() if self.settings.supports_tools else None

        for round_index in range(self.preferences.max_tool_rounds):
            trace.rounds = round_index + 1
            content, tool_calls, usage = self.client.chat(messages, tools=schemas)
            _accumulate_usage(trace.usage, usage)

            # Normalize defensively rather than trusting that a particular client
            # implementation already did it. Providers disagree on the shape, and
            # a loop that assumed one of them would fail silently.
            tool_calls = [normalize_tool_call(c) for c in tool_calls]

            if not tool_calls:
                trace.elapsed_ms = (time.perf_counter() - started) * 1000
                return AssistantReply(
                    text=content or "（模型没有返回内容，请重试或调整配置）",
                    trace=trace,
                    round_trips=trace.rounds,
                )

            messages.append(
                AssistantMessage(role="assistant", content=content, tool_calls=tool_calls)
            )

            for call in tool_calls:
                result = run_tool(call["name"], call["arguments"], ctx)
                if extra_context and call["name"] == "get_dataset_quality":
                    result = {**result, **extra_context.get("quality", {})}
                trace.tool_calls.append(
                    {"name": call["name"], "arguments": call["arguments"], "result": result}
                )
                messages.append(
                    AssistantMessage(
                        role="tool",
                        content=_render_tool_result(result),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )

        # Round cap reached without a text answer. Reported rather than returning
        # whatever partial tool trace exists as if it were an answer.
        trace.elapsed_ms = (time.perf_counter() - started) * 1000
        trace.truncated = True
        return AssistantReply(
            text=(
                f"这个问题需要查询的数据超出了单轮上限（已调用 "
                f"{len(trace.tool_calls)} 次工具）。请把问题拆小一些，"
                "或者直接告诉我你卡在哪一步。"
            ),
            trace=trace,
            round_trips=trace.rounds,
        )

    def suggest(self, ctx: ToolContext, *, has_data: bool, has_quality: bool) -> str:
        """Contextual nudge for the input placeholder."""
        return suggestion_prompt(ctx, has_data=has_data, has_quality=has_quality)

    def probe(self) -> dict[str, Any]:
        return self.client.probe()


def _render_tool_result(result: dict[str, Any]) -> str:
    from .tools import dumps

    return dumps(result)


def _render_extra_context(context: dict[str, Any]) -> str:

    from .tools import dumps

    return dumps(context)


def _accumulate_usage(target: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


__all__ = [
    "AssistantReply",
    "LlmError",
    "OpenAiCompatClient",
    "ToolContext",
    "TrainingAssistant",
    "TurnTrace",
]
