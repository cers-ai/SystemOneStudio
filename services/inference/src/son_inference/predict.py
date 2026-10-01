"""Inference: prompt formatting, schema enforcement, and llama.cpp wiring.

Two responsibilities, and the split is the point.

The business layer owns *enforcement*: schema validation, the mandatory
``reason`` field, the 200-character cap, and the timing breakdown. llama.cpp's
native endpoint cannot do those, and the requirement needs them (格式合规率 >= 99%,
and P50/TTFT tracked separately because they need different optimisations).

The engine layer is a thin wrapper. llama.cpp is chosen over a hosted stack
because 需求方案.txt 13 makes P95 <= 100ms an architectural input, not a nice
to-have.

UNVERIFIED: no request has been served here. This machine has no GPU and no
llama.cpp. The enforcement and timing logic is tested; the latency is not, and
Q1 (技术方案.md 1.1) additionally leaves the P95 target itself unsettled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from son_evaluator import LatencySample
from son_jev import build_grammar, format_jev_prompt, parse_completion

from son_contracts import (
    REASON_MAX_LENGTH,
    PredictRequest,
    PredictResponse,
    validate_jev_payload,
)


class SchemaViolation(ValueError):
    """A prediction that does not satisfy the JEV output schema.

    Raised instead of repairing the payload: silently truncating an over-long
    ``reason`` would falsify the format-compliance metric, which is a tracked
    product indicator (需求方案.txt 14.2).
    """


@dataclass
class Prediction:
    """One served request, with the timings tracked separately."""

    response: PredictResponse
    latency: LatencySample
    raw_completion: str = ""
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        return f"{self.response.decision.value} / {self.latency.total_ms:.0f}ms"


class InferenceEngine(ABC):
    """The llama.cpp side. Implemented by a subprocess or HTTP client."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        max_tokens: int = 256,
        stop: tuple[str, ...] = (),
    ) -> str:
        """Generate a completion for `prompt`."""

    @abstractmethod
    def is_ready(self) -> bool: ...


@dataclass
class PredictionService:
    """Business layer over an inference engine."""

    engine: InferenceEngine
    jev_format_compat: bool = True
    max_tokens: int = 256
    measurements: list[LatencySample] = field(default_factory=list)

    def predict(self, request: PredictRequest, *, measure: bool = False) -> Prediction:
        """Format, generate, validate, return.

        The prompt is rendered by :mod:`son_jev` so the format switch has exactly
        one implementation, and the grammar is attached only when JEV format
        compat is on -- otherwise a non-JEV scene is still constrained to JSON,
        which is not what turning the switch off means.
        """
        from son_contracts import JevFlags

        flags = JevFlags(jev_format_compat=self.jev_format_compat)
        prompt = format_jev_prompt(request.sample, flags)

        grammar = build_grammar().grammar if self.jev_format_compat else None

        ttft: float | None = None
        total = 0.0
        completion = ""

        if measure:
            total, ttft, completion = self._timed_generate(prompt, grammar)
        else:
            completion = self.engine.complete(prompt, grammar=grammar, max_tokens=self.max_tokens)
            total = 0.0

        payload = self._coerce(completion, flags)
        response = self._build_response(payload)

        sample = LatencySample(total_ms=total, ttft_ms=ttft)
        self.measurements.append(sample)
        return Prediction(
            response=response,
            latency=sample,
            raw_completion=completion,
            notes=self._notes(completion),
        )

    def _build_response(self, payload: dict[str, Any]) -> PredictResponse:
        """Coerce a validated payload into the response type.

        ``decision`` is typed as :class:`~son_contracts.Decision`, so the
        three-way enum holds even with ``jev_format_compat`` off. That is
        deliberate: the switch controls the output *schema* -- prompt shape,
        grammar, reason cap -- not whether the product is a three-class decision
        model. Letting ``decision`` be an arbitrary string would make the
        accuracy and recall metrics uncomputable.

        Pydantic errors are converted into :class:`SchemaViolation` so callers
        get a reason they can act on rather than a library-specific exception
        escaping the service layer.
        """
        from pydantic import ValidationError

        try:
            return PredictResponse(**payload, jev_compatible=self.jev_format_compat)
        except ValidationError as exc:
            problems = "；".join(
                f"{'.'.join(str(p) for p in error['loc']) or 'payload'}: {error['msg']}"
                for error in exc.errors()
            )
            raise SchemaViolation(f"输出未通过响应契约校验：{problems}") from exc

    def _coerce(self, completion: str, flags: Any) -> dict[str, Any]:
        """Parse and validate without repairing.

        Validation runs on the model's actual output, before coercion into the
        response type, so format compliance measures the model rather than this
        function's leniency.
        """
        try:
            payload = parse_completion(completion)
        except ValueError as exc:
            raise SchemaViolation(f"模型输出不符合输出格式：{exc}") from exc

        if self.jev_format_compat:
            result = validate_jev_payload(payload, flags)
            if not result.compliant:
                raise SchemaViolation("；".join(result.errors))

        if not str(payload.get("reason", "")).strip():
            raise SchemaViolation("reason 为空：每条判定都必须给出决策依据")

        return {
            "decision": payload["decision"],
            "score": _as_float(payload["score"], "score"),
            "confidence": _as_float(payload["confidence"], "confidence"),
            "reason": payload["reason"],
        }

    def _notes(self, completion: str) -> tuple[str, ...]:
        notes: list[str] = []
        if len(completion) > REASON_MAX_LENGTH * 2:
            notes.append("输出明显偏长，reason 可能触及长度上限")
        return tuple(notes)

    def _timed_generate(self, prompt: str, grammar: str | None) -> tuple[float, float | None, str]:
        """Time a generation, separating TTFT from total.

        The engine interface returns only the final text, so TTFT is unavailable
        unless the engine reports it. Returning None rather than equating it to
        the total keeps the two metrics distinguishable instead of quietly
        double-counting one number.
        """
        import time

        start = time.perf_counter()
        completion = self.engine.complete(prompt, grammar=grammar, max_tokens=self.max_tokens)
        total_ms = (time.perf_counter() - start) * 1000
        return total_ms, None, completion

    def measured_samples(self) -> list[LatencySample]:
        return list(self.measurements)


def grammar_prompt_suffix_for(flags: Any) -> str:
    """Hint appended to the prompt when grammar-constrained decoding is on."""
    return build_grammar().grammar if getattr(flags, "jev_format_compat", False) else ""


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SchemaViolation(f"{field} 必须是数值，实际为 {type(value).__name__}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaViolation(f"{field} 不是合法数值：{value!r}") from exc
