"""V3 probability contracts. Legacy generative/JEV contracts remain separate."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from son_contracts.enums import Decision

STATE_RESERVED_FIELDS = frozenset(
    {
        "decision",
        "label",
        "target",
        "reason",
        "evidence",
        "score",
        "confidence",
        "prediction",
        "origin",
    }
)


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FieldRole(StrEnum):
    STATE_INPUT = "STATE_INPUT"
    TARGET_CHOICE = "TARGET_CHOICE"
    EVIDENCE = "EVIDENCE"
    GROUP_ID = "GROUP_ID"
    ENTITY_ID = "ENTITY_ID"
    IGNORE = "IGNORE"


class StateValueType(StrEnum):
    TEXT = "TEXT"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    JSON = "JSON"


class DecisionOption(FrozenContract):
    id: Decision
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)

    @field_validator("name", "description")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("候选含义不能为空")
        return value.strip()


class DecisionSchema(FrozenContract):
    schema_id: str = Field(min_length=1)
    type: Literal["choice"] = "choice"
    version: int = Field(ge=1, strict=True)
    question: str = Field(min_length=1, max_length=4000)
    options: tuple[DecisionOption, DecisionOption, DecisionOption]

    @field_validator("question")
    @classmethod
    def meaningful_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("判断问题不能为空")
        return value.strip()

    @model_validator(mode="after")
    def three_choices(self) -> Self:
        if {option.id for option in self.options} != set(Decision):
            raise ValueError("必须恰好定义 black / white / gray 三类")
        return self


class FieldMapping(FrozenContract):
    column: str = Field(min_length=1)
    role: FieldRole
    value_type: StateValueType = StateValueType.TEXT

    @model_validator(mode="after")
    def no_label_input(self) -> Self:
        if (
            self.role == FieldRole.STATE_INPUT
            and self.column.strip().casefold() in STATE_RESERVED_FIELDS
        ):
            raise ValueError(f"{self.column} 不能作为模型输入")
        return self


def validate_state(state: dict[str, JsonValue]) -> None:
    """Reject non-finite values and audit/target keys at every nesting level."""

    def visit(value: JsonValue) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.strip().casefold() in STATE_RESERVED_FIELDS:
                    raise ValueError(f"State 包含目标或审计字段：{key}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("State 不允许 NaN/Infinity")

    if not state:
        raise ValueError("State 不能为空")
    visit(state)


class ChoiceTarget(FrozenContract):
    choice: Decision


class DecisionSample(FrozenContract):
    sample_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    state: dict[str, JsonValue]
    target: ChoiceTarget
    evidence: str | None = None
    source: Literal["seed", "constructed"] = "seed"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("sample_id", "group_id")
    @classmethod
    def meaningful_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("样本和分组标识不能为空")
        return value.strip()

    @field_validator("state")
    @classmethod
    def safe_state(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        validate_state(value)
        return value


class ChoiceProbabilities(FrozenContract):
    black: float = Field(ge=0, le=1)
    white: float = Field(ge=0, le=1)
    gray: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def normalized(self) -> Self:
        if not math.isclose(self.black + self.white + self.gray, 1.0, abs_tol=1e-6):
            raise ValueError("三类概率之和必须为 1")
        return self


class DecisionAction(StrEnum):
    AUTO = "AUTO"
    REVIEW = "REVIEW"


class DecisionPredictRequest(FrozenContract):
    model: str = Field(min_length=1)
    state: dict[str, JsonValue]

    @field_validator("state")
    @classmethod
    def safe_state(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        validate_state(value)
        return value


class DecisionPredictResponse(FrozenContract):
    trace_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    schema_version: int = Field(ge=1, strict=True)
    choice: Decision
    probabilities: ChoiceProbabilities
    confidence: float = Field(ge=0, le=1)
    action: DecisionAction
    policy: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_readout(self) -> Self:
        probs = self.probabilities.model_dump()
        maximum = max(probs.values())
        if not math.isclose(self.confidence, maximum, abs_tol=1e-6):
            raise ValueError("confidence 必须为最大候选概率")
        if not math.isclose(probs[self.choice.value], maximum, abs_tol=1e-6):
            raise ValueError("choice 必须为最大概率候选")
        return self
