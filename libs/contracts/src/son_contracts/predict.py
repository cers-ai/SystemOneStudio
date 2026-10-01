"""Prediction request/response contract for the deployed `/v1/predict` API."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from son_contracts.enums import Decision
from son_contracts.jev import JEV_OUTPUT_SCHEMA, REASON_MAX_LENGTH, JevFlags

#: Fields consumed by the platform itself rather than by the model.
_RESERVED_FIELDS = frozenset({"model_version_id"})


class PredictRequest(BaseModel):
    """A single decision request.

    The sample payload is schema-free: the field set is defined per scene, and
    the platform must not reject a scene-specific field it has never seen. So
    unknown fields are allowed and passed through to the prompt template.
    """

    model_config = ConfigDict(extra="allow")

    model_version_id: str | None = Field(
        default=None,
        description="Pin a specific model version; omit to use the active deployment.",
    )

    @property
    def sample(self) -> dict[str, Any]:
        """The user-supplied feature payload, without platform fields."""
        return {k: v for k, v in self.model_dump().items() if k not in _RESERVED_FIELDS}

    def render_features(self) -> str:
        """Render the sample for the `{features}` slot of the input template.

        Stable key order so that prompts are reproducible and cacheable, which
        matters because training and evaluation must see byte-identical prompts.
        """
        return "\n".join(f"{key}: {self.sample[key]}" for key in sorted(self.sample))


class PredictResponse(BaseModel):
    """Decision output. Field set is identical whether or not JEV format
    compatibility is on, so callers need only one parser.

    ``jev_compatible`` tells the caller which contract they are actually
    holding. It is false when the scene runs with ``jev_format_compat``
    disabled, in which case `reason` is no longer bound by the JEV schema.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Decision = Field(description="判定结果: black / white / gray")
    score: float = Field(ge=0.0, le=1.0, description="风险评分")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    reason: str = Field(
        min_length=1,
        description="决策依据 (required; capped at 200 chars under JEV format compat)",
    )
    jev_compatible: bool = Field(
        default=True,
        description="False when the scene runs with jev_format_compat disabled.",
    )

    @model_validator(mode="after")
    def _check_reason_bound(self) -> Self:
        """Enforce the JEV 200-char cap only when JEV format compat is on.

        The cap is not declared as a field-level ``max_length`` because that
        would apply it unconditionally, and it must not: with
        ``jev_format_compat`` disabled the reason field is no longer governed
        by the JEV schema.

        Rejecting with a specific message is deliberate. Silently truncating
        would corrupt the format-compliance metric, which is one of the
        product's tracked success indicators (需求方案.txt 14.2).
        """
        if self.jev_compatible and len(self.reason) > REASON_MAX_LENGTH:
            raise ValueError(
                f"reason exceeds {REASON_MAX_LENGTH} chars "
                f"(got {len(self.reason)}) under jev_format_compat"
            )
        return self

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank: every decision needs a stated basis")
        return value

    def to_jev_dict(self) -> dict[str, Any]:
        """Project onto the JEV output schema, dropping the compat marker."""
        return {
            "decision": self.decision.value,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class JevValidationResult(BaseModel):
    """Outcome of checking a prediction against `JEV_OUTPUT_SCHEMA`."""

    compliant: bool
    errors: list[str] = Field(default_factory=list)
    schema_version: str = "jev-1.0"

    def add(self, error: str) -> None:
        self.errors.append(error)
        self.compliant = False


def validate_jev_payload(payload: dict[str, Any], flags: JevFlags) -> JevValidationResult:
    """Check a raw prediction payload against the JEV output schema.

    Kept separate from `PredictResponse` on purpose: the inference service
    validates whatever the model actually emitted *before* it is coerced into
    a response, so that format-compliance is measured on the model's real
    output rather than on a repair of it.
    """
    result = JevValidationResult(compliant=True)
    if not flags.jev_format_compat:
        return result

    schema = JEV_OUTPUT_SCHEMA
    for key in schema["required"]:
        if key not in payload:
            result.add(f"missing required field: {key}")

    extra = set(payload) - set(schema["properties"])
    if extra:
        result.add(f"unexpected fields: {sorted(extra)}")

    for key in schema["properties"]:
        if key not in payload:
            continue
        spec = schema["properties"][key]
        value = payload[key]
        if "enum" in spec and value not in spec["enum"]:
            result.add(f"{key} must be one of {spec['enum']}, got {value!r}")
        if spec.get("type") == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                result.add(f"{key} must be a number, got {type(value).__name__}")
            else:
                if "minimum" in spec and value < spec["minimum"]:
                    result.add(f"{key} below minimum {spec['minimum']} (got {value})")
                if "maximum" in spec and value > spec["maximum"]:
                    result.add(f"{key} above maximum {spec['maximum']} (got {value})")
        if "maxLength" in spec and isinstance(value, str) and len(value) > spec["maxLength"]:
            result.add(f"{key} exceeds maxLength {spec['maxLength']} (got {len(value)})")
    return result
