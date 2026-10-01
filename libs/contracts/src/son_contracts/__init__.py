"""Cross-service contracts for SystemOneStudio.

This package is the single source of truth for types that cross a service
boundary. Services must import from here and never from each other's internals.
The OpenAPI document served by apps/api is the source for the generated
TypeScript types consumed by apps/web.
"""

from son_contracts.enums import (
    RAPID_MODE_LAST_MANUAL_STEP,
    WIZARD_STEP_ORDER,
    ArtifactFormat,
    DataOrigin,
    Decision,
    JevCompatLevel,
    QuantLevel,
    RunState,
    Split,
    StepState,
    SynthMethod,
    TrainingMethod,
    UiMode,
)
from son_contracts.jev import (
    DEFAULT_JEV_FLAGS,
    EFFECT_TARGETS,
    JEV_BASELINE_QUANT,
    JEV_EVAL_BASELINE,
    JEV_INPUT_TEMPLATE,
    JEV_OUTPUT_SCHEMA,
    JEV_SPEC_VERSION,
    PERFORMANCE_TARGETS,
    REASON_MAX_LENGTH,
    JevEvalBaseline,
    JevFlags,
)
from son_contracts.lineage import Lineage, QualityReport, SplitSummary
from son_contracts.predict import (
    JevValidationResult,
    PredictRequest,
    PredictResponse,
    validate_jev_payload,
)

__all__ = [
    "DEFAULT_JEV_FLAGS",
    "EFFECT_TARGETS",
    "JEV_BASELINE_QUANT",
    "JEV_EVAL_BASELINE",
    "JEV_INPUT_TEMPLATE",
    "JEV_OUTPUT_SCHEMA",
    "JEV_SPEC_VERSION",
    "PERFORMANCE_TARGETS",
    "RAPID_MODE_LAST_MANUAL_STEP",
    "REASON_MAX_LENGTH",
    "WIZARD_STEP_ORDER",
    "ArtifactFormat",
    "DataOrigin",
    "Decision",
    "JevCompatLevel",
    "JevEvalBaseline",
    "JevFlags",
    "JevValidationResult",
    "Lineage",
    "PredictRequest",
    "PredictResponse",
    "QualityReport",
    "QuantLevel",
    "RunState",
    "Split",
    "SplitSummary",
    "StepState",
    "SynthMethod",
    "TrainingMethod",
    "UiMode",
    "validate_jev_payload",
]
