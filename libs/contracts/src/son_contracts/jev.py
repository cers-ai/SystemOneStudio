"""JEV alignment constants.

PROVENANCE WARNING
------------------
`需求方案.txt` section 6.1 reproduces a *paraphrase* of the JEV spec. It is not
the upstream document, and neither the full JEV style definition nor the JEV
evaluation benchmark dataset is present in this repository.

Anything asserting L2 or L3 alignment must be backed by an externally verified
source, recorded here with its provenance, before it is claimed. Until then
only L1 is implementable and assertable. Tracked as Q3 in 技术方案.md.
"""

from typing import Any, Final, Literal

from pydantic import BaseModel

JEV_SPEC_VERSION: Final = "jev-1.0"

#: Default quantization for both deployment and evaluation. Evaluations are
#: only comparable when every model is measured at the JEV baseline quant.
JEV_BASELINE_QUANT: Final = "Q4_K_M"

#: Upper bound on the `reason` field. Enforced, not advisory: a decision
#: without a stated justification is not shippable for compliance review.
REASON_MAX_LENGTH: Final = 200

JEV_INPUT_TEMPLATE: Final = """<system>
你是一个专业的决策模型，负责对输入样本进行判定。
</system>
<task>
根据以下特征，判断该样本是否属于风险样本。
</task>
<features>
{features}
</features>
<output_format>
请输出 JSON 格式：
{"decision": "...", "score": ..., "confidence": ..., "reason": "..."}
</output_format>
"""

#: JSON Schema for the JEV prediction output. Used for runtime validation and
#: for generating the GBNF grammar that constrains decoding, so that
#: format compliance is structural rather than something we repair after the
#: fact with string surgery.
JEV_OUTPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "score", "confidence", "reason"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["black", "white", "gray"],
            "description": "判定结果",
        },
        "score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "风险评分",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "置信度",
        },
        "reason": {
            "type": "string",
            "maxLength": REASON_MAX_LENGTH,
            "description": "决策依据",
        },
    },
}


class JevEvalBaseline(BaseModel):
    """Evaluation baseline and target thresholds from JEV spec section 6.1."""

    model_config = {"frozen": True}

    quant: str = JEV_BASELINE_QUANT
    latency_metric: str = "p95_end_to_end_ms"
    accuracy_metric: str = "decision_accuracy"
    target_latency_p95_ms: int = 100
    target_accuracy: float = 0.90


JEV_EVAL_BASELINE: Final = JevEvalBaseline()


class JevFlags(BaseModel):
    """The two JEV compatibility switches.

    These are two independent switches by requirement, not one toggle. They
    control different layers and must be allowed to differ:

    * ``jev_format_compat``   -> input/output schema, prompt format, API shape
    * ``jev_training_compat`` -> training paradigm, hyperparameters, eval basis

    Both default to enabled. When ``jev_format_compat`` is off the API still
    returns the same four fields so callers keep a single parser, but the
    response is explicitly flagged as non-JEV via ``PredictResponse``'s
    ``jev_compatible`` field.
    """

    model_config = {"frozen": True}

    jev_format_compat: bool = True
    jev_training_compat: bool = True

    def any_enabled(self) -> bool:
        """Whether any switch is on.

        Deliberately a named method rather than ``__bool__``: a falsey
        ``JevFlags`` would make ``flags or JevFlags()`` silently discard a
        caller's deliberate "both off" configuration.
        """
        return self.jev_format_compat or self.jev_training_compat


DEFAULT_JEV_FLAGS: Final = JevFlags()

#: Performance targets are first priority, effect targets second. This ordering
#: is an architectural input, not a reporting preference: the P95 target is
#: what rules out heavier inference stacks. See 技术方案.md section 1.1.
PERFORMANCE_TARGETS: Final[dict[str, float]] = {
    "p50_ms": 50.0,
    "p95_ms": 100.0,
    "p99_ms": 200.0,
    "ttft_ms": 30.0,
    "qps": 100.0,
    "peak_memory_mb": 2048.0,
}

EFFECT_TARGETS: Final[dict[str, float]] = {
    "accuracy": 0.90,
    "recall": 0.85,
    "false_kill_rate": 0.05,
    "f1": 0.88,
    "auc_roc": 0.90,
    "format_compliance": 0.99,
}

MetricGroup = Literal["performance", "effect"]
