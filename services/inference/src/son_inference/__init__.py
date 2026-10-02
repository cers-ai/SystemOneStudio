"""Inference layer: llama.cpp orchestration and JEV schema enforcement.

NOT VERIFIED: no request has been served on this machine. There is no GPU and
no llama.cpp checkout here, so schema enforcement and timing *bookkeeping* are
tested while latency is not measured. Q1 (技术方案.md 1.1) additionally leaves
the P95 target itself unsettled, and Q3 leaves L2/L3 alignment unclaimed.
"""

from son_inference.predict import (
    InferenceEngine,
    Prediction,
    PredictionService,
    SchemaViolation,
    grammar_prompt_suffix_for,
)
from son_inference.server import (
    ServerConfig,
    ServerError,
    ServerHandle,
    deployment_summary,
    llama_cpp_available,
    start_server,
)

__all__ = [
    "InferenceEngine",
    "Prediction",
    "PredictionService",
    "SchemaViolation",
    "ServerConfig",
    "ServerError",
    "ServerHandle",
    "deployment_summary",
    "grammar_prompt_suffix_for",
    "llama_cpp_available",
    "start_server",
]
from son_inference.deployment import (
    Deployment,
    DeploymentFailed,
    DeploymentStatus,
    binary_available,
    curl_example,
    health_check,
    start,
    stop,
)
from son_inference.evaluate import (
    EvaluationOutcome,
    InferenceUnavailable,
    PredictionRecord,
    build_predict_response,
    read_test_rows,
    run_evaluation,
    summarize_latency,
)

__all__ += [
    "Deployment",
    "DeploymentFailed",
    "DeploymentStatus",
    "EvaluationOutcome",
    "InferenceUnavailable",
    "PredictionRecord",
    "binary_available",
    "build_predict_response",
    "curl_example",
    "health_check",
    "read_test_rows",
    "run_evaluation",
    "start",
    "stop",
    "summarize_latency",
]
