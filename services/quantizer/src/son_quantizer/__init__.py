"""LoRA merge and GGUF quantization.

Q4_K_M is the default because it matches the JEV evaluation baseline; numbers
measured at different levels are not comparable.

NOT VERIFIED: nothing in this package has been run against a real model. The
development machine has no GPU and no llama.cpp checkout, so only the
orchestration -- ordering, default level, artifact registration -- is exercised
by tests. See AGENTS.md.
"""

from son_quantizer.pipeline import (
    AUTO_ARTIFACT_LEVELS,
    Artifact,
    LlamaCppTools,
    QuantizeRequest,
    QuantizeResult,
    SubprocessLlamaCppTools,
    ToolchainError,
    artifacts_for_registration,
    build_gguf_filename,
    quantize_all,
)

__all__ = [
    "AUTO_ARTIFACT_LEVELS",
    "Artifact",
    "LlamaCppTools",
    "QuantizeRequest",
    "QuantizeResult",
    "SubprocessLlamaCppTools",
    "ToolchainError",
    "artifacts_for_registration",
    "build_gguf_filename",
    "quantize_all",
]
