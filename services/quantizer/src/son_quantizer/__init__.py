"""LoRA merge and GGUF quantization.

Plan: M2 `[GPU]`. Drives llama.cpp tooling (convert + llama-quantize); Q4_K_M is
the default because it matches the JEV eval baseline.
"""
