"""Training pipeline: SFT -> LoRA/QLoRA -> DPO preference alignment.

Plan: M2 `[GPU]`. torch / transformers / peft / trl / bitsandbytes are optional
extras with Python pinned to 3.11; see AGENTS.md. Hyperparameters arrive
already resolved from the recommender, never defaulted here.
"""
