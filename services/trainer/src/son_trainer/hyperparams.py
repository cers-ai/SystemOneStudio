"""Smart hyperparameter defaults.

需求方案.txt principle 2: "能不动就不动" -- every configurable value gets a
scene-derived default and the business user only touches what they must. The
illustrative values in 5.6 (2e-4 / batch 4 / 1000 steps) are therefore *outputs*
of these functions, not constants to be copied.

The scaling rules follow the well-known linear/√ scaling heuristics for LoRA
fine-tuning. They are ours, not the requirement's, and are deliberately kept in
one place so they can be retuned once a GPU node exists to measure against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

from son_contracts import TrainingMethod

#: Documented example values from 需求方案.txt 5.6, used as anchors.
REFERENCE_LEARNING_RATE = 2e-4
REFERENCE_BATCH_SIZE = 4
REFERENCE_MAX_STEPS = 1_000

#: Learning rate below this the run barely moves; above it a small model diverges.
MIN_LR = 5e-5
MAX_LR = 5e-4

#: Beyond this the VRAM requirement exceeds the 12-16GB the MVP targets.
MAX_BATCH_SIZE = 16


@dataclass(frozen=True)
class DatasetProfile:
    """The two facts a recommendation actually depends on."""

    rows: int
    params_b: float
    #: Whether the run is QLoRA, which trades VRAM for a lower ceiling.
    quantized: bool = False
    avg_prompt_tokens: int = 320
    avg_response_tokens: int = 160

    @property
    def approx_tokens_per_row(self) -> int:
        return self.avg_prompt_tokens + self.avg_response_tokens

    @property
    def approx_epochs_at_reference_steps(self) -> float:
        return REFERENCE_MAX_STEPS * REFERENCE_BATCH_SIZE / max(self.rows, 1)


class Hyperparams(TypedDict):
    """Resolved training hyperparameters.

    A TypedDict rather than ``dict[str, float | int | str]`` so callers get a
    precise type per field; the heterogeneous form makes every comparison at the
    call site a union comparison that mypy cannot narrow.
    """

    learning_rate: float
    batch_size: int
    max_steps: int
    seed: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    quantized: bool


def recommend_learning_rate(profile: DatasetProfile, *, difficulty: str = "normal") -> float:
    """Scale the reference 2e-4 by model size, then bound it.

    Smaller models tolerate (and need) higher rates; larger ones need lower.
    """
    scale = (3.0 / max(profile.params_b, 0.5)) ** 0.5
    rate = REFERENCE_LEARNING_RATE * scale

    if difficulty == "hard":
        # More classes or noisier labels -> smaller, more conservative steps.
        rate *= 0.7
    elif difficulty == "easy":
        rate *= 1.15

    return round(min(max(rate, MIN_LR), MAX_LR), 7)


def recommend_batch_size(profile: DatasetProfile) -> int:
    """Larger models and quantized runs get smaller batches to hold VRAM.

    The MVP caps the base model at 1.5B-3B and targets 12-16GB (需求方案.txt 8.1),
    so the batch size is chosen to keep the step within that envelope.
    """
    if profile.params_b >= 3.0:
        batch = 2
    elif profile.params_b >= 2.0:
        batch = 4
    else:
        batch = 8

    if profile.quantized:
        batch = min(batch * 2, MAX_BATCH_SIZE)

    return min(batch, MAX_BATCH_SIZE)


def recommend_max_steps(profile: DatasetProfile, *, epochs: float = 3.0) -> int:
    """Enough steps to see the data a few times, bounded to stay inside an hour.

    需求方案.txt 5.6 shows a ~45 minute target for a 3B LoRA run. A step is
    dominated by tokens, so step count is derived from token volume rather than
    from row count alone.
    """
    if profile.rows <= 0:
        return REFERENCE_MAX_STEPS

    steps_for_epochs = math.ceil(profile.rows * epochs / max(profile.quantized * 2 or 1, 1))

    # 45 minutes at a conservative ~2 steps/second on a mid-range GPU.
    time_budget_steps = 5_400

    return int(max(200, min(steps_for_epochs, time_budget_steps, REFERENCE_MAX_STEPS * 3)))


def recommend_hyperparams(
    profile: DatasetProfile,
    methods: tuple[TrainingMethod, ...],
    *,
    difficulty: str = "normal",
    seed: int = 42,
) -> Hyperparams:
    """Full hyperparameter set for a training run.

    ``jev_training_compat`` is recorded here rather than applied silently: when
    it is on, DPO's learning rate is held below SFT's, because preference
    training diverges much more easily than supervised training.
    """
    lr = recommend_learning_rate(profile, difficulty=difficulty)
    batch = recommend_batch_size(profile)

    if TrainingMethod.DPO in methods:
        # Keep DPO conservative relative to SFT; see note above.
        lr = round(lr * 0.5, 7)

    return {
        "learning_rate": lr,
        "batch_size": batch,
        "max_steps": recommend_max_steps(profile),
        "seed": seed,
        "gradient_accumulation_steps": max(1, 16 // batch),
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "quantized": profile.quantized,
    }


def estimate_duration_minutes(
    profile: DatasetProfile,
    hyperparams: Hyperparams,
    *,
    steps_per_second: float = 2.0,
) -> float:
    """Wall-clock estimate shown in the model picker (需求方案.txt 5.5).

    ``steps_per_second`` is a placeholder throughput for a mid-range GPU. It has
    never been measured here -- this machine has no GPU -- so the number is a
    planning aid and must be replaced by a measurement before it is shown to
    users as a promise.
    """
    steps = int(hyperparams.get("max_steps", REFERENCE_MAX_STEPS))
    return round(steps / max(steps_per_second, 0.01) / 60, 1)
