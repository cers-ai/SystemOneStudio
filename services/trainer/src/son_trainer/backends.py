"""Training backend interface.

The pipeline, hyperparameters, checkpointing and preference-pair generation in
this package are all plain Python and fully testable. Actual optimization needs
torch + PEFT + TRL + bitsandbytes and a GPU, neither of which exists on the
development machine (see AGENTS.md). So the heavy work sits behind this
interface: the orchestration is verified here, and swapping in the real backend
requires implementing one method.

This is a deliberate split, not a stub pretending to be a trainer. A fake trainer
that silently "succeeds" would let the pipeline's control flow be tested while
proving nothing about training -- but a protocol boundary does let the control
flow, resume semantics and preference bookkeeping be tested for real.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from son_contracts import TrainingMethod
from son_trainer.hyperparams import Hyperparams


@dataclass(frozen=True)
class TrainSample:
    """One prompt/response pair for supervised training."""

    prompt: str
    response: str
    origin: str = "seed"
    #: Preference pairs (需求方案.txt 8.2) carry chosen/rejected instead.
    chosen: str | None = None
    rejected: str | None = None

    def is_preference(self) -> bool:
        return self.chosen is not None and self.rejected is not None


@dataclass(frozen=True)
class StageResult:
    """Outcome of one pipeline stage."""

    stage: str
    status: str
    step: int
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_uri: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


@dataclass(frozen=True)
class StageRequest:
    """What a stage needs in order to run."""

    stage: TrainingMethod
    samples: tuple[TrainSample, ...]
    base_model: str
    hyperparams: Hyperparams
    adapter_path: str | None = None
    resume_from: str | None = None
    seed: int = 42
    output_dir: str | None = None


class TrainingBackend(ABC):
    """Implement this to run real optimization on a GPU node."""

    #: Reported in stage notes so a log always says what actually ran.
    name: str = "abstract"

    @abstractmethod
    def run_stage(self, request: StageRequest) -> StageResult:
        """Execute one stage, optionally resuming from a checkpoint."""

    def sample_predictions(
        self,
        adapter_path: str,
        prompts: tuple[str, ...],
        *,
        n_per_prompt: int = 4,
    ) -> tuple[str, ...]:
        """Sample candidate outputs, used to build DPO preference pairs.

        Returns (prompt, output) pairs flattened. Only needed by pipelines that
        include a DPO stage, so the default is an explicit refusal rather than
        fabricated samples.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sampling; "
            "a DPO stage needs it (需求方案.txt 8.2)"
        )
