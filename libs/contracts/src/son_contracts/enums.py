"""Domain enums shared across every service.

These values are part of the product contract, not internal implementation
details: they are persisted in PostgreSQL, exposed over HTTP and asserted
against the JEV output schema. Changing a member is a breaking change.
"""

from enum import StrEnum


class Decision(StrEnum):
    """Three-way judgement outcome.

    Deliberately NOT a boolean. The `gray` label is a first-class outcome that
    threads through synthesis ratios, quality reports and per-label evaluation
    metrics. Collapsing it into black/white silently drops the hardest cases.
    """

    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"


class DataOrigin(StrEnum):
    """Row-level provenance.

    This is the field that makes the "synthetic data must never enter the test
    split" constraint enforceable. Split logic filters on it, it is not a
    convention callers are trusted to honour.
    """

    SEED = "seed"
    SYNTH = "synth"
    PLAYGROUND = "playground"


class Split(StrEnum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class QuantLevel(StrEnum):
    """GGUF quantization levels.

    Q4_K_M is the default because the JEV eval baseline is defined at Q4_K_M;
    performance numbers are only comparable at equal quantization.
    """

    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q8_0 = "Q8_0"


class ArtifactFormat(StrEnum):
    GGUF = "gguf"
    NATIVE = "native"


class JevCompatLevel(StrEnum):
    """JEV alignment levels.

    Assigned by measurement, never by hand. L3 in particular requires a
    comparison test against native JEV output before it may be written.
    """

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class TrainingMethod(StrEnum):
    """Training methods, named by their technical identifier.

    The UI must never render these strings. See libs/ui-terminology.
    """

    LORA = "lora"
    QLORA = "qlora"
    SFT = "sft"
    DPO = "dpo"
    DISTILLATION = "distillation"
    QAT = "qat"
    FEWSHOT = "fewshot"


class UiMode(StrEnum):
    """Interaction modes. All four share one engine and one flow definition."""

    WIZARD = "wizard"
    CANVAS = "canvas"
    RAPID = "rapid"
    EXPERT = "expert"


class SynthMethod(StrEnum):
    DISTRIBUTION_FIT = "distribution_fit"
    SMALL_SAMPLE_DERIVE = "small_sample_derive"
    RULE_INJECTION = "rule_injection"


class RunState(StrEnum):
    """Lifecycle for every long-running job.

    The orchestrator state machine and every `status` column are generated from
    this enum so the two cannot drift apart.
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepState(StrEnum):
    """The 7-step wizard pipeline.

    SKIPPED_WITH_DEFAULT exists because every wizard step offers a
    "skip, use defaults" affordance; those runs must remain distinguishable
    from runs where the user actually configured the step.
    """

    SCENE_SELECTED = "scene_selected"
    SEED_UPLOADED = "seed_uploaded"
    DATA_GOVERNED = "data_governed"
    SYNTH_DONE = "synth_done"
    BASE_MODEL_SELECTED = "base_model_selected"
    TRAINING_CONFIGURED = "training_configured"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    DEPLOYED = "deployed"


WIZARD_STEP_ORDER: tuple[StepState, ...] = (
    StepState.SCENE_SELECTED,
    StepState.SEED_UPLOADED,
    StepState.DATA_GOVERNED,
    StepState.SYNTH_DONE,
    StepState.BASE_MODEL_SELECTED,
    StepState.TRAINING_CONFIGURED,
    StepState.TRAINED,
    StepState.EVALUATED,
    StepState.DEPLOYED,
)

#: In rapid mode the user only completes steps 1-2 (scene + seed upload);
#: the orchestrator drives everything after that. Keep this a constant so the
#: API, the orchestrator and the UI cannot disagree about where "rapid" ends.
RAPID_MODE_LAST_MANUAL_STEP: StepState = StepState.SEED_UPLOADED
