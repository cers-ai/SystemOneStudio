"""Base model registry and adapter abstraction.

Solves the "vendor lock-in" pain point from 需求方案.txt section 1.2: a new base
model is added by registering an adapter, not by editing the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from son_contracts import ArtifactFormat, JevCompatLevel, QuantLevel, TrainingMethod


class RegistryModel(BaseModel):
    """A base model entry.

    Field set mirrors the `model_registry` YAML in 需求方案.txt section 7.1.

    ``jev_compat_level`` must be written from a measurement, not typed in by
    hand: L3 requires a comparison test against native JEV output before it may
    be claimed. See 技术方案.md section 5.2.
    """

    model_config = {"extra": "forbid"}

    model_id: str
    display_name: str
    family: str
    params: str = Field(description="Human-readable size, e.g. 3B")
    license: str
    modality: str = "text"
    supported_tasks: list[str] = Field(default_factory=list)
    jev_compatible: bool = False
    jev_compat_level: JevCompatLevel | None = None
    jev_level_evidence: Literal["declared", "measured"] | None = Field(
        default=None,
        description=(
            "How jev_compat_level was established. 'measured' requires a "
            "comparison test against native JEV output."
        ),
    )
    min_gpu_memory: str = Field(description="Minimum GPU memory, e.g. 12GB")
    available_formats: list[ArtifactFormat] = Field(default_factory=list)
    adapter_type: str = Field(description="Key into the adapter factory registry")
    source_url: str | None = None
    local_cache_path: str | None = None
    in_mvp_scope: bool = Field(
        default=True,
        description=(
            "False for models outside 需求方案.txt 10.2. The MVP is capped at "
            "1.5B-3B with LoRA/QLoRA as a deliberate answer to limited VRAM."
        ),
    )

    @model_validator(mode="after")
    def _l3_requires_measurement(self) -> "RegistryModel":
        """L3 style alignment must be earned by a comparison test.

        需求方案.txt 6.2 states L3 is verified by 对比测试, not by declaration.
        Encoding that here prevents a seed file or an API call from asserting
        L3 on nothing but a claim.
        """
        if self.jev_compat_level is JevCompatLevel.L3 and self.jev_level_evidence != "measured":
            raise ValueError(
                "JEV L3 (style compatibility) requires jev_level_evidence='measured'; "
                "a declared level is not sufficient verification"
            )
        return self

    @model_validator(mode="after")
    def _measured_level_requires_compatible(self) -> "RegistryModel":
        if self.jev_compat_level is not None and not self.jev_compatible:
            raise ValueError("jev_compat_level is set but jev_compatible is False")
        return self


class TrainConfig(BaseModel):
    """Training configuration passed to `BaseModelAdapter.build_trainer`.

    Hyperparameters arrive already resolved: defaults are computed from model
    size and dataset size per UX principle 2, so nothing downstream should
    second-guess them.
    """

    model_config = {"extra": "allow"}

    methods: list[TrainingMethod]
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_steps: int = 1000
    seed: int = 42
    output_dir: str | None = None


class BaseModelAdapter(ABC):
    """Unified adapter interface.

    Signatures follow 需求方案.txt section 7.3 and must not be renamed.

    Note that `format_prompt` and `format_jev_prompt` are both required on every
    adapter. Turning off `jev_format_compat` means falling back to
    `format_prompt`, not having no prompt at all: a single adapter must stay
    usable in both modes so we never maintain two implementations per family.
    """

    #: Registry key matching `RegistryModel.adapter_type`.
    adapter_type: ClassVar[str] = "base"

    def __init__(self, model_path: str) -> None:
        """Bind the adapter to one model path.

        Every adapter needs it, and passing it per call would mean threading it
        through every method. The seven abstract signatures below are unchanged
        from 需求方案.txt 7.3.
        """
        self.model_path = model_path

    @abstractmethod
    def load_tokenizer(self, model_path: str) -> Any:
        """Load and return the tokenizer for `model_path`."""

    @abstractmethod
    def format_prompt(self, sample: dict[str, Any]) -> str:
        """Plain instruction prompt, used when jev_format_compat is disabled."""

    @abstractmethod
    def format_jev_prompt(self, sample: dict[str, Any]) -> str:
        """JEV-aligned prompt using `JEV_INPUT_TEMPLATE`."""

    @abstractmethod
    def build_trainer(self, train_config: TrainConfig, dataset: Any) -> Any:
        """Construct the PEFT/TRL trainer for this base model."""

    @abstractmethod
    def merge_lora(self, base_path: str, lora_path: str, output_path: str) -> str:
        """Merge adapter weights into the base model. Returns the merged path."""

    @abstractmethod
    def export_gguf(
        self, model_path: str, output_path: str, quant_level: QuantLevel = QuantLevel.Q4_K_M
    ) -> str:
        """Convert to GGUF and quantize. Q4_K_M is the JEV baseline quant."""

    @abstractmethod
    def infer(self, model_path: str, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run a single prediction and return a JEV-shaped payload."""


AdapterFactory = type[BaseModelAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}


def register_adapter(adapter_cls: type[BaseModelAdapter]) -> type[BaseModelAdapter]:
    """Class decorator adding an adapter to the factory registry."""
    key = adapter_cls.adapter_type
    if not key or key == "base":
        raise ValueError(f"{adapter_cls.__name__} must define a concrete adapter_type")
    if key in _REGISTRY:
        raise ValueError(f"adapter_type {key!r} already registered by {_REGISTRY[key].__name__}")
    _REGISTRY[key] = adapter_cls
    return adapter_cls


def get_adapter(adapter_type: str) -> AdapterFactory:
    try:
        return _REGISTRY[adapter_type]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown adapter_type {adapter_type!r}; registered: {known}") from None


def registered_adapters() -> list[str]:
    return sorted(_REGISTRY)
