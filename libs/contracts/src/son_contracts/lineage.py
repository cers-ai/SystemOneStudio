"""Full-chain version lineage.

Every artifact must be traceable back to the scene, dataset, synthesis run,
base model and training method that produced it. The linkage is *materialized*
as a snapshot on the model version rather than derived through foreign keys:
an exported GGUF can be copied and served independently of this database, and
a materialized snapshot survives upstream renames.
"""

import re
from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CODE_PATTERNS: dict[str, re.Pattern[str]] = {
    "scene": re.compile(r"^sc_[A-Za-z0-9_]+$"),
    "dataset": re.compile(r"^ds_[A-Za-z0-9_]+$"),
    "synth": re.compile(r"^syn_[A-Za-z0-9_]+$"),
    "model_version": re.compile(r"^mv_[A-Za-z0-9_]+$"),
}


class Lineage(BaseModel):
    """Immutable provenance snapshot attached to a model version.

    Codes follow the naming style in 需求方案.txt section 5.7.1, e.g.
    ``sc_v3`` / ``ds_v2`` / ``syn_v5`` / ``mv_0017``.
    """

    model_config = ConfigDict(frozen=True)

    scene: str = Field(description="Scene code, e.g. sc_v3")
    dataset: str = Field(description="Dataset code, e.g. ds_v2")
    synth: str | None = Field(
        default=None,
        description="Synthesis run code, e.g. syn_v5. None when synthesis was skipped.",
    )
    base_model: str = Field(description="Base model id, e.g. qwen2.5-3b-instruct")
    method: str = Field(description="Training method id, e.g. jev_lora_dpo")
    model_version: str = Field(description="Model version code, e.g. mv_0017")

    @field_validator("scene", "dataset", "synth", "model_version")
    @classmethod
    def _check_code_shape(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        key = str(getattr(info, "field_name", ""))
        pattern = _CODE_PATTERNS.get(key)
        if pattern is not None and not pattern.match(value):
            raise ValueError(f"{key} code {value!r} does not match {pattern.pattern}")
        return value

    def describe(self) -> str:
        """One-line lineage for audit logs and CLI output."""
        parts = [
            f"场景 {self.scene}",
            f"数据 {self.dataset}",
            f"合成 {self.synth or '(未合成)'}",
            f"基座 {self.base_model}",
            f"方法 {self.method}",
        ]
        return " | ".join(parts)

    def as_snapshot(self) -> dict[str, str | None]:
        return self.model_dump()

    @classmethod
    def reconstruct(cls, snapshot: Mapping[str, object]) -> Self:
        """Rebuild a lineage from a persisted snapshot, validating shapes."""
        return cls.model_validate(dict(snapshot))


class QualityReport(BaseModel):
    """Data quality report (需求方案.txt section 5.3)."""

    model_config = {"extra": "forbid"}

    score: float = Field(ge=0.0, le=100.0, description="Composite quality score")
    sample_count: int = Field(ge=0)
    missing_rate: float = Field(ge=0.0, le=1.0)
    black_white_ratio: float = Field(gt=0.0, description="black:white ratio, excludes gray")
    label_distribution: dict[str, int] = Field(
        description="Row counts per label. Gray must be reported explicitly.",
    )
    anomalies: list[str] = Field(default_factory=list)
    masked_fields: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class SplitSummary(BaseModel):
    """Result of the fixed 7:1.5:1.5 split."""

    model_config = {"extra": "forbid"}

    train_rows: int = Field(ge=0)
    valid_rows: int = Field(ge=0)
    test_rows: int = Field(ge=0)
    #: Hard invariant, asserted at split time. Synthetic rows never reach test.
    test_contains_synth: bool = Field(default=False)
    label_distribution: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-split label counts, so gray coverage is auditable.",
    )
