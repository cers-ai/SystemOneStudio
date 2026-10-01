"""Seed data for the base model registry.

Scope is 需求方案.txt 10.2: the four families the MVP supports, all within the
1.5B-3B band (plus the 1B Llama and the sub-1B in-house classifier, both of
which the requirement lists explicitly). 7B variants from 需求方案.txt 7.2 are
registered with ``in_mvp_scope=False`` rather than omitted, so the shelf can
show them as unavailable instead of pretending they do not exist.

JEV levels: 需求方案.txt 7.2 *claims* a level per family, but per 6.2 only L3
has a defined verification method and it has not been run here. So every entry
carries ``jev_level_evidence="declared"`` and no entry claims L3. Do not flip
an entry to L3 or to ``"measured"`` without attaching the comparison report.
"""

from son_contracts import ArtifactFormat, JevCompatLevel
from son_model_registry.adapter import RegistryModel

MCP = ArtifactFormat


def _base(
    model_id: str,
    display_name: str,
    family: str,
    params: str,
    license_: str,
    min_gpu_memory: str,
    adapter_type: str,
    *,
    tasks: list[str],
    jev_level: JevCompatLevel | None,
    in_mvp_scope: bool = True,
    supported_tasks_extra: list[str] | None = None,
) -> RegistryModel:
    return RegistryModel(
        model_id=model_id,
        display_name=display_name,
        family=family,
        params=params,
        license=license_,
        min_gpu_memory=min_gpu_memory,
        adapter_type=adapter_type,
        supported_tasks=tasks + (supported_tasks_extra or []),
        jev_compatible=jev_level is not None,
        jev_compat_level=jev_level,
        jev_level_evidence="declared" if jev_level is not None else None,
        available_formats=[MCP.GGUF, MCP.NATIVE],
        in_mvp_scope=in_mvp_scope,
    )


DECISION_TASKS = ["classification", "scoring"]

REGISTRY_SEED: tuple[RegistryModel, ...] = (
    _base(
        "qwen2.5-1.5b-instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen",
        "1.5B",
        "Apache-2.0",
        "8GB",
        "qwen_adapter",
        tasks=DECISION_TASKS,
        supported_tasks_extra=["instruction"],
        jev_level=None,
    ),
    _base(
        "qwen2.5-3b-instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen",
        "3B",
        "Apache-2.0",
        "12GB",
        "qwen_adapter",
        tasks=DECISION_TASKS,
        supported_tasks_extra=["instruction"],
        jev_level=None,
    ),
    _base(
        "gemma-2-2b-it",
        "Gemma 2 2B-it",
        "Gemma",
        "2B",
        "Gemma Terms of Use",
        "8GB",
        "gemma_adapter",
        tasks=DECISION_TASKS,
        supported_tasks_extra=["instruction"],
        jev_level=JevCompatLevel.L1,
    ),
    _base(
        "llama-3.2-1b-instruct",
        "Llama 3.2 1B-Instruct",
        "Llama",
        "1B",
        "Llama 3.2 Community License",
        "4GB",
        "llama_adapter",
        tasks=DECISION_TASKS,
        supported_tasks_extra=["instruction"],
        jev_level=JevCompatLevel.L1,
    ),
    _base(
        "llama-3.2-3b-instruct",
        "Llama 3.2 3B-Instruct",
        "Llama",
        "3B",
        "Llama 3.2 Community License",
        "12GB",
        "llama_adapter",
        tasks=DECISION_TASKS,
        supported_tasks_extra=["instruction"],
        jev_level=JevCompatLevel.L1,
    ),
    _base(
        "son-tabular-classifier-lt1b",
        "自研轻量分类器 <1B",
        "自研",
        "<1B",
        "Proprietary",
        "4GB",
        "tabular_classifier_adapter",
        tasks=["classification"],
        jev_level=JevCompatLevel.L1,
    ),
    # --- Outside MVP scope: shown on the shelf but not selectable ---
    _base(
        "qwen2.5-7b-instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen",
        "7B",
        "Apache-2.0",
        "24GB",
        "qwen_adapter",
        tasks=DECISION_TASKS,
        supported_tasks_extra=["instruction"],
        jev_level=None,
        in_mvp_scope=False,
    ),
    _base(
        "gemma-2-9b-it",
        "Gemma 2 9B-it",
        "Gemma",
        "9B",
        "Gemma Terms of Use",
        "24GB",
        "gemma_adapter",
        tasks=DECISION_TASKS,
        supported_tasks_extra=["instruction"],
        jev_level=JevCompatLevel.L1,
        in_mvp_scope=False,
    ),
)


def mvp_models() -> list[RegistryModel]:
    return [m for m in REGISTRY_SEED if m.in_mvp_scope]
