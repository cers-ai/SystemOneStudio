"""Model shelf, artifact registry, deployment and evaluation endpoints (M5).

Covers 需求方案.txt 5.7.1 (产物与版本关联), 5.7.4 (一键部署) and 9 (评测报告).

Two things this module deliberately does *not* do:

* start a real llama.cpp server -- it returns the launch plan and the deployment
  summary. There is no GPU here and no llama.cpp, so a route that claimed to have
  started a server would be lying. See AGENTS.md.
* report a latency number. The evaluator computes percentiles from samples; with
  no samples the performance block comes back unmeasured, and Q1 additionally
  leaves the P95 target unsettled (技术方案.md 1.1).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from son_evaluator import (
    EffectReport,
    LatencySample,
    PerformanceReport,
    build_effect_report,
    build_performance_report,
    feature_importance,
)
from son_inference import ServerConfig, deployment_summary
from son_jev import assert_level_claimable

from son_contracts import (
    ArtifactFormat,
    Decision,
    JevCompatLevel,
    JevFlags,
    Lineage,
    QuantLevel,
    RunState,
    TrainingMethod,
)
from son_model_registry import REGISTRY_SEED, RegistryModel, mvp_models

router = APIRouter(tags=["models"])


# --------------------------------------------------------------------------
# Base model shelf (需求方案.txt 5.5)
# --------------------------------------------------------------------------


class ModelCard(BaseModel):
    """One shelf entry, in the shape 5.5's mockup shows."""

    model_id: str
    display_name: str
    family: str
    params: str
    min_gpu_memory: str
    jev_compat_level: JevCompatLevel | None
    license: str
    in_mvp_scope: bool
    adapter_type: str


class ModelShelfEntry(ModelCard):
    """A shelf entry plus the provenance of every number on it.

    ``vram_source`` and ``duration_source`` exist because the shelf is the one
    screen where a plausible-looking hardcoded number would read as a
    measurement. The UI renders the source next to the figure.
    """

    vram_source: str = "registry_estimate"
    duration_source: str = "unmeasured"


class ModelShelfResponse(BaseModel):
    recommended: list[ModelCard]
    all: list[ModelCard]
    notes: list[str]


def _card(model: RegistryModel) -> ModelCard:
    return ModelCard(
        model_id=model.model_id,
        display_name=model.display_name,
        family=model.family,
        params=model.params,
        min_gpu_memory=model.min_gpu_memory,
        jev_compat_level=model.jev_compat_level,
        license=model.license,
        in_mvp_scope=model.in_mvp_scope,
        adapter_type=model.adapter_type,
    )


@router.get("/api/models", response_model=ModelShelfResponse, summary="底座模型货架")
def list_models() -> ModelShelfResponse:
    """Top-3 in-scope recommendations plus the full shelf.

    Out-of-scope entries are listed but flagged, so the shelf shows a 7B model as
    unavailable instead of pretending it does not exist. The MVP cap at 1.5B-3B
    is a deliberate answer to limited VRAM (需求方案.txt 13), not an omission.

    ``min_gpu_memory`` comes from the registry and is an *estimate*, not a
    measurement; no duration is produced at all because none has been measured.
    The UI is expected to show that distinction rather than rendering the number
    as if it were measured.
    """
    in_scope = mvp_models()
    return ModelShelfResponse(
        recommended=[_card(m) for m in in_scope[:3]],
        all=[_card(m) for m in REGISTRY_SEED],
        notes=[
            "显存需求为注册表中的估算值，未在 GPU 节点实测",
            "训练时长暂无数据：本机无 GPU，任何时长都未经测量",
            "JEV 兼容等级以实测为准：仓库内的等级为需求文档声明值",
        ],
    )


@router.get("/api/models/{model_id}/plan", summary="训练方案与耗时估算")
def training_plan(
    model_id: str,
    rows: int = 10000,
    qlora: bool = False,
) -> dict[str, object]:
    """Computed hyperparameters plus a duration estimate, clearly labelled.

    需求方案.txt 5.6 says the advanced values are recommended rather than typed,
    so they have to be *computed* rather than hardcoded in the UI. Every figure
    here is an estimate; ``measured`` is false and ``basis`` says why.
    """
    model = next((m for m in REGISTRY_SEED if m.model_id == model_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail=f"未找到底座模型 {model_id}")

    try:
        params = float(model.params.rstrip("B"))
    except ValueError:
        params = 3.0

    from son_trainer import DatasetProfile, estimate_duration_minutes, recommend_hyperparams

    profile = DatasetProfile(rows=max(rows, 1), params_b=params, quantized=qlora)
    hyperparams = recommend_hyperparams(profile, (TrainingMethod.SFT, TrainingMethod.DPO))

    return {
        "model_id": model_id,
        "hyperparams": hyperparams,
        "estimated_minutes": estimate_duration_minutes(profile, hyperparams),
        # Explicit, so no caller can mistake this for a measurement.
        "measured": False,
        "basis": ("按数据量与模型规格推算的吞吐估算；尚未在 GPU 节点实测，实际时长可能显著不同"),
        "vram_estimate": model.min_gpu_memory,
        "vram_measured": False,
    }


@router.get("/api/models/{model_id}/adapter", summary="适配器可用性")
def adapter_status(model_id: str) -> dict[str, object]:
    """Whether the adapter for a model is registered.

    No adapter is registered yet: the real ones need torch/PEFT and a GPU. This
    reports that honestly instead of returning a placeholder.
    """
    model = next((m for m in REGISTRY_SEED if m.model_id == model_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail=f"未找到底座模型 {model_id}")

    registered = model.adapter_type in _REGISTERED_ADAPTER_KEYS
    return {
        "model_id": model_id,
        "adapter_type": model.adapter_type,
        "registered": registered,
        "note": (
            "适配器已注册"
            if registered
            else "适配器尚未实现；训练侧实现 TrainingBackend 与适配器后接入"
        ),
    }


_REGISTERED_ADAPTER_KEYS: set[str] = set()


# --------------------------------------------------------------------------
# Artifacts and lineage (需求方案.txt 5.7.1)
# --------------------------------------------------------------------------


class ArtifactRecord(BaseModel):
    format: ArtifactFormat
    quant: QuantLevel | None = None
    size_bytes: int | None = None
    is_recommended: bool = False


class ModelVersionResponse(BaseModel):
    model_version: str
    lineage: Lineage
    artifacts: list[ArtifactRecord]
    created_at: datetime
    notes: list[str]


@router.post("/api/model-versions", response_model=ModelVersionResponse, summary="注册模型版本")
def register_model_version(
    lineage: Lineage,
    artifacts: list[ArtifactRecord],
) -> ModelVersionResponse:
    """Register a trained model version and its artifacts.

    Lineage codes are validated by the ``Lineage`` model, so a malformed
    ``sc_``/``ds_``/``mv_`` code is a 422 rather than a broken artifact later.
    """
    notes: list[str] = []
    levels = {a.quant for a in artifacts if a.quant is not None}
    if levels and QuantLevel.Q4_K_M not in levels:
        notes.append("未产出标准压缩（推荐）档位，评测结果无法与 JEV 基线对比")
    if not any(a.format is ArtifactFormat.NATIVE for a in artifacts):
        notes.append("未登记原生权重，仅有压缩产物")

    return ModelVersionResponse(
        model_version=lineage.model_version,
        lineage=lineage,
        artifacts=artifacts,
        created_at=datetime.now(UTC),
        notes=notes,
    )


# --------------------------------------------------------------------------
# Evaluation report (需求方案.txt 9)
# --------------------------------------------------------------------------


class EvaluationRequest(BaseModel):
    """Evaluation input.

    ``format_compliant`` / ``format_total`` are explicit because 格式合规率 is a
    tracked product indicator (需求方案.txt 14.2). They cannot be inferred from
    the request: a caller supplying predictions says nothing about whether the
    model's raw output parsed. Omit them and the metric reports 未测量.
    """

    y_true: list[Decision]
    y_pred: list[Decision]
    scores: list[float] | None = None
    #: Raw model outputs that parsed against the JEV schema. Leave None unless
    #: measured from an actual inference run.
    format_compliant: int | None = Field(default=None, ge=0)
    format_total: int | None = Field(default=None, ge=0)
    contributions: dict[str, float] = Field(default_factory=dict)
    latency_samples: list[LatencySample] = Field(default_factory=list)
    quant_level: QuantLevel = QuantLevel.Q4_K_M
    concurrency: int = 1
    wall_clock_s: float | None = None
    peak_memory_mb: float | None = None


class EvaluationResponse(BaseModel):
    performance: dict[str, object]
    effect: dict[str, object]
    top_factors: list[tuple[str, float]]
    notes: list[str]


def _build_effect(request: EvaluationRequest) -> EffectReport:
    """Build the effect block, turning a data mismatch into a 422.

    The evaluator raises ``ValueError`` on a length mismatch; letting that escape
    would surface as a 500, which tells the caller nothing about what to fix.

    Format compliance is passed through as supplied. It used to be computed as
    ``len(y_true)/len(y_true)``, which certified a hard product metric at 100%
    for every request regardless of what the model actually emitted.
    """
    try:
        return build_effect_report(
            tuple(request.y_true),
            tuple(request.y_pred),
            scores=tuple(request.scores) if request.scores is not None else None,
            format_compliant=request.format_compliant,
            format_total=request.format_total,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _build_performance(request: EvaluationRequest) -> PerformanceReport:
    """Build the performance block.

    ``LatencySample`` is a plain dataclass, not a pydantic model, so it has no
    ``model_dump``; ``dataclasses.asdict`` is the right conversion.
    """
    samples = [LatencySample(**asdict(sample)) for sample in request.latency_samples]
    return build_performance_report(
        samples,
        quant_level=request.quant_level.value,
        concurrent=request.concurrency,
        wall_clock_s=request.wall_clock_s,
        peak_memory_mb=request.peak_memory_mb,
    )


@router.post("/api/evaluations", response_model=EvaluationResponse, summary="评测报告")
def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    """Compute the report, performance block first (需求方案.txt 9.1/9.2).

    The two blocks are returned separately and never merged: the priority order
    is an architectural input, not a reporting preference.
    """
    effect = _build_effect(request)
    performance = _build_performance(request)

    notes = [*performance.notes, *effect.notes]
    if not request.latency_samples:
        notes.append("性能指标未测量：本机无 GPU，无法产生真实延迟样本")

    return EvaluationResponse(
        performance={
            "sample_count": performance.sample_count,
            "p50_ms": performance.p50_ms,
            "p95_ms": performance.p95_ms,
            "p99_ms": performance.p99_ms,
            "ttft_p95_ms": performance.ttft_p95_ms,
            "qps": performance.qps,
            "peak_memory_mb": performance.peak_memory_mb,
            "quant_level": performance.quant_level,
            "targets": performance.targets,
        },
        effect={
            "accuracy": round(effect.accuracy, 4),
            "macro_f1": round(effect.macro_f1, 4),
            "auc_roc": effect.auc_roc,
            "recall": None if effect.recall is None else round(effect.recall, 4),
            "false_kill_rate": None
            if effect.false_kill_rate is None
            else round(effect.false_kill_rate, 4),
            "format_compliance": None
            if effect.format_compliance is None
            else round(effect.format_compliance, 4),
            "per_label": {k: v.as_dict() for k, v in effect.per_label.items()},
            "confusion": effect.confusion,
            "targets": effect.target_check(),
        },
        top_factors=feature_importance(request.contributions),
        notes=notes,
    )


# --------------------------------------------------------------------------
# Deployment (需求方案.txt 5.7.4)
# --------------------------------------------------------------------------


class DeployRequest(BaseModel):
    model_version: str
    artifact_path: str
    quant_level: QuantLevel = QuantLevel.Q4_K_M
    port: int = 8080
    concurrency: int = 8
    gpu_layers: int = 0
    jev_format_compat: bool = True
    jev_training_compat: bool = True


class DeployResponse(BaseModel):
    status: RunState
    api_url: str
    docs_url: str
    curl: str
    launch_plan: list[str]
    jev_format_compat: bool
    notes: list[str]


@router.post("/api/deployments", response_model=DeployResponse, summary="一键部署")
def deploy(request: DeployRequest) -> DeployResponse:
    """Produce the deployment plan and call example.

    Returns ``status=pending`` with the exact launch command rather than claiming
    a service is running. Starting llama.cpp needs a GPU node and a llama.cpp
    checkout; neither exists here, and reporting a running service from this
    process would be false (AGENTS.md).
    """
    config = ServerConfig(
        model_path=request.artifact_path,
        port=request.port,
        gpu_layers=request.gpu_layers,
        concurrency=request.concurrency,
    )
    summary = deployment_summary(config, jev_format_compat=request.jev_format_compat)

    flags = JevFlags(
        jev_format_compat=request.jev_format_compat,
        jev_training_compat=request.jev_training_compat,
    )
    notes = list(summary["notes"])
    if not request.jev_format_compat:
        notes.append("已关闭输出格式对齐：调用方将收到 jev_compatible=false 标记")
    notes.append("本接口只生成启动方案；实际启动需要 GPU 节点上的 llama.cpp")

    return DeployResponse(
        status=RunState.PENDING,
        api_url=summary["api_url"],
        docs_url=summary["docs_url"],
        curl=summary["curl"],
        launch_plan=[" ".join(config.argv())],
        jev_format_compat=flags.jev_format_compat,
        notes=notes,
    )


# --------------------------------------------------------------------------
# JEV compatibility claim (需求方案.txt 6.2)
# --------------------------------------------------------------------------


class JevLevelClaimRequest(BaseModel):
    model_id: str
    level: JevCompatLevel
    evidence: str | None = None


@router.post("/api/jev/level-claim", summary="校验 JEV 兼容等级声明")
def claim_jev_level(request: JevLevelClaimRequest) -> dict[str, object]:
    """Validate a compatibility claim before it is written.

    L3 requires a comparison test; L2 requires the JEV benchmark, which is not in
    this repository. Both are refused here rather than noted in a comment.
    """
    model = next((m for m in REGISTRY_SEED if m.model_id == request.model_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail=f"未找到底座模型 {request.model_id}")

    try:
        assert_level_claimable(request.level, evidence=request.evidence)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"model_id": request.model_id, "level": request.level.value, "accepted": True}
