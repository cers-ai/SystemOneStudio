"""Projects, scenes and platform status (需求方案.txt 10.1 P0).

Covers the three P0 modules that had no surface at all:

* 项目管理 —— 新建、模板、JEV 开关
* 场景定义 —— 字段配置、标签、指标、JEV 对齐
* 系统管理 —— 模型注册表、兼容等级校验、环境状态

PERSISTENCE: an in-process store, not PostgreSQL. SQLAlchemy is a dependency and
the schema is designed in 技术方案.md 3.1, but no ORM models exist yet. This is
stated in the store's docstring and surfaced by ``/api/system/status`` so a
restart losing data is never a surprise. Swapping in the ORM is a change to this
module only.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from son_contracts import Decision, JevFlags, UiMode

router = APIRouter(tags=["platform"])


# --------------------------------------------------------------------------
# Scenes (需求方案.txt 5.1)
# --------------------------------------------------------------------------


class FieldSpec(BaseModel):
    """One field in a scene's schema."""

    name: str
    label: str
    kind: str = Field(default="numeric", pattern="^(numeric|categorical|text|datetime)$")
    required: bool = False
    sensitive: bool = False
    constraint_min: float | None = None
    constraint_max: float | None = None


class SceneTemplate(BaseModel):
    """A predefined scene (需求方案.txt 5.1 card grid)."""

    id: str
    name: str
    summary: str
    recommended: bool = False
    #: The requirement says the fraud scene has 32 fields. The real schema is
    #: outstanding (技术方案.md Q4), so this is a documented placeholder.
    field_count: int
    labels: list[Decision]
    recommended_base_model: str
    recommended_methods: list[str]
    core_metrics: list[str]
    example_dataset: str | None = None


class SceneSummary(BaseModel):
    id: str
    code: str
    name: str
    summary: str
    field_count: int
    labels: list[Decision]
    recommended_base_model: str
    recommended_methods: list[str]
    core_metrics: list[str]
    jev_format_compat: bool
    jev_training_compat: bool
    source: str = Field(description="template | custom")


SCENE_TEMPLATES: tuple[SceneTemplate, ...] = (
    SceneTemplate(
        id="fraud_account",
        name="反诈账户判定",
        summary="从账号属性、交易特征、设备信息判定是否涉诈，含灰样本待定区间",
        recommended=True,
        field_count=32,
        labels=[Decision.BLACK, Decision.WHITE, Decision.GRAY],
        recommended_base_model="qwen2.5-3b-instruct",
        recommended_methods=["lora", "dpo"],
        core_metrics=["false_kill_rate", "recall", "p95_ms"],
        example_dataset="fraud_seed_example.csv",
    ),
    SceneTemplate(
        id="payment_risk",
        name="支付风控",
        summary="基于交易特征与设备指纹判定交易风险",
        field_count=18,
        labels=[Decision.BLACK, Decision.WHITE, Decision.GRAY],
        recommended_base_model="qwen2.5-1.5b-instruct",
        recommended_methods=["lora"],
        core_metrics=["false_kill_rate", "p95_ms"],
    ),
    SceneTemplate(
        id="compliance_review",
        name="合规审核",
        summary="对业务材料做合规性判定",
        field_count=12,
        labels=[Decision.BLACK, Decision.WHITE, Decision.GRAY],
        recommended_base_model="gemma-2-2b-it",
        recommended_methods=["lora"],
        core_metrics=["accuracy", "format_compliance"],
    ),
    SceneTemplate(
        id="marketing_decision",
        name="营销决策",
        summary="用户分层与触达策略判定",
        field_count=14,
        labels=[Decision.BLACK, Decision.WHITE, Decision.GRAY],
        recommended_base_model="qwen2.5-1.5b-instruct",
        recommended_methods=["lora"],
        core_metrics=["accuracy"],
    ),
    SceneTemplate(
        id="credit_approval",
        name="信贷审批",
        summary="基于征信字段判定审批结论",
        field_count=24,
        labels=[Decision.BLACK, Decision.WHITE, Decision.GRAY],
        recommended_base_model="qwen2.5-3b-instruct",
        recommended_methods=["lora", "dpo"],
        core_metrics=["false_kill_rate", "auc_roc"],
    ),
    SceneTemplate(
        id="content_safety",
        name="内容安全",
        summary="对文本内容做安全判定",
        field_count=8,
        labels=[Decision.BLACK, Decision.WHITE, Decision.GRAY],
        recommended_base_model="gemma-2-2b-it",
        recommended_methods=["lora"],
        core_metrics=["accuracy", "format_compliance"],
    ),
)

TEMPLATE_BY_ID = {t.id: t for t in SCENE_TEMPLATES}


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class Store:
    """In-memory store. Lost on restart; see the module docstring.

    Also process-local: with more than one uvicorn worker each has its own copy,
    so a record written through one worker is invisible to another. The container
    therefore runs a single worker until the ORM lands. Both facts are reported by
    ``/api/system/status`` rather than left to be discovered.
    """

    def __init__(self) -> None:
        self.projects: dict[str, dict[str, Any]] = {}
        self.scenes: dict[str, dict[str, Any]] = {}
        self.audit: list[dict[str, Any]] = []

    def next_scene_code(self) -> str:
        return f"sc_v{len(self.scenes) + 1}"

    def next_project_code(self) -> str:
        return f"pr_{len(self.projects) + 1}"

    def record(self, actor: str, action: str, target: str, detail: str) -> dict[str, Any]:
        entry = {
            "id": uuid.uuid4().hex[:12],
            "actor": actor,
            "action": action,
            "target": target,
            "detail": detail,
            "ts": datetime.now(UTC).isoformat(),
        }
        self.audit.insert(0, entry)
        return entry


STORE = Store()


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------


def _to_summary(record: dict[str, Any], template: SceneTemplate | None) -> SceneSummary:
    return SceneSummary(
        id=str(record["id"]),
        code=str(record["code"]),
        name=str(record["name"]),
        summary=str(record["summary"]),
        field_count=len(record.get("fields") or []) or (template.field_count if template else 0),
        labels=[Decision(d) for d in (record.get("labels") or [])],
        recommended_base_model=str(
            record.get("recommended_base_model")
            or (template.recommended_base_model if template else "")
        ),
        recommended_methods=list(
            record.get("recommended_methods") or (template.recommended_methods if template else [])
        ),
        core_metrics=list(
            record.get("core_metrics") or (template.core_metrics if template else [])
        ),
        jev_format_compat=bool(record["jev_format_compat"]),
        jev_training_compat=bool(record["jev_training_compat"]),
        source=str(record["source"]),
    )


@router.get("/api/scenes/templates", response_model=list[SceneTemplate], summary="场景模板列表")
def list_templates() -> list[SceneTemplate]:
    """The gallery source (需求方案.txt 5.1: 六个预置模板)."""
    return list(SCENE_TEMPLATES)


@router.get("/api/scenes", response_model=list[SceneSummary], summary="已创建场景")
def list_scenes() -> list[SceneSummary]:
    return [
        _to_summary(rec, TEMPLATE_BY_ID.get(str(rec.get("template_id", ""))))
        for rec in STORE.scenes.values()
    ]


class CreateSceneRequest(BaseModel):
    template_id: str | None = None
    name: str | None = None
    summary: str = ""
    fields: list[FieldSpec] = Field(default_factory=list)
    labels: list[Decision] = Field(
        default_factory=lambda: [Decision.BLACK, Decision.WHITE, Decision.GRAY]
    )
    jev_format_compat: bool = True
    jev_training_compat: bool = True

    @field_validator("labels")
    @classmethod
    def _must_be_three_way(cls, value: list[Decision]) -> list[Decision]:
        """Three-class is a product constraint, not a default.

        A binary scene would make recall and the gray handling meaningless, so it
        is refused here rather than defaulted silently.
        """
        if len(value) != 3 or len(set(value)) != 3:
            raise ValueError(
                f"标签体系必须是 black / white / gray 三分类，收到 {[v.value for v in value]}"
            )
        return value

    @field_validator("name")
    @classmethod
    def _name_required_when_custom(cls, value: str | None, info: object) -> str | None:
        return value


@router.post("/api/scenes", response_model=SceneSummary, summary="从模板或从零创建场景")
def create_scene(request: CreateSceneRequest) -> SceneSummary:
    """Instantiate a template, or build a scene from scratch.

    Creating from scratch requires a name; instantiating a template inherits its
    recommended model, methods and metrics.
    """
    template = TEMPLATE_BY_ID.get(request.template_id or "")
    if request.template_id and template is None:
        raise HTTPException(status_code=404, detail=f"未找到场景模板 {request.template_id}")

    if template is None and not (request.name or "").strip():
        raise HTTPException(status_code=422, detail="从零创建场景时必须填写场景名称")

    name = request.name or (template.name if template else "")
    scene_id = uuid.uuid4().hex[:12]
    code = STORE.next_scene_code()

    record: dict[str, Any] = {
        "id": scene_id,
        "code": code,
        "name": name,
        "summary": request.summary or (template.summary if template else ""),
        "fields": [f.model_dump() for f in request.fields]
        or (
            [{"name": f"field_{i}", "label": f"字段 {i + 1}"} for i in range(template.field_count)]
            if template
            else []
        ),
        "labels": [d.value for d in request.labels],
        "recommended_base_model": template.recommended_base_model if template else "",
        "recommended_methods": template.recommended_methods if template else [],
        "core_metrics": template.core_metrics if template else [],
        "jev_format_compat": request.jev_format_compat,
        "jev_training_compat": request.jev_training_compat,
        "template_id": request.template_id or "",
        "source": "template" if template else "custom",
        "created_at": datetime.now(UTC).isoformat(),
    }
    STORE.scenes[scene_id] = record
    STORE.record("当前用户", "create_scene", code, f"创建场景「{name}」")

    return _to_summary(record, template)


@router.patch("/api/scenes/{scene_id}/jev", response_model=SceneSummary, summary="更新 JEV 开关")
def update_jev_flags(scene_id: str, flags: JevFlags) -> SceneSummary:
    """The two switches stay independent (需求方案.txt 6.3)."""
    record = STORE.scenes.get(scene_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"未找到场景 {scene_id}")
    record["jev_format_compat"] = flags.jev_format_compat
    record["jev_training_compat"] = flags.jev_training_compat
    STORE.record(
        "当前用户",
        "update_jev_flags",
        str(record["code"]),
        f"格式对齐={flags.jev_format_compat} 训练对齐={flags.jev_training_compat}",
    )
    return _to_summary(record, TEMPLATE_BY_ID.get(str(record.get("template_id", ""))))


# --------------------------------------------------------------------------
# Projects (需求方案.txt 10.1)
# --------------------------------------------------------------------------


class Project(BaseModel):
    id: str
    code: str
    name: str
    mode: UiMode
    scene_code: str | None
    created_at: str
    note: str = ""


class CreateProjectRequest(BaseModel):
    name: str
    scene_id: str | None = None
    mode: UiMode = UiMode.WIZARD


@router.get("/api/projects", response_model=list[Project], summary="项目列表")
def list_projects() -> list[Project]:
    return [Project(**rec) for rec in STORE.projects.values()]


@router.post("/api/projects", response_model=Project, summary="新建项目")
def create_project(request: CreateProjectRequest) -> Project:
    scene_code: str | None = None
    if request.scene_id:
        scene = STORE.scenes.get(request.scene_id)
        if scene is None:
            raise HTTPException(status_code=404, detail=f"未找到场景 {request.scene_id}")
        scene_code = str(scene["code"])

    project = Project(
        id=uuid.uuid4().hex[:12],
        code=STORE.next_project_code(),
        name=request.name,
        mode=request.mode,
        scene_code=scene_code,
        created_at=datetime.now(UTC).isoformat(),
    )
    STORE.projects[project.id] = project.model_dump()
    STORE.record("当前用户", "create_project", project.code, f"新建项目「{request.name}」")
    return project


# --------------------------------------------------------------------------
# System management (需求方案.txt 10.1 系统基础)
# --------------------------------------------------------------------------


class ServiceStatus(BaseModel):
    name: str
    ok: bool
    detail: str


class SystemStatus(BaseModel):
    gpu_available: bool
    gpu_detail: str
    services: list[ServiceStatus]
    persistence: str
    model_count: int
    in_scope_model_count: int
    notes: list[str]


@router.get("/api/system/status", response_model=SystemStatus, summary="平台运行状态")
def system_status() -> SystemStatus:
    """What this deployment can actually do.

    Reports the GPU rather than assuming one. The whole verification story
    depends on whether a number came from a real device.
    """
    from son_inference.llamacpp_client import gpu_report

    from son_model_registry import REGISTRY_SEED, mvp_models

    try:
        gpu = gpu_report()
    except Exception as exc:
        gpu = {"cuda_available": False, "reason": str(exc)}

    has_gpu = bool(gpu.get("cuda_available"))
    gpu_detail = (
        "、".join(f"{d['name']} {d['total_memory_gb']}GB" for d in gpu.get("devices", []))
        if has_gpu
        else str(gpu.get("reason", "未检测到 GPU"))
    )

    services = [
        ServiceStatus(name="控制面 API", ok=True, detail="运行中"),
        ServiceStatus(
            name="训练后端",
            ok=has_gpu,
            detail="已就绪" if has_gpu else "未就绪：需要 NVIDIA GPU 与训练依赖",
        ),
        ServiceStatus(
            name="推理服务",
            ok=has_gpu,
            detail="已就绪" if has_gpu else "未就绪：需要 NVIDIA GPU 与推理运行时",
        ),
    ]

    notes = [
        "数据持久化当前为进程内存：重启即丢失，且容器固定单进程运行"
        "（多进程会导致各进程数据互不可见）。接入数据库后此项会改变。",
    ]
    if not has_gpu:
        notes.append(
            "本节点无 GPU：训练、量化、推理与延迟指标均无法执行。相关能力已实现，"
            "需要在 GPU 节点上运行 deploy/gpu。"
        )

    return SystemStatus(
        gpu_available=has_gpu,
        gpu_detail=gpu_detail,
        services=services,
        persistence="in-memory",
        model_count=len(REGISTRY_SEED),
        in_scope_model_count=len(mvp_models()),
        notes=notes,
    )


class AuditEntry(BaseModel):
    id: str
    actor: str
    action: str
    target: str
    detail: str
    ts: str


@router.get("/api/system/audit", response_model=list[AuditEntry], summary="审计日志")
def audit_log(limit: int = 50) -> list[AuditEntry]:
    """变更记录（需求方案.txt 原则 5 可审计）."""
    return [AuditEntry(**e) for e in STORE.audit[:limit]]
