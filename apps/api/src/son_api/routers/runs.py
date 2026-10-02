"""Run API: the seven wizard steps as backend actions (改造开发方案.md 10).

This is where the frontend stops being the source of truth. Every step is a
route that produces an artifact and only then advances the Run's state; the
browser re-reads the Run and renders that. A click cannot mark a step done,
because the transition requires the asset id it stands for.

Route map (改造开发方案.md 26):

    POST  /api/projects                     GET /api/projects
    POST  /api/projects/{id}/runs           GET /api/runs/{run_id}
    PATCH /api/runs/{id}/scene
    POST  /api/runs/{id}/dataset
    POST  /api/runs/{id}/prepare-data
    POST  /api/runs/{id}/synth
    PATCH /api/runs/{id}/model
    PATCH /api/runs/{id}/training-config
    POST  /api/runs/{id}/start
    GET   /api/jobs/{job_id}                POST /api/jobs/{job_id}/cancel

Steps 1-4 run inline (seconds); step 7 enqueues a job the worker executes,
because training is minutes to hours and an HTTP request must not block on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from son_db import Database, now_iso
from son_db.assets import (
    log_path,
    sub_dir,
)
from son_db.models import (
    DatasetRow,
    DeploymentRow,
    EvaluationRow,
    JobRow,
    ModelVersionRow,
    ProjectRow,
    RunRow,
    SplitRow,
    SynthRow,
)
from son_db.repositories import dump, load, record_audit
from son_orchestrator.runstate import (
    IllegalTransition,
    MissingEvidence,
    RunState,
    available_actions,
    current_step,
    transition,
)
from son_orchestrator.runstate import (
    parse as parse_state,
)
from son_worker.prepare import AssetError, generate_synth, ingest_csv, prepare_split
from son_worker.queue import JobType, enqueue
from son_worker.queue import serialize as serialize_job

from son_contracts import JEV_OUTPUT_SCHEMA, REASON_MAX_LENGTH, SynthMethod

router = APIRouter(tags=["runs"])

#: Models this stage genuinely supports. 改造开发方案.md 30: Qwen2.5-1.5B first.
SUPPORTED_MODELS: tuple[str, ...] = ("qwen2.5-1.5b-instruct",)
SUPPORTED_METHODS: tuple[str, ...] = ("lora",)
SUPPORTED_QUANT: tuple[str, ...] = ("Q4_K_M",)


def _db() -> Database:
    """The application-scoped database.

    Set by ``create_app`` at startup; a module-level default keeps tests able to
    construct the app without a lifespan.
    """
    database = _DB_HOLDER.get("db")
    if database is None:
        database = Database.open()
        _DB_HOLDER["db"] = database
    return database


_DB_HOLDER: dict[str, Database] = {}


def configure(database: Database) -> None:
    _DB_HOLDER["db"] = database


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    deleted_at: str | None = None


class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class RunOut(BaseModel):
    id: str
    project_id: str
    state: str
    mode: str
    current_step: int
    scene_code: str | None
    base_model_id: str | None
    dataset_id: str | None
    split_id: str | None
    synth_id: str | None
    job_id: str | None
    model_version_id: str | None
    evaluation_id: str | None
    deployment_id: str | None
    error_message: str | None
    available_actions: list[str]
    created_at: str
    updated_at: str


class RunDetail(RunOut):
    dataset: dict[str, Any] | None = None
    split: dict[str, Any] | None = None
    synth: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    job: dict[str, Any] | None = None
    model_version: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    deployment: dict[str, Any] | None = None
    lineage: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# Projects and runs
# --------------------------------------------------------------------------


@router.post("/api/projects", response_model=ProjectOut, summary="新建项目")
def create_project(request: CreateProjectRequest) -> ProjectOut:
    database = _db()
    with database.session() as session:
        project = ProjectRow.new(request.name, request.description)
        session.add(project)
        session.flush()
        record_audit(session, "create_project", project.id, project.name)
        return ProjectOut(**_project_fields(project))


@router.get("/api/projects", response_model=list[ProjectOut], summary="项目列表")
def list_projects() -> list[ProjectOut]:
    """Soft-deleted projects are hidden but not destroyed (需求方案.txt principle 5)."""
    database = _db()
    with database.read() as session:
        rows = (
            session.query(ProjectRow)
            .filter(ProjectRow.deleted_at.is_(None))
            .order_by(ProjectRow.created_at.asc())
            .all()
        )
        return [ProjectOut(**_project_fields(row)) for row in rows]


@router.post("/api/projects/{project_id}/runs", response_model=RunDetail, summary="新建 Run")
def create_run(project_id: str) -> RunDetail:
    database = _db()
    with database.session() as session:
        project = session.get(ProjectRow, project_id)
        if project is None or project.deleted_at is not None:
            raise HTTPException(status_code=404, detail="项目不存在")
        run = RunRow(project_id=project_id, state=RunState.CREATED.value)
        session.add(run)
        session.flush()
        record_audit(session, "create_run", run.id, project.name)
        return _run_detail(session, run)


@router.get("/api/runs/{run_id}", response_model=RunDetail, summary="读取 Run（前端唯一真相源）")
def get_run(run_id: str) -> RunDetail:
    database = _db()
    with database.read() as session:
        run = _require_run(session, run_id)
        return _run_detail(session, run)


@router.get("/api/runs", response_model=list[RunOut], summary="Run 列表")
def list_runs(project_id: str | None = None) -> list[RunOut]:
    database = _db()
    with database.read() as session:
        query = session.query(RunRow)
        if project_id:
            query = query.filter(RunRow.project_id == project_id)
        return [_run_out(run) for run in query.order_by(RunRow.created_at.desc()).all()]


# --------------------------------------------------------------------------
# Step 1: scene
# --------------------------------------------------------------------------


class SetSceneRequest(BaseModel):
    scene_code: str = Field(min_length=1, max_length=64)


@router.patch("/api/runs/{run_id}/scene", response_model=RunDetail, summary="第1步 选择场景")
def set_scene(run_id: str, request: SetSceneRequest) -> RunDetail:
    from son_api.routers.platform import SCENE_TEMPLATES

    if not any(t.id == request.scene_code for t in SCENE_TEMPLATES):
        raise HTTPException(status_code=422, detail=f"未找到场景 {request.scene_code}")

    database = _db()
    with database.session() as session:
        run = _require_run(session, run_id)
        run.scene_code = request.scene_code
        session.flush()
        _advance(session, run, RunState.SCENE_READY)
        return _run_detail(session, run)


# --------------------------------------------------------------------------
# Step 2: dataset (upload once)
# --------------------------------------------------------------------------


@router.post("/api/runs/{run_id}/dataset", response_model=RunDetail, summary="第2步 上传种子数据")
async def upload_dataset(run_id: str, upload: UploadFile = File(...)) -> RunDetail:
    """Store the upload. Everything downstream references ``dataset_id``.

    The file is copied into the run's own tree rather than referenced in place,
    so a run stays reproducible after the upload is deleted.
    """
    database = _db()
    with database.read() as session:
        run = _require_run(session, run_id)
        if run.state not in (RunState.SCENE_READY.value, RunState.DATASET_READY.value):
            raise HTTPException(
                status_code=409, detail=f"当前状态 {run.state} 不能上传数据，请先选择场景"
            )

    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=422, detail="上传文件为空")

    target = sub_dir(run_id, "dataset") / "source.csv"
    target.write_bytes(payload)

    try:
        prepared = ingest_csv(target, upload.filename or "upload.csv")
    except AssetError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with database.session() as session:
        run = _require_run(session, run_id)
        existing = session.query(DatasetRow).filter(DatasetRow.run_id == run_id).one_or_none()
        if existing is not None:
            # Re-upload replaces the record rather than accumulating them: two
            # datasets for one run is never what the user meant.
            session.delete(existing)
            session.flush()

        from son_db.repositories import next_dataset_code

        dataset = DatasetRow(
            run_id=run_id,
            code=next_dataset_code(session, run_id),
            original_filename=upload.filename or "upload.csv",
            source_path=str(prepared.path),
            rows=prepared.rows,
            cols=prepared.cols,
            label_column=prepared.label_column,
            label_mapping_json=dump(prepared.label_mapping),
            checksum=prepared.checksum,
            status="READY",
        )
        session.add(dataset)
        session.flush()

        run.dataset_id = dataset.id
        record_audit(session, "upload_dataset", dataset.id, f"{dataset.rows} 行")
        _advance(session, run, RunState.DATASET_READY)
        return _run_detail(session, run)


# --------------------------------------------------------------------------
# Step 3: prepare (quality + split)
# --------------------------------------------------------------------------


@router.post(
    "/api/runs/{run_id}/prepare-data", response_model=RunDetail, summary="第3步 数据治理与划分"
)
def prepare(run_id: str) -> RunDetail:
    """Quality, masking and the 7:1.5:1.5 split, in one backend action.

    One call rather than three: the old UI let the user re-enter the upload
    screen here, and the state machine has no edge for "quality done but not
    split".
    """
    database = _db()
    with database.read() as session:
        run = _require_run(session, run_id)
        if not run.dataset_id:
            raise HTTPException(status_code=409, detail="尚未上传数据")
        dataset_path = Path(_dataset_of(session, run).source_path)

    try:
        result = prepare_split(dataset_path, run_id, dataset_dir=sub_dir(run_id, "dataset"))
    except AssetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if result.test_synth_rows:
        raise HTTPException(
            status_code=500,
            detail=f"划分结果中有 {result.test_synth_rows} 条合成数据，违反硬性约束",
        )

    with database.session() as session:
        run = _require_run(session, run_id)
        from son_db.repositories import next_split_code

        split = SplitRow(
            dataset_id=run.dataset_id,
            run_id=run_id,
            code=next_split_code(session, run_id),
            train_path=str(result.train_path),
            valid_path=str(result.valid_path),
            test_path=str(result.test_path),
            train_rows=result.train_rows,
            valid_rows=result.valid_rows,
            test_rows=result.test_rows,
            test_synth_rows=result.test_synth_rows,
            quality_json=dump(result.quality),
        )
        session.add(split)
        session.flush()

        run.split_id = split.id
        record_audit(session, "prepare_data", split.id, f"{split.train_rows}/{split.test_rows}")
        _advance(session, run, RunState.DATA_QUALITY_READY)
        return _run_detail(session, run)


# --------------------------------------------------------------------------
# Step 4: synth
# --------------------------------------------------------------------------


class SynthRequest(BaseModel):
    method: SynthMethod = SynthMethod.DISTRIBUTION_FIT
    target_rows: int | None = Field(default=None, ge=1, le=200_000)
    seed: int = 42


@router.post("/api/runs/{run_id}/synth", response_model=RunDetail, summary="第4步 数据合成")
def synth(run_id: str, request: SynthRequest) -> RunDetail:
    """Generate ``syn_v1.csv`` from the training split.

    Training input becomes train + syn; test stays seed-only for ever
    (需求方案.txt 5.4).
    """
    database = _db()
    with database.read() as session:
        run = _require_run(session, run_id)
        if not run.split_id:
            raise HTTPException(status_code=409, detail="请先完成数据治理与划分")
        split = session.get(SplitRow, run.split_id)
        assert split is not None
        train_path = Path(split.train_path)

    try:
        result = generate_synth(
            train_path,
            output_path=sub_dir(run_id, "synth") / "syn_v1.csv",
            method=request.method,
            target_rows=request.target_rows,
            seed=request.seed,
        )
    except AssetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with database.session() as session:
        run = _require_run(session, run_id)
        from son_db.repositories import next_synth_code

        row = SynthRow(
            run_id=run_id,
            dataset_id=run.dataset_id,
            code=next_synth_code(session, run_id),
            method=request.method.value,
            config_json=dump({"target_rows": request.target_rows, "seed": request.seed}),
            output_path=str(result.path),
            rows=result.rows,
            checksum=result.checksum,
            fidelity_score=result.fidelity_score,
            privacy_json=dump(result.privacy),
            status="READY",
        )
        session.add(row)
        session.flush()

        run.synth_id = row.id
        record_audit(session, "synth", row.id, f"{row.rows} 行")
        _advance(session, run, RunState.TRAIN_SET_READY)
        return _run_detail(session, run)


# --------------------------------------------------------------------------
# Step 5: model
# --------------------------------------------------------------------------


class SetModelRequest(BaseModel):
    model_id: str


@router.patch("/api/runs/{run_id}/model", response_model=RunDetail, summary="第5步 选择基座模型")
def set_model(run_id: str, request: SetModelRequest) -> RunDetail:
    """Only models this stage actually wires up are selectable (改造开发方案.md 30)."""
    if request.model_id not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{request.model_id} 本阶段未打通；当前可用：{list(SUPPORTED_MODELS)}。"
                "其余型号仅保留注册信息。"
            ),
        )

    database = _db()
    with database.session() as session:
        run = _require_run(session, run_id)
        run.base_model_id = request.model_id
        session.flush()
        _advance(session, run, RunState.MODEL_SELECTED)
        return _run_detail(session, run)


# --------------------------------------------------------------------------
# Step 6: training config
# --------------------------------------------------------------------------


class TrainingConfigRequest(BaseModel):
    method: str = "lora"
    max_steps: int | None = Field(default=None, ge=1, le=100_000)
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    qlora: bool = False


@router.patch(
    "/api/runs/{run_id}/training-config", response_model=RunDetail, summary="第6步 训练配置"
)
def set_training_config(run_id: str, request: TrainingConfigRequest) -> RunDetail:
    """LoRA SFT only this stage (改造开发方案.md 30)."""
    if request.method not in SUPPORTED_METHODS:
        raise HTTPException(
            status_code=422,
            detail=(f"{request.method} 本阶段未打通；当前可用：{list(SUPPORTED_METHODS)}"),
        )
    if request.qlora:
        raise HTTPException(
            status_code=422,
            detail="低显存快速适配属于后续阶段；本阶段仅支持标准快速风格适配",
        )

    database = _db()
    with database.session() as session:
        run = _require_run(session, run_id)
        config: dict[str, Any] = {"method": request.method}
        if request.max_steps:
            config["max_steps"] = request.max_steps
        if request.learning_rate:
            config["learning_rate"] = request.learning_rate
        run.training_config_json = dump(config)
        session.flush()
        _advance(session, run, RunState.TRAINING_CONFIGURED)
        return _run_detail(session, run)


# --------------------------------------------------------------------------
# Step 7: start
# --------------------------------------------------------------------------


@router.post("/api/runs/{run_id}/start", response_model=RunDetail, summary="第7步 提交训练任务")
def start(run_id: str) -> RunDetail:
    """Create the PIPELINE job. This route never trains anything itself.

    Training is minutes to hours; doing it inside a request would time out and
    report nothing. The worker picks the job up (改造开发方案.md 11).
    """
    database = _db()
    with database.session() as session:
        run = _require_run(session, run_id)
        if not all([run.split_id, run.synth_id, run.base_model_id, run.training_config_json]):
            raise HTTPException(
                status_code=409,
                detail="配置不完整：需要完成划分、合成、模型选择与训练配置",
            )
        split = session.get(SplitRow, run.split_id)
        synth = session.get(SynthRow, run.synth_id)
        assert split is not None and synth is not None

        payload = {
            "base_model": run.base_model_id,
            "train_path": split.train_path,
            "synth_path": synth.output_path,
            "config": load(run.training_config_json) or {},
        }
        job = enqueue(
            session,
            run_id,
            JobType.PIPELINE,
            payload=payload,
            log_path=str(log_path(run_id, "training", f"{run_id}.log")),
        )
        run.job_id = job.id
        session.flush()
        _advance(session, run, RunState.QUEUED, require_evidence=False)
        record_audit(session, "start_training", job.id, run_id)
        return _run_detail(session, run)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


@router.get("/api/jobs/{job_id}", summary="任务进度（前端每 1-2 秒轮询）")
def job_status(job_id: str) -> dict[str, Any]:
    """The polling endpoint from 改造开发方案.md 16. No WebSocket yet."""
    database = _db()
    with database.read() as session:
        job = session.get(JobRow, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return serialize_job(job)


@router.post("/api/jobs/{job_id}/cancel", summary="取消任务")
def cancel_job(job_id: str) -> dict[str, Any]:
    from son_worker.queue import IllegalJobTransition, cancel

    database = _db()
    with database.session() as session:
        try:
            job = cancel(session, job_id)
        except IllegalJobTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return serialize_job(job)


# --------------------------------------------------------------------------
# Serialization and state advance
# --------------------------------------------------------------------------


def _require_run(session: Any, run_id: str) -> RunRow:
    run = session.get(RunRow, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run 不存在")
    return run


def _dataset_of(session: Any, run: RunRow) -> DatasetRow:
    dataset = session.get(DatasetRow, run.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return dataset


def _advance(session: Any, run: RunRow, target: RunState, *, require_evidence: bool = True) -> None:
    """Move the Run forward, or fail loudly.

    Every step route calls this after writing its asset, so the evidence check
    is redundant by construction -- and that redundancy is the point. A route
    that forgets to attach an asset fails here instead of producing a Run that
    claims a dataset it does not have.
    """
    current = parse_state(run.state)
    try:
        transition(current, target, run if require_evidence else None)
    except MissingEvidence as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    run.state = target.value
    run.updated_at = now_iso()


def _run_out(run: RunRow) -> RunOut:
    state = parse_state(run.state)
    return RunOut(
        id=run.id,
        project_id=run.project_id,
        state=run.state,
        mode=run.mode,
        current_step=current_step(state),
        scene_code=run.scene_code,
        base_model_id=run.base_model_id,
        dataset_id=run.dataset_id,
        split_id=run.split_id,
        synth_id=run.synth_id,
        job_id=run.job_id,
        model_version_id=run.model_version_id,
        evaluation_id=run.evaluation_id,
        deployment_id=run.deployment_id,
        error_message=run.error_message,
        available_actions=list(available_actions(state)),
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _run_detail(session: Any, run: RunRow) -> RunDetail:
    """Everything the wizard renders, derived from the run's own records."""
    detail = _run_out(run).model_dump()

    dataset = session.get(DatasetRow, run.dataset_id) if run.dataset_id else None
    split = session.get(SplitRow, run.split_id) if run.split_id else None
    synth = session.get(SynthRow, run.synth_id) if run.synth_id else None
    job = session.get(JobRow, run.job_id) if run.job_id else None
    version = session.get(ModelVersionRow, run.model_version_id) if run.model_version_id else None
    evaluation = session.get(EvaluationRow, run.evaluation_id) if run.evaluation_id else None
    deployment = session.get(DeploymentRow, run.deployment_id) if run.deployment_id else None

    detail["dataset"] = (
        {
            "id": dataset.id,
            "code": dataset.code,
            "original_filename": dataset.original_filename,
            "rows": dataset.rows,
            "cols": dataset.cols,
            "label_column": dataset.label_column,
            "checksum": dataset.checksum,
        }
        if dataset
        else None
    )
    detail["split"] = (
        {
            "id": split.id,
            "code": split.code,
            "train_rows": split.train_rows,
            "valid_rows": split.valid_rows,
            "test_rows": split.test_rows,
            "test_synth_rows": split.test_synth_rows,
            "quality": load(split.quality_json),
        }
        if split
        else None
    )
    detail["synth"] = (
        {
            "id": synth.id,
            "code": synth.code,
            "method": synth.method,
            "rows": synth.rows,
            "fidelity_score": synth.fidelity_score,
            "privacy": load(synth.privacy_json),
        }
        if synth
        else None
    )
    detail["training_config"] = load(run.training_config_json)
    detail["job"] = serialize_job(job) if job else None
    detail["model_version"] = (
        {
            "id": version.id,
            "code": version.code,
            "base_model_id": version.base_model_id,
            "training_method": version.training_method,
            "adapter_path": version.adapter_path,
            "merged_model_path": version.merged_model_path,
        }
        if version
        else None
    )
    detail["evaluation"] = (
        {
            "id": evaluation.id,
            "metrics": load(evaluation.metrics_json),
            "performance": load(evaluation.performance_json),
            "predictions_path": evaluation.predictions_path,
        }
        if evaluation
        else None
    )
    detail["deployment"] = (
        {
            "id": deployment.id,
            "runtime": deployment.runtime,
            "port": deployment.port,
            "status": deployment.status,
            "pid": deployment.pid,
        }
        if deployment
        else None
    )
    detail["lineage"] = build_lineage(session, run)
    return RunDetail(**detail)


def build_lineage(session: Any, run: RunRow) -> dict[str, Any] | None:
    """Real provenance, assembled from the run's own records.

    改造开发方案.md 18: never client-supplied. Every id here was written by a
    route that produced the corresponding artifact.
    """
    if run.state == RunState.CREATED.value:
        return None

    def _code(model: Any, row_id: str | None) -> str | None:
        if not row_id:
            return None
        row = session.get(model, row_id)
        return str(row.code) if row is not None else None

    lineage = {
        "run": run.id,
        "scene": run.scene_code,
        "dataset": _code(DatasetRow, run.dataset_id),
        "split": _code(SplitRow, run.split_id),
        "synth": _code(SynthRow, run.synth_id),
        "base_model": run.base_model_id,
        "job": run.job_id,
        "model_version": _code(ModelVersionRow, run.model_version_id),
    }
    return {k: v for k, v in lineage.items() if v}


def _project_fields(project: ProjectRow) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at,
        "deleted_at": project.deleted_at,
    }


@router.get("/api/runtime/capabilities", summary="本阶段真正支持的能力")
def capabilities() -> dict[str, Any]:
    """What this build genuinely does, for the UI to grey out everything else.

    改造开发方案.md 31: never show a choice that does not work.
    """
    return {
        "scene": "fraud_account",
        "upload": "csv",
        "labels": ["black", "white", "gray"],
        "synth_method": SynthMethod.DISTRIBUTION_FIT.value,
        "base_models": list(SUPPORTED_MODELS),
        "training_methods": list(SUPPORTED_METHODS),
        "quant": list(SUPPORTED_QUANT),
        "runtime": "llama_cpp",
        "jev_level": "L1",
        "output_schema": JEV_OUTPUT_SCHEMA,
        "reason_max_length": REASON_MAX_LENGTH,
        "unsupported": {
            "note": "以下能力仅保留注册信息，界面应隐藏或灰掉",
            "base_models": ["gemma-2-2b-it", "llama-3.2-1b-instruct", "qwen2.5-3b-instruct"],
            "training_methods": ["dpo", "qlora", "qat", "distillation", "fewshot"],
            "quant": ["Q5_K_M", "Q8_0"],
            "runtimes": ["ollama", "vllm", "file_export"],
            "jev_levels": ["L2", "L3"],
        },
    }
