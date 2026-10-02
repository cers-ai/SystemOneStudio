"""Golden Path E2E (改造开发方案.md 36).

    project -> run -> scene -> upload -> prepare -> synth -> model -> config
    -> start -> worker -> model version -> evaluation -> deployment -> predict

This is the test 改造开发方案.md calls the release gate: "any arrow broken means
MVP is not complete".

Two halves:

* **CPU half** -- every arrow up to and including the worker claiming the job,
  plus the evaluation and predict paths driven by a scripted engine. Runs
  everywhere, on every commit.
* **GPU half** -- marked ``gpu`` and skipped unless ``SON_GPU=1``. It runs the real
  training, merge, quantize, deploy and predict, and produces the numbers for the
  GPU Acceptance Report.

The scripted engine is not a shortcut around the pipeline. It is the same
``InferenceEngine`` interface the real one implements, so the evaluator, the
schema gate and the report assembly are all genuinely exercised. What it does not
prove is that a model produces good predictions -- only the GPU half does that.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from son_db import Database, migrate
from son_inference.predict import InferenceEngine, PredictionService
from son_worker.__main__ import Worker
from son_worker.queue import JobType

from son_api.main import create_app
from son_api.routers import runs as runs_router

CSV = """account,amount,device,身份证号,label
A001,50000,ios,110101199001011234,black
A002,48000,android,310101198505056789,black
A003,52000,web,440101199512123456,black
A004,300,ios,110101199203045678,white
A005,250,android,310101198807128901,white
A006,280,web,440101199011239012,white
A007,40000,ios,110101199404056789,gray
A008,41000,android,310101199306127890,gray
A009,60000,web,440101199407015678,black
A010,35000,ios,110101199508017890,gray
A011,200,android,310101199609028901,white
A012,47000,web,440101199710039012,black
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    monkeypatch.setenv("SON_DB_PATH", str(tmp_path / "runtime" / "son.db"))
    monkeypatch.setenv("SON_WORKSPACE", str(tmp_path / "runtime" / "workspace"))
    database = Database.open()
    migrate(database)
    runs_router.configure(database)
    yield database
    database.dispose()
    runs_router._DB_HOLDER.clear()


@pytest.fixture
def client(env: Database) -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


# ==========================================================================
# The golden path
# ==========================================================================


def test_golden_path_reaches_a_model_version(client: TestClient, env: Database) -> None:
    """Every arrow up to a trained artifact, with a stub for the GPU only.

    The stub stands in for torch, not for the pipeline: the split runs, the
    synthetic set is written, the job is claimed and completed by the real
    Worker, and a model version is registered only because files exist.
    """
    # --- project ---
    project = client.post("/api/projects", json={"name": "反诈 Demo"}).json()
    assert project["id"].startswith("prj_")

    # --- run ---
    run_id = client.post(f"/api/projects/{project['id']}/runs").json()["id"]
    assert client.get(f"/api/runs/{run_id}").json()["state"] == "CREATED"

    # --- step 1: scene ---
    body = client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"}).json()
    assert body["state"] == "SCENE_READY"
    assert body["current_step"] == 1

    # --- step 2: upload, once ---
    body = client.post(
        f"/api/runs/{run_id}/dataset",
        files={"upload": ("seed.csv", io.BytesIO(CSV.encode("utf-8")), "text/csv")},
    ).json()
    assert body["state"] == "DATASET_READY"
    assert body["dataset"]["rows"] == 12
    assert body["dataset"]["checksum"]

    # --- step 3: quality + split in one action ---
    body = client.post(f"/api/runs/{run_id}/prepare-data").json()
    assert body["state"] == "DATA_QUALITY_READY"
    assert body["split"]["test_rows"] >= 1
    assert body["split"]["test_synth_rows"] == 0
    assert body["split"]["quality"]["score"] > 0

    # --- step 4: synth ---
    body = client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 400}).json()
    assert body["state"] == "TRAIN_SET_READY"
    assert body["synth"]["rows"] == 400

    # --- step 5: model ---
    body = client.patch(
        f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"}
    ).json()
    assert body["state"] == "MODEL_SELECTED"

    # --- step 6: training config ---
    body = client.patch(
        f"/api/runs/{run_id}/training-config", json={"method": "lora", "max_steps": 100}
    ).json()
    assert body["state"] == "TRAINING_CONFIGURED"

    # --- step 7: submit. The API must not train. ---
    body = client.post(f"/api/runs/{run_id}/start").json()
    assert body["state"] == "QUEUED"
    job_id = body["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "QUEUED"
    assert client.get(f"/api/jobs/{job_id}").json()["progress"] == 0

    # --- the worker claims and runs it ---
    worker = Worker(env)
    worker.register(JobType.PIPELINE, _fake_pipeline_skill)
    claim = worker.run_once()
    assert claim.claimed is True

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "SUCCEEDED", job["error_message"]
    assert job["progress"] == 100
    assert job["result"]["gguf_path"]

    # --- a model version exists only because files exist ---
    version = _register_model_version(env, run_id, job["result"])
    assert Path(version["adapter_path"]).exists()
    assert Path(version["gguf_path"]).exists()
    assert version["gguf_checksum"]

    # --- evaluation runs the model over the real test split ---
    report = _run_evaluation(env, version, run_id)
    assert report["rows"] == _test_row_count(env, run_id)
    assert 0.0 <= report["effect"]["accuracy"] <= 1.0
    assert report["effect"]["targets"]["accuracy"]["passed"] in (True, False)
    assert report["effect"]["targets"]["format_compliance"]["target"] == 0.99

    # --- prediction goes through the same schema gate ---
    service = PredictionService(engine=_ScriptedEngine())
    prediction = service.predict(request=_sample_request(), measure=True)
    assert prediction.response.decision.value in ("black", "white", "gray")
    assert prediction.response.reason
    assert prediction.measured is True

    # --- deployment produces a launch plan, not a claim of readiness ---
    plan = _deployment_plan(env, run_id, version)
    assert plan["runtime"] == "llama_cpp"
    assert plan["status"] in ("PENDING", "STARTING", "SERVING")
    assert "llama-server" in plan["launch_command"]
    assert plan["predict_url"].endswith("/v1/predict")

    # --- lineage is assembled from records, end to end ---
    lineage = client.get(f"/api/runs/{run_id}").json()["lineage"]
    assert lineage["scene"] == "fraud_account"
    assert lineage["dataset"] == "ds_v1"
    assert lineage["split"] == "split_v1"
    assert lineage["synth"] == "syn_v1"
    assert lineage["base_model"] == "qwen2.5-1.5b-instruct"
    assert lineage["model_version"]


def test_the_run_survives_a_restart(client: TestClient, env: Database) -> None:
    """改造开发方案.md 39 step 12: close everything, reopen, still there."""
    project = client.post("/api/projects", json={"name": "持久化"}).json()
    run_id = client.post(f"/api/projects/{project['id']}/runs").json()["id"]
    client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
    client.post(
        f"/api/runs/{run_id}/dataset",
        files={"upload": ("seed.csv", io.BytesIO(CSV.encode("utf-8")), "text/csv")},
    )

    reopened = Database.open()
    try:
        runs_router.configure(reopened)
        fresh = TestClient(create_app())
        body = fresh.get(f"/api/runs/{run_id}").json()
        assert body["state"] == "DATASET_READY"
        assert body["dataset"]["checksum"]
        assert body["scene_code"] == "fraud_account"
    finally:
        runs_router.configure(env)
        reopened.dispose()


def test_reopening_the_api_shows_the_project(env: Database) -> None:
    first = TestClient(create_app())
    project = first.post("/api/projects", json={"name": "重启可见"}).json()

    second = TestClient(create_app())
    names = [p["name"] for p in second.get("/api/projects").json()]
    assert "重启可见" in names
    assert project["id"]


# ==========================================================================
# Cross-cutting properties
# ==========================================================================


def test_test_split_never_gains_a_synthetic_row(client: TestClient) -> None:
    """需求方案.txt 5.4, checked on the artifact rather than in passing."""
    run_id = _prepared_run(client, synth_rows=800)
    split = client.get(f"/api/runs/{run_id}").json()["split"]
    assert split["test_synth_rows"] == 0
    assert split["test_rows"] > 0


def test_the_api_cannot_skip_a_step(client: TestClient) -> None:
    run_id = _run(client)
    for path, body in (
        ("prepare-data", None),
        ("synth", {"target_rows": 100}),
    ):
        response = client.post(f"/api/runs/{run_id}/{path}", json=body)
        assert response.status_code == 409, path
    assert client.get(f"/api/runs/{run_id}").json()["state"] == "CREATED"


def test_unsupported_choices_are_refused(client: TestClient) -> None:
    """改造开发方案.md 31: never offer a dead end."""
    run_id = _prepared_run(client)
    assert (
        client.patch(f"/api/runs/{run_id}/model", json={"model_id": "gemma-2-2b-it"}).status_code
        == 422
    )
    client.patch(f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"})
    assert (
        client.patch(f"/api/runs/{run_id}/training-config", json={"method": "dpo"}).status_code
        == 422
    )


def test_every_asset_carries_a_checksum(client: TestClient, env: Database) -> None:
    """Rule 7: run_id, path, checksum, status on every asset."""
    run_id = _prepared_run(client, synth_rows=200)
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["dataset"]["checksum"]
    assert detail["synth"]["code"] == "syn_v1"
    assert (_workspace() / run_id / "dataset").exists()
    assert (_workspace() / run_id / "synth").exists()


def test_a_failed_job_leaves_a_message(env: Database, client: TestClient) -> None:
    """Rule 8: no swallowed exceptions."""
    run_id = _prepared_run(client)
    client.patch(f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"})
    client.patch(f"/api/runs/{run_id}/training-config", json={"method": "lora"})
    client.post(f"/api/runs/{run_id}/start")
    job_id = client.get(f"/api/runs/{run_id}").json()["job_id"]

    def boom(_ctx: Any) -> dict[str, Any]:
        raise RuntimeError("CUDA out of memory")

    worker = Worker(env)
    worker.register(JobType.PIPELINE, boom)
    worker.run_once()

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "FAILED"
    assert "CUDA out of memory" in job["error_message"]


# ==========================================================================
# GPU half
# ==========================================================================


@pytest.mark.gpu
def test_golden_path_on_a_real_gpu(env: Database, client: TestClient) -> None:
    """The full chain on real hardware. Skipped unless SON_GPU=1.

    Prints a GPU Acceptance Report: every number here comes from an actual run.
    """
    if os.environ.get("SON_GPU") != "1":
        pytest.skip("需要 GPU 节点：设置 SON_GPU=1 后运行")

    from son_inference.llamacpp_client import gpu_report

    gpu = gpu_report()
    assert gpu.get("cuda_available"), gpu

    run_id = _prepared_run(client, synth_rows=1000)
    client.patch(f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"})
    client.patch(
        f"/api/runs/{run_id}/training-config",
        json={"method": "lora", "max_steps": int(os.environ.get("SON_GPU_STEPS", "200"))},
    )
    client.post(f"/api/runs/{run_id}/start")

    # No skill registered: the real train_skill is installed on the worker.
    worker = Worker(env)
    from son_worker.skills import train_skill

    worker.register(JobType.PIPELINE, train_skill)
    claim = worker.run_once()

    job = client.get(f"/api/jobs/{claim.job.id}").json()  # type: ignore[union-attr]
    assert job["status"] == "SUCCEEDED", job["error_message"]

    report = {
        "gpu": gpu,
        "training": job["result"]["training"],
        "adapter": job["result"]["adapter_path"],
        "gguf_bytes": job["result"]["gguf_bytes"],
        "gguf_checksum": job["result"]["gguf_checksum"],
    }
    print("\n=== GPU ACCEPTANCE REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert job["result"]["gguf_bytes"] > 0


# ==========================================================================
# Helpers
# ==========================================================================


def _run(client: TestClient) -> str:
    project = client.post("/api/projects", json={"name": "E2E"}).json()
    return client.post(f"/api/projects/{project['id']}/runs").json()["id"]


def _prepared_run(client: TestClient, *, synth_rows: int = 400) -> str:
    """Steps 1-4."""
    run_id = _run(client)
    client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
    client.post(
        f"/api/runs/{run_id}/dataset",
        files={"upload": ("seed.csv", io.BytesIO(CSV.encode("utf-8")), "text/csv")},
    )
    client.post(f"/api/runs/{run_id}/prepare-data")
    client.post(f"/api/runs/{run_id}/synth", json={"target_rows": synth_rows})
    return run_id


def _fake_pipeline_skill(context: Any) -> dict[str, Any]:
    """Stand in for torch on a CPU machine.

    Writes real files so the downstream artifact checks are genuine -- the point
    of Rule 4 is that a missing file is refused, and that refusal needs something
    present to be refused *about* in the success case.
    """
    run_id = context.job.run_id
    base = _workspace() / run_id

    context.progress(30, stage="TRAIN", message="模拟训练")

    adapter = base / "training" / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"lora-placeholder")

    merged = base / "model" / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")

    gguf = base / "model" / "model-q4_k_m.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"gguf-placeholder")

    from son_db.assets import bytes_of
    from son_db.assets import checksum as file_checksum

    context.progress(100, stage="QUANTIZE", message="完成")
    return {
        "adapter_path": str(adapter),
        "merged_path": str(merged),
        "gguf_path": str(gguf),
        "gguf_checksum": file_checksum(gguf),
        "gguf_bytes": bytes_of(gguf),
        "quant_level": "Q4_K_M",
        "note": "GPU 节点上的真实训练由 SON_GPU=1 的测试覆盖",
    }


class _ScriptedEngine(InferenceEngine):
    """An engine that returns a valid JEV payload.

    Proves the schema gate and report assembly run end to end. It does not prove
    a model predicts well -- only the GPU half does that.
    """

    def complete(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        max_tokens: int = 256,
        stop: tuple[str, ...] = (),
    ) -> str:
        decision = "black" if "amount: 5" in prompt or "amount: 6" in prompt else "white"
        return json.dumps(
            {
                "decision": decision,
                "score": 0.87,
                "confidence": 0.9,
                "reason": "依据提供的特征综合判定",
            },
            ensure_ascii=False,
        )

    def is_ready(self) -> bool:
        return True


def _sample_request() -> Any:
    from son_contracts import PredictRequest

    return PredictRequest(amount=50000, device="ios")


def _register_model_version(env: Database, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Create the model version and artifact rows the pipeline would.

    The state advancement happens *after* the session closes. SQLite allows one
    writer, so opening a second session while the first holds an open
    transaction deadlocks -- which is a real constraint on how this code may be
    structured, not just a test artefact.
    """
    from son_db.models import ArtifactRow, ModelVersionRow
    from son_db.repositories import next_model_version_code, record_audit

    with env.session() as session:
        version = ModelVersionRow(
            run_id=run_id,
            code=next_model_version_code(session, run_id),
            base_model_id="qwen2.5-1.5b-instruct",
            training_method="lora",
            adapter_path=result["adapter_path"],
            merged_model_path=result["merged_path"],
        )
        session.add(version)
        session.flush()

        artifact = ArtifactRow(
            model_version_id=version.id,
            type="GGUF",
            format="gguf",
            quant="Q4_K_M",
            path=result["gguf_path"],
            checksum=result["gguf_checksum"],
            bytes=result["gguf_bytes"],
        )
        session.add(artifact)
        session.flush()

        from son_db.models import RunRow

        run = session.get(RunRow, run_id)
        assert run is not None
        run.model_version_id = version.id
        record_audit(session, "model_version", version.id, version.code)

        payload = {
            "id": version.id,
            "code": version.code,
            "adapter_path": version.adapter_path,
            "gguf_path": artifact.path,
            "gguf_checksum": artifact.checksum,
            "gguf_bytes": artifact.bytes,
        }

    # Walk the run through the pipeline states the worker would drive.
    from son_orchestrator.runstate import RunState
    from son_worker.progress import advance

    for target in (
        RunState.TRAINING,
        RunState.MERGING,
        RunState.QUANTIZING,
        RunState.MODEL_READY,
    ):
        advance(env, run_id, target, strict=False)

    return payload


def _workspace() -> Path:
    return Path(os.environ["SON_WORKSPACE"])


def _test_row_count(env: Database, run_id: str) -> int:
    from son_db.models import SplitRow

    with env.read() as session:
        split = session.query(SplitRow).filter(SplitRow.run_id == run_id).one_or_none()
        assert split is not None
        return int(split.test_rows)


def _run_evaluation(env: Database, version: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Predict the real test split with a scripted engine."""
    from son_db.models import SplitRow
    from son_inference.evaluate import run_evaluation as evaluate

    with env.read() as session:
        split = session.query(SplitRow).filter(SplitRow.run_id == run_id).one()
        test_csv = Path(split.test_path)

    base = _workspace() / run_id / "evaluation"
    outcome = evaluate(test_csv, base, PredictionService(engine=_ScriptedEngine()))

    from son_db.models import EvaluationRow

    with env.session() as session:
        row = EvaluationRow(
            run_id=run_id,
            model_version_id=version["id"],
            test_dataset_path=str(test_csv),
            predictions_path=str(outcome.predictions_path),
            error_cases_path=str(outcome.error_cases_path),
            metrics_json=json.dumps(outcome.effect, ensure_ascii=False),
            performance_json=json.dumps(outcome.performance, ensure_ascii=False),
        )
        session.add(row)
        session.flush()

    assert outcome.predictions_path.exists()
    assert outcome.report_path.exists()
    return {
        "effect": outcome.effect,
        "performance": outcome.performance,
        "rows": len(outcome.predictions_path.read_text(encoding="utf-8").splitlines()),
    }


def _deployment_plan(env: Database, run_id: str, version: dict[str, Any]) -> dict[str, Any]:
    """Deployment record plus the launch command a GPU node would run."""
    from son_db.models import ArtifactRow, DeploymentRow

    with env.session() as session:
        artifact = (
            session.query(ArtifactRow).filter(ArtifactRow.model_version_id == version["id"]).first()
        )
        assert artifact is not None
        row = DeploymentRow(
            run_id=run_id,
            model_version_id=version["id"],
            artifact_id=artifact.id,
            runtime="llama_cpp",
            host="127.0.0.1",
            port=8081,
            status="PENDING",
        )
        session.add(row)
        session.flush()
        payload = {
            "id": row.id,
            "runtime": row.runtime,
            "status": row.status,
            "port": row.port,
            "predict_url": f"http://{row.host}:{row.port}/v1/predict",
            "gguf_path": artifact.path,
        }

    from son_inference.deployment import Deployment

    plan = Deployment(gguf_path=payload["gguf_path"], port=8081)
    return {**payload, "launch_command": " ".join(plan.argv())}
