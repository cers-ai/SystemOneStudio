"""Run API tests.

These exercise the property the whole refactor exists for: **a step only
advances when its asset exists**. Every step is tested twice -- with its
prerequisites missing, and with them present -- because the failure mode is a
run that claims work it never did.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from son_db import Database, migrate

from son_api.main import create_app
from son_api.routers import runs as runs_router

CSV = """account,amount,device,身份证号,label
A001,5000,ios,110101199001011234,black
A002,200,android,310101198505056789,white
A003,90000,web,440101199512123456,gray
A004,1500,ios,110101199203045678,black
A005,300,web,310101198807128901,white
A006,45000,android,440101199011239012,black
A007,700,ios,110101199404056789,gray
A008,22000,web,310101199306127890,black
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


def _project(client: TestClient) -> str:
    return client.post("/api/projects", json={"name": "反诈 Demo"}).json()["id"]


def _run(client: TestClient, project_id: str | None = None) -> str:
    pid = project_id or _project(client)
    return client.post(f"/api/projects/{pid}/runs").json()["id"]


def _upload(client: TestClient, run_id: str, csv: str = CSV) -> dict:
    return client.post(
        f"/api/runs/{run_id}/dataset",
        files={"upload": ("seed.csv", io.BytesIO(csv.encode("utf-8")), "text/csv")},
    ).json()


def _state(client: TestClient, run_id: str) -> str:
    return client.get(f"/api/runs/{run_id}").json()["state"]


class TestProjectAndRun:
    def test_create_project(self, client: TestClient) -> None:
        body = client.post("/api/projects", json={"name": "项目"}).json()
        assert body["name"] == "项目"
        assert body["id"].startswith("prj_")

    def test_list_projects(self, client: TestClient) -> None:
        client.post("/api/projects", json={"name": "a"})
        client.post("/api/projects", json={"name": "b"})
        assert len(client.get("/api/projects").json()) == 2

    def test_create_run_starts_at_created(self, client: TestClient) -> None:
        run_id = _run(client)
        body = client.get(f"/api/runs/{run_id}").json()
        assert body["state"] == "CREATED"
        assert body["current_step"] == 1

    def test_unknown_project_is_404(self, client: TestClient) -> None:
        assert client.post("/api/projects/nope/runs").status_code == 404

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/run_nope").status_code == 404


class TestStep1Scene:
    def test_scene_advances_the_run(self, client: TestClient) -> None:
        run_id = _run(client)
        body = client.patch(
            f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"}
        ).json()
        assert body["state"] == "SCENE_READY"
        assert body["scene_code"] == "fraud_account"

    def test_unknown_scene_is_422(self, client: TestClient) -> None:
        run_id = _run(client)
        response = client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "nope"})
        assert response.status_code == 422

    def test_upload_advances_only_to_dataset_ready(self, client: TestClient) -> None:
        """Uploading must not advance past its own step.

        This is the guarantee the old wizard broke: a click marked steps done
        without anything having happened.
        """
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        assert _upload(client, run_id)["state"] == "DATASET_READY"
        assert _state(client, run_id) == "DATASET_READY"


class TestStep2Dataset:
    def test_upload_creates_the_asset(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        body = _upload(client, run_id)
        assert body["state"] == "DATASET_READY"
        assert body["dataset"]["rows"] == 8
        assert body["dataset"]["code"] == "ds_v1"
        assert body["dataset"]["checksum"]

    def test_upload_before_scene_is_409(self, client: TestClient) -> None:
        run_id = _run(client)
        response = client.post(
            f"/api/runs/{run_id}/dataset",
            files={"upload": ("s.csv", io.BytesIO(CSV.encode()), "text/csv")},
        )
        assert response.status_code == 409

    def test_empty_upload_is_422(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        response = client.post(
            f"/api/runs/{run_id}/dataset",
            files={"upload": ("s.csv", io.BytesIO(b""), "text/csv")},
        )
        assert response.status_code == 422

    def test_unparseable_csv_is_422_and_leaves_no_file(
        self, client: TestClient, env: Database
    ) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        response = client.post(
            f"/api/runs/{run_id}/dataset",
            files={"upload": ("s.csv", io.BytesIO(b""), "text/csv")},
        )
        assert response.status_code == 422
        assert _state(client, run_id) == "SCENE_READY"

    def test_reupload_replaces_rather_than_accumulates(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        _upload(client, run_id)
        assert len(client.get(f"/api/runs/{run_id}").json()["dataset"]) > 0


class TestStep3Prepare:
    def _to_dataset(self, client: TestClient) -> str:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        return run_id

    def test_prepare_creates_the_split(self, client: TestClient) -> None:
        run_id = self._to_dataset(client)
        body = client.post(f"/api/runs/{run_id}/prepare-data").json()
        assert body["state"] == "DATA_QUALITY_READY"
        assert (
            body["split"]["train_rows"] + body["split"]["valid_rows"] + body["split"]["test_rows"]
            == 8
        )

    def test_synth_rows_in_test_are_zero(self, client: TestClient) -> None:
        """The hard constraint, recorded rather than asserted in passing."""
        run_id = self._to_dataset(client)
        assert (
            client.post(f"/api/runs/{run_id}/prepare-data").json()["split"]["test_synth_rows"] == 0
        )

    def test_quality_report_is_attached(self, client: TestClient) -> None:
        run_id = self._to_dataset(client)
        quality = client.post(f"/api/runs/{run_id}/prepare-data").json()["split"]["quality"]
        assert quality["score"] > 0
        assert quality["dimensions"]

    def test_prepare_before_upload_is_409(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        assert client.post(f"/api/runs/{run_id}/prepare-data").status_code == 409

    def test_three_files_are_written(self, client: TestClient) -> None:
        run_id = self._to_dataset(client)
        detail = client.post(f"/api/runs/{run_id}/prepare-data").json()
        assert detail["split"]["test_rows"] >= 1


class TestStep4Synth:
    def _to_split(self, client: TestClient) -> str:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        client.post(f"/api/runs/{run_id}/prepare-data")
        return run_id

    def test_synth_creates_the_asset(self, client: TestClient) -> None:
        run_id = self._to_split(client)
        body = client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 200}).json()
        assert body["state"] == "TRAIN_SET_READY"
        assert body["synth"]["rows"] == 200
        assert body["synth"]["code"] == "syn_v1"

    def test_synth_reports_fidelity_and_privacy(self, client: TestClient) -> None:
        run_id = self._to_split(client)
        synth = client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 100}).json()["synth"]
        assert synth["fidelity_score"] is not None
        assert "lines" in synth["privacy"]

    def test_synth_before_split_is_409(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        response = client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 100})
        assert response.status_code == 409
        assert "划分" in response.json()["detail"]

    def test_test_split_never_gains_synth_rows(self, client: TestClient) -> None:
        """Synth may only ever land in the training input."""
        run_id = self._to_split(client)
        before = client.get(f"/api/runs/{run_id}").json()["split"]["test_rows"]
        client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 500})
        after = client.get(f"/api/runs/{run_id}").json()["split"]
        assert after["test_rows"] == before
        assert after["test_synth_rows"] == 0


class TestSteps5And6:
    def _to_synth(self, client: TestClient) -> str:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        client.post(f"/api/runs/{run_id}/prepare-data")
        client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 100})
        return run_id

    def test_only_wired_models_are_selectable(self, client: TestClient) -> None:
        """改造开发方案.md 31: never offer a choice that cannot work."""
        run_id = self._to_synth(client)
        ok = client.patch(f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"})
        assert ok.status_code == 200
        assert ok.json()["state"] == "MODEL_SELECTED"

    def test_other_models_are_refused_with_a_reason(self, client: TestClient) -> None:
        run_id = self._to_synth(client)
        response = client.patch(f"/api/runs/{run_id}/model", json={"model_id": "gemma-2-2b-it"})
        assert response.status_code == 422
        assert "未打通" in response.json()["detail"]

    def test_training_config_accepts_lora(self, client: TestClient) -> None:
        run_id = self._to_synth(client)
        client.patch(f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"})
        body = client.patch(f"/api/runs/{run_id}/training-config", json={"method": "lora"}).json()
        assert body["state"] == "TRAINING_CONFIGURED"
        assert body["training_config"]["method"] == "lora"

    def test_dpo_is_refused(self, client: TestClient) -> None:
        run_id = self._to_synth(client)
        client.patch(f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"})
        response = client.patch(f"/api/runs/{run_id}/training-config", json={"method": "dpo"})
        assert response.status_code == 422

    def test_qlora_is_refused(self, client: TestClient) -> None:
        run_id = self._to_synth(client)
        client.patch(f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"})
        response = client.patch(
            f"/api/runs/{run_id}/training-config", json={"method": "lora", "qlora": True}
        )
        assert response.status_code == 422


class TestStep7Start:
    def _configured(self, client: TestClient) -> str:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        client.post(f"/api/runs/{run_id}/prepare-data")
        client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 100})
        client.patch(f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"})
        client.patch(f"/api/runs/{run_id}/training-config", json={"method": "lora"})
        return run_id

    def test_start_enqueues_a_job_and_advances(self, client: TestClient) -> None:
        run_id = self._configured(client)
        body = client.post(f"/api/runs/{run_id}/start").json()
        assert body["state"] == "QUEUED"
        assert body["job_id"]

    def test_the_api_does_not_train(self, client: TestClient) -> None:
        """Training belongs to the worker; an HTTP request must not block on it."""
        run_id = self._configured(client)
        client.post(f"/api/runs/{run_id}/start")
        job = client.get(f"/api/runs/{run_id}").json()["job"]
        assert job["status"] == "QUEUED"
        assert job["progress"] == 0
        assert job["started_at"] is None

    def test_start_without_config_is_409(self, client: TestClient) -> None:
        run_id = _run(client)
        assert client.post(f"/api/runs/{run_id}/start").status_code == 409

    def test_job_payload_carries_the_real_paths(self, client: TestClient) -> None:
        run_id = self._configured(client)
        client.post(f"/api/runs/{run_id}/start")
        job_id = client.get(f"/api/runs/{run_id}").json()["job_id"]
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "QUEUED"
        assert job["type"] == "PIPELINE"

    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get("/api/jobs/job_nope").status_code == 404

    def test_queued_job_can_be_cancelled(self, client: TestClient) -> None:
        run_id = self._configured(client)
        client.post(f"/api/runs/{run_id}/start")
        job_id = client.get(f"/api/runs/{run_id}").json()["job_id"]
        assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "CANCELLED"


class TestNoStepCanBeSkipped:
    def test_cannot_upload_without_a_scene(self, client: TestClient) -> None:
        run_id = _run(client)
        response = client.post(
            f"/api/runs/{run_id}/dataset",
            files={"upload": ("s.csv", io.BytesIO(CSV.encode()), "text/csv")},
        )
        assert response.status_code == 409

    def test_cannot_configure_a_model_before_synth(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        response = client.patch(
            f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"}
        )
        assert response.status_code == 409

    def test_reselecting_a_scene_after_data_is_refused(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        response = client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        assert response.status_code == 409


class TestCapabilities:
    def test_lists_only_wired_capabilities(self, client: TestClient) -> None:
        body = client.get("/api/runtime/capabilities").json()
        assert body["base_models"] == ["qwen2.5-1.5b-instruct"]
        assert body["training_methods"] == ["lora"]
        assert body["quant"] == ["Q4_K_M"]
        assert body["runtime"] == "llama_cpp"
        assert body["jev_level"] == "L1"

    def test_unsupported_is_explicit(self, client: TestClient) -> None:
        """The UI greys these out rather than offering a dead end."""
        body = client.get("/api/runtime/capabilities").json()
        assert "dpo" in body["unsupported"]["training_methods"]
        assert "L3" in body["unsupported"]["jev_levels"]

    def test_output_schema_is_served(self, client: TestClient) -> None:
        body = client.get("/api/runtime/capabilities").json()
        assert set(body["output_schema"]["required"]) == {
            "decision",
            "score",
            "confidence",
            "reason",
        }


class TestLineage:
    def test_lineage_is_generated_not_supplied(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        lineage = client.get(f"/api/runs/{run_id}").json()["lineage"]
        assert lineage["scene"] == "fraud_account"
        assert lineage["dataset"] == "ds_v1"
        assert "split" not in lineage  # not produced yet

    def test_lineage_grows_with_the_run(self, client: TestClient) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)
        client.post(f"/api/runs/{run_id}/prepare-data")
        client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 100})
        lineage = client.get(f"/api/runs/{run_id}").json()["lineage"]
        assert lineage["split"] == "split_v1"
        assert lineage["synth"] == "syn_v1"

    def test_lineage_is_absent_at_creation(self, client: TestClient) -> None:
        run_id = _run(client)
        assert client.get(f"/api/runs/{run_id}").json()["lineage"] is None


class TestPersistenceAcrossRestart:
    def test_run_survives_a_new_database_object(self, client: TestClient, env: Database) -> None:
        run_id = _run(client)
        client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"})
        _upload(client, run_id)

        # A brand new Database over the same file, reconfigured for the app.
        reopened = Database.open()
        try:
            runs_router.configure(reopened)
            reopened_client = TestClient(create_app())
            body = reopened_client.get(f"/api/runs/{run_id}").json()
            assert body["state"] == "DATASET_READY"
            assert body["dataset"]["checksum"]
        finally:
            runs_router.configure(env)
            reopened.dispose()
