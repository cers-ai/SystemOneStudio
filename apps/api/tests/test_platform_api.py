"""Platform router tests: scenes, projects, system status, audit.

Covers the two P0 modules that had no surface (项目管理 / 场景定义) plus 系统管理.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from son_api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def detail_text(response: object) -> str:
    """Flatten a FastAPI error body.

    Pydantic validation errors come back as a list of objects while our own
    HTTPException detail is a string, so a test asserting on the message has to
    cope with both.
    """
    body = response.json()  # type: ignore[attr-defined]
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, str):
        return detail
    return " ".join(str(d.get("msg", d)) for d in detail if isinstance(d, dict))


class TestSceneTemplates:
    def test_gallery_has_the_six_presets_from_requirement_5_1(self) -> None:
        templates = _client().get("/api/scenes/templates").json()
        assert [t["id"] for t in templates] == [
            "fraud_account",
            "payment_risk",
            "compliance_review",
            "marketing_decision",
            "credit_approval",
            "content_safety",
        ]

    def test_fraud_scene_is_flagged_recommended(self) -> None:
        """需求方案.txt 5.1 marks 反诈账户判定 as [推荐]."""
        recommended = [t for t in _client().get("/api/scenes/templates").json() if t["recommended"]]
        assert len(recommended) == 1
        assert recommended[0]["id"] == "fraud_account"

    def test_every_template_is_three_way(self) -> None:
        for template in _client().get("/api/scenes/templates").json():
            assert set(template["labels"]) == {"black", "white", "gray"}

    def test_fraud_scene_declares_32_fields(self) -> None:
        fraud = _client().get("/api/scenes/templates").json()[0]
        assert fraud["field_count"] == 32

    def test_fraud_scene_offers_an_example_dataset(self) -> None:
        """需求方案.txt 5.2: 提供示例数据集，零数据也能体验."""
        fraud = _client().get("/api/scenes/templates").json()[0]
        assert fraud["example_dataset"]

    def test_templates_carry_a_recommended_base_model(self) -> None:
        for template in _client().get("/api/scenes/templates").json():
            assert template["recommended_base_model"]

    def test_templates_declare_core_metrics(self) -> None:
        for template in _client().get("/api/scenes/templates").json():
            assert template["core_metrics"]


class TestCreateSceneFromTemplate:
    def test_inherits_recommendations(self) -> None:
        response = _client().post(
            "/api/scenes", json={"template_id": "fraud_account", "name": "我的反诈项目"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_base_model"] == "qwen2.5-3b-instruct"
        assert body["core_metrics"]
        assert body["source"] == "template"

    def test_generates_a_scene_code(self) -> None:
        body = _client().post("/api/scenes", json={"template_id": "fraud_account"}).json()
        assert body["code"].startswith("sc_v")

    def test_jev_switches_default_on_and_are_independent(self) -> None:
        """需求方案.txt 6.3: two switches, both on by default, independently set."""
        body = _client().post("/api/scenes", json={"template_id": "fraud_account"}).json()
        assert body["jev_format_compat"] is True
        assert body["jev_training_compat"] is True

        off = (
            _client()
            .post(
                "/api/scenes",
                json={"template_id": "fraud_account", "jev_format_compat": False},
            )
            .json()
        )
        assert off["jev_format_compat"] is False
        assert off["jev_training_compat"] is True

    def test_unknown_template_is_404(self) -> None:
        assert _client().post("/api/scenes", json={"template_id": "nope"}).status_code == 404


class TestCreateSceneFromScratch:
    def test_requires_a_name(self) -> None:
        response = _client().post("/api/scenes", json={})
        assert response.status_code == 422
        assert "名称" in detail_text(response)

    def test_creates_with_the_given_name(self) -> None:
        body = _client().post("/api/scenes", json={"name": "商户风险判定"}).json()
        assert body["name"] == "商户风险判定"
        assert body["source"] == "custom"

    def test_binary_label_system_is_rejected(self) -> None:
        """Three-class is a product constraint, not a preference.

        A binary scene makes recall and the gray handling meaningless, so it is
        refused rather than quietly defaulted.
        """
        response = _client().post(
            "/api/scenes", json={"name": "二分类", "labels": ["black", "white"]}
        )
        assert response.status_code == 422
        assert "三分类" in detail_text(response)

    def test_duplicate_label_is_rejected(self) -> None:
        response = _client().post(
            "/api/scenes", json={"name": "重复标签", "labels": ["black", "white", "white"]}
        )
        assert response.status_code == 422

    def test_custom_scene_has_no_fabricated_recommendation(self) -> None:
        body = _client().post("/api/scenes", json={"name": "空白场景"}).json()
        assert body["recommended_base_model"] == ""
        assert body["core_metrics"] == []


class TestUpdateJevFlags:
    def _scene_id(self) -> str:
        return _client().post("/api/scenes", json={"template_id": "fraud_account"}).json()["id"]

    def test_can_turn_format_off_without_touching_training(self) -> None:
        scene_id = self._scene_id()
        body = (
            _client()
            .patch(
                f"/api/scenes/{scene_id}/jev",
                json={"jev_format_compat": False, "jev_training_compat": True},
            )
            .json()
        )
        assert body["jev_format_compat"] is False
        assert body["jev_training_compat"] is True

    def test_can_turn_training_off_without_touching_format(self) -> None:
        scene_id = self._scene_id()
        body = (
            _client()
            .patch(
                f"/api/scenes/{scene_id}/jev",
                json={"jev_format_compat": True, "jev_training_compat": False},
            )
            .json()
        )
        assert body["jev_format_compat"] is True
        assert body["jev_training_compat"] is False

    def test_unknown_scene_is_404(self) -> None:
        assert (
            _client()
            .patch(
                "/api/scenes/nope/jev",
                json={"jev_format_compat": True, "jev_training_compat": True},
            )
            .status_code
            == 404
        )


class TestProjects:
    def test_create_and_list(self) -> None:
        created = _client().post("/api/projects", json={"name": "Q3 反诈"}).json()
        assert created["code"].startswith("pr_")
        codes = [p["code"] for p in _client().get("/api/projects").json()]
        assert created["code"] in codes

    def test_binds_a_scene_code(self) -> None:
        scene_id = _client().post("/api/scenes", json={"template_id": "fraud_account"}).json()["id"]
        project = (
            _client().post("/api/projects", json={"name": "绑定场景", "scene_id": scene_id}).json()
        )
        assert project["scene_code"].startswith("sc_v")

    def test_unknown_scene_is_404(self) -> None:
        assert (
            _client().post("/api/projects", json={"name": "x", "scene_id": "nope"}).status_code
            == 404
        )

    @pytest.mark.parametrize("mode", ["wizard", "rapid", "canvas", "expert"])
    def test_all_four_modes_are_accepted(self, mode: str) -> None:
        """需求方案.txt principle 1: four modes over one engine."""
        body = _client().post("/api/projects", json={"name": f"m-{mode}", "mode": mode}).json()
        assert body["mode"] == mode

    def test_invalid_mode_is_rejected(self) -> None:
        assert (
            _client().post("/api/projects", json={"name": "x", "mode": "telepathy"}).status_code
            == 422
        )


class TestSystemStatus:
    def test_reports_gpu_truthfully(self) -> None:
        """The whole verification story depends on whether a number came from a
        real device, so this must not default to available."""
        body = _client().get("/api/system/status").json()
        assert isinstance(body["gpu_available"], bool)
        assert body["gpu_detail"]

    def test_lists_service_availability(self) -> None:
        services = _client().get("/api/system/status").json()["services"]
        names = {s["name"] for s in services}
        assert {"控制面 API", "训练后端", "推理服务"} <= names

    def test_declares_in_memory_persistence(self) -> None:
        """A restart losing data must be discoverable, not a surprise."""
        body = _client().get("/api/system/status").json()
        assert body["persistence"] == "in-memory"
        assert any("重启" in n for n in body["notes"])

    def test_counts_the_registry(self) -> None:
        body = _client().get("/api/system/status").json()
        assert body["in_scope_model_count"] <= body["model_count"]
        assert body["in_scope_model_count"] > 0

    def test_always_has_at_least_one_note(self) -> None:
        assert _client().get("/api/system/status").json()["notes"]


class TestAudit:
    def test_records_scene_creation(self) -> None:
        _client().post("/api/scenes", json={"template_id": "fraud_account"})
        actions = [e["action"] for e in _client().get("/api/system/audit").json()]
        assert "create_scene" in actions

    def test_records_jev_flag_changes(self) -> None:
        scene_id = _client().post("/api/scenes", json={"template_id": "fraud_account"}).json()["id"]
        _client().patch(
            f"/api/scenes/{scene_id}/jev",
            json={"jev_format_compat": False, "jev_training_compat": True},
        )
        audit = _client().get("/api/system/audit").json()
        assert any(e["action"] == "update_jev_flags" for e in audit)
        assert any("格式对齐" in e["detail"] for e in audit)

    def test_records_project_creation(self) -> None:
        _client().post("/api/projects", json={"name": "审计测试"})
        assert any(
            e["action"] == "create_project" for e in _client().get("/api/system/audit").json()
        )

    def test_entries_have_a_timestamp(self) -> None:
        _client().post("/api/projects", json={"name": "时间戳"})
        entry = _client().get("/api/system/audit").json()[0]
        assert entry["ts"]

    def test_newest_first(self) -> None:
        _client().post("/api/projects", json={"name": "第一条"})
        _client().post("/api/projects", json={"name": "第二条"})
        details = [e["detail"] for e in _client().get("/api/system/audit").json()]
        assert "第二条" in details[0]


class TestOpenApi:
    def test_platform_routes_present(self) -> None:
        paths = _client().get("/openapi.json").json()["paths"]
        for path in (
            "/api/scenes/templates",
            "/api/scenes",
            "/api/scenes/{scene_id}/jev",
            "/api/projects",
            "/api/system/status",
            "/api/system/audit",
        ):
            assert path in paths, f"{path} missing"
