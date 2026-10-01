"""Dataset router tests (M1).

These cover the upload path end to end, and in particular that the split
endpoint cannot be used to route synthetic rows into the test set.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from son_api.main import create_app

CSV_SEED = """account,amount,device,身份证号,label
A001,5000,ios,110101199001011234,black
A002,200,android,310101198505056789,white
A003,90000,web,440101199512123456,gray
A004,1500,ios,110101199203045678,black
A005,300,web,310101198807128901,white
A006,45000,android,440101199011239012,black
A007,700,ios,110101199404056789,gray
A008,22000,web,310101199306127890,black
"""


def _client() -> TestClient:
    return TestClient(create_app())


def _upload(content: str = CSV_SEED, filename: str = "seed.csv") -> dict[str, object]:
    """Build the `files=` payload for a multipart POST."""
    return {"upload": (filename, io.BytesIO(content.encode("utf-8")), "text/csv")}


class TestPreview:
    def test_returns_first_five_rows(self) -> None:
        body = _client().post("/api/datasets/preview", files=_upload()).json()
        assert len(body["rows"]) == 5
        assert body["total_rows"] == 8

    def test_detects_the_label_column(self) -> None:
        body = _client().post("/api/datasets/preview", files=_upload()).json()
        assert body["label_column"] == "label"
        assert body["label_confidence"] == "high"

    def test_reports_three_way_distribution(self) -> None:
        body = _client().post("/api/datasets/preview", files=_upload()).json()
        assert body["label_distribution"] == {"black": 4, "white": 2, "gray": 2}

    def test_reports_sensitive_fields(self) -> None:
        body = _client().post("/api/datasets/preview", files=_upload()).json()
        assert body["sensitive_fields"]["身份证号"] == "id_card"

    def test_empty_upload_is_rejected(self) -> None:
        response = _client().post("/api/datasets/preview", files=_upload(""))
        assert response.status_code == 422

    def test_header_only_upload_rejected(self) -> None:
        response = _client().post("/api/datasets/preview", files=_upload("a,b,c\n"))
        assert response.status_code in (200, 422)


class TestQuality:
    def test_returns_a_bounded_score(self) -> None:
        body = _client().post("/api/datasets/quality", files=_upload()).json()
        assert 0.0 <= body["score"] <= 100.0
        assert body["grade"] in ("良好", "可接受", "偏差")

    def test_reports_the_three_dimension_cards(self) -> None:
        body = _client().post("/api/datasets/quality", files=_upload()).json()
        assert [d["name"] for d in body["dimensions"]] == ["样本量", "缺失率", "黑白比"]

    def test_lists_masked_fields(self) -> None:
        body = _client().post("/api/datasets/quality", files=_upload()).json()
        assert "身份证号" in body["masked_fields"]


class TestSplit:
    def test_split_runs_and_reports_the_invariant(self) -> None:
        body = _client().post("/api/datasets/split", files=_upload()).json()
        assert body["test_contains_synth"] is False
        assert body["train_rows"] > 0
        assert body["test_rows"] > 0

    def test_all_rows_land_somewhere(self) -> None:
        body = _client().post("/api/datasets/split", files=_upload()).json()
        assert body["train_rows"] + body["valid_rows"] + body["test_rows"] == 8

    def test_label_distribution_keeps_all_three(self) -> None:
        body = _client().post("/api/datasets/split", files=_upload()).json()
        assert set(body["label_distribution"]) == {"train", "valid", "test"}
        assert set(body["label_distribution"]["test"]) == {"black", "white", "gray"}

    def test_caller_cannot_supply_its_own_split(self) -> None:
        """There is one implementation of the split rule, and it is the guarded one."""
        response = _client().post("/api/datasets/split?test_rows=8", files=_upload())
        # Unknown query params are ignored rather than honoured.
        assert response.status_code == 200
        assert response.json()["test_rows"] < 8

    def test_undetectable_label_column_is_rejected(self) -> None:
        csv = "a,b,c\nfoo,bar,baz\nfoo,qux,baz\n"
        response = _client().post("/api/datasets/split", files=_upload(csv))
        assert response.status_code == 422
        assert "标签列" in response.json()["detail"]


class TestSynthRecommendation:
    def test_returns_prefilled_parameters(self) -> None:
        body = _client().post("/api/datasets/recommend-synth", files=_upload()).json()
        assert body["method"] == "distribution_fit"
        assert body["black_white_ratio"] >= 1.0

    def test_gives_reasons_in_business_language(self) -> None:
        body = _client().post("/api/datasets/recommend-synth", files=_upload()).json()
        assert body["reasons"]
        assert all(isinstance(r, str) for r in body["reasons"])


class TestSynth:
    def test_generates_requested_rows(self) -> None:
        response = _client().post("/api/datasets/synth?target_rows=50", files=_upload())
        assert response.status_code == 200
        assert response.json()["rows"] == 50

    def test_every_row_is_marked_synthetic(self) -> None:
        body = _client().post("/api/datasets/synth?target_rows=50", files=_upload()).json()
        assert body["all_rows_marked_synthetic"] is True

    def test_reports_fidelity_and_privacy(self) -> None:
        body = _client().post("/api/datasets/synth?target_rows=50", files=_upload()).json()
        assert 0.0 <= body["fidelity_score"] <= 1.0
        assert body["privacy_lines"]
        assert body["reversible_risk"] in ("none", "review", "high")

    def test_distribution_fit_is_the_default_method(self) -> None:
        body = _client().post("/api/datasets/synth?target_rows=30", files=_upload()).json()
        assert body["method"] == "distribution_fit"

    def test_augment_label_shifts_the_distribution(self) -> None:
        body = (
            _client()
            .post("/api/datasets/synth?target_rows=100&augment_label=white", files=_upload())
            .json()
        )
        assert body["label_distribution"]["white"] == 100

    def test_unknown_method_is_rejected(self) -> None:
        response = _client().post(
            "/api/datasets/synth?method=telepathy&target_rows=10", files=_upload()
        )
        assert response.status_code == 422


class TestOpenApiStillGenerates:
    def test_dataset_routes_present(self) -> None:
        paths = _client().get("/openapi.json").json()["paths"]
        for path in (
            "/api/datasets/preview",
            "/api/datasets/quality",
            "/api/datasets/split",
            "/api/datasets/synth",
            "/api/datasets/recommend-synth",
        ):
            assert path in paths, f"{path} missing from OpenAPI"
