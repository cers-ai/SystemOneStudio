"""Dataset and synthesis endpoints (M1: 需求方案.txt 5.2 - 5.4).

The API is the only place a dataset enters the system. Everything that can
poison the evaluation set is guarded here or in
:func:`son_data_pipeline.stratified_split`:

* rows carry an ``origin`` stamp from the moment they are created
* the split endpoint always routes through the guarded splitter
* no endpoint accepts a caller-supplied split assignment
"""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field
from son_data_pipeline import (
    apply_masking,
    build_report,
    detect_label_column,
    label_counts,
    recommend,
    report_dimensions,
    stratified_split,
    with_origin,
)
from son_synth import SynthRequest, assess_fidelity, assess_privacy, synthesize

from son_contracts import DataOrigin, Decision, SynthMethod

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class PreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, object]]
    total_rows: int
    label_column: str | None
    label_confidence: str
    label_reason: str
    label_distribution: dict[str, int]
    sensitive_fields: dict[str, str]
    missing_cells: int
    missing_rate: float
    preview_limit: int = Field(default=5, description="需求方案.txt 5.2 shows the first 5 rows")


class QualityResponse(BaseModel):
    score: float
    grade: str
    dimensions: list[dict[str, object]]
    label_distribution: dict[str, int]
    anomalies: list[str]
    masked_fields: list[str]
    suggestions: list[str]


class SplitResponse(BaseModel):
    train_rows: int
    valid_rows: int
    test_rows: int
    test_contains_synth: bool
    label_distribution: dict[str, dict[str, int]]
    undersized_groups: list[str]


class SynthRecommendationResponse(BaseModel):
    method: str
    total_rows: int
    black_white_ratio: float
    augment_label: str | None
    augment_rows: int
    reasons: list[str]
    describe: str


class SynthResponse(BaseModel):
    method: str
    rows: int
    label_distribution: dict[str, int]
    fidelity_score: float
    fidelity_verdict: str
    fidelity_notes: list[str]
    privacy_lines: list[str]
    reversible_risk: str
    all_rows_marked_synthetic: bool


def _read_upload(upload: UploadFile) -> list[dict[str, object]]:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="上传文件为空")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="文件编码无法解析，请另存为 UTF-8 CSV") from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=422, detail="CSV 缺少表头行")

    return [dict(row) for row in reader]


@router.post("/preview", response_model=PreviewResponse, summary="上传并预览")
async def preview(upload: UploadFile) -> PreviewResponse:
    """Step 2 of the wizard: upload, preview, auto-detect (需求方案.txt 5.2)."""
    import pandas as pd

    records = _read_upload(upload)
    frame = pd.DataFrame(records)

    detection = detect_label_column(frame)
    counts = (
        label_counts(frame, detection.column, detection.mapping)
        if detection.column
        else {d.value: 0 for d in Decision}
    )

    from son_data_pipeline.masking import detect_sensitive_columns

    sensitive = detect_sensitive_columns(frame)
    missing_cells = int(frame.isna().sum().sum()) + int((frame == "").sum().sum())
    total_cells = frame.size

    return PreviewResponse(
        columns=[str(c) for c in frame.columns],
        rows=records[:5],
        total_rows=len(frame),
        label_column=detection.column,
        label_confidence=detection.confidence,
        label_reason=detection.reason,
        label_distribution=counts,
        sensitive_fields={k: v.value for k, v in sensitive.items()},
        missing_cells=missing_cells,
        missing_rate=missing_cells / total_cells if total_cells else 0.0,
    )


@router.post("/quality", response_model=QualityResponse, summary="数据质量报告")
async def quality(upload: UploadFile) -> QualityResponse:
    """Step 3 of the wizard: composite score plus one-click suggestions."""
    import pandas as pd

    frame = pd.DataFrame(_read_upload(upload))
    detection = detect_label_column(frame)
    counts = (
        label_counts(frame, detection.column, detection.mapping)
        if detection.column
        else {d.value: 0 for d in Decision}
    )

    masked, masking_report = apply_masking(frame)
    report = build_report(masked, counts, masked_fields=tuple(masking_report.masked_fields))

    return QualityResponse(
        score=report.score,
        grade="良好" if report.score >= 80 else "可接受" if report.score >= 60 else "偏差",
        dimensions=[
            {"name": d.name, "value": d.value, "verdict": d.verdict, "detail": d.detail}
            for d in report_dimensions(report, counts)
        ],
        label_distribution=report.label_distribution,
        anomalies=report.anomalies,
        masked_fields=report.masked_fields,
        suggestions=report.suggestions,
    )


@router.post("/split", response_model=SplitResponse, summary="数据划分 7:1.5:1.5")
async def split(upload: UploadFile) -> SplitResponse:
    """Fixed 7:1.5:1.5 split.

    The caller cannot pass in their own split. There is one implementation of
    this rule and it is the one that enforces the no-synthetic-in-test
    constraint.
    """
    import pandas as pd

    frame = with_origin(pd.DataFrame(_read_upload(upload)), DataOrigin.SEED.value)
    detection = detect_label_column(frame)

    if not detection.column:
        raise HTTPException(
            status_code=422,
            detail="未能识别标签列，请提供 black/white/gray 三分类标签",
        )

    frame = frame.rename(columns={detection.column: "label"})
    result = stratified_split(frame)

    return SplitResponse(
        train_rows=result.summary.train_rows,
        valid_rows=result.summary.valid_rows,
        test_rows=result.summary.test_rows,
        test_contains_synth=result.summary.test_contains_synth,
        label_distribution=result.summary.label_distribution,
        undersized_groups=result.undersized_groups,
    )


@router.post("/recommend-synth", response_model=SynthRecommendationResponse, summary="合成参数推荐")
async def recommend_synth(upload: UploadFile) -> SynthRecommendationResponse:
    """Step 4 of the wizard: pre-filled synthesis parameters (需求方案.txt 5.4)."""
    import pandas as pd

    frame = pd.DataFrame(_read_upload(upload))
    detection = detect_label_column(frame)
    counts = (
        label_counts(frame, detection.column, detection.mapping)
        if detection.column
        else {d.value: 0 for d in Decision}
    )

    recommendation = recommend(counts, seed_rows=len(frame))
    return SynthRecommendationResponse(
        method=recommendation.method.value,
        total_rows=recommendation.total_rows,
        black_white_ratio=recommendation.black_white_ratio,
        augment_label=recommendation.augment_label.value if recommendation.augment_label else None,
        augment_rows=recommendation.augment_rows,
        reasons=list(recommendation.reasons),
        describe=recommendation.describe(),
    )


@router.post("/synth", response_model=SynthResponse, summary="智能数据合成")
async def synth(
    upload: UploadFile,
    method: SynthMethod = SynthMethod.DISTRIBUTION_FIT,
    target_rows: int = 1_000,
    augment_label: Decision | None = None,
    target_ratio: float | None = None,
) -> SynthResponse:
    """Step 4 of the wizard: generate samples and report fidelity + privacy."""
    import pandas as pd

    frame = with_origin(pd.DataFrame(_read_upload(upload)), DataOrigin.SEED.value)
    detection = detect_label_column(frame)
    if detection.column:
        frame = frame.rename(columns={detection.column: "label"})

    categorical = tuple(c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c]))
    request = SynthRequest(
        target_rows=target_rows,
        method=method,
        categorical_columns=categorical,
        augment_label=augment_label,
        target_ratio=target_ratio,
    )

    result = synthesize(frame, request)
    fidelity = assess_fidelity(frame, result.frame)
    privacy = assess_privacy(frame, result.frame)

    return SynthResponse(
        method=result.method.value,
        rows=result.rows,
        label_distribution=result.label_counts,
        fidelity_score=fidelity.score,
        fidelity_verdict=fidelity.verdict,
        fidelity_notes=list(fidelity.notes),
        privacy_lines=privacy.describe(),
        reversible_risk=privacy.reversible_risk,
        all_rows_marked_synthetic=True,
    )
