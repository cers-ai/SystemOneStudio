"""Data preparation: the asset-izing half of step 3 (改造开发方案.md 10).

Everything the wizard used to do in a browser tab now happens once, on the
server, and lands on the workspace with a checksum:

    upload -> dataset/source.csv + datasets row
    prepare -> dataset/{train,valid,test}.csv + dataset_splits row
    synth   -> synth/syn_v1.csv + synth_runs row

The split keeps the hard constraint from 需求方案.txt 5.4 -- no synthetic row in
test -- and the count of any that slipped through is stored in
``test_synth_rows`` so it is auditable rather than merely asserted in passing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from son_data_pipeline import (
    apply_masking,
    build_report,
    detect_label_column,
    label_counts,
    stratified_split,
)
from son_data_pipeline.quality import report_dimensions
from son_data_pipeline.recommend import recommend as recommend_synth
from son_evaluator import build_effect_report  # noqa: F401  (re-exported for callers)
from son_synth import (
    Constraint,
    SynthRequest,
    assess_fidelity,
    assess_privacy,
    synthesize,
)

from son_contracts import DataOrigin, Decision, SynthMethod


class AssetError(RuntimeError):
    """A step could not produce its asset."""


@dataclass(frozen=True)
class PreparedDataset:
    """Result of ingesting an uploaded file."""

    path: Path
    rows: int
    cols: int
    label_column: str | None
    label_mapping: dict[str, str]
    checksum: str
    preview: list[dict[str, Any]]
    sensitive_fields: dict[str, str]
    missing_rate: float
    label_counts: dict[str, int] | None


@dataclass(frozen=True)
class PreparedSplit:
    train_path: Path
    valid_path: Path
    test_path: Path
    train_rows: int
    valid_rows: int
    test_rows: int
    test_synth_rows: int
    quality: dict[str, Any]


@dataclass(frozen=True)
class PreparedSynth:
    path: Path
    rows: int
    checksum: str
    fidelity_score: float | None
    fidelity_verdict: str | None
    privacy: dict[str, Any]
    label_counts: dict[str, int]
    recommendation: dict[str, Any]


ORIGIN_COLUMN = "origin"
LABEL_COLUMN = "label"


def ingest_csv(destination: Path, filename: str) -> PreparedDataset:
    """Write the upload to the workspace and detect what is in it.

    The file is written first and re-read from disk, so every later step operates
    on the artifact rather than on an in-memory copy the caller might mutate.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame = pd.read_csv(destination)
    except Exception as exc:
        raise AssetError(f"无法解析 CSV（{filename}）：{exc}") from exc

    if frame.empty:
        raise AssetError(f"CSV（{filename}）没有任何数据行")

    from son_data_pipeline.masking import detect_sensitive_columns
    from son_db.assets import checksum as file_checksum

    detection = detect_label_column(frame)
    if detection.column:
        frame = frame.rename(columns={detection.column: LABEL_COLUMN})
        frame = _normalize_labels(frame)

    from son_data_pipeline.ingest import with_origin

    frame = with_origin(frame, DataOrigin.SEED.value, column=ORIGIN_COLUMN)
    frame = frame.to_csv(index=False)

    destination.write_text(frame, encoding="utf-8")

    reloaded = pd.read_csv(destination, dtype=str, keep_default_na=False)
    cells = reloaded.size or 1
    missing = int((reloaded == "").sum().sum())

    return PreparedDataset(
        path=destination,
        rows=len(reloaded),
        cols=len(reloaded.columns),
        label_column=detection.column,
        label_mapping={k: v.value for k, v in detection.mapping.items()},
        checksum=file_checksum(destination),
        preview=reloaded.head(5).to_dict("records"),
        sensitive_fields={k: v.value for k, v in detect_sensitive_columns(reloaded).items()},
        missing_rate=missing / cells,
        label_counts=label_counts(reloaded, LABEL_COLUMN, detection.mapping),
    )


def prepare_split(
    dataset_path: Path,
    run_id: str,
    *,
    dataset_dir: Path,
    seed: int = 42,
) -> PreparedSplit:
    """Mask, score and split -- steps 3's single backend action.

    The mask is applied before the split, so the files written to the workspace
    are already de-identified. A masked copy that only existed in memory would
    mean training on data the audit trail claims was scrubbed.
    """
    frame = pd.read_csv(dataset_path, keep_default_na=False)
    if LABEL_COLUMN not in frame.columns:
        raise AssetError("数据缺少标签列，无法划分")

    masked, masking = apply_masking(frame)
    normalized = _normalize_labels(masked)

    detection = detect_label_column(normalized)
    if not detection.column:
        raise AssetError("未能识别三分类标签（black / white / gray）")
    counts = label_counts(normalized, LABEL_COLUMN, detection.mapping) or {
        d.value: 0 for d in Decision
    }

    report = build_report(normalized, counts)
    dimensions = report_dimensions(report, counts)

    split = stratified_split(normalized, seed=seed)
    if split.summary.test_contains_synth:
        raise AssetError("划分结果包含合成数据，违反硬性约束")

    train_path = _write_frame(dataset_dir / "train.csv", split.train)
    valid_path = _write_frame(dataset_dir / "valid.csv", split.valid)
    test_path = _write_frame(dataset_dir / "test.csv", split.test)

    leaked = int((split.test[ORIGIN_COLUMN] == DataOrigin.SYNTH.value).sum())

    return PreparedSplit(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        train_rows=len(split.train),
        valid_rows=len(split.valid),
        test_rows=len(split.test),
        test_synth_rows=leaked,
        quality={
            "score": report.score,
            "label_distribution": report.label_distribution,
            "dimensions": [
                {"name": d.name, "value": d.value, "verdict": d.verdict} for d in dimensions
            ],
            "anomalies": report.anomalies,
            "suggestions": report.suggestions,
            "masked_fields": masking.masked_fields,
            "masking_skipped": list(masking.skipped_fields),
        },
    )


def generate_synth(
    train_path: Path,
    *,
    output_path: Path,
    method: SynthMethod = SynthMethod.DISTRIBUTION_FIT,
    seed: int = 42,
    target_rows: int | None = None,
    constraints: tuple[Constraint, ...] = (),
) -> PreparedSynth:
    """Synthesize from the training split and write the artifact.

    Reads the *train* split, never the test one. The test split is the evaluation
    set and synthetic data must never touch it (需求方案.txt 5.4), so the function
    takes the train path explicitly rather than a dataset id it could pick the
    wrong split from.
    """
    seed_frame = pd.read_csv(train_path, keep_default_na=False)
    if seed_frame.empty:
        raise AssetError("训练集为空，无法合成")

    detection = detect_label_column(seed_frame)
    counts = label_counts(seed_frame, LABEL_COLUMN, detection.mapping) or {
        d.value: 0 for d in Decision
    }

    recommendation = recommend_synth(counts, seed_rows=len(seed_frame))

    categorical = tuple(
        c for c in seed_frame.columns if not pd.api.types.is_numeric_dtype(seed_frame[c])
    )
    rows = recommendation.total_rows if target_rows is None else target_rows
    rows = max(0, min(int(rows), 200_000))

    request = SynthRequest(
        target_rows=rows,
        method=method,
        label_column=LABEL_COLUMN,
        origin_column=ORIGIN_COLUMN,
        categorical_columns=categorical,
        augment_label=recommendation.augment_label,
        target_ratio=recommendation.black_white_ratio or None,
        constraints=constraints,
        seed=seed,
    )

    from son_synth.generators import JointRowGenerator

    if method != SynthMethod.DISTRIBUTION_FIT:
        raise AssetError("本阶段仅支持保留标签与依据的整行重采样扩增")
    frame = (
        synthesize(seed_frame, request, generator=JointRowGenerator()).frame
        if rows
        else seed_frame.iloc[:0].copy()
    )

    from son_db.assets import checksum as file_checksum

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    fidelity = assess_fidelity(seed_frame, frame, label_column=LABEL_COLUMN)
    privacy = assess_privacy(seed_frame, frame)

    return PreparedSynth(
        path=output_path,
        rows=len(frame),
        checksum=file_checksum(output_path),
        fidelity_score=fidelity.score,
        fidelity_verdict=fidelity.verdict,
        privacy={
            "duplicates_found": privacy.duplicates_found,
            "duplicate_check_ran": privacy.duplicate_check_ran,
            "nearest_neighbour_distance": privacy.nearest_neighbour_distance,
            "nn_check_ran": privacy.nn_check_ran,
            "reversible_risk": privacy.reversible_risk,
            "notes": [
                *privacy.notes,
                "整行重采样会产生重复样本，不增加新知识；可选择仅用种子",
            ],
            "lines": privacy.describe(),
        },
        label_counts={str(k): int(v) for k, v in frame[LABEL_COLUMN].value_counts().items()},
        recommendation={
            "method": recommendation.method.value,
            "total_rows": recommendation.total_rows,
            "black_white_ratio": recommendation.black_white_ratio,
            "ratio_computable": recommendation.ratio_computable,
            "augment_label": (
                recommendation.augment_label.value if recommendation.augment_label else None
            ),
            "reasons": list(recommendation.reasons),
        },
    )


def training_frame(
    train_path: Path,
    synth_path: Path | None = None,
) -> list[dict[str, str]]:
    """Concatenate train + synth into the rows training will actually see.

    The test split is not reachable from here by construction: this function
    takes paths, and the only caller that has them is the run's own config.
    """
    rows = pd.read_csv(train_path, dtype=str, keep_default_na=False).to_dict("records")
    if synth_path is not None and synth_path.exists():
        extra = pd.read_csv(synth_path, dtype=str, keep_default_na=False).to_dict("records")
        rows = rows + extra
    return rows


def _normalize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Map every recognised label spelling onto black / white / gray."""
    out = frame.copy()
    if LABEL_COLUMN not in out.columns:
        detection = detect_label_column(out)
        if detection.column:
            out = out.rename(columns={detection.column: LABEL_COLUMN})
    if LABEL_COLUMN in out.columns:
        from son_data_pipeline.ingest import normalize_label_value

        raw = out[LABEL_COLUMN].astype(str)
        values = raw.map(normalize_label_value)
        invalid = raw[values.isna()].unique().tolist()
        if invalid:
            raise AssetError(
                f"标签存在无法识别的值：{invalid[:10]}；请使用 black/white/gray 或 黑/白/灰"
            )
        out[LABEL_COLUMN] = values.map(lambda value: value.value)
    return out


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
