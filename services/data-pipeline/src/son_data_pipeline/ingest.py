"""Ingestion and label column detection.

需求方案.txt 5.2: on upload, auto-detect the label column, sensitive fields and
missing values. Detection is heuristic and every decision is reported back, so
the user can override it instead of trusting a guess they cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import pandas as pd

from son_contracts import Decision
from son_data_pipeline.masking import SensitiveKind, detect_sensitive_columns

#: Column names that commonly hold the ground-truth label.
_LABEL_NAME_HINTS: tuple[str, ...] = (
    "label",
    "labels",
    "target",
    "y",
    "class",
    "decision",
    "is_fraud",
    "fraud",
    "标签",
    "结果",
    "判定",
)

#: Chinese and English surface forms of the three-way decision.
_LABEL_ALIASES: dict[str, Decision] = {
    "黑": Decision.BLACK,
    "黑样本": Decision.BLACK,
    "涉诈": Decision.BLACK,
    "black": Decision.BLACK,
    "1": Decision.BLACK,
    "白": Decision.WHITE,
    "白样本": Decision.WHITE,
    "正常": Decision.WHITE,
    "white": Decision.WHITE,
    "0": Decision.WHITE,
    "灰": Decision.GRAY,
    "灰样本": Decision.GRAY,
    "待定": Decision.GRAY,
    "gray": Decision.GRAY,
    "grey": Decision.GRAY,
    "-1": Decision.GRAY,
}


#: A label takes a handful of distinct values, so a column with more is a
#: feature column rather than a label column.
MAX_LABEL_CARDINALITY = 20


class SourceKind(StrEnum):
    CSV = "csv"
    EXCEL = "excel"
    DATABASE = "database"


@dataclass(frozen=True)
class LabelDetection:
    """Outcome of label column inference."""

    column: str | None
    mapping: dict[str, Decision]
    unmapped_values: tuple[str, ...] = ()
    confidence: Literal["high", "medium", "low"] = "low"
    reason: str = ""


@dataclass
class IngestResult:
    """Everything the upload step reports back (需求方案.txt 5.2)."""

    frame: pd.DataFrame
    source: SourceKind
    label: LabelDetection
    sensitive_columns: dict[str, SensitiveKind]
    missing_cells: int
    missing_by_column: dict[str, int] = field(default_factory=dict)

    @property
    def missing_rate(self) -> float:
        total = self.frame.size
        return self.missing_cells / total if total else 0.0

    @property
    def preview(self) -> pd.DataFrame:
        """需求方案.txt 5.2 asks for a preview; 5.3 shows the first 5 rows."""
        return self.frame.head(5)


def normalize_label_value(value: Any) -> Decision | None:
    text = str(value).strip().lower()
    return _LABEL_ALIASES.get(text)


def _looks_categorical(series: pd.Series, *, name_hinted: bool) -> bool:
    """Whether a column is plausible as a label column.

    Two gates:

    * absolute cardinality -- a label takes a handful of distinct values, while
      a feature column of amounts can have thousands
    * dtype -- a numeric column is only considered when its name is
      label-like. Without this, a feature column whose values happen to include
      "0" or "1" matches the alias table and gets reported as the label.
    """
    if series.dropna().empty:
        return False
    if series.nunique(dropna=True) > MAX_LABEL_CARDINALITY:
        return False
    return name_hinted or not pd.api.types.is_numeric_dtype(series)


def detect_label_column(frame: pd.DataFrame) -> LabelDetection:
    """Infer the label column.

    Priority order, strongest evidence first:

    1. a categorical column whose values are all recognizable decisions
    2. a label-like name, reported at low confidence if its values are unknown
    3. nothing, so the user picks it
    """
    candidates: list[tuple[str, dict[str, Decision], tuple[str, ...], int]] = []

    for column in frame.columns:
        name = str(column)
        series = frame[column].dropna()
        is_hinted = any(hint in name.lower() for hint in _LABEL_NAME_HINTS)
        if not _looks_categorical(series, name_hinted=is_hinted):
            continue

        distinct = series.unique()
        mapping: dict[str, Decision] = {}
        unmapped: list[str] = []
        for raw in distinct:
            decision = normalize_label_value(raw)
            if decision is None:
                unmapped.append(str(raw))
            else:
                mapping[str(raw)] = decision
        if not mapping:
            continue
        # Coverage is over distinct values, not rows: dividing by the row count
        # would report 2 labels in 1500 rows as 0% coverage.
        coverage = len(mapping) / len(distinct)
        candidates.append((name, mapping, tuple(unmapped), int(coverage * 100)))

    if candidates:
        name_hinted = [
            c for c in candidates if any(hint in c[0].lower() for hint in _LABEL_NAME_HINTS)
        ]
        pool = name_hinted or candidates
        # Distinct names: `unmapped` is already bound to a list by the loop above.
        best_column, best_mapping, best_unmapped, best_score = max(pool, key=lambda c: c[3])
        is_hinted = any(hint in best_column.lower() for hint in _LABEL_NAME_HINTS)
        confidence: Literal["high", "medium", "low"] = (
            "high" if is_hinted and best_score == 100 else "medium" if best_score >= 90 else "low"
        )
        return LabelDetection(
            column=best_column,
            mapping=best_mapping,
            unmapped_values=best_unmapped,
            confidence=confidence,
            reason=(
                f"column '{best_column}' has {best_score}% recognizable decision values"
                + (" and a label-like name" if is_hinted else "")
            ),
        )

    named = [c for c in frame.columns if any(h in str(c).lower() for h in _LABEL_NAME_HINTS)]
    if named:
        return LabelDetection(
            column=str(named[0]),
            mapping={},
            confidence="low",
            reason=f"column '{named[0]}' has a label-like name but its values are not recognized",
        )

    return LabelDetection(
        column=None,
        mapping={},
        confidence="low",
        reason="no column contains recognizable black/white/gray values",
    )


def read_table(source: str, *, kind: SourceKind = SourceKind.CSV) -> pd.DataFrame:
    """Load a CSV or Excel file into a frame."""
    if kind is SourceKind.CSV:
        return pd.read_csv(source)
    if kind is SourceKind.EXCEL:
        return pd.read_excel(source)
    raise ValueError(f"database connections are wired up in M1's api layer, not here: {kind}")


def ingest(source: str, *, kind: SourceKind = SourceKind.CSV) -> IngestResult:
    """Load a dataset and run the full detection pass."""
    frame = read_table(source, kind=kind)
    missing_by_column = {str(c): int(frame[c].isna().sum()) for c in frame.columns}
    return IngestResult(
        frame=frame,
        source=kind,
        label=detect_label_column(frame),
        sensitive_columns=detect_sensitive_columns(frame),
        missing_cells=int(frame.isna().sum().sum()),
        missing_by_column=missing_by_column,
    )


def with_origin(
    frame: pd.DataFrame,
    origin: str,
    *,
    column: str = "origin",
) -> pd.DataFrame:
    """Stamp row-level provenance.

    This is what makes the synth-rows-never-enter-test invariant enforceable
    rather than conventional: the split layer filters on this column, and a
    missing or unknown origin is a hard error, not a default.
    """
    from son_contracts import DataOrigin

    frame = frame.copy()
    frame[column] = DataOrigin(origin).value
    return frame


def label_counts(
    frame: pd.DataFrame, label_column: str | None, mapping: dict[str, Decision]
) -> dict[str, int] | None:
    """Count rows per decision, or None when there is no label column.

    Returning an all-zero dict for a missing column made "no label column"
    indistinguishable from "every class happens to be empty", and the
    suggestions layer then reported gray samples as absent as if it had
    inspected the data.
    """
    if not label_column or label_column not in frame.columns:
        return None
    counts = {d.value: 0 for d in Decision}
    for raw in frame[label_column]:
        decision = mapping.get(str(raw))
        if decision is not None:
            counts[decision.value] += 1
    return counts
    return counts
