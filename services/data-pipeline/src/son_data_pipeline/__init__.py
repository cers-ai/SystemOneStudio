"""Data pipeline: ingestion, masking, quality scoring, splitting and parameters.

The split invariant (synthetic rows never reach the test split) is the reason
this package exists as more than a CSV reader. See
:mod:`son_data_pipeline.split` for how it is enforced structurally.
"""

from son_data_pipeline.ingest import (
    IngestResult,
    LabelDetection,
    SourceKind,
    detect_label_column,
    ingest,
    label_counts,
    normalize_label_value,
    read_table,
    with_origin,
)
from son_data_pipeline.masking import (
    MASKING_RULES,
    MaskingReport,
    MaskingRule,
    SensitiveKind,
    apply_masking,
    detect_sensitive_columns,
)
from son_data_pipeline.quality import (
    Dimension,
    balance_score,
    build_report,
    build_suggestions,
    completeness_score,
    detect_anomalies,
    report_dimensions,
    volume_score,
)
from son_data_pipeline.recommend import SynthRecommendation, recommend, recommend_ratio
from son_data_pipeline.split import (
    SPLIT_RATIOS,
    SplitResult,
    SyntheticDataLeakError,
    label_share,
    stratified_split,
)

__all__ = [
    "MASKING_RULES",
    "SPLIT_RATIOS",
    "Dimension",
    "IngestResult",
    "LabelDetection",
    "MaskingReport",
    "MaskingRule",
    "SensitiveKind",
    "SourceKind",
    "SplitResult",
    "SynthRecommendation",
    "SyntheticDataLeakError",
    "apply_masking",
    "balance_score",
    "build_report",
    "build_suggestions",
    "completeness_score",
    "detect_anomalies",
    "detect_label_column",
    "detect_sensitive_columns",
    "ingest",
    "label_counts",
    "label_share",
    "normalize_label_value",
    "read_table",
    "recommend",
    "recommend_ratio",
    "report_dimensions",
    "stratified_split",
    "volume_score",
    "with_origin",
]
