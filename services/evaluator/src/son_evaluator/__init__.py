"""Evaluation: performance block first, effect block second.

需求方案.txt 9.1 / 9.2 are two separate blocks with a fixed priority order, and
they are stored and reported separately here. Merging them would lose the one
architectural fact the whole inference choice rests on: latency is the binding
constraint, effect metrics come second.

Verified locally: the whole effect block, and performance *aggregation*.
Not verified: any actual latency or memory measurement -- that needs a GPU node
(see AGENTS.md).
"""

from son_evaluator.effect import (
    ClassMetrics,
    EffectReport,
    accuracy_score,
    binary_auc_roc,
    build_effect_report,
    confusion_matrix,
    false_kill_rate,
    feature_importance,
    macro_f1,
    per_label_metrics,
)
from son_evaluator.performance import (
    DISPUTED_TARGETS,
    LatencySample,
    PerformanceReport,
    build_performance_report,
    check_target,
    percentile,
)

__all__ = [
    "DISPUTED_TARGETS",
    "ClassMetrics",
    "EffectReport",
    "LatencySample",
    "PerformanceReport",
    "accuracy_score",
    "binary_auc_roc",
    "build_effect_report",
    "build_performance_report",
    "check_target",
    "confusion_matrix",
    "false_kill_rate",
    "feature_importance",
    "macro_f1",
    "per_label_metrics",
    "percentile",
]
