"""Effect metrics (需求方案.txt 9.2, second priority).

These are pure functions over labelled predictions, so they run and are verified
without a GPU. The whole block is computed here; nothing is delegated.

Two decisions that follow from the product's shape rather than from statistics
convention:

* This is **three-way**. `gray` gets its own precision/recall. Averaging it away
  would let a model look competent by never hedging.
* ``false_kill_rate`` is defined against ``white`` only: killing a normal
  account is the error the fraud scenario actually cares about (需求方案.txt 1.2
  and 14.3). Killing a ``gray`` sample is not a false kill.

All metrics are reported alongside their target so the UI can show ✓/⚠ without
re-deriving thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from son_contracts import EFFECT_TARGETS, Decision


@dataclass(frozen=True)
class ClassMetrics:
    """Per-class precision / recall / F1 / support."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class EffectReport:
    """The second-priority block of an evaluation (需求方案.txt 9.2)."""

    accuracy: float
    macro_f1: float
    auc_roc: float | None
    false_kill_rate: float
    format_compliance: float
    per_label: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    sample_count: int = 0
    notes: tuple[str, ...] = ()

    def target_check(self) -> dict[str, dict[str, float | bool]]:
        """Metric vs target, so the UI never re-implements the thresholds."""
        pairs = {
            "accuracy": (self.accuracy, EFFECT_TARGETS["accuracy"], "min"),
            "macro_f1": (self.macro_f1, EFFECT_TARGETS["f1"], "min"),
            "false_kill_rate": (self.false_kill_rate, EFFECT_TARGETS["false_kill_rate"], "max"),
            "format_compliance": (
                self.format_compliance,
                EFFECT_TARGETS["format_compliance"],
                "min",
            ),
        }
        if self.auc_roc is not None:
            pairs["auc_roc"] = (self.auc_roc, EFFECT_TARGETS["auc_roc"], "min")

        result: dict[str, dict[str, float | bool]] = {}
        for name, (value, target, direction) in pairs.items():
            passed = value >= target if direction == "min" else value <= target
            result[name] = {"value": round(value, 4), "target": target, "passed": passed}
        return result


def confusion_matrix(
    y_true: tuple[Decision, ...],
    y_pred: tuple[Decision, ...],
) -> dict[str, dict[str, int]]:
    """Count matrix over the three decisions."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    matrix = {t.value: {p.value: 0 for p in Decision} for t in Decision}
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[truth.value][pred.value] += 1
    return matrix


def per_label_metrics(matrix: dict[str, dict[str, int]]) -> dict[str, ClassMetrics]:
    """Precision / recall / F1 for each decision."""
    out: dict[str, ClassMetrics] = {}
    for label in Decision:
        tp = matrix[label.value][label.value]
        fn = sum(matrix[label.value].values()) - tp
        fp = sum(row[label.value] for row in matrix.values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label.value] = ClassMetrics(
            label=label.value,
            precision=precision,
            recall=recall,
            f1=f1,
            support=tp + fn,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
    return out


def accuracy_score(y_true: tuple[Decision, ...], y_pred: tuple[Decision, ...]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t is p)
    return correct / len(y_true)


def false_kill_rate(matrix: dict[str, dict[str, int]]) -> float:
    """Share of ``white`` samples wrongly judged ``black``.

    误杀率 in 需求方案.txt 9.2 is a false-positive rate on the white class.
    Killing a gray sample is not counted: gray means "insufficient evidence",
    and rejecting it is not the harm the metric is protecting against.
    """
    white_total = sum(matrix[Decision.WHITE.value].values())
    if white_total == 0:
        return 0.0
    return matrix[Decision.WHITE.value][Decision.BLACK.value] / white_total


def binary_auc_roc(
    y_true: tuple[Decision, ...],
    scores: tuple[float, ...],
    *,
    positive: Decision = Decision.BLACK,
) -> float | None:
    """AUC-ROC for one class treated as positive.

    Returns None when the task is not binary in practice -- i.e. when one class
    is absent -- rather than reporting a number that means nothing.
    """
    if len(y_true) != len(scores):
        raise ValueError("scores and labels must be the same length")

    positives = sum(1 for t in y_true if t is positive)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(
        (
            (score, 1 if truth is positive else 0)
            for score, truth in zip(scores, y_true, strict=True)
        ),
        key=lambda p: p[0],
    )

    # Mann-Whitney U with tie correction.
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            if pairs[position][1] == 1:
                rank_sum += average_rank
        index = end + 1

    u = rank_sum - positives * (positives + 1) / 2
    return u / (positives * negatives)


def macro_f1(per_label: dict[str, ClassMetrics]) -> float:
    present = [m for m in per_label.values() if m.support > 0]
    if not present:
        return 0.0
    return sum(m.f1 for m in present) / len(present)


def build_effect_report(
    y_true: tuple[Decision, ...],
    y_pred: tuple[Decision, ...],
    *,
    scores: tuple[float, ...] | None = None,
    format_compliant: int = 0,
    format_total: int = 0,
) -> EffectReport:
    """Assemble the whole second-priority block."""
    matrix = confusion_matrix(y_true, y_pred)
    per_label = per_label_metrics(matrix)

    notes: list[str] = []
    absent = [d.value for d in Decision if per_label[d.value].support == 0]
    if absent:
        notes.append(f"测试集中缺少 {', '.join(absent)} 样本，其指标不计入宏平均")

    if scores is None:
        notes.append("未提供风险评分，AUC-ROC 无法计算")
        auc: float | None = None
    else:
        auc = binary_auc_roc(y_true, scores)
        if auc is None:
            notes.append("测试集只有单一类别，AUC-ROC 不适用")

    compliance = format_compliant / format_total if format_total else 0.0

    return EffectReport(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=macro_f1(per_label),
        auc_roc=auc,
        false_kill_rate=false_kill_rate(matrix),
        format_compliance=compliance,
        per_label=per_label,
        confusion=matrix,
        sample_count=len(y_true),
        notes=tuple(notes),
    )


def feature_importance(
    contributions: dict[str, float],
    *,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Top-N key decision factors (需求方案.txt 9.3).

    Returned as a ranking, not a chart config: the UI decides how to draw it and
    relabels it as 关键判定因素 (see libs/ui-terminology).
    """
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    return [(name, round(weight, 4)) for name, weight in ranked[:top_n]]
