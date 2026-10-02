"""Fidelity and privacy reporting (需求方案.txt 5.4).

The requirement asks for three things after a synthesis run: a fidelity score
comparing seed and synthetic distributions, a duplicate check, and a nearest
neighbour distance as the privacy signal.

Threshold honesty: 需求方案.txt shows illustrative values (保真度 0.92,
最近邻距离 0.87) but never states what counts as acceptable. The gates below are
therefore advisory and report their numbers; they do not block. Technical
方案.md Q2 tracks getting real thresholds. Blocking training on an invented
number would be worse than reporting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Verdict = Literal["excellent", "good", "acceptable", "poor"]

#: Verdict on the nearest-neighbour privacy signal, distinct from the fidelity
#: verdict vocabulary above.
NNVerdict = Literal["safe", "review", "poor"]
RiskLevel = Literal["none", "review", "high"]

#: Illustrative bounds, not product decisions. See module docstring.
FIDELITY_EXCELLENT = 0.90
FIDELITY_GOOD = 0.80
FIDELITY_ACCEPTABLE = 0.65

NN_DISTANCE_SAFE = 0.50
NN_DISTANCE_REVIEW = 0.20


@dataclass
class FidelityReport:
    """How close the synthetic distribution is to the seed distribution."""

    score: float
    verdict: Verdict
    per_column: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        return f"保真度评分：{self.score:.2f}（{_VERDICT_LABEL[self.verdict]}）"


@dataclass
class PrivacyReport:
    """Duplicate and memorisation checks (需求方案.txt 5.4).

    ``duplicate_check_ran`` and ``nn_check_ran`` exist because a check that
    could not execute must not render as a green tick. Returning 0 duplicates
    and a distance of 1.0 -- the best possible values -- produced
    "✓ 去重检查通过 / ✓ 最近邻距离：平均 1.00（安全）" for a dataset with no
    comparable columns at all.
    """

    duplicates_found: int | None
    nearest_neighbour_distance: float | None
    nn_verdict: NNVerdict
    reversible_risk: RiskLevel
    duplicate_check_ran: bool = True
    nn_check_ran: bool = True
    notes: tuple[str, ...] = ()

    def describe(self) -> list[str]:
        if self.duplicate_check_ran:
            clean = self.duplicates_found == 0
            dup_line = f"{'✓' if clean else '⚠'} 去重检查" + (
                "通过" if clean else f"：{self.duplicates_found} 条重复"
            )
        else:
            dup_line = "⚠ 去重检查未执行（没有可对比的字段）"

        if self.nn_check_ran:
            safe = self.nn_verdict == "safe"
            nn_line = (
                f"{'✓' if safe else '⚠'} 最近邻距离：平均 {self.nearest_neighbour_distance:.2f}"
                f"（{_NN_VERDICT_LABEL[self.nn_verdict]}）"
            )
        else:
            nn_line = "⚠ 最近邻距离未测量（没有可对比的数值字段）"

        no_risk = self.reversible_risk == "none"
        risk_line = f"{'✓' if no_risk else '⚠'} " + (
            "无可逆还原风险" if no_risk else f"可逆风险：{self.reversible_risk}"
        )
        return [dup_line, nn_line, risk_line]


_VERDICT_LABEL: dict[str, str] = {
    "excellent": "优秀",
    "good": "良好",
    "acceptable": "可接受",
    "poor": "偏差",
}

_NN_VERDICT_LABEL: dict[str, str] = {
    "safe": "安全",
    "review": "需复核",
    "poor": "风险过高",
}


def _verdict_for(score: float) -> Verdict:
    if score >= FIDELITY_EXCELLENT:
        return "excellent"
    if score >= FIDELITY_GOOD:
        return "good"
    if score >= FIDELITY_ACCEPTABLE:
        return "acceptable"
    return "poor"


def _column_fidelity(seed: pd.Series, synth: pd.Series) -> float:
    """Agreement between two columns on a 0-1 scale.

    Numeric columns compare sorted value quantiles (distribution shape);
    categorical columns compare the Jensen-Shannon similarity of value
    frequencies. Comparing the same statistic on both sides keeps the score
    interpretable when a dataset mixes the two.
    """
    seed_values = seed.dropna()
    synth_values = synth.dropna()
    if seed_values.empty or synth_values.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(seed_values) and pd.api.types.is_numeric_dtype(synth_values):
        quantiles = np.linspace(0, 1, 21)
        seed_q = seed_values.sort_values().to_numpy()
        synth_q = synth_values.sort_values().to_numpy()
        seed_q = np.quantile(seed_q, quantiles)
        synth_q = np.quantile(synth_q, quantiles)
        scale = float(np.max(np.abs(seed_q))) or 1.0
        return float(max(0.0, 1.0 - np.mean(np.abs(seed_q - synth_q)) / scale))

    seed_freq = seed_values.astype(str).value_counts(normalize=True)
    synth_freq = synth_values.astype(str).value_counts(normalize=True)
    categories = seed_freq.index.union(synth_freq.index)
    p = seed_freq.reindex(categories, fill_value=0.0).to_numpy()
    q = synth_freq.reindex(categories, fill_value=0.0).to_numpy()

    # Jensen-Shannon divergence, base 2, bounded in [0, 1].
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / np.where(m > 0, m, 1)), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / np.where(m > 0, m, 1)), 0.0)
    jsd = float(np.sum(kl_pm) / 2 + np.sum(kl_qm) / 2)
    return float(max(0.0, 1.0 - min(jsd, 1.0)))


def assess_fidelity(
    seed_frame: pd.DataFrame,
    synth_frame: pd.DataFrame,
    *,
    label_column: str | None = "label",
) -> FidelityReport:
    """Compare synthetic columns against their seed counterparts.

    ``label_column`` is excluded from the feature score. It used to be hardcoded
    to "label", which meant a dataset whose label column was named ``is_fraud``
    averaged a perfect label match (1.0) into the score and reported 0.5 for a
    completely broken feature distribution.
    """
    shared = [c for c in synth_frame.columns if c in seed_frame.columns and c != "origin"]
    if not shared:
        return FidelityReport(score=0.0, verdict="poor", notes=("没有可对比的字段",))

    per_column = {c: _column_fidelity(seed_frame[c], synth_frame[c]) for c in shared}
    feature_cols = [c for c in per_column if c != label_column]
    basis = feature_cols or list(per_column)
    score = float(np.mean([per_column[c] for c in basis]))

    notes: list[str] = []
    if label_column and label_column in per_column:
        notes.append(
            f"标签列 {label_column!r} 一致度 {per_column[label_column]:.2f}（未计入特征分布得分）"
        )
    weak = [c for c in basis if per_column[c] < FIDELITY_ACCEPTABLE]
    if weak:
        notes.append(f"以下字段分布偏差较大：{', '.join(weak)}")

    return FidelityReport(
        score=round(score, 4),
        verdict=_verdict_for(score),
        per_column={k: round(v, 4) for k, v in per_column.items()},
        notes=tuple(notes),
    )


def find_duplicates(seed_frame: pd.DataFrame, synth_frame: pd.DataFrame) -> int | None:
    """Count synthetic rows that exactly reproduce a seed row.

    Returns None when there is nothing to compare, which is not the same as
    "found zero duplicates" -- the caller renders those differently.
    """
    shared = [c for c in synth_frame.columns if c in seed_frame.columns and c != "origin"]
    if not shared:
        return None
    seed_keys = {tuple(row) for row in seed_frame[shared].astype(str).to_numpy()}
    synth_keys = [tuple(row) for row in synth_frame[shared].astype(str).to_numpy()]
    return sum(1 for key in synth_keys if key in seed_keys)


def nearest_neighbour_distance(
    seed_frame: pd.DataFrame,
    synth_frame: pd.DataFrame,
    *,
    max_rows: int = 500,
    sample: int = 200,
    seed: int = 42,
) -> float | None:
    """Mean distance from each synthetic row to its closest seed row.

    Small distance means the synthetic row sits almost on top of a real record,
    which is a memorisation risk rather than a quality win. Normalised per column
    so unit differences do not dominate.

    Returns None when it cannot be measured -- no shared numeric columns, or an
    empty side. It used to return 1.0, which is above the "safe" threshold and
    therefore reported a clean bill of health for a check that never ran.
    """
    shared = [c for c in synth_frame.columns if c in seed_frame.columns and c != "origin"]
    numeric = [
        c
        for c in shared
        if pd.api.types.is_numeric_dtype(seed_frame[c])
        and pd.api.types.is_numeric_dtype(synth_frame[c])
    ]
    if not numeric:
        return None

    seed_rows = seed_frame[numeric].dropna()
    synth_rows = synth_frame[numeric].dropna()
    if seed_rows.empty or synth_rows.empty:
        return None

    if len(seed_rows) > max_rows:
        seed_rows = seed_rows.sample(max_rows, random_state=seed)
    if len(synth_rows) > sample:
        synth_rows = synth_rows.sample(sample, random_state=seed)

    scales = seed_rows.std(ddof=0).replace(0, 1).fillna(1)
    seed_norm = (seed_rows / scales).to_numpy(dtype=float)
    synth_norm = (synth_rows / scales).to_numpy(dtype=float)

    distances = [float(np.sqrt(((seed_norm - row) ** 2).sum(axis=1)).min()) for row in synth_norm]
    return float(np.mean(distances))


def assess_privacy(
    seed_frame: pd.DataFrame,
    synth_frame: pd.DataFrame,
    *,
    seed: int = 42,
) -> PrivacyReport:
    """Run the duplicate and memorisation checks.

    A check that could not execute is reported as unexecuted and forces the risk
    level to at least ``review``; it is never reported as clean.
    """
    duplicates = find_duplicates(seed_frame, synth_frame)
    distance = nearest_neighbour_distance(seed_frame, synth_frame, seed=seed)

    dup_ran = duplicates is not None
    nn_ran = distance is not None

    if not nn_ran:
        nn_verdict: NNVerdict = "poor"
    elif distance is not None and distance >= NN_DISTANCE_SAFE:
        nn_verdict = "safe"
    elif distance is not None and distance >= NN_DISTANCE_REVIEW:
        nn_verdict = "review"
    else:
        nn_verdict = "poor"

    duplicates_hit = bool(duplicates)
    if not dup_ran or not nn_ran:
        risk: RiskLevel = "review"
    elif duplicates_hit or nn_verdict == "poor":
        risk = "high"
    elif nn_verdict == "safe":
        risk = "none"
    else:
        risk = "review"

    notes: list[str] = []
    if not dup_ran:
        notes.append("种子数据与合成数据没有可对比的字段，去重检查未执行")
    if not nn_ran:
        notes.append("没有可对比的数值字段，最近邻距离未测量")
    if duplicates_hit:
        notes.append(f"{duplicates} 条合成样本与种子数据完全重复")
    if nn_ran and nn_verdict != "safe":
        notes.append("合成样本与真实样本过于接近，存在记忆风险")

    return PrivacyReport(
        duplicates_found=duplicates,
        nearest_neighbour_distance=None if distance is None else round(distance, 4),
        nn_verdict=nn_verdict,
        reversible_risk=risk,
        duplicate_check_ran=dup_ran,
        nn_check_ran=nn_ran,
        notes=tuple(notes),
    )
