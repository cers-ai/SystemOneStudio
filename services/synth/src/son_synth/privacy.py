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
    """Duplicate and memorisation checks (需求方案.txt 5.4)."""

    duplicates_found: int
    nearest_neighbour_distance: float
    nn_verdict: NNVerdict
    reversible_risk: RiskLevel
    notes: tuple[str, ...] = ()

    def describe(self) -> list[str]:
        clean = self.duplicates_found == 0
        safe = self.nn_verdict == "safe"
        no_risk = self.reversible_risk == "none"
        return [
            f"{'✓' if clean else '⚠'} 去重检查"
            + ("通过" if clean else f"：{self.duplicates_found} 条重复"),
            f"{'✓' if safe else '⚠'} 最近邻距离：平均 {self.nearest_neighbour_distance:.2f}"
            f"（{'安全' if safe else _NN_VERDICT_LABEL[self.nn_verdict]}）",
            f"{'✓' if no_risk else '⚠'} "
            + ("无可逆还原风险" if no_risk else f"可逆风险：{self.reversible_risk}"),
        ]


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


def assess_fidelity(seed_frame: pd.DataFrame, synth_frame: pd.DataFrame) -> FidelityReport:
    """Compare synthetic columns against their seed counterparts."""
    shared = [c for c in synth_frame.columns if c in seed_frame.columns and c != "origin"]
    if not shared:
        return FidelityReport(score=0.0, verdict="poor", notes=("没有可对比的字段",))

    per_column = {c: _column_fidelity(seed_frame[c], synth_frame[c]) for c in shared}
    label_column = "label"
    numeric = [v for c, v in per_column.items() if c != label_column]
    score = float(np.mean(numeric)) if numeric else float(np.mean(list(per_column.values())))

    notes: list[str] = []
    weak = [c for c, v in per_column.items() if v < FIDELITY_ACCEPTABLE]
    if weak:
        notes.append(f"以下字段分布偏差较大：{', '.join(weak)}")

    return FidelityReport(
        score=round(score, 4),
        verdict=_verdict_for(score),
        per_column={k: round(v, 4) for k, v in per_column.items()},
        notes=tuple(notes),
    )


def find_duplicates(seed_frame: pd.DataFrame, synth_frame: pd.DataFrame) -> int:
    """Count synthetic rows that exactly reproduce a seed row."""
    shared = [c for c in synth_frame.columns if c in seed_frame.columns and c != "origin"]
    if not shared:
        return 0
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
) -> float:
    """Mean distance from each synthetic row to its closest seed row.

    Small distance means the synthetic row sits almost on top of a real record,
    which is a memorisation risk rather than a quality win. Normalised per
    column so unit differences do not dominate.
    """
    shared = [c for c in synth_frame.columns if c in seed_frame.columns and c != "origin"]
    numeric = [
        c
        for c in shared
        if pd.api.types.is_numeric_dtype(seed_frame[c])
        and pd.api.types.is_numeric_dtype(synth_frame[c])
    ]
    if not numeric:
        return 1.0

    seed_rows = seed_frame[numeric].dropna()
    synth_rows = synth_frame[numeric].dropna()
    if seed_rows.empty or synth_rows.empty:
        return 1.0

    if len(seed_rows) > max_rows:
        seed_rows = seed_rows.sample(max_rows, random_state=seed)
    if len(synth_rows) > sample:
        synth_rows = synth_rows.sample(sample, random_state=seed)

    scales = seed_rows.std(ddof=0).replace(0, 1).fillna(1)
    seed_norm = (seed_rows / scales).to_numpy(dtype=float)
    synth_norm = (synth_rows / scales).to_numpy(dtype=float)

    distances = []
    for row in synth_norm:
        distances.append(float(np.sqrt(((seed_norm - row) ** 2).sum(axis=1)).min()))
    return float(np.mean(distances))


def assess_privacy(
    seed_frame: pd.DataFrame,
    synth_frame: pd.DataFrame,
    *,
    seed: int = 42,
) -> PrivacyReport:
    """Run the duplicate and memorisation checks."""
    duplicates = find_duplicates(seed_frame, synth_frame)
    distance = nearest_neighbour_distance(seed_frame, synth_frame, seed=seed)

    if distance >= NN_DISTANCE_SAFE:
        nn_verdict: NNVerdict = "safe"
    elif distance >= NN_DISTANCE_REVIEW:
        nn_verdict = "review"
    else:
        nn_verdict = "poor"

    if duplicates == 0 and nn_verdict == "safe":
        risk: RiskLevel = "none"
    elif duplicates > 0 or nn_verdict == "poor":
        risk = "high"
    else:
        risk = "review"

    notes: list[str] = []
    if duplicates:
        notes.append(f"{duplicates} 条合成样本与种子数据完全重复")
    if nn_verdict != "safe":
        notes.append("合成样本与真实样本过于接近，存在记忆风险")

    return PrivacyReport(
        duplicates_found=duplicates,
        nearest_neighbour_distance=round(distance, 4),
        nn_verdict=nn_verdict,
        reversible_risk=risk,
        notes=tuple(notes),
    )
