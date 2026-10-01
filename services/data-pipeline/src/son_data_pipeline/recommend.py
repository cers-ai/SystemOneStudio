"""Synthetic dataset parameter recommendations.

需求方案.txt principle 2: every configurable value needs a scene-derived
default, and the business user should not have to touch it. 5.4 shows the
black:white ratio arriving pre-filled as "1 : 50（自动推荐）".

The recommendation is a pure function of the label counts. It is also the
*inverse* of the balance advice in :mod:`son_data_pipeline.quality`: the target
ratio is what the augmented dataset should reach, so it always points at
correcting the current imbalance rather than preserving it.
"""

from __future__ import annotations

from dataclasses import dataclass

from son_contracts import Decision, SynthMethod

#: Upper bound on the recommended ratio. A 1:50 target (as in 5.4) is the
#: extreme end; beyond this the synthetics start to dominate training and the
#: result stops reflecting the real distribution.
MAX_RECOMMENDED_RATIO = 50.0

#: Below this many seed rows of a label, synthesizing more of it produces
#: samples too far from the seed distribution to be trustworthy.
MIN_SEED_ROWS_FOR_AUGMENT = 20


@dataclass(frozen=True)
class SynthRecommendation:
    """Pre-filled synthesis parameters (需求方案.txt 5.4)."""

    method: SynthMethod
    total_rows: int
    black_white_ratio: float
    augment_label: Decision | None
    augment_rows: int
    reasons: tuple[str, ...]

    def describe(self) -> str:
        ratio = (
            f"{self.black_white_ratio:.0f} : 1"
            if self.black_white_ratio >= 1
            else f"1 : {1 / self.black_white_ratio:.0f}"
        )
        return (
            f"{self.method.value} | 合成总量 {self.total_rows:,} | 黑白比例 {ratio} | "
            f"重点扩增 {self.augment_label.value if self.augment_label else '无'}"
        )


def recommend_ratio(counts: dict[str, int]) -> float:
    """Recommended black:white ratio for the augmented dataset.

    Expresses the larger class over the smaller one, so it is always >= 1. Gray
    is excluded: it is a separate outcome and scaling it by the black/white
    imbalance would be meaningless.
    """
    black = counts.get(Decision.BLACK.value, 0)
    white = counts.get(Decision.WHITE.value, 0)
    if black == 0 or white == 0:
        return 1.0
    return max(1.0, min(max(black, white) / min(black, white), MAX_RECOMMENDED_RATIO))


def recommend(
    counts: dict[str, int],
    *,
    seed_rows: int,
    target_total: int | None = None,
) -> SynthRecommendation:
    """Compute the pre-filled synthesis parameters for a scene."""
    reasons: list[str] = []

    black = counts.get(Decision.BLACK.value, 0)
    white = counts.get(Decision.WHITE.value, 0)
    gray = counts.get(Decision.GRAY.value, 0)

    ratio = recommend_ratio(counts)
    minority_label = Decision.WHITE if white < black else Decision.BLACK

    minority_rows = min(black, white)
    if ratio <= 1.5:
        augment_label: Decision | None = None
        reasons.append("黑白样本已均衡，无需定向扩增")
    elif minority_rows < MIN_SEED_ROWS_FOR_AUGMENT:
        augment_label = None
        reasons.append(
            f"占比低的一类仅 {minority_rows} 行，少于 {MIN_SEED_ROWS_FOR_AUGMENT} 行，"
            "定向扩增会偏离种子分布，建议先补充种子数据"
        )
    else:
        augment_label = minority_label
        reasons.append(f"黑白比 {ratio:.1f}:1，建议扩增 {augment_label.value} 以拉平分布")

    if gray == 0:
        reasons.append("未发现灰样本，建议在校验约束中覆盖待定区间")

    total = target_total if target_total is not None else max(seed_rows, SUFFICIENT_SEED_ROWS)
    augment_rows = 0
    if augment_label is not None:
        augment_rows = max(0, total - seed_rows)
        reasons.append(f"在合成阶段新增 {augment_rows:,} 行 {augment_label.value} 样本")

    return SynthRecommendation(
        method=SynthMethod.DISTRIBUTION_FIT,
        total_rows=total,
        black_white_ratio=ratio,
        augment_label=augment_label,
        augment_rows=augment_rows,
        reasons=tuple(reasons),
    )


#: Smallest seed set the recommendation will scale up from.
SUFFICIENT_SEED_ROWS = 5_000
