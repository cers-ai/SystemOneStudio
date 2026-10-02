"""Data quality report and recommendations.

需求方案.txt 5.3 asks for a composite score the user can read at a glance, plus
per-dimension verdicts, anomaly detection, a masking check and one-click
suggestions.

The composite score is ours, not the requirement's: 5.3 shows an example of
"87 分（良好）" with no formula behind it. The weights below are provisional and
labelled as such. What the requirement does fix are the dimensions, the 7:1.5:1.5
split, and the "apply suggestions" affordance, and those are implemented as
written.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from son_contracts import Decision, QualityReport

#: Rows at or above which sample volume is reported as sufficient.
#: Provisional: the requirement states "✓ 充足" without a threshold.
SUFFICIENT_ROWS = 10_000

#: Missing-rate ceilings, matching the "✓ 良好" verdict in 5.3.
MISSING_GOOD = 0.01
MISSING_ACCEPTABLE = 0.05

#: Weights of the composite score. Provisional, see module docstring.
WEIGHT_VOLUME = 0.30
WEIGHT_COMPLETENESS = 0.35
WEIGHT_BALANCE = 0.35


@dataclass(frozen=True)
class Dimension:
    """One scored dimension with the verdict shown next to it in the UI."""

    name: str
    value: str
    score: float
    verdict: str
    detail: str = ""


def volume_score(rows: int) -> tuple[float, str]:
    score = min(rows / SUFFICIENT_ROWS, 1.0)
    verdict = "充足" if rows >= SUFFICIENT_ROWS else "偏少"
    return score, verdict


def completeness_score(missing_rate: float | None) -> tuple[float, str]:
    """Completeness verdict.

    ``missing_rate=None`` means "not measurable", which is distinct from 0.0: an
    empty upload used to score a perfect 1.0 ("良好") for a dimension nobody
    measured.
    """
    if missing_rate is None:
        return 0.0, "无数据"
    score = max(0.0, 1.0 - missing_rate * 10)
    if missing_rate <= MISSING_GOOD:
        verdict = "良好"
    elif missing_rate <= MISSING_ACCEPTABLE:
        verdict = "可接受"
    else:
        verdict = "偏差"
    return score, verdict


def balance_score(counts: dict[str, int]) -> tuple[float, str, float | None]:
    """Score how evenly black and white are represented.

    Gray is deliberately excluded: it is a third outcome, not a minority class
    to be balanced away. Imbalance is measured as (max-min)/(max+min) over the
    black/white counts, so 2:1 scores 0.667 and 1:1 scores 1.0.

    The ratio is None rather than 0 when it cannot be computed, because a
    single-class dataset is not "balanced" and reporting 0 would read that way.
    """
    black = counts.get(Decision.BLACK.value, 0)
    white = counts.get(Decision.WHITE.value, 0)
    total = black + white
    if total == 0:
        return 0.0, "无黑白样本", None
    if min(black, white) == 0:
        return 0.0, "缺少单侧样本", None

    ratio = max(black, white) / min(black, white)
    score = max(0.0, 1.0 - (max(black, white) - min(black, white)) / total)
    verdict = "均衡" if ratio <= 1.5 else "不均衡" if ratio <= 5 else "严重不均衡"
    return score, verdict, ratio


def detect_anomalies(
    frame: pd.DataFrame,
    *,
    modified_z_threshold: float = 3.5,
    degenerate_band_ratio: float = 10.0,
    max_reported_columns: int = 10,
) -> list[str]:
    """Outlier detection on numeric columns.

    需求方案.txt 5.3 shows "金额字段有 23 个异常值（> 100 万）", i.e. outliers
    making up ~30% of the column. That is well past the ~5% contamination the
    IQR rule assumes, and IQR fails on it outright: with 50 values at 1000 and
    23 at 2,000,000, the third quartile lands inside the outlier block, so the
    outlier-inflated IQR swallows the very rows we are looking for.

    So the primary statistic is the modified z-score (Iglewicz & Hoaglin):
    ``0.6745 * (x - median) / MAD``. It is robust to nearly 50% contamination.

    When the MAD collapses to zero -- a column that is one value plus a thin
    outlier tail -- no robust spread can help, because the outliers are the only
    source of spread. That case falls back to a multiplicative band around the
    median, which is what catches the requirement's example.
    """
    findings: list[str] = []
    numeric = list(frame.select_dtypes(include="number").columns)[:max_reported_columns]

    for column in numeric:
        series = frame[column].dropna()
        if len(series) < 4:
            continue

        median = float(series.median())
        mad = float((series - median).abs().median())
        scale = 1.4826 * mad

        if scale > 0:
            scores = (0.6745 * (series - median) / scale).abs()
            outliers = series[scores > modified_z_threshold]
            detail = f"修正 z 分数 > {modified_z_threshold}"
        elif median != 0:
            low, high = median / degenerate_band_ratio, median * degenerate_band_ratio
            outliers = series[(series < low) | (series > high)]
            detail = f"偏离中位数 {median:.0f} 超过 {degenerate_band_ratio:g} 倍"
        else:
            # Median is zero, so a multiplicative band is meaningless. Every
            # non-zero value is a deviation.
            outliers = series[series != median]
            detail = "存在非零值"

        if len(outliers):
            findings.append(f"{column} 有 {len(outliers)} 个异常值（{detail}）")
    return findings


def build_suggestions(
    counts: dict[str, int],
    missing_rate: float,
    anomalies: list[str],
) -> list[str]:
    """The one-click actions offered in 需求方案.txt 5.3."""
    suggestions: list[str] = []

    black = counts.get(Decision.BLACK.value, 0)
    white = counts.get(Decision.WHITE.value, 0)
    if black and white and max(black, white) / min(black, white) > 1.5:
        suggestions.append("黑白样本不均衡，建议在数据合成阶段扩增占比低的一类")

    if missing_rate > MISSING_GOOD:
        suggestions.append(f"缺失率 {missing_rate:.2%} 偏高，建议填充或剔除缺失行")

    if anomalies:
        suggestions.append(f"{len(anomalies)} 个字段存在异常值，建议剔除或修正")

    if counts.get(Decision.GRAY.value, 0) == 0:
        suggestions.append("未发现灰样本，建议补充待定样本以覆盖三分类")

    return suggestions


def build_report(
    frame: pd.DataFrame,
    counts: dict[str, int],
    *,
    masked_fields: tuple[str, ...] = (),
) -> QualityReport:
    """Build the composite quality report for 需求方案.txt 5.3."""
    rows = len(frame)
    cells = frame.size
    if rows == 0 or cells == 0:
        # Nothing to analyse. Report zero rather than a score that reads as if
        # an analysis had been performed on an empty upload.
        return QualityReport(
            score=0.0,
            sample_count=0,
            missing_rate=0.0,
            black_white_ratio=None,
            label_distribution={},
            anomalies=[],
            masked_fields=list(masked_fields),
            suggestions=["数据为空，请先上传样本数据"],
        )

    missing_cells = int(frame.isna().sum().sum())
    missing_rate = missing_cells / cells

    v_score, _ = volume_score(rows)
    c_score, _ = completeness_score(missing_rate)
    b_score, _, ratio = balance_score(counts)
    anomalies = detect_anomalies(frame)

    composite = 100 * (
        WEIGHT_VOLUME * v_score + WEIGHT_COMPLETENESS * c_score + WEIGHT_BALANCE * b_score
    )

    return QualityReport(
        score=round(composite, 1),
        sample_count=rows,
        missing_rate=round(missing_rate, 6),
        black_white_ratio=round(ratio, 4) if ratio is not None else None,
        label_distribution=dict(counts),
        anomalies=anomalies,
        masked_fields=list(masked_fields),
        suggestions=build_suggestions(counts, missing_rate, anomalies),
    )


def report_dimensions(
    report: QualityReport,
    counts: dict[str, int],
) -> list[Dimension]:
    """Per-dimension breakdown, mirroring the three cards in 需求方案.txt 5.3."""
    v_score, v_verdict = volume_score(report.sample_count)
    c_score, c_verdict = completeness_score(report.missing_rate)
    b_score, b_verdict, ratio = balance_score(counts)
    return [
        Dimension("样本量", f"{report.sample_count:,}", v_score, v_verdict),
        Dimension(
            "缺失率",
            f"{report.missing_rate:.2%}",
            c_score,
            c_verdict,
            detail=f"{report.sample_count} 行" if report.sample_count else "",
        ),
        Dimension(
            "黑白比",
            f"{ratio:.2f}:1" if ratio is not None else "不可比",
            b_score,
            b_verdict,
            detail=(
                f"黑 {counts.get(Decision.BLACK.value, 0)} / "
                f"白 {counts.get(Decision.WHITE.value, 0)} / "
                f"灰 {counts.get(Decision.GRAY.value, 0)}"
            ),
        ),
    ]
