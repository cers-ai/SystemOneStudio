"""Tests for label detection, quality scoring and synthesis recommendations."""

from __future__ import annotations

import pandas as pd
import pytest
from son_data_pipeline import (
    balance_score,
    build_report,
    build_suggestions,
    completeness_score,
    detect_anomalies,
    detect_label_column,
    label_counts,
    normalize_label_value,
    recommend,
    recommend_ratio,
    report_dimensions,
    volume_score,
)

from son_contracts import Decision, SynthMethod


class TestLabelNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("黑", Decision.BLACK),
            ("黑样本", Decision.BLACK),
            ("涉诈", Decision.BLACK),
            ("black", Decision.BLACK),
            ("BLACK", Decision.BLACK),
            ("白", Decision.WHITE),
            ("正常", Decision.WHITE),
            ("white", Decision.WHITE),
            ("灰", Decision.GRAY),
            ("待定", Decision.GRAY),
            ("gray", Decision.GRAY),
            ("0", Decision.WHITE),
            ("1", Decision.BLACK),
        ],
    )
    def test_recognizes_spellings(self, raw: str, expected: Decision) -> None:
        assert normalize_label_value(raw) is expected

    def test_unknown_value_is_none(self) -> None:
        assert normalize_label_value("maybe") is None

    def test_surrounding_whitespace_tolerated(self) -> None:
        assert normalize_label_value("  黑  ") is Decision.BLACK


class TestLabelColumnDetection:
    def test_detects_the_label_column(self, seed_frame: pd.DataFrame) -> None:
        detected = detect_label_column(seed_frame)
        assert detected.column == "label"
        assert detected.confidence == "high"

    def test_maps_every_distinct_value(self, seed_frame: pd.DataFrame) -> None:
        assert detect_label_column(seed_frame).unmapped_values == ()

    def test_prefers_a_label_named_column(self) -> None:
        frame = pd.DataFrame({"标签": ["黑", "白", "灰"], "status": ["黑", "白", "灰"]})
        assert detect_label_column(frame).column == "标签"

    def test_reports_unknown_values(self) -> None:
        frame = pd.DataFrame({"label": ["黑", "白", "maybe"]})
        detected = detect_label_column(frame)
        assert "maybe" in detected.unmapped_values
        assert detected.confidence != "high"

    def test_no_recognizable_labels_reports_no_column(self) -> None:
        frame = pd.DataFrame({"amount": [1, 2, 3], "device": ["a", "b", "c"]})
        detected = detect_label_column(frame)
        assert detected.column is None
        assert detected.confidence == "low"

    def test_numeric_feature_column_is_not_mistaken_for_a_label(self) -> None:
        """A column of 0/1/2/3 values is a feature, not a decision column."""
        frame = pd.DataFrame({"amount": [1, 2, 3], "other": [0, 1, 0]})
        assert detect_label_column(frame).column is None

    def test_low_cardinality_three_value_column_is_still_detected(self) -> None:
        frame = pd.DataFrame(
            {
                "amount": [1, 2, 3] * 4,
                "status": ["black", "white", "gray"] * 4,
            }
        )
        assert detect_label_column(frame).column == "status"

    def test_label_like_name_with_unrecognized_values_is_low_confidence(self) -> None:
        frame = pd.DataFrame({"label": ["x", "y", "z"]})
        detected = detect_label_column(frame)
        assert detected.column == "label"
        assert detected.confidence == "low"

    def test_counts_ignore_unmapped_values(self) -> None:
        frame = pd.DataFrame({"label": ["黑", "白", "灰", "maybe"]})
        mapping = detect_label_column(frame).mapping
        counts = label_counts(frame, "label", mapping)
        assert sum(counts.values()) == 3


class TestQualityDimensions:
    def test_volume_saturates_at_the_target(self) -> None:
        assert volume_score(10_000)[0] == 1.0
        assert volume_score(5_000)[0] == 0.5

    def test_volume_verdict_matches_5_3(self) -> None:
        assert volume_score(15_000)[1] == "充足"
        assert volume_score(200)[1] == "偏少"

    def test_completeness_verdicts_match_5_3(self) -> None:
        assert completeness_score(0.0008)[1] == "良好"
        assert completeness_score(0.03)[1] == "可接受"
        assert completeness_score(0.2)[1] == "偏差"

    def test_zero_missing_scores_full(self) -> None:
        assert completeness_score(0.0)[0] == 1.0

    def test_balance_detects_the_2_1_case_shown_in_5_3(self) -> None:
        counts = {Decision.BLACK.value: 10_000, Decision.WHITE.value: 5_000, Decision.GRAY.value: 0}
        score, verdict, ratio = balance_score(counts)
        assert verdict == "不均衡"
        assert ratio == 2.0
        assert score < 1.0

    def test_balanced_scores_full(self) -> None:
        counts = {Decision.BLACK.value: 1000, Decision.WHITE.value: 1000, Decision.GRAY.value: 500}
        assert balance_score(counts)[0] == 1.0

    def test_gray_does_not_affect_balance(self) -> None:
        """gray is a separate outcome, not a class to be balanced away."""
        base = {Decision.BLACK.value: 1000, Decision.WHITE.value: 1000, Decision.GRAY.value: 0}
        with_gray = {**base, Decision.GRAY.value: 9000}
        assert balance_score(base) == balance_score(with_gray)

    def test_no_black_or_white_is_reported(self) -> None:
        counts = {Decision.GRAY.value: 500}
        assert balance_score(counts)[1] == "无黑白样本"


class TestAnomalyDetection:
    def test_finds_the_outliers_5_3_shows(self) -> None:
        frame = pd.DataFrame({"金额": [1000.0] * 50 + [2_000_000.0] * 23})
        findings = detect_anomalies(frame)
        assert findings and "23 个异常值" in findings[0]

    def test_clean_column_reports_nothing(self) -> None:
        frame = pd.DataFrame({"amount": [1000.0 + i for i in range(50)]})
        assert detect_anomalies(frame) == []

    def test_tiny_column_is_skipped(self) -> None:
        assert detect_anomalies(pd.DataFrame({"a": [1.0, 99.0]})) == []


class TestSuggestions:
    def test_suggests_augmenting_the_imbalanced_class(self) -> None:
        counts = {Decision.BLACK.value: 10_000, Decision.WHITE.value: 5_000, Decision.GRAY.value: 0}
        suggestions = build_suggestions(counts, 0.0008, [])
        assert any("扩增" in s for s in suggestions)

    def test_no_augmentation_advice_when_balanced(self) -> None:
        counts = {Decision.BLACK.value: 1000, Decision.WHITE.value: 1000, Decision.GRAY.value: 5}
        assert not any("扩增" in s for s in build_suggestions(counts, 0.0, []))

    def test_flags_missing_gray(self) -> None:
        counts = {Decision.BLACK.value: 1000, Decision.WHITE.value: 1000, Decision.GRAY.value: 0}
        assert any("灰样本" in s for s in build_suggestions(counts, 0.0, []))

    def test_clean_data_needs_no_suggestions(self) -> None:
        counts = {
            Decision.BLACK.value: 20_000,
            Decision.WHITE.value: 20_000,
            Decision.GRAY.value: 50,
        }
        assert build_suggestions(counts, 0.0, []) == []


class TestQualityReport:
    def test_composite_is_bounded(self, seed_frame: pd.DataFrame) -> None:
        mapping = detect_label_column(seed_frame).mapping
        report = build_report(seed_frame, label_counts(seed_frame, "label", mapping))
        assert 0.0 <= report.score <= 100.0
        assert report.black_white_ratio == pytest.approx(2.0)

    def test_score_drops_as_the_dataset_gets_worse(self) -> None:
        good = pd.DataFrame([{"amount": 100.0} for _ in range(20_000)])
        bad = pd.DataFrame([{"amount": None} for _ in range(20)])
        good_score = build_report(
            good, {Decision.BLACK.value: 10_000, Decision.WHITE.value: 10_000}
        ).score
        bad_score = build_report(bad, {Decision.BLACK.value: 10, Decision.WHITE.value: 10}).score
        assert good_score > bad_score

    def test_report_carries_all_three_labels(self, three_way_frame: pd.DataFrame) -> None:
        counts = {
            Decision.BLACK.value: 600,
            Decision.WHITE.value: 400,
            Decision.GRAY.value: 300,
        }
        report = build_report(three_way_frame, counts)
        assert set(report.label_distribution) == {d.value for d in Decision}

    def test_dimensions_mirror_the_three_cards_in_5_3(self, three_way_frame: pd.DataFrame) -> None:
        counts = {
            Decision.BLACK.value: 600,
            Decision.WHITE.value: 400,
            Decision.GRAY.value: 300,
        }
        dimensions = report_dimensions(build_report(three_way_frame, counts), counts)
        assert [d.name for d in dimensions] == ["样本量", "缺失率", "黑白比"]
        assert all(0.0 <= d.score <= 1.0 for d in dimensions)

    def test_masked_fields_reach_the_report(self, three_way_frame: pd.DataFrame) -> None:
        report = build_report(
            three_way_frame,
            {Decision.BLACK.value: 600, Decision.WHITE.value: 400},
            masked_fields=("身份证号",),
        )
        assert report.masked_fields == ["身份证号"]

    def test_ratio_is_none_when_not_computable(self) -> None:
        """A single-class dataset has no ratio; reporting 0 would read as balanced."""
        report = build_report(pd.DataFrame({"amount": [1.0] * 10}), {Decision.GRAY.value: 10})
        assert report.black_white_ratio is None


class TestSynthesisRecommendation:
    def test_ratio_points_at_correcting_imbalance(self) -> None:
        counts = {Decision.BLACK.value: 10_000, Decision.WHITE.value: 500}
        assert recommend_ratio(counts) == 20.0

    def test_ratio_capped(self) -> None:
        counts = {Decision.BLACK.value: 100_000, Decision.WHITE.value: 1}
        assert recommend_ratio(counts) == 50.0

    def test_ratio_is_symmetric_and_capped(self) -> None:
        """Ratio is always the larger class over the smaller one, so >= 1."""
        assert recommend_ratio({Decision.BLACK.value: 9_000, Decision.WHITE.value: 10}) == 50.0
        assert recommend_ratio({Decision.BLACK.value: 10, Decision.WHITE.value: 9_000}) == 50.0

    def test_balanced_input_yields_one(self) -> None:
        assert recommend_ratio({Decision.BLACK.value: 1_000, Decision.WHITE.value: 1_000}) == 1.0

    def test_gray_excluded_from_the_ratio(self) -> None:
        counts = {
            Decision.BLACK.value: 1000,
            Decision.WHITE.value: 1000,
            Decision.GRAY.value: 99_000,
        }
        assert recommend_ratio(counts) == 1.0

    def test_recommends_augmenting_the_minority_label(self) -> None:
        rec = recommend({Decision.BLACK.value: 10_000, Decision.WHITE.value: 500}, seed_rows=10_500)
        assert rec.augment_label is Decision.WHITE
        assert rec.method is SynthMethod.DISTRIBUTION_FIT

    def test_declines_to_augment_a_tiny_minority(self) -> None:
        """Synthesizing on top of 8 seed rows produces rows that are not realistic."""
        rec = recommend({Decision.BLACK.value: 10_000, Decision.WHITE.value: 8}, seed_rows=10_008)
        assert rec.augment_label is None
        assert any("种子" in r for r in rec.reasons)

    def test_balanced_data_needs_no_targeted_augmentation(self) -> None:
        rec = recommend({Decision.BLACK.value: 1000, Decision.WHITE.value: 1050}, seed_rows=2050)
        assert rec.augment_label is None

    def test_reasonmentions_missing_gray(self) -> None:
        rec = recommend({Decision.BLACK.value: 1000, Decision.WHITE.value: 1000}, seed_rows=2000)
        assert any("灰样本" in r for r in rec.reasons)

    def test_describe_is_readable(self) -> None:
        rec = recommend({Decision.BLACK.value: 10_000, Decision.WHITE.value: 500}, seed_rows=10_500)
        text = rec.describe()
        assert "黑白比例 20 : 1" in text
        assert "扩增 white" in text
