"""Evaluator tests.

The effect block is fully verified here. The performance tests verify
*aggregation* only -- no latency number in this file was measured, because this
machine has no GPU.
"""

from __future__ import annotations

import pytest
from son_evaluator import (
    EffectReport,
    LatencySample,
    accuracy_score,
    binary_auc_roc,
    build_effect_report,
    build_performance_report,
    check_target,
    confusion_matrix,
    false_kill_rate,
    feature_importance,
    per_label_metrics,
    percentile,
)

from son_contracts import Decision

BLACK, WHITE, GRAY = Decision.BLACK, Decision.WHITE, Decision.GRAY


class TestConfusionMatrix:
    def test_counts_land_in_the_right_cells(self) -> None:
        matrix = confusion_matrix((BLACK, WHITE, GRAY), (BLACK, BLACK, GRAY))
        assert matrix["black"]["black"] == 1
        assert matrix["white"]["black"] == 1
        assert matrix["gray"]["gray"] == 1

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            confusion_matrix((BLACK,), (BLACK, WHITE))

    def test_matrix_covers_all_three_decisions(self) -> None:
        matrix = confusion_matrix((BLACK,), (BLACK,))
        assert set(matrix) == {"black", "white", "gray"}

    def test_row_totals_equal_support(self) -> None:
        truth = (BLACK, BLACK, WHITE, GRAY)
        matrix = confusion_matrix(truth, (BLACK, WHITE, WHITE, BLACK))
        metrics = per_label_metrics(matrix)
        assert sum(metrics[d.value].support for d in Decision) == len(truth)


class TestThreeWayMetrics:
    """gray must not be averaged away."""

    def test_gray_gets_its_own_recall(self) -> None:
        matrix = confusion_matrix((GRAY, GRAY, BLACK), (BLACK, GRAY, BLACK))
        metrics = per_label_metrics(matrix)
        assert metrics["gray"].recall == 0.5
        assert metrics["gray"].support == 2

    def test_gray_absent_from_test_is_disclosed(self) -> None:
        report = build_effect_report((BLACK, WHITE), (BLACK, WHITE))
        assert any("gray" in note for note in report.notes)

    def test_macro_f1_ignores_absent_classes(self) -> None:
        with_gray = build_effect_report((BLACK, WHITE, GRAY), (BLACK, WHITE, GRAY))
        without_gray = build_effect_report((BLACK, WHITE), (BLACK, WHITE))
        assert with_gray.macro_f1 == without_gray.macro_f1 == 1.0

    def test_perfect_prediction_scores_one(self) -> None:
        report = build_effect_report((BLACK, WHITE, GRAY), (BLACK, WHITE, GRAY))
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0

    def test_accuracy_counts_exact_matches_only(self) -> None:
        assert accuracy_score((BLACK, WHITE, GRAY), (BLACK, WHITE, BLACK)) == pytest.approx(2 / 3)


class TestFalseKillRate:
    def test_white_killed_as_black_is_counted(self) -> None:
        matrix = confusion_matrix((WHITE, WHITE, BLACK), (BLACK, WHITE, BLACK))
        assert false_kill_rate(matrix) == 0.5

    def test_killing_gray_is_not_a_false_kill(self) -> None:
        """误杀 protects normal accounts; rejecting an undecidable sample is not it."""
        matrix = confusion_matrix((GRAY, GRAY), (BLACK, BLACK))
        assert false_kill_rate(matrix) == 0.0

    def test_no_white_samples_is_zero_not_a_crash(self) -> None:
        assert false_kill_rate(confusion_matrix((BLACK,), (BLACK,))) == 0.0


class TestAucRoc:
    def test_perfect_separation_is_one(self) -> None:
        truth = (BLACK, BLACK, WHITE, WHITE)
        scores = (0.9, 0.8, 0.2, 0.1)
        assert binary_auc_roc(truth, scores) == 1.0

    def test_inverted_ranking_is_zero(self) -> None:
        truth = (BLACK, BLACK, WHITE, WHITE)
        scores = (0.1, 0.2, 0.8, 0.9)
        assert binary_auc_roc(truth, scores) == 0.0

    def test_ties_get_credit(self) -> None:
        """Tied scores must land at 0.5, not be counted as correctly ordered."""
        truth = (BLACK, BLACK, WHITE, WHITE)
        scores = (0.5, 0.5, 0.5, 0.5)
        assert binary_auc_roc(truth, scores) == 0.5

    def test_single_class_returns_none(self) -> None:
        assert binary_auc_roc((WHITE, WHITE), (0.5, 0.6)) is None

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            binary_auc_roc((BLACK,), (0.5, 0.6))

    def test_missing_scores_is_disclosed(self) -> None:
        report = build_effect_report((BLACK, WHITE), (BLACK, WHITE))
        assert report.auc_roc is None
        assert any("AUC" in note for note in report.notes)


class TestEffectReportAssembly:
    def _report(self) -> EffectReport:
        truth = (BLACK, BLACK, WHITE, WHITE, GRAY, BLACK)
        pred = (BLACK, WHITE, WHITE, GRAY, GRAY, BLACK)
        return build_effect_report(
            truth, pred, scores=(0.9, 0.6, 0.2, 0.4, 0.5, 0.7), format_compliant=5, format_total=6
        )

    def test_every_effect_target_is_checked(self) -> None:
        checks = self._report().target_check()
        for name in ("accuracy", "macro_f1", "false_kill_rate", "format_compliance", "auc_roc"):
            assert name in checks
            assert "passed" in checks[name]

    def test_format_compliance_is_a_ratio(self) -> None:
        assert self._report().format_compliance == pytest.approx(5 / 6)

    def test_zero_format_samples_is_zero_not_a_division_error(self) -> None:
        report = build_effect_report((BLACK,), (BLACK,))
        assert report.format_compliance == 0.0

    def test_sample_count_is_recorded(self) -> None:
        assert self._report().sample_count == 6

    def test_confusion_matrix_travels_with_the_report(self) -> None:
        assert self._report().confusion["black"]["white"] == 1

    def test_target_check_flags_a_failing_metric(self) -> None:
        report = build_effect_report((WHITE,) * 10, (BLACK,) * 10)
        assert report.target_check()["false_kill_rate"]["passed"] is False


class TestFeatureImportance:
    def test_ranked_by_weight_descending(self) -> None:
        ranked = feature_importance({"a": 0.1, "b": 0.5, "c": 0.3})
        assert [name for name, _ in ranked] == ["b", "c", "a"]

    def test_truncated_to_top_n(self) -> None:
        assert len(feature_importance({f"f{i}": i / 100 for i in range(30)}, top_n=10)) == 10

    def test_empty_is_handled(self) -> None:
        assert feature_importance({}) == []


class TestPercentile:
    def test_single_value(self) -> None:
        assert percentile([42.0], 95) == 42.0

    def test_interpolates_between_samples(self) -> None:
        assert percentile([0.0, 100.0], 50) == 50.0

    def test_matches_known_values(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0
        assert percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0

    def test_empty_is_none(self) -> None:
        assert percentile([], 95) is None

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 100"):
            percentile([1.0], 101)


class TestPerformanceAggregation:
    def test_no_samples_reports_unmeasured(self) -> None:
        report = build_performance_report([])
        assert report.p50_ms is None
        assert "未测量" in report.describe()

    def test_percentiles_are_computed(self) -> None:
        samples = [LatencySample(total_ms=float(v)) for v in range(1, 101)]
        report = build_performance_report(samples, wall_clock_s=10.0)
        assert report.p50_ms == pytest.approx(50.5, abs=1)
        assert report.p95_ms == pytest.approx(95.05, abs=1)
        assert report.p99_ms == pytest.approx(99.01, abs=1)

    def test_ttft_is_tracked_separately(self) -> None:
        """TTFT and end-to-end need different optimisations, so they are separate."""
        samples = [LatencySample(total_ms=90.0, ttft_ms=12.0) for _ in range(10)]
        report = build_performance_report(samples, wall_clock_s=1.0)
        assert report.ttft_p95_ms == 12.0
        assert report.p50_ms == 90.0

    def test_ttft_missing_when_not_measured(self) -> None:
        report = build_performance_report([LatencySample(total_ms=10.0)])
        assert report.ttft_p95_ms is None

    def test_qps_needs_wall_clock(self) -> None:
        """Deriving throughput from latency alone reports single-stream as concurrent."""
        samples = [LatencySample(total_ms=50.0) for _ in range(20)]
        assert build_performance_report(samples).qps is None
        assert build_performance_report(samples, wall_clock_s=2.0).qps == 10.0

    def test_missing_measurements_are_disclosed(self) -> None:
        report = build_performance_report([LatencySample(total_ms=10.0)], wall_clock_s=1.0)
        assert any("峰值内存" in note for note in report.notes)

    def test_quant_level_is_recorded_for_comparability(self) -> None:
        report = build_performance_report(
            [LatencySample(total_ms=10.0)], wall_clock_s=1.0, quant_level="Q4_K_M"
        )
        assert report.quant_level == "Q4_K_M"
        assert any("可比" in note for note in report.notes)


class TestDisputedTargets:
    """Q1: P95/P99 cannot be judged until the requirement's conflict is settled."""

    def test_p95_is_not_judged(self) -> None:
        samples = [LatencySample(total_ms=250.0) for _ in range(20)]
        report = build_performance_report(samples, wall_clock_s=1.0)
        assert report.targets["p95_ms"]["passed"] is None

    def test_p95_value_is_still_reported(self) -> None:
        """Undecided target does not mean the measurement is hidden."""
        samples = [LatencySample(total_ms=250.0) for _ in range(20)]
        assert build_performance_report(samples, wall_clock_s=1.0).p95_ms == 250.0

    def test_the_conflict_is_explained_in_the_notes(self) -> None:
        report = build_performance_report([LatencySample(total_ms=250.0)], wall_clock_s=1.0)
        assert any("Q1" in note for note in report.notes)

    def test_undisputed_targets_are_judged(self) -> None:
        samples = [LatencySample(total_ms=10.0, ttft_ms=5.0) for _ in range(20)]
        report = build_performance_report(samples, wall_clock_s=1.0, peak_memory_mb=100.0)
        assert report.targets["ttft_ms"]["passed"] is True
        assert report.targets["peak_memory_mb"]["passed"] is True

    def test_check_target_direction_differs_by_metric(self) -> None:
        assert check_target("p50_ms", 10.0)["passed"] is True
        assert check_target("p50_ms", 500.0)["passed"] is False
        assert check_target("qps", 500.0)["passed"] is True
        assert check_target("qps", 1.0)["passed"] is False

    def test_unmeasured_metric_is_undecided_not_passed(self) -> None:
        assert check_target("p50_ms", None)["passed"] is None
