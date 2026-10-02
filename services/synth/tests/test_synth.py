"""Tests for synthesis, fidelity and the privacy gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from son_synth import (
    Constraint,
    Generator,
    SynthRequest,
    assess_fidelity,
    assess_privacy,
    find_duplicates,
    get_generator,
    nearest_neighbour_distance,
    synthesize,
)

from son_contracts import DataOrigin, Decision, SynthMethod


@pytest.fixture
def seed_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "amount": rng.normal(5_000, 1_200, n).round(2),
            "device": rng.choice(["ios", "android", "web"], n),
            "age": rng.integers(18, 70, n),
            "label": [Decision.BLACK.value] * 300 + [Decision.WHITE.value] * 100,
            "origin": DataOrigin.SEED.value,
        }
    )


def _request(**overrides: object) -> SynthRequest:
    base: dict[str, object] = {
        "target_rows": 200,
        "categorical_columns": ("device",),
        "label_column": "label",
    }
    base.update(overrides)
    return SynthRequest(**base)  # type: ignore[arg-type]


class TestProvenanceStamping:
    """The stamp is what makes the test-split guard enforceable."""

    def test_every_generated_row_is_marked_synthetic(self, seed_frame: pd.DataFrame) -> None:
        result = synthesize(seed_frame, _request())
        assert (result.frame["origin"] == DataOrigin.SYNTH.value).all()

    def test_result_assert_passes(self, seed_frame: pd.DataFrame) -> None:
        synthesize(seed_frame, _request()).assert_synthetic()

    def test_missing_stamp_is_caught(self, seed_frame: pd.DataFrame) -> None:
        result = synthesize(seed_frame, _request())
        tampered = result.frame.copy()
        tampered.loc[0, "origin"] = DataOrigin.SEED.value
        with pytest.raises(AssertionError, match="not marked as synthetic"):
            type(result)(
                frame=tampered, method=result.method, rows=len(tampered), label_counts={}
            ).assert_synthetic()

    def test_missing_column_is_caught(self, seed_frame: pd.DataFrame) -> None:
        result = synthesize(seed_frame, _request())
        with pytest.raises(AssertionError, match="origin"):
            type(result)(
                frame=result.frame.drop(columns=["origin"]),
                method=result.method,
                rows=len(result.frame),
                label_counts={},
            ).assert_synthetic()


class TestGeneration:
    def test_returns_exactly_the_requested_row_count(self, seed_frame: pd.DataFrame) -> None:
        assert synthesize(seed_frame, _request(target_rows=137)).rows == 137

    def test_wrong_row_count_is_caught(self, seed_frame: pd.DataFrame) -> None:
        class ShortCircuit(Generator):
            def generate(self, seed_frame: pd.DataFrame, request: SynthRequest) -> pd.DataFrame:
                return pd.DataFrame({"origin": [DataOrigin.SYNTH.value]})

        with pytest.raises(AssertionError, match="expected"):
            synthesize(seed_frame, _request(), generator=ShortCircuit())

    def test_non_positive_target_rejected(self, seed_frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="positive"):
            synthesize(seed_frame, _request(target_rows=0))

    def test_feature_columns_exclude_label_and_origin(self, seed_frame: pd.DataFrame) -> None:
        features = _request().resolved_features(seed_frame)
        assert "label" not in features
        assert "origin" not in features

    def test_empty_seed_frame_rejected(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            synthesize(pd.DataFrame(), _request())

    def test_all_three_methods_dispatch(self) -> None:
        assert get_generator(SynthMethod.DISTRIBUTION_FIT) is not None
        assert get_generator(SynthMethod.SMALL_SAMPLE_DERIVE) is not None
        assert get_generator(SynthMethod.RULE_INJECTION) is not None

    def test_deterministic_for_a_fixed_seed(self, seed_frame: pd.DataFrame) -> None:
        a = synthesize(seed_frame, _request(seed=7)).frame
        b = synthesize(seed_frame, _request(seed=7)).frame
        pd.testing.assert_frame_equal(a, b)


class TestConstraints:
    def test_amount_clamp_is_enforced(self, seed_frame: pd.DataFrame) -> None:
        request = _request(constraints=(Constraint("amount", minimum=0, maximum=10_000),))
        frame = synthesize(seed_frame, request).frame
        assert frame["amount"].max() <= 10_000
        assert frame["amount"].min() >= 0

    def test_age_range_from_requirement_5_4(self, seed_frame: pd.DataFrame) -> None:
        request = _request(constraints=(Constraint("age", minimum=18, maximum=70),))
        frame = synthesize(seed_frame, request).frame
        assert frame["age"].between(18, 70).all()

    def test_unknown_constraint_column_is_ignored(self, seed_frame: pd.DataFrame) -> None:
        request = _request(constraints=(Constraint("nope", maximum=1),))
        assert len(synthesize(seed_frame, request).frame) == 200


class TestLabelTargeting:
    def test_augmenting_the_minority_label_hits_the_ratio(self, seed_frame: pd.DataFrame) -> None:
        request = _request(
            target_rows=1_000,
            augment_label=Decision.BLACK,
            target_ratio=4.0,
        )
        counts = synthesize(seed_frame, request).label_counts
        assert counts[Decision.BLACK.value] == 800
        assert counts[Decision.WHITE.value] == 200

    def test_ratio_of_one_is_even(self, seed_frame: pd.DataFrame) -> None:
        request = _request(
            target_rows=400,
            augment_label=Decision.WHITE,
            target_ratio=1.0,
        )
        counts = synthesize(seed_frame, request).label_counts
        assert counts[Decision.WHITE.value] == 200

    def test_augmenting_without_a_ratio_yields_only_that_label(
        self, seed_frame: pd.DataFrame
    ) -> None:
        """No ratio means a pure top-up of one label; the caller supplies the ratio."""
        counts = synthesize(seed_frame, _request(augment_label=Decision.WHITE)).label_counts
        assert set(counts) == {Decision.WHITE.value}

    def test_no_target_follows_the_seed_distribution(self, seed_frame: pd.DataFrame) -> None:
        """A distribution-fit run must still produce labeled rows, never None."""
        result = synthesize(seed_frame, _request(target_rows=2_000))
        assert result.frame["label"].notna().all()
        assert set(result.label_counts) == {Decision.BLACK.value, Decision.WHITE.value}
        # Seed frame is 300 black out of 400, so the synthetic set should follow.
        share = result.label_counts[Decision.BLACK.value] / 2_000
        assert 0.70 < share < 0.80


class TestFidelity:
    def test_matching_distribution_scores_well(self, seed_frame: pd.DataFrame) -> None:
        report = assess_fidelity(seed_frame, synthesize(seed_frame, _request()).frame)
        assert report.score >= 0.60
        assert report.verdict != "poor"

    def test_shifted_distribution_scores_worse(self, seed_frame: pd.DataFrame) -> None:
        shifted = seed_frame.copy()
        shifted["amount"] = shifted["amount"] * 50
        good = assess_fidelity(seed_frame, synthesize(seed_frame, _request()).frame)
        bad = assess_fidelity(seed_frame, shifted)
        assert bad.score < good.score

    def test_categorical_column_is_compared_by_frequency(self, seed_frame: pd.DataFrame) -> None:
        flipped = seed_frame.copy()
        flipped["device"] = "unknown"
        frame = synthesize(seed_frame, _request()).frame
        assert assess_fidelity(flipped, frame).per_column["device"] < 0.5

    def test_score_is_bounded(self, seed_frame: pd.DataFrame) -> None:
        frame = synthesize(seed_frame, _request()).frame
        report = assess_fidelity(seed_frame, frame)
        assert 0.0 <= report.score <= 1.0
        assert all(0.0 <= v <= 1.0 for v in report.per_column.values())

    def test_no_shared_columns_is_reported_not_crashed(self, seed_frame: pd.DataFrame) -> None:
        report = assess_fidelity(seed_frame, pd.DataFrame({"zzz": [1]}))
        assert report.verdict == "poor"
        assert report.notes

    def test_weak_columns_are_named(self, seed_frame: pd.DataFrame) -> None:
        shifted = seed_frame.copy()
        shifted["age"] = shifted["age"] * 100
        report = assess_fidelity(shifted, synthesize(seed_frame, _request()).frame)
        assert any("age" in note for note in report.notes)


class TestPrivacy:
    def test_independent_synthesis_finds_no_exact_duplicates(
        self, seed_frame: pd.DataFrame
    ) -> None:
        assert find_duplicates(seed_frame, synthesize(seed_frame, _request()).frame) == 0

    def test_copied_rows_are_detected_as_duplicates(self, seed_frame: pd.DataFrame) -> None:
        copied = seed_frame.head(5).copy()
        assert find_duplicates(seed_frame, copied) == 5

    def test_copied_rows_raise_the_risk_level(self, seed_frame: pd.DataFrame) -> None:
        report = assess_privacy(seed_frame, seed_frame.head(20).copy())
        assert (report.duplicates_found or 0) > 0
        assert report.reversible_risk == "high"

    def test_privacy_report_lines_match_requirement_5_4(self, seed_frame: pd.DataFrame) -> None:
        report = assess_privacy(seed_frame, synthesize(seed_frame, _request()).frame)
        lines = report.describe()
        assert lines[0].startswith("✓ 去重检查通过")
        assert "最近邻距离" in lines[1]

    def test_nearest_neighbour_distance_is_positive(self, seed_frame: pd.DataFrame) -> None:
        frame = synthesize(seed_frame, _request()).frame
        distance = nearest_neighbour_distance(seed_frame, frame)
        assert distance is not None
        assert distance > 0

    def test_verbatim_copies_sit_far_closer_than_independent_rows(
        self, seed_frame: pd.DataFrame
    ) -> None:
        independent = synthesize(seed_frame, _request(target_rows=50)).frame
        verbatim = seed_frame.head(50).copy()
        near_verbatim = nearest_neighbour_distance(seed_frame, verbatim)
        near_independent = nearest_neighbour_distance(seed_frame, independent)
        assert near_verbatim is not None and near_independent is not None
        assert near_verbatim < near_independent

    def test_no_numeric_columns_is_unmeasured_not_safe(self, seed_frame: pd.DataFrame) -> None:
        """1.0 is above the safe threshold, so returning it claimed a clean check."""
        assert nearest_neighbour_distance(seed_frame, pd.DataFrame({"device": ["ios"]})) is None

    def test_incomplete_checks_never_report_no_risk(self, seed_frame: pd.DataFrame) -> None:
        """All-categorical synthetic output: nothing was comparable."""
        report = assess_privacy(seed_frame[["device", "label"]], seed_frame[["device", "label"]])
        assert report.duplicate_check_ran is True
        assert report.nn_check_ran is False
        assert report.reversible_risk != "none"

    def test_unmeasured_nn_is_spelled_out(self, seed_frame: pd.DataFrame) -> None:
        """Categorical-only output: the duplicate check ran, the distance did not."""
        report = assess_privacy(seed_frame[["device", "label"]], seed_frame[["device", "label"]])
        assert report.duplicate_check_ran is True
        assert any("未测量" in line for line in report.describe())
        assert any("最近邻距离" in note for note in report.notes)

    def test_fully_unrun_checks_appear_in_both(self) -> None:
        report = assess_privacy(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]}))
        lines = report.describe()
        assert any("未执行" in line for line in lines)
        assert any("未测量" in line for line in lines)

    def test_no_shared_columns_skips_both_checks(self) -> None:
        report = assess_privacy(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]}))
        assert report.duplicate_check_ran is False
        assert report.nn_check_ran is False
        assert report.reversible_risk == "review"

    def test_empty_seed_side_is_unmeasured(self) -> None:
        report = assess_privacy(pd.DataFrame({"amount": []}), pd.DataFrame({"amount": [1.0]}))
        assert report.nn_check_ran is False


class TestFidelityPrivacyIntegration:
    def test_generated_set_passes_into_the_split_guarded_path(
        self, seed_frame: pd.DataFrame
    ) -> None:
        """The end-to-end invariant: synthesized rows must not reach test."""
        from son_data_pipeline import stratified_split

        synthetic = synthesize(seed_frame, _request(target_rows=300)).frame
        combined = pd.concat([seed_frame, synthetic], ignore_index=True)
        result = stratified_split(combined)
        assert (result.test["origin"] == DataOrigin.SEED.value).all()
        assert result.summary.test_contains_synth is False
