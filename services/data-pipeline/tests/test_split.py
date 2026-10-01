"""Tests for the split invariant.

需求方案.txt 5.4 states as a hard constraint that synthetic data must not enter
the test split. These tests exist to make sure a future refactor cannot quietly
turn that into a convention.
"""

from __future__ import annotations

import pandas as pd
import pytest
from son_data_pipeline.split import (
    SPLIT_RATIOS,
    SyntheticDataLeakError,
    label_share,
    stratified_split,
)

from son_contracts import DataOrigin, Decision, SplitSummary


class TestRatiosAreFixed:
    def test_documented_7_1_5_1_5(self) -> None:
        assert SPLIT_RATIOS == {"train": 0.70, "valid": 0.15, "test": 0.15}

    def test_close_to_target_on_a_realistic_dataset(self, three_way_frame: pd.DataFrame) -> None:
        result = stratified_split(three_way_frame)
        total = len(three_way_frame)
        for name, expected in SPLIT_RATIOS.items():
            actual = len(result.frames()[name]) / total
            assert actual == pytest.approx(expected, abs=0.02), f"{name} off target"


class TestSyntheticDataNeverEntersTest:
    """The core hard constraint."""

    def test_test_split_has_zero_synthetic_rows(self, mixed_frame: pd.DataFrame) -> None:
        result = stratified_split(mixed_frame)
        assert (result.test["origin"] == DataOrigin.SEED.value).all()

    def test_summary_reports_the_invariant(self, mixed_frame: pd.DataFrame) -> None:
        assert stratified_split(mixed_frame).summary.test_contains_synth is False

    def test_synthetic_rows_still_reach_train(self, mixed_frame: pd.DataFrame) -> None:
        """Augmentation is the point; excluding synth everywhere would be useless."""
        result = stratified_split(mixed_frame)
        assert (result.train["origin"] == DataOrigin.SYNTH.value).any()

    def test_all_synthetic_input_is_rejected(self) -> None:
        """Not silently reassigned to train, and not quietly made into a valid split."""
        frame = pd.DataFrame(
            [{"account": "A", "label": Decision.BLACK.value, "origin": DataOrigin.SYNTH.value}] * 50
        )
        with pytest.raises(ValueError, match="no seed rows"):
            stratified_split(frame)

    def test_leak_error_is_an_assertion(self) -> None:
        """Consumers testing with AssertionError keep catching it."""
        assert issubclass(SyntheticDataLeakError, AssertionError)

    def test_guard_catches_a_leak_if_one_is_injected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proves the post-split assertion actually fires.

        Without this, the assertion could be dead code that never runs and the
        tests above would still pass.
        """
        import son_data_pipeline.split as split_mod

        real = split_mod._assert_no_synthetic_in_test

        def leaky(test: pd.DataFrame, origin_column: str) -> None:
            poisoned = pd.concat(
                [
                    test,
                    pd.DataFrame([{"origin": DataOrigin.SYNTH.value, "label": "black"}]),
                ],
                ignore_index=True,
            )
            real(poisoned, origin_column)

        monkeypatch.setattr(split_mod, "_assert_no_synthetic_in_test", leaky)
        with pytest.raises(SyntheticDataLeakError, match="hard product constraint"):
            stratified_split(
                pd.DataFrame(
                    [
                        {
                            "account": f"A{i}",
                            "label": Decision.BLACK.value,
                            "origin": DataOrigin.SEED.value,
                        }
                        for i in range(200)
                    ]
                )
            )

    def test_unknown_origin_is_rejected_not_assumed_safe(self) -> None:
        frame = pd.DataFrame(
            [{"account": "A", "label": Decision.BLACK.value, "origin": "mystery"}] * 50
        )
        with pytest.raises(ValueError, match="unknown origin"):
            stratified_split(frame)

    def test_missing_origin_column_is_rejected(self) -> None:
        frame = pd.DataFrame([{"account": "A", "label": Decision.BLACK.value}] * 50)
        with pytest.raises(ValueError, match="origin"):
            stratified_split(frame)


class TestGrayCoverage:
    """gray is a first-class outcome, so it must not vanish from a split."""

    def test_gray_appears_in_test_when_present(self, three_way_frame: pd.DataFrame) -> None:
        result = stratified_split(three_way_frame)
        assert (result.test["label"] == Decision.GRAY.value).any()

    def test_gray_share_stays_close_across_splits(self, three_way_frame: pd.DataFrame) -> None:
        result = stratified_split(three_way_frame)
        shares = [label_share(result.summary, s, Decision.GRAY) for s in ("train", "valid", "test")]
        assert max(shares) - min(shares) < 0.01, f"gray drifted across splits: {shares}"

    def test_all_three_labels_present_in_test(self, three_way_frame: pd.DataFrame) -> None:
        counts = stratified_split(three_way_frame).summary.label_distribution["test"]
        assert set(counts) == {d.value for d in Decision}
        assert all(v > 0 for v in counts.values())

    def test_nonzero_labels_are_reported(self, seed_frame: pd.DataFrame) -> None:
        counts = stratified_split(seed_frame).summary.label_distribution["test"]
        assert counts[Decision.GRAY.value] == 0


class TestRowAccounting:
    def test_no_rows_lost(self, mixed_frame: pd.DataFrame) -> None:
        result = stratified_split(mixed_frame)
        frames = result.frames()
        assert sum(len(f) for f in frames.values()) == len(mixed_frame)

    def test_no_rows_duplicated(self, mixed_frame: pd.DataFrame) -> None:
        result = stratified_split(mixed_frame)
        seen: list[str] = []
        for frame in result.frames().values():
            seen.extend(frame["account"].tolist())
        assert len(seen) == len(set(seen))

    def test_non_decision_labels_rejected_before_splitting(self) -> None:
        frame = pd.DataFrame(
            [{"account": "A", "label": "fraud", "origin": DataOrigin.SEED.value}] * 50
        )
        with pytest.raises(ValueError, match="non-decision"):
            stratified_split(frame)

    def test_empty_frame_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            stratified_split(pd.DataFrame(columns=["account", "label", "origin"]))


class TestDeterminism:
    def test_same_seed_same_split(self, mixed_frame: pd.DataFrame) -> None:
        a = stratified_split(mixed_frame, seed=7)
        b = stratified_split(mixed_frame, seed=7)
        assert a.test["account"].tolist() == b.test["account"].tolist()

    def test_different_seed_moves_rows(self, mixed_frame: pd.DataFrame) -> None:
        a = stratified_split(mixed_frame, seed=1)
        b = stratified_split(mixed_frame, seed=99)
        assert a.test["account"].tolist() != b.test["account"].tolist()


class TestUndersizedStrata:
    def test_small_dataset_still_produces_a_test_split(self) -> None:
        frame = pd.DataFrame(
            [
                {"account": f"A{i}", "label": Decision.BLACK.value, "origin": DataOrigin.SEED.value}
                for i in range(30)
            ]
            + [
                {
                    "account": f"G{i}",
                    "label": Decision.GRAY.value,
                    "origin": DataOrigin.SEED.value,
                }
                for i in range(3)
            ]
        )
        result = stratified_split(frame)
        assert len(result.test) > 0
        assert result.summary.test_contains_synth is False

    def test_undersized_groups_are_disclosed(self) -> None:
        frame = pd.DataFrame(
            [
                {"account": f"A{i}", "label": Decision.BLACK.value, "origin": DataOrigin.SEED.value}
                for i in range(300)
            ]
            + [
                {
                    "account": f"G{i}",
                    "label": Decision.GRAY.value,
                    "origin": DataOrigin.SEED.value,
                }
                for i in range(4)
            ]
        )
        result = stratified_split(frame)
        assert any("gray" in group for group in result.undersized_groups)


class TestSummaryShape:
    def test_summary_always_declares_the_invariant_false(self, mixed_frame: pd.DataFrame) -> None:
        summary: SplitSummary = stratified_split(mixed_frame).summary
        assert summary.test_contains_synth is False
        assert summary.test_rows == len(stratified_split(mixed_frame).test)
