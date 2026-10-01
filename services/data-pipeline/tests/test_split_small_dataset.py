"""Split behaviour on datasets too small for a per-label share.

An 8-row dataset cannot give every label 15% of its rows: the share rounds to
zero and the fallback decides the composition of the evaluation set. With 8 rows
the 7:1.5:1.5 contract yields exactly one test row, so no allocation can put
three labels in it.

Two things therefore have to hold, and neither is "pad the test set" -- that
would break the fixed ratios, which are a hard requirement:

* the single test row goes to the largest class, deterministically. A previous
  implementation used ``head()`` after the shuffle, which handed the evaluation
  set to whichever label happened to sort first.
* the coverage shortfall is *reported* rather than hidden: the per-label counts
  travel with the summary, and a reader can see the evaluation set is thin.
"""

from __future__ import annotations

import pandas as pd
import pytest
from son_data_pipeline import stratified_split

from son_contracts import DataOrigin, Decision


def _frame(counts: dict[Decision, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, count in counts.items():
        rows += [
            {
                "account": f"{label.value[0].upper()}{i}",
                "label": label.value,
                "origin": DataOrigin.SEED.value,
            }
            for i in range(count)
        ]
    return pd.DataFrame(rows)


@pytest.fixture
def eight_rows() -> pd.DataFrame:
    return _frame({Decision.BLACK: 4, Decision.WHITE: 2, Decision.GRAY: 2})


class TestTinyDatasetTestSet:
    def test_test_split_is_never_empty(self, eight_rows: pd.DataFrame) -> None:
        assert len(stratified_split(eight_rows).test) > 0

    def test_single_row_goes_to_the_largest_class(self, eight_rows: pd.DataFrame) -> None:
        """Deterministic and explainable, rather than shuffle order."""
        test = stratified_split(eight_rows).test
        assert len(test) == 1
        assert test.iloc[0]["label"] == Decision.BLACK.value

    def test_choice_does_not_depend_on_the_shuffle_seed(self, eight_rows: pd.DataFrame) -> None:
        for seed in (0, 1, 42, 999):
            test = stratified_split(eight_rows, seed=seed).test
            assert test.iloc[0]["label"] == Decision.BLACK.value

    def test_coverage_shortfall_is_visible_in_the_summary(self, eight_rows: pd.DataFrame) -> None:
        """The reviewer must be able to see the evaluation set is one row."""
        counts = stratified_split(eight_rows).summary.label_distribution["test"]
        assert set(counts) == {d.value for d in Decision}
        assert sum(counts.values()) == 1

    def test_undersized_strata_are_disclosed(self, eight_rows: pd.DataFrame) -> None:
        assert stratified_split(eight_rows).undersized_groups


class TestBalanceEngagesWhenThereIsRoom:
    """With enough rows for more than one test row, balancing must kick in."""

    def test_all_three_labels_appear(self) -> None:
        frame = _frame({Decision.BLACK: 40, Decision.WHITE: 30, Decision.GRAY: 30})
        counts = stratified_split(frame, seed=1).summary.label_distribution["test"]
        assert all(counts[d.value] > 0 for d in Decision), counts

    def test_counts_are_proportional(self) -> None:
        frame = _frame({Decision.BLACK: 40, Decision.WHITE: 30, Decision.GRAY: 30})
        counts = stratified_split(frame, seed=1).summary.label_distribution["test"]
        assert counts[Decision.BLACK.value] > counts[Decision.WHITE.value]

    def test_gray_share_is_close_to_the_source(self) -> None:
        frame = _frame({Decision.BLACK: 40, Decision.WHITE: 30, Decision.GRAY: 30})
        result = stratified_split(frame, seed=1)
        total = sum(result.summary.label_distribution["test"].values())
        gray_share = result.summary.label_distribution["test"][Decision.GRAY.value] / total
        assert 0.25 < gray_share < 0.35


class TestInvariantsStillHold:
    def test_no_synthetic_in_test(self, eight_rows: pd.DataFrame) -> None:
        result = stratified_split(eight_rows)
        assert result.summary.test_contains_synth is False
        assert (result.test["origin"] == DataOrigin.SEED.value).all()

    def test_no_rows_lost(self, eight_rows: pd.DataFrame) -> None:
        result = stratified_split(eight_rows)
        assert sum(len(f) for f in result.frames().values()) == len(eight_rows)

    def test_no_rows_duplicated(self, eight_rows: pd.DataFrame) -> None:
        result = stratified_split(eight_rows)
        accounts = [a for f in result.frames().values() for a in f["account"]]
        assert len(accounts) == len(set(accounts))

    def test_ratios_still_hold_on_a_tiny_dataset(self, eight_rows: pd.DataFrame) -> None:
        """Padding the test set to cover labels would break the fixed ratios.

        Asserted as the documented rounding rule rather than as 15%: with 8 rows
        a 15% share is 1.2, and integer arithmetic can only produce 1 row (12.5%).
        Requiring exactly 15% would mean inventing rows.
        """
        summary = stratified_split(eight_rows).summary
        total = summary.train_rows + summary.valid_rows + summary.test_rows
        assert total == len(eight_rows)
        assert summary.test_rows == max(1, round(total * 0.15))
        assert summary.test_rows <= round(total * 0.15) + 1
