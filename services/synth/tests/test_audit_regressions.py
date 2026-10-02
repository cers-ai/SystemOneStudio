"""Regression tests for the defects found in the September audit.

Each test names the defect it guards. They are grouped here rather than
scattered because they share a theme: **a number or a stage that looked fine
while being wrong**. Three of them were cases where an old test actively
asserted the buggy behaviour.
"""

from __future__ import annotations

import pandas as pd
import pytest
from son_data_pipeline import recommend_ratio, stratified_split
from son_synth import (
    Constraint,
    SynthRequest,
    assess_fidelity,
    synthesize,
)
from son_trainer.preferences import build_preference_pairs

from son_contracts import DataOrigin, Decision, SynthMethod

# ===========================================================================
# 1. Preference pairs: samples were handed to the wrong prompt
# ===========================================================================


class TestPreferencePairAlignment:
    """`samples` is laid out in *prompt* order.

    Filtering out non-seed rows before indexing shifted every later prompt's
    block. Since synthetic rows are always skipped and always land in train, an
    augmented dataset -- the normal case -- mis-assigned every sample.
    """

    def _prompts(self) -> tuple[tuple[str, Decision, str], ...]:
        return (
            ("PROMPT_P0", Decision.BLACK, DataOrigin.SEED.value),
            ("PROMPT_P1", Decision.BLACK, DataOrigin.SYNTH.value),  # skipped
            ("PROMPT_P2", Decision.BLACK, DataOrigin.SEED.value),
        )

    def test_p2_does_not_receive_p1s_samples(self) -> None:
        samples = (
            # P0: one right, one wrong -> pair
            '{"decision":"black","reason":"right"}',
            '{"decision":"white","reason":"wrong"}',
            # P1 (skipped)
            '{"decision":"white","reason":"WRONG-ONLY-1"}',
            '{"decision":"gray","reason":"WRONG-ONLY-2"}',
            # P2: both wrong -> no pair
            '{"decision":"white","reason":"p2-wrong-1"}',
            '{"decision":"gray","reason":"p2-wrong-2"}',
        )
        report = build_preference_pairs(self._prompts(), samples, n_per_prompt=2)

        assert [p.prompt for p in report.pairs] == ["PROMPT_P0"]
        assert report.prompts_without_a_valid_pair == 1
        assert "WRONG-ONLY" not in report.pairs[0].rejected

    def test_a_later_seed_prompt_still_gets_its_own_samples(self) -> None:
        samples = tuple(
            [
                '{"decision":"black","reason":"ok"}',
                '{"decision":"white","reason":"no"}',
            ]
            * 3
        )
        report = build_preference_pairs(self._prompts(), samples, n_per_prompt=2)
        assert {p.prompt for p in report.pairs} == {"PROMPT_P0", "PROMPT_P2"}

    def test_unparsed_samples_are_disclosed(self) -> None:
        samples = (
            "total nonsense",
            '{"decision":"white"}',
            "also nonsense",
            '{"decision":"white"}',
        )
        report = build_preference_pairs(
            (("P", Decision.BLACK, DataOrigin.SEED.value),), samples, n_per_prompt=2
        )
        assert any("无法解析" in note for note in report.notes)


# ===========================================================================
# 2. Synthesizer: label=None and a dimensional error in the jitter
# ===========================================================================


def _seed_frame(rows: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [100.0 + i for i in range(rows)],
            "label": [Decision.BLACK.value] * (rows // 2) + [Decision.WHITE.value] * (rows // 2),
            "origin": DataOrigin.SEED.value,
        }
    )


class TestFeatureDeriveLabels:
    def test_label_is_never_none(self) -> None:
        """The else branch assigned the (None) augment label to every row."""
        frame = synthesize(
            _seed_frame(),
            SynthRequest(
                target_rows=50,
                method=SynthMethod.SMALL_SAMPLE_DERIVE,
                categorical_columns=(),
            ),
        ).frame
        assert frame["label"].notna().all()
        assert set(frame["label"]) <= {Decision.BLACK.value, Decision.WHITE.value}

    def test_labels_follow_the_seed_distribution(self) -> None:
        frame = synthesize(
            _seed_frame(),
            SynthRequest(
                target_rows=400,
                method=SynthMethod.SMALL_SAMPLE_DERIVE,
                categorical_columns=(),
            ),
        ).frame
        share = (frame["label"] == Decision.BLACK.value).mean()
        assert 0.4 < share < 0.6

    def test_constraints_are_honoured(self) -> None:
        """This backend used to skip constraint clamping entirely."""
        frame = synthesize(
            _seed_frame(),
            SynthRequest(
                target_rows=200,
                method=SynthMethod.SMALL_SAMPLE_DERIVE,
                categorical_columns=(),
                constraints=(Constraint("amount", minimum=0, maximum=150),),
            ),
        ).frame
        assert frame["amount"].between(0, 150).all()


class TestJitterDimensionalError:
    def test_values_stay_within_the_seed_range(self) -> None:
        """An absolute std used as a multiplicative noise term produced -51x/+55x."""
        frame = synthesize(
            _seed_frame(),
            SynthRequest(
                target_rows=300,
                method=SynthMethod.SMALL_SAMPLE_DERIVE,
                categorical_columns=(),
            ),
        ).frame
        assert (frame["amount"] > 0).all()
        assert frame["amount"].max() < 5000

    def test_no_extreme_multiples_of_the_source(self) -> None:
        """The seed spans 100..299, so compare against its own maximum."""
        seed = _seed_frame()
        ceiling = float(seed["amount"].max()) * 1.5
        frame = synthesize(
            seed,
            SynthRequest(
                target_rows=300,
                method=SynthMethod.SMALL_SAMPLE_DERIVE,
                categorical_columns=(),
            ),
        ).frame
        assert frame["amount"].max() < ceiling, frame["amount"].max()


class TestRatioFalsyTrap:
    def test_zero_ratio_is_rejected_not_reinterpreted(self) -> None:
        """`if request.target_ratio:` made 0.0 mean "no ratio given"."""
        with pytest.raises(ValueError, match="必须大于 0"):
            synthesize(
                _seed_frame(),
                SynthRequest(
                    target_rows=100,
                    method=SynthMethod.DISTRIBUTION_FIT,
                    categorical_columns=(),
                    augment_label=Decision.BLACK,
                    target_ratio=0.0,
                ),
            )

    def test_ratio_of_four_hits_the_requested_split(self) -> None:
        frame = synthesize(
            _seed_frame(),
            SynthRequest(
                target_rows=100,
                method=SynthMethod.DISTRIBUTION_FIT,
                categorical_columns=(),
                augment_label=Decision.BLACK,
                target_ratio=4.0,
            ),
        ).frame
        counts = frame["label"].value_counts()
        assert counts[Decision.BLACK.value] == 80
        assert counts[Decision.WHITE.value] == 20

    def test_negative_ratio_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="必须大于 0"):
            synthesize(
                _seed_frame(),
                SynthRequest(
                    target_rows=10,
                    method=SynthMethod.DISTRIBUTION_FIT,
                    categorical_columns=(),
                    augment_label=Decision.BLACK,
                    target_ratio=-1.0,
                ),
            )

    def test_gray_augmentation_complements_with_white(self) -> None:
        """The old expression made white unreachable when augmenting gray."""
        frame = synthesize(
            _seed_frame(),
            SynthRequest(
                target_rows=100,
                method=SynthMethod.DISTRIBUTION_FIT,
                categorical_columns=(),
                augment_label=Decision.GRAY,
                target_ratio=1.0,
            ),
        ).frame
        assert Decision.WHITE.value in set(frame["label"])


class TestRuleInjectionEmptyValues:
    def test_empty_rule_raises_instead_of_index_error(self) -> None:
        from son_synth import RuleInjectionGenerator

        generator = RuleInjectionGenerator(rules={"amount": {"black": [], "default": []}})
        with pytest.raises(ValueError, match="没有取值"):
            generator.generate(
                _seed_frame(),
                SynthRequest(
                    target_rows=10,
                    method=SynthMethod.RULE_INJECTION,
                    categorical_columns=(),
                    augment_label=Decision.BLACK,
                ),
            )


# ===========================================================================
# 3. recommend_ratio: a missing class reported as "balanced"
# ===========================================================================


class TestRecommendRatioMissingClass:
    def test_missing_white_is_not_reported_as_balanced(self) -> None:
        """1.0 means perfectly balanced; the API then said 黑白样本已均衡."""
        assert recommend_ratio({Decision.BLACK.value: 100}) is None

    def test_missing_black_is_not_reported_as_balanced(self) -> None:
        assert recommend_ratio({Decision.WHITE.value: 100}) is None

    def test_both_present_still_reports_the_ratio(self) -> None:
        assert recommend_ratio({Decision.BLACK.value: 100, Decision.WHITE.value: 5}) == 20.0


# ===========================================================================
# 4. Fidelity: the label column was hardcoded, distorting the score
# ===========================================================================


class TestFidelityLabelColumn:
    def test_custom_label_column_is_excluded_from_the_feature_score(self) -> None:
        seed = _seed_frame().rename(columns={"label": "is_fraud"})
        synth = seed.copy()
        synth["amount"] = synth["amount"] * 100  # break the feature distribution

        report = assess_fidelity(seed, synth, label_column="is_fraud")
        assert report.score < 0.2, report.score
        assert report.per_column["is_fraud"] == 1.0
        assert any("is_fraud" in note for note in report.notes)

    def test_default_label_column_still_excluded(self) -> None:
        report = assess_fidelity(_seed_frame(), _seed_frame())
        assert report.score > 0.9


# ===========================================================================
# 5. Split: gray rows must not vanish behind a tiny seed pool
# ===========================================================================


class TestSplitNeverLosesGray:
    def test_gray_survives_when_seed_is_tiny(self) -> None:
        rows = []
        for label, count in ((Decision.BLACK, 3), (Decision.WHITE, 3), (Decision.GRAY, 3)):
            rows += [
                {
                    "account": f"{label.value[0].upper()}{i}",
                    "label": label.value,
                    "origin": DataOrigin.SEED.value,
                }
                for i in range(count)
            ]
        frame = pd.DataFrame(rows)
        result = stratified_split(frame)
        assert len(result.test) > 0
        assert result.summary.test_contains_synth is False
