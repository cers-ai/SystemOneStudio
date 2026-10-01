"""Tests that lock down the product's non-negotiable constraints.

These are not coverage tests. Each one guards a rule from 需求方案.txt that a
future refactor could plausibly break: the three-way label, the mandatory
`reason` field, the two independent JEV switches, and lineage code shapes.
"""

import pytest
from pydantic import ValidationError

from son_contracts import (
    DEFAULT_JEV_FLAGS,
    JEV_OUTPUT_SCHEMA,
    RAPID_MODE_LAST_MANUAL_STEP,
    REASON_MAX_LENGTH,
    WIZARD_STEP_ORDER,
    Decision,
    JevFlags,
    Lineage,
    PredictRequest,
    PredictResponse,
    Split,
    SplitSummary,
    StepState,
    validate_jev_payload,
)


class TestDecisionIsThreeWay:
    def test_exactly_black_white_gray(self) -> None:
        assert {d.value for d in Decision} == {"black", "white", "gray"}

    def test_gray_is_not_optional(self) -> None:
        assert Decision.GRAY in set(Decision)

    def test_bool_is_not_accepted(self) -> None:
        with pytest.raises(ValidationError):
            PredictResponse(decision=True, score=0.5, confidence=0.5, reason="x")


class TestReasonIsMandatory:
    def test_reason_required(self) -> None:
        with pytest.raises(ValidationError):
            PredictResponse(decision=Decision.BLACK, score=0.9, confidence=0.9)  # type: ignore[call-arg]

    def test_reason_cannot_be_blank(self) -> None:
        with pytest.raises(ValidationError):
            PredictResponse(decision=Decision.BLACK, score=0.9, confidence=0.9, reason="   ")

    def test_reason_cap_is_200(self) -> None:
        assert REASON_MAX_LENGTH == 200
        PredictResponse(decision=Decision.BLACK, score=0.9, confidence=0.9, reason="x" * 200)

    def test_reason_over_cap_rejected_not_truncated(self) -> None:
        """Over-length must fail loudly; silent truncation would falsify the
        format-compliance metric."""
        with pytest.raises(ValidationError):
            PredictResponse(decision=Decision.BLACK, score=0.9, confidence=0.9, reason="x" * 201)

    def test_oversized_reason_allowed_when_jev_disabled(self) -> None:
        resp = PredictResponse(
            decision=Decision.GRAY,
            score=0.1,
            confidence=0.2,
            reason="x" * 400,
            jev_compatible=False,
        )
        assert len(resp.reason) == 400


class TestJevPayloadValidation:
    def _valid(self) -> dict[str, object]:
        return {
            "decision": "black",
            "score": 0.87,
            "confidence": 0.92,
            "reason": "交易金额异常度高，设备风险评分高",
        }

    def test_valid_payload_passes(self) -> None:
        assert validate_jev_payload(self._valid(), DEFAULT_JEV_FLAGS).compliant

    def test_rejects_binary_decision(self) -> None:
        payload = self._valid() | {"decision": "fraud"}
        result = validate_jev_payload(payload, DEFAULT_JEV_FLAGS)
        assert not result.compliant
        assert any("decision" in e for e in result.errors)

    def test_rejects_missing_reason(self) -> None:
        payload = self._valid()
        del payload["reason"]
        result = validate_jev_payload(payload, DEFAULT_JEV_FLAGS)
        assert not result.compliant
        assert any("reason" in e for e in result.errors)

    def test_rejects_oversized_reason(self) -> None:
        payload = self._valid() | {"reason": "x" * 201}
        result = validate_jev_payload(payload, DEFAULT_JEV_FLAGS)
        assert not result.compliant
        assert any("maxLength" in e for e in result.errors)

    def test_rejects_out_of_range_score(self) -> None:
        assert not validate_jev_payload(self._valid() | {"score": 1.5}, DEFAULT_JEV_FLAGS).compliant
        assert not validate_jev_payload(
            self._valid() | {"confidence": -0.1}, DEFAULT_JEV_FLAGS
        ).compliant

    def test_rejects_unexpected_fields(self) -> None:
        result = validate_jev_payload(self._valid() | {"debug": 1}, DEFAULT_JEV_FLAGS)
        assert not result.compliant
        assert any("unexpected" in e for e in result.errors)

    def test_skipped_when_format_compat_off(self) -> None:
        """Non-JEV payloads are legal once jev_format_compat is disabled."""
        flags = JevFlags(jev_format_compat=False, jev_training_compat=True)
        assert validate_jev_payload({"whatever": True}, flags).compliant

    def test_schema_requires_exactly_four_fields(self) -> None:
        assert set(JEV_OUTPUT_SCHEMA["required"]) == {
            "decision",
            "score",
            "confidence",
            "reason",
        }


class TestJevSwitchesAreIndependent:
    def test_both_default_on(self) -> None:
        assert DEFAULT_JEV_FLAGS.jev_format_compat
        assert DEFAULT_JEV_FLAGS.jev_training_compat

    def test_format_can_be_off_while_training_stays_on(self) -> None:
        flags = JevFlags(jev_format_compat=False, jev_training_compat=True)
        assert not flags.jev_format_compat
        assert flags.jev_training_compat

    def test_training_can_be_off_while_format_stays_on(self) -> None:
        flags = JevFlags(jev_format_compat=True, jev_training_compat=False)
        assert flags.jev_format_compat
        assert not flags.jev_training_compat

    def test_flags_are_frozen(self) -> None:
        with pytest.raises(ValidationError):
            DEFAULT_JEV_FLAGS.jev_format_compat = False  # type: ignore[misc]

    def test_both_off_is_truthy_as_an_object(self) -> None:
        """A falsey model would make `flags or default` drop a deliberate
        'both off' configuration on the floor."""
        both_off = JevFlags(jev_format_compat=False, jev_training_compat=False)
        assert bool(both_off) is True
        assert both_off.any_enabled() is False

    def test_any_enabled_reports_true_when_one_is_on(self) -> None:
        assert JevFlags(jev_format_compat=False, jev_training_compat=True).any_enabled() is True


class TestLineageCodes:
    def _valid(self) -> Lineage:
        return Lineage(
            scene="sc_v3",
            dataset="ds_v2",
            synth="syn_v5",
            base_model="qwen2.5-3b-instruct",
            method="jev_lora_dpo",
            model_version="mv_0017",
        )

    def test_accepts_documented_code_style(self) -> None:
        assert self._valid().model_version == "mv_0017"

    def test_rejects_wrong_prefix(self) -> None:
        with pytest.raises(ValidationError):
            Lineage(
                scene="scene_v3",
                dataset="ds_v2",
                base_model="m",
                method="x",
                model_version="mv_0017",
            )

    def test_synth_is_optional_when_synthesis_skipped(self) -> None:
        lineage = self._valid().model_copy(update={"synth": None})
        assert Lineage.reconstruct(lineage.as_snapshot()).synth is None

    def test_describe_mentions_unset_synth(self) -> None:
        lineage = self._valid().model_copy(update={"synth": None})
        assert "(未合成)" in lineage.describe()

    def test_roundtrips_through_snapshot(self) -> None:
        assert Lineage.reconstruct(self._valid().as_snapshot()) == self._valid()


class TestSplitInvariants:
    def test_test_split_must_not_contain_synthetic(self) -> None:
        summary = SplitSummary(train_rows=70, valid_rows=15, test_rows=15)
        assert summary.test_contains_synth is False

    def test_violation_is_representable_so_it_can_be_asserted(self) -> None:
        """The field exists so an assertion can fail loudly on a bad split
        rather than the violation being invisible."""
        summary = SplitSummary(train_rows=70, valid_rows=15, test_rows=15, test_contains_synth=True)
        assert summary.test_contains_synth is True


class TestWizardFlow:
    def test_seven_user_facing_steps_then_evaluate_and_deploy(self) -> None:
        assert WIZARD_STEP_ORDER[:7] == (
            StepState.SCENE_SELECTED,
            StepState.SEED_UPLOADED,
            StepState.DATA_GOVERNED,
            StepState.SYNTH_DONE,
            StepState.BASE_MODEL_SELECTED,
            StepState.TRAINING_CONFIGURED,
            StepState.TRAINED,
        )

    def test_rapid_mode_stops_after_step_2(self) -> None:
        """Rapid mode only requires the user to complete steps 1 and 2."""
        assert RAPID_MODE_LAST_MANUAL_STEP == StepState.SEED_UPLOADED
        assert WIZARD_STEP_ORDER.index(RAPID_MODE_LAST_MANUAL_STEP) == 1

    def test_split_names(self) -> None:
        assert {s.value for s in Split} == {"train", "valid", "test"}


class TestPredictRequestPassthrough:
    def test_collects_arbitrary_scene_fields(self) -> None:
        req = PredictRequest(account="A12345", amount=50000, device="iOS 15.0")
        assert req.sample == {"account": "A12345", "amount": 50000, "device": "iOS 15.0"}

    def test_platform_fields_excluded_from_sample(self) -> None:
        req = PredictRequest(model_version_id="mv_0017", account="A1")
        assert "model_version_id" not in req.sample

    def test_feature_rendering_is_stable(self) -> None:
        a = PredictRequest(amount=1, account="A", device="d").render_features()
        b = PredictRequest(device="d", account="A", amount=1).render_features()
        assert a == b
        assert a == "account: A\namount: 1\ndevice: d"

    def test_to_jev_dict_drops_compat_marker(self) -> None:
        resp = PredictResponse(decision=Decision.GRAY, score=0.1, confidence=0.2, reason="证据不足")
        assert set(resp.to_jev_dict()) == {"decision", "score", "confidence", "reason"}
