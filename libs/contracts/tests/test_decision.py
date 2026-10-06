"""V3 semantics remain distinct from the legacy generative contract."""

import pytest
from pydantic import ValidationError

from son_contracts.decision import (
    ChoiceProbabilities,
    ChoiceTarget,
    DecisionOption,
    DecisionPredictRequest,
    DecisionPredictResponse,
    DecisionSample,
    DecisionSchema,
    FieldMapping,
    FieldRole,
)
from son_contracts.enums import Decision


def test_schema_is_three_way_and_semantics_are_frozen() -> None:
    options = tuple(
        DecisionOption(id=choice, name=choice.value, description="业务含义") for choice in Decision
    )
    schema = DecisionSchema(
        schema_id="dec_schema_1", version=1, question="如何分类？", options=options
    )
    with pytest.raises(ValidationError):
        schema.options[0].description = "修改历史语义"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        DecisionSchema(
            schema_id="dec_schema_1",
            version=1,
            question="如何分类？",
            options=(options[0], options[0], options[2]),
        )


@pytest.mark.parametrize("key", ["label", "reason", "evidence", "confidence", "TARGET"])
def test_target_and_audit_fields_cannot_enter_state(key: str) -> None:
    with pytest.raises(ValidationError):
        DecisionPredictRequest(model="mv_1", state={"nested": [{key: "secret"}]})
    with pytest.raises(ValidationError):
        FieldMapping(column=key, role=FieldRole.STATE_INPUT)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_and_probabilities_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        DecisionPredictRequest(model="mv_1", state={"x": [value]})
    with pytest.raises(ValidationError):
        ChoiceProbabilities(black=value, white=0, gray=0)


def test_evidence_is_optional_and_never_required_for_prediction() -> None:
    sample = DecisionSample(
        sample_id="s1", group_id="g1", state={"x": 1}, target=ChoiceTarget(choice=Decision.GRAY)
    )
    assert sample.evidence is None
    response = DecisionPredictResponse(
        trace_id="trace_1",
        model="mv_1",
        schema_version=1,
        choice=Decision.GRAY,
        probabilities=ChoiceProbabilities(black=0.05, white=0.05, gray=0.9),
        confidence=0.9,
        action="AUTO",
        policy="pol_1",
    )
    assert "reason" not in response.model_dump()
    assert response.action == "AUTO"  # gray is a class, not the REVIEW action.


@pytest.mark.parametrize(
    "updates",
    [
        {"confidence": 0.4},
        {"choice": "white"},
        {"probabilities": {"black": 0.4, "white": 0.2, "gray": 0.1}},
        {"reason": "not an output"},
        {"action": "auto"},
    ],
)
def test_probability_readout_is_consistent(updates: dict[str, object]) -> None:
    payload = {
        "trace_id": "trace_1",
        "model": "mv_1",
        "schema_version": 1,
        "choice": "black",
        "probabilities": {"black": 0.8, "white": 0.1, "gray": 0.1},
        "confidence": 0.8,
        "action": "REVIEW",
        "policy": "pol_1",
    }
    with pytest.raises(ValidationError):
        DecisionPredictResponse.model_validate(payload | updates)
