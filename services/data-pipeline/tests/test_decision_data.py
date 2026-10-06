"""Guard V3 isolation, canonical serialization and candidate remapping."""

import itertools

import pytest
from son_data_pipeline.decision_data import (
    SplitMode,
    canonical_state,
    render_decision_input,
    split_decision_seeds,
    target_slot,
)

from son_contracts.decision import ChoiceTarget, DecisionOption, DecisionSample, DecisionSchema
from son_contracts.enums import Decision


def seeds(count: int = 120) -> list[DecisionSample]:
    choices = list(Decision)
    return [
        DecisionSample(
            sample_id=f"s{i}",
            group_id=f"g{i // 2}",
            state={"feature": i},
            target=ChoiceTarget(choice=choices[(i // 2) % 3]),
        )
        for i in range(count)
    ]


def test_canonical_state_preserves_text_and_types_and_normalizes_numbers() -> None:
    assert (
        canonical_state({"z": -0.0, "a": {"y": 1.0, "x": None}, "id": "001"})
        == '{"a":{"x":null,"y":1},"id":"001","z":0}'
    )
    assert canonical_state({"b": True, "a": " 中文 "}) == canonical_state(
        {"a": " 中文 ", "b": True}
    )


@pytest.mark.parametrize("mode", ["independent", "shared_valid_cal"])
def test_split_is_deterministic_group_isolated_and_frozen(mode: SplitMode) -> None:
    rows = seeds()
    result = split_decision_seeds(rows, mode=mode)
    result.verify()
    assert (
        result.manifest_hash == split_decision_seeds(list(reversed(rows)), mode=mode).manifest_hash
    )
    partitions = [result.train, result.valid, result.test]
    if mode == "independent":
        partitions.append(result.cal)
    else:
        assert result.cal == result.valid
        assert any("独立性降低" in warning for warning in result.warnings)
    seen: set[str] = set()
    for partition in partitions:
        groups = {row.group_id for row in partition}
        assert partition and not seen & groups
        seen.update(groups)
        assert all(row.source == "seed" for row in partition)
    assert sum(map(len, partitions)) == len(rows)
    rows[0].state["feature"] = 999999
    result.verify()  # Original input changes cannot modify the frozen copy.
    result.test[0].state["feature"] = 999999
    with pytest.raises(ValueError, match="已被修改"):
        result.verify()


def test_constructed_inputs_duplicate_ids_and_cross_group_duplicates_fail() -> None:
    rows = seeds()
    with pytest.raises(ValueError, match="构造数据"):
        split_decision_seeds([*rows, rows[0].model_copy(update={"source": "constructed"})])
    with pytest.raises(ValueError, match="唯一"):
        split_decision_seeds([*rows, rows[0]])
    with pytest.raises(ValueError, match="相同 State"):
        split_decision_seeds(
            [*rows, rows[0].model_copy(update={"sample_id": "dup", "group_id": "other"})]
        )


def test_insufficient_groups_never_silently_switch_split_mode() -> None:
    with pytest.raises(ValueError, match="group 数不足"):
        split_decision_seeds(seeds(6))
    result = split_decision_seeds(seeds(6), mode="shared_valid_cal")
    assert any("链路验证" in warning for warning in result.warnings)
    assert any("缺少类别" in warning for warning in result.warnings)


def test_candidate_permutations_remap_target_and_do_not_render_evidence() -> None:
    row = seeds()[0].model_copy(update={"evidence": "audit-secret"})
    schema = DecisionSchema(
        schema_id="dec_schema_1",
        version=1,
        question="如何判断？",
        options=tuple(
            DecisionOption(id=choice, name=choice.value, description="含义") for choice in Decision
        ),
    )
    for permutation in itertools.permutations(Decision):
        order = (permutation[0], permutation[1], permutation[2])
        prompt = render_decision_input(row.state, schema, order)
        assert order[target_slot(row, order)] == row.target.choice
        assert prompt.endswith("<DECISION>")
        assert f"{'ABC'[target_slot(row, order)]} = {row.target.choice.value}" in prompt
        assert "audit-secret" not in prompt
