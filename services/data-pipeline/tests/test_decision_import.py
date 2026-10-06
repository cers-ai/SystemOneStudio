"""Exercise the CSV-to-frozen-artifact path and preserve source files."""

import json
from pathlib import Path

import pytest
from son_data_pipeline.decision_cli import prepare
from son_data_pipeline.decision_data import split_decision_seeds
from son_data_pipeline.decision_import import import_decision_csv

from son_contracts.decision import FieldMapping, FieldRole, StateValueType


def mappings() -> list[FieldMapping]:
    return [
        FieldMapping(column="item", role=FieldRole.ENTITY_ID),
        FieldMapping(column="value", role=FieldRole.STATE_INPUT, value_type=StateValueType.NUMBER),
        FieldMapping(column="decision", role=FieldRole.TARGET_CHOICE),
        FieldMapping(column="reason", role=FieldRole.EVIDENCE),
    ]


def csv_data(count: int = 30) -> bytes:
    return (
        "item,value,decision,reason\n"
        + "".join(
            f"{i:03},{i},{('black', 'white', 'gray')[i % 3]},audit-{i}\n" for i in range(count)
        )
    ).encode()


def test_explicit_csv_roles_preserve_ids_and_exclude_evidence() -> None:
    samples = import_decision_csv(csv_data(), mappings())
    assert samples[0].group_id == "000"
    assert samples[0].state == {"value": 0}
    assert samples[0].evidence == "audit-0"
    assert samples == import_decision_csv(csv_data(), mappings())
    split_decision_seeds(samples).verify()


def test_group_is_required_unless_independent_rows_explicitly_confirmed() -> None:
    roles = [
        mapping.model_copy(update={"role": FieldRole.IGNORE})
        if mapping.column == "item"
        else mapping
        for mapping in mappings()
    ]
    with pytest.raises(ValueError, match="独立种子"):
        import_decision_csv(csv_data(), roles)
    samples = import_decision_csv(csv_data(), roles, independent_rows=True)
    assert samples[0].metadata["independent_rows"] is True


@pytest.mark.parametrize(
    "content",
    [
        b"item,value,decision,reason\na,not-a-number,black,secret\n",
        b"item,value,decision,reason\na,NaN,black,secret\n",
        b"item,value,decision,reason\na,1,unknown,secret\n",
        b"item,value,decision,reason\na,1,black,secret,extra\n",
        b"item,value,decision,reason\na,1,black\n",
    ],
)
def test_bad_csv_is_rejected_without_echoing_audit_content(content: bytes) -> None:
    with pytest.raises(ValueError) as failure:
        import_decision_csv(content, mappings())
    assert "secret" not in str(failure.value)


def test_same_entity_cannot_cross_groups() -> None:
    samples = import_decision_csv(csv_data(), mappings())
    samples[1].metadata["entity_id"] = samples[0].metadata["entity_id"]
    with pytest.raises(ValueError, match="同一entity"):
        split_decision_seeds(samples)


def test_cli_artifacts_are_real_data_and_never_overwrite(tmp_path: Path) -> None:
    dataset, schema, mapping = (
        tmp_path / name for name in ("seed.csv", "schema.json", "mapping.json")
    )
    dataset.write_bytes(csv_data())
    schema.write_text(
        json.dumps(
            {
                "schema_id": "dec_schema_1",
                "version": 1,
                "question": "如何分类？",
                "options": [
                    {"id": choice, "name": choice, "description": "业务含义"}
                    for choice in ("black", "white", "gray")
                ],
            }
        ),
        encoding="utf-8",
    )
    mapping.write_text(
        json.dumps([role.model_dump(mode="json") for role in mappings()]), encoding="utf-8"
    )
    output = tmp_path / "prepared"
    result = prepare(dataset, schema, output, mapping_path=mapping)
    assert result["status"] == "DATA_PREPARED" and result["model_trained"] is False
    inputs = json.loads((output / "training_inputs.json").read_text())
    assert "test" not in inputs
    assert inputs["schema_sha256"]
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert set(manifest["files"]) == {
        "schema.json",
        "train.jsonl",
        "valid.jsonl",
        "cal.jsonl",
        "test.jsonl",
        "split_manifest.json",
        "training_inputs.json",
    }
    with pytest.raises(ValueError, match="拒绝覆盖"):
        prepare(dataset, schema, output, mapping_path=mapping)
    assert dataset.read_bytes() == csv_data()
