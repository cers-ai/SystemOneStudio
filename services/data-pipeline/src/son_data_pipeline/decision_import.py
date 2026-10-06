"""Explicit CSV roles and value types; never infer labels from feature values."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from son_contracts.decision import (
    ChoiceTarget,
    DecisionSample,
    FieldMapping,
    FieldRole,
    StateValueType,
)
from son_contracts.enums import Decision


def _value(text: str | None, value_type: StateValueType) -> JsonValue:
    if text is None or text == "":
        return None
    if value_type == StateValueType.TEXT:
        return text  # Preserve identifiers, whitespace, dates and leading zeros.
    if value_type == StateValueType.BOOLEAN:
        if text.strip().casefold() not in ("true", "false"):
            raise ValueError("布尔字段必须使用 true/false")
        return text.strip().casefold() == "true"
    parsed: JsonValue = json.loads(text)
    if value_type == StateValueType.NUMBER and (
        isinstance(parsed, bool) or not isinstance(parsed, int | float)
    ):
        raise ValueError("数值字段必须是有限数字")
    return parsed


def import_decision_csv(
    content: bytes,
    mappings: Sequence[FieldMapping],
    *,
    label_aliases: Mapping[str, Decision] | None = None,
    independent_rows: bool = False,
) -> list[DecisionSample]:
    """No group column requires an explicit declaration of independent rows.

    Source IDs identify a file and row; group/ENTITY IDs are never State unless
    a separate business field is explicitly mapped as an input. Every source
    column needs an explicit role so unknown audit fields cannot slip through.
    """
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    columns = reader.fieldnames
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any(not name.strip() for name in columns)
    ):
        raise ValueError("CSV表头为空或存在重复/空字段")
    by_column = {mapping.column: mapping for mapping in mappings}
    if len(by_column) != len(mappings) or set(by_column) != set(columns):
        raise ValueError("每个CSV字段必须且只能设置一个角色")
    by_role = {
        role: [mapping.column for mapping in mappings if mapping.role == role] for role in FieldRole
    }
    if len(by_role[FieldRole.TARGET_CHOICE]) != 1 or not by_role[FieldRole.STATE_INPUT]:
        raise ValueError("必须指定一个标签字段和至少一个模型输入字段")
    if len(by_role[FieldRole.GROUP_ID]) > 1 or len(by_role[FieldRole.ENTITY_ID]) > 1:
        raise ValueError("最多指定一个group字段和一个entity字段")
    group_columns = by_role[FieldRole.GROUP_ID] or by_role[FieldRole.ENTITY_ID]
    if not group_columns and not independent_rows:
        raise ValueError("请选择group/entity字段，或明确确认各行是独立种子")
    source_hash = hashlib.sha256(content).hexdigest()
    aliases = {choice.value: choice for choice in Decision}
    if label_aliases:
        aliases.update(label_aliases)
    samples = []
    for number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"第{number}行的字段数量与表头不一致")
        label = row[by_role[FieldRole.TARGET_CHOICE][0]].strip()
        if label not in aliases:
            raise ValueError(f"第{number}行标签未映射到black/white/gray")
        state: dict[str, JsonValue] = {}
        for column in by_role[FieldRole.STATE_INPUT]:
            try:
                state[column] = _value(row[column], by_column[column].value_type)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"第{number}行字段{column}类型无效") from exc
        sample_id = f"sample_{source_hash[:16]}_{number}"
        group_id = row[group_columns[0]].strip() if group_columns else sample_id
        evidence = "\n".join(row[column] for column in by_role[FieldRole.EVIDENCE]) or None
        metadata: dict[str, JsonValue] = {
            "source_sha256": source_hash,
            "row": number,
            "independent_rows": independent_rows and not group_columns,
        }
        if by_role[FieldRole.ENTITY_ID]:
            entity = row[by_role[FieldRole.ENTITY_ID][0]].strip()
            if not entity:
                raise ValueError(f"第{number}行entity为空")
            metadata["entity_id"] = entity
        try:
            samples.append(
                DecisionSample(
                    sample_id=sample_id,
                    group_id=group_id,
                    state=state,
                    target=ChoiceTarget(choice=aliases[label]),
                    evidence=evidence,
                    metadata=metadata,
                )
            )
        except ValueError as exc:
            raise ValueError(f"第{number}行样本结构或输入违反契约") from exc
    if not samples:
        raise ValueError("CSV没有种子样本")
    return samples
