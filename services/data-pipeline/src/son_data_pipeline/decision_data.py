"""V3 canonical samples and group splits, independent of the legacy JEV path."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from son_contracts.decision import DecisionSample, DecisionSchema, validate_state
from son_contracts.enums import Decision

SERIALIZER_VERSION = "state-json-v1"
SplitMode = Literal["independent", "shared_valid_cal"]


def canonical_state(state: dict[str, JsonValue]) -> str:
    """One deterministic JSON representation, used for both train and runtime.

    Dates are already ISO strings in the input contract. Finite integral floats
    and negative zero normalize to integers; array order and string content stay
    intact. No implicit date, boolean, or identifier coercion takes place.
    """
    validate_state(state)

    def normalize(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {key: normalize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [normalize(child) for child in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    return json.dumps(
        normalize(state), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_decision_input(
    state: dict[str, JsonValue],
    schema: DecisionSchema,
    order: tuple[Decision, Decision, Decision] = (Decision.BLACK, Decision.WHITE, Decision.GRAY),
) -> str:
    """Render choices in requested order. Token resolution is a GPU Spike gate."""
    if len(order) != 3 or set(order) != set(Decision):
        raise ValueError("候选顺序必须恰好包含三个业务类别")
    options = {option.id: option for option in schema.options}
    lines = []
    for slot, choice in zip("ABC", order, strict=True):
        option = options[choice]
        lines.append(f"{slot} = {choice.value}: {option.name} — {option.description}")
    options_text = "\n".join(lines)
    return (
        f"<STATE>\n{canonical_state(state)}\n</STATE>\n\n"
        f"<QUESTION>\n{schema.question}\n</QUESTION>\n\n"
        f"<OPTIONS>\n{options_text}\n</OPTIONS>\n\n<DECISION>"
    )


def target_slot(sample: DecisionSample, order: tuple[Decision, Decision, Decision]) -> int:
    if len(order) != 3 or set(order) != set(Decision):
        raise ValueError("候选顺序无效")
    return order.index(sample.target.choice)


@dataclass(frozen=True)
class DecisionDataSplit:
    train: tuple[DecisionSample, ...]
    valid: tuple[DecisionSample, ...]
    cal: tuple[DecisionSample, ...]
    test: tuple[DecisionSample, ...]
    mode: SplitMode
    seed: int
    warnings: tuple[str, ...]
    manifest_hash: str

    def verify(self) -> None:
        encoded = json.dumps(
            self.manifest(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if content_hash(encoded) != self.manifest_hash:
            raise ValueError("冻结数据或清单已被修改")

    def manifest(self) -> dict[str, object]:
        return {
            "serializer": SERIALIZER_VERSION,
            "mode": self.mode,
            "seed": self.seed,
            "partitions": {
                name: [
                    {
                        "sample_id": row.sample_id,
                        "group_id": row.group_id,
                        "sha256": sample_hash(row),
                    }
                    for row in getattr(self, name)
                ]
                for name in ("train", "valid", "cal", "test")
            },
            "warnings": list(self.warnings),
        }


def sample_hash(sample: DecisionSample) -> str:
    """Audit metadata is included in the frozen sample fingerprint."""
    return content_hash(
        json.dumps(
            sample.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def split_decision_seeds(
    samples: list[DecisionSample], *, mode: SplitMode = "independent", seed: int = 42
) -> DecisionDataSplit:
    """Split only original seeds before any augmentation; never silently switch modes.

    Greedy whole-group stratification minimizes normalized size/class deficits.
    Group isolation has precedence over exact ratios. Shared mode intentionally
    aliases valid/cal and makes the reduced independence visible in its manifest.
    """
    if mode not in ("independent", "shared_valid_cal"):
        raise ValueError("未知划分模式")
    if not samples:
        raise ValueError("不能划分空数据")
    if any(row.source != "seed" for row in samples):
        raise ValueError("冻结划分只接受原始种子；构造数据必须在划分之后加入 train")
    if len({row.sample_id for row in samples}) != len(samples):
        raise ValueError("sample_id 必须唯一")
    samples = [DecisionSample.model_validate(row.model_dump(mode="json")) for row in samples]

    groups: dict[str, list[DecisionSample]] = defaultdict(list)
    state_groups: dict[str, str] = {}
    entity_groups: dict[str, str] = {}
    for row in samples:
        # Contracts contain mutable JSON trees: revalidate before freezing them.
        validate_state(row.state)
        fingerprint = content_hash(canonical_state(row.state))
        previous = state_groups.setdefault(fingerprint, row.group_id)
        if previous != row.group_id:
            raise ValueError("相同 State 被分配到不同 group；请合并分组，避免重复样本泄漏")
        entity = row.metadata.get("entity_id")
        if entity is not None:
            if not isinstance(entity, str) or not entity.strip():
                raise ValueError("entity_id 必须为非空字符串")
            if entity_groups.setdefault(entity, row.group_id) != row.group_id:
                raise ValueError("同一entity被分配到不同group；请合并分组")
        groups[row.group_id].append(row)

    ratios = {"train": 0.7, "valid": 0.1, "cal": 0.1, "test": 0.1}
    if mode == "shared_valid_cal":
        ratios = {"train": 0.7, "valid": 0.15, "test": 0.15}
    if len(groups) < len(ratios):
        raise ValueError("独立 group 数不足以形成非空分区；请增加种子或显式选择共享验证校准模式")

    total_classes = Counter(row.target.choice for row in samples)
    allocated: dict[str, list[DecisionSample]] = {name: [] for name in ratios}
    class_counts: dict[str, Counter[Decision]] = {name: Counter() for name in ratios}
    ordered = sorted(
        groups, key=lambda group: (-len(groups[group]), content_hash(f"{seed}:{group}"))
    )
    for index, group in enumerate(ordered):
        rows = groups[group]
        counts = Counter(row.target.choice for row in rows)
        empty = [name for name, partition in allocated.items() if not partition]
        candidates = empty if len(ordered) - index == len(empty) else list(ratios)

        def cost(
            name: str,
            group_size: int = len(rows),
            group_counts: Counter[Decision] = counts,
        ) -> float:
            # Increase in squared deficits, evaluated globally by each delta.
            expected = max(1.0, len(samples) * ratios[name])
            before = len(allocated[name]) - expected
            score = ((before + group_size) ** 2 - before**2) / expected
            for choice, total in total_classes.items():
                target = max(1.0, total * ratios[name])
                delta = class_counts[name][choice] - target
                score += ((delta + group_counts[choice]) ** 2 - delta**2) / target
            return score

        chosen = min(candidates, key=cost)
        allocated[chosen].extend(rows)
        class_counts[chosen].update(counts)

    for rows in allocated.values():
        rows.sort(key=lambda row: row.sample_id)
    if mode == "shared_valid_cal":
        allocated["cal"] = allocated["valid"]
    warnings = []
    if len(samples) < 200:
        warnings.append("真实种子少于200条，仅用于链路验证，不代表业务效果验收")
    if mode == "shared_valid_cal":
        warnings.append("验证与校准共用样本，校准独立性降低")
    for name, rows in allocated.items():
        missing = set(Decision) - {row.target.choice for row in rows}
        if missing:
            warnings.append(f"{name} 缺少类别：{', '.join(sorted(missing))}")
    warnings.append("按完整group划分，实际比例可能偏离目标；以清单数量为准")
    result = DecisionDataSplit(
        train=tuple(allocated["train"]),
        valid=tuple(allocated["valid"]),
        cal=tuple(allocated["cal"]),
        test=tuple(allocated["test"]),
        mode=mode,
        seed=seed,
        warnings=tuple(warnings),
        manifest_hash="",
    )
    manifest_hash = content_hash(
        json.dumps(result.manifest(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return DecisionDataSplit(
        train=result.train,
        valid=result.valid,
        cal=result.cal,
        test=result.test,
        mode=mode,
        seed=seed,
        warnings=result.warnings,
        manifest_hash=manifest_hash,
    )
