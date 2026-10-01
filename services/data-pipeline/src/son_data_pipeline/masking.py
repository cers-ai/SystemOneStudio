"""Sensitive field detection and masking.

需求方案.txt 5.2: on upload, sensitive fields are masked automatically and the
user can inspect the masking rules. Section 5.3 shows the post-governance report
asserting 身份证 / 手机号 / 银行卡号 are all masked.

Two detection passes, because column names alone are not enough:

* by column name (the usual case, and the cheapest)
* by value shape (catches `col_17` full of 18-digit strings)

Masking is non-reversible by construction: we keep a prefix and a suffix for
human readability and discard the middle. There is deliberately no unmask path,
because 需求方案.txt 13 treats synthetic-data privacy as a compliance risk and a
reversible mask is not a mask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd


class SensitiveKind(StrEnum):
    ID_CARD = "id_card"
    PHONE = "phone"
    BANK_CARD = "bank_card"
    EMAIL = "email"


@dataclass(frozen=True)
class MaskingRule:
    """How one kind of sensitive value is masked."""

    kind: SensitiveKind
    label: str
    keep_prefix: int
    keep_suffix: int
    value_pattern: re.Pattern[str]
    name_pattern: re.Pattern[str]
    description: str

    def mask_value(self, value: Any) -> Any:
        """Mask a single value. Non-strings pass through untouched."""
        text = str(value)
        if len(text) <= self.keep_prefix + self.keep_suffix:
            return "*" * len(text)
        hidden = len(text) - self.keep_prefix - self.keep_suffix
        # Guard both slices: `text[-0:]` is the whole string, not an empty one,
        # so a keep_suffix of 0 has to be spelled out.
        prefix = text[: self.keep_prefix]
        suffix = text[len(text) - self.keep_suffix :] if self.keep_suffix else ""
        return f"{prefix}{'*' * hidden}{suffix}"

    def matches_value(self, value: Any) -> bool:
        return bool(self.value_pattern.fullmatch(str(value).strip()))


MASKING_RULES: tuple[MaskingRule, ...] = (
    MaskingRule(
        kind=SensitiveKind.ID_CARD,
        label="身份证号",
        keep_prefix=6,
        keep_suffix=4,
        value_pattern=re.compile(r"\d{17}[\dXx]"),
        name_pattern=re.compile(r"id_?card|identity|身份证|证件号|sfz"),
        description="保留前 6 位与后 4 位，中间 8 位掩码",
    ),
    MaskingRule(
        kind=SensitiveKind.PHONE,
        label="手机号",
        keep_prefix=3,
        keep_suffix=4,
        value_pattern=re.compile(r"1[3-9]\d{9}"),
        name_pattern=re.compile(r"phone|mobile|tel|手机号|电话"),
        description="保留前 3 位与后 4 位，中间 4 位掩码",
    ),
    MaskingRule(
        kind=SensitiveKind.BANK_CARD,
        label="银行卡号",
        keep_prefix=4,
        keep_suffix=4,
        value_pattern=re.compile(r"\d{16,19}"),
        name_pattern=re.compile(r"bank_?card|account_?no|银行卡|卡号"),
        description="保留前 4 位与后 4 位，中间掩码",
    ),
    MaskingRule(
        kind=SensitiveKind.EMAIL,
        label="邮箱",
        keep_prefix=1,
        keep_suffix=0,
        value_pattern=re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+"),
        name_pattern=re.compile(r"e-?mail|邮箱"),
        description="保留首字符，@ 之后全部掩码",
    ),
)


@dataclass(frozen=True)
class MaskingReport:
    """What was detected and what was masked, per 需求方案.txt 5.3."""

    masked_fields: dict[str, SensitiveKind]
    masked_cells: int
    rules: tuple[MaskingRule, ...]

    def describe(self) -> list[str]:
        return [f"{rule.label}已脱敏" for rule in self.rules]


def detect_sensitive_columns(
    frame: pd.DataFrame,
    *,
    value_sample_ratio: float = 0.5,
    min_rows_for_value_scan: int = 5,
) -> dict[str, SensitiveKind]:
    """Detect sensitive columns by name, then confirm or extend by value shape.

    A value-shape pass alone would flag an ID column of sequential test fixtures;
    requiring either a matching name or a high match rate keeps false positives
    down.
    """
    detected: dict[str, SensitiveKind] = {}

    for column in frame.columns:
        text = str(column)
        for rule in MASKING_RULES:
            if rule.name_pattern.search(text):
                detected[str(column)] = rule.kind
                break

    for column in frame.columns:
        name = str(column)
        if name in detected:
            continue
        series = frame[column].dropna()
        if len(series) < min_rows_for_value_scan:
            continue
        for rule in MASKING_RULES:
            hits = sum(1 for value in series.head(200) if rule.matches_value(value))
            if hits / min(len(series), 200) >= value_sample_ratio:
                detected[name] = rule.kind
                break

    return detected


def apply_masking(
    frame: pd.DataFrame,
    detected: dict[str, SensitiveKind] | None = None,
) -> tuple[pd.DataFrame, MaskingReport]:
    """Return a masked copy plus a report of what was masked."""
    detected = detected if detected is not None else detect_sensitive_columns(frame)
    rules_by_kind = {rule.kind: rule for rule in MASKING_RULES}

    masked = frame.copy()
    masked_cells = 0
    applied: list[MaskingRule] = []

    for column, kind in detected.items():
        if column not in masked.columns:
            continue
        rule = rules_by_kind[kind]
        non_null = int(masked[column].notna().sum())
        masked[column] = masked[column].map(rule.mask_value)
        masked_cells += non_null
        applied.append(rule)

    return masked, MaskingReport(
        masked_fields=detected,
        masked_cells=masked_cells,
        rules=tuple(applied),
    )
