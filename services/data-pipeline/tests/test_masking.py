"""Tests for sensitive field detection and masking (需求方案.txt 5.2 / 5.3)."""

from __future__ import annotations

import pandas as pd
import pytest
from son_data_pipeline.masking import (
    MASKING_RULES,
    SensitiveKind,
    apply_masking,
    detect_sensitive_columns,
)


@pytest.fixture
def sensitive_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account": ["A001", "A002", "A003"],
            "身份证号": ["110101199001011234", "310101198505056789", "440101199512123456"],
            "手机号": ["13812345678", "15900001111", "18612340000"],
            "银行卡号": ["6222021234567890123", "6212261234567890123", "6214830212345678"],
            "amount": [100, 200, 300],
            "label": ["黑", "白", "灰"],
        }
    )


class TestDetectionByName:
    def test_finds_all_three_documented_kinds(self, sensitive_frame: pd.DataFrame) -> None:
        detected = detect_sensitive_columns(sensitive_frame)
        assert detected["身份证号"] is SensitiveKind.ID_CARD
        assert detected["手机号"] is SensitiveKind.PHONE
        assert detected["银行卡号"] is SensitiveKind.BANK_CARD

    def test_leaves_business_columns_alone(self, sensitive_frame: pd.DataFrame) -> None:
        detected = detect_sensitive_columns(sensitive_frame)
        assert "amount" not in detected
        assert "account" not in detected
        assert "label" not in detected

    def test_english_column_names_detected(self) -> None:
        frame = pd.DataFrame({"id_card": ["110101199001011234"] * 6})
        assert detect_sensitive_columns(frame)["id_card"] is SensitiveKind.ID_CARD


class TestDetectionByValue:
    def test_opaque_column_name_still_detected(self) -> None:
        """Real uploads have columns like `col_17`; a name-only rule misses those."""
        frame = pd.DataFrame({"col_17": ["110101199001011234"] * 10})
        detected = detect_sensitive_columns(frame)
        assert detected.get("col_17") is SensitiveKind.ID_CARD

    def test_low_match_rate_not_flagged(self) -> None:
        frame = pd.DataFrame({"col_x": ["110101199001011234"] + [f"value-{i}" for i in range(19)]})
        assert "col_x" not in detect_sensitive_columns(frame)

    def test_tiny_sample_is_not_value_scanned(self) -> None:
        frame = pd.DataFrame({"col_y": ["110101199001011234"] * 3})
        assert "col_y" not in detect_sensitive_columns(frame)


class TestMasking:
    def test_id_card_keeps_prefix_and_suffix(self) -> None:
        rule = next(r for r in MASKING_RULES if r.kind is SensitiveKind.ID_CARD)
        assert rule.mask_value("110101199001011234") == "110101********1234"

    def test_phone_keeps_prefix_and_suffix(self) -> None:
        rule = next(r for r in MASKING_RULES if r.kind is SensitiveKind.PHONE)
        assert rule.mask_value("13812345678") == "138****5678"

    def test_bank_card_keeps_prefix_and_suffix(self) -> None:
        rule = next(r for r in MASKING_RULES if r.kind is SensitiveKind.BANK_CARD)
        masked = rule.mask_value("6222021234567890123")
        assert masked.startswith("6222") and masked.endswith("0123")
        assert "*" in masked

    def test_masked_output_is_not_reversible(self, sensitive_frame: pd.DataFrame) -> None:
        """There is deliberately no unmask path; verify the middle is truly gone."""
        masked, _ = apply_masking(sensitive_frame)
        for original, hidden in zip(sensitive_frame["身份证号"], masked["身份证号"], strict=True):
            assert hidden != original
            assert original[6:14] not in hidden

    def test_masking_leaves_business_columns_untouched(self, sensitive_frame: pd.DataFrame) -> None:
        masked, _ = apply_masking(sensitive_frame)
        assert masked["amount"].tolist() == sensitive_frame["amount"].tolist()

    def test_masking_does_not_mutate_the_input(self, sensitive_frame: pd.DataFrame) -> None:
        before = sensitive_frame["手机号"].tolist()
        apply_masking(sensitive_frame)
        assert sensitive_frame["手机号"].tolist() == before

    def test_short_values_are_fully_masked(self) -> None:
        rule = next(r for r in MASKING_RULES if r.kind is SensitiveKind.PHONE)
        assert rule.mask_value("1381234") == "*******"

    def test_report_describes_what_5_3_shows(self, sensitive_frame: pd.DataFrame) -> None:
        _, report = apply_masking(sensitive_frame)
        described = report.describe()
        assert described == ["身份证号已脱敏", "手机号已脱敏", "银行卡号已脱敏"]

    def test_report_counts_masked_cells(self, sensitive_frame: pd.DataFrame) -> None:
        _, report = apply_masking(sensitive_frame)
        assert report.masked_cells == 9

    def test_email_masking(self) -> None:
        rule = next(r for r in MASKING_RULES if r.kind is SensitiveKind.EMAIL)
        assert "@" not in rule.mask_value("user@example.com")

    def test_explicit_detection_is_honoured(self, sensitive_frame: pd.DataFrame) -> None:
        masked, report = apply_masking(sensitive_frame, {"amount": SensitiveKind.PHONE})
        assert "amount" in report.masked_fields
        assert "身份证号" not in report.masked_fields
        assert masked["身份证号"].tolist() == sensitive_frame["身份证号"].tolist()
