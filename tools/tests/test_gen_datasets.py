"""Tests for the bundled example datasets.

A fixture nobody has run is decoration. These load every generated file through
the real ingestion path and assert the properties the README claims, so a
regenerated dataset that quietly stops exercising a behaviour fails here rather
than in a bug report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from son_data_pipeline import detect_label_column, label_counts, recommend_ratio, stratified_split

from son_contracts import DataOrigin, Decision

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def _csv(relative: str) -> Path:
    path = DATASET_DIR / relative
    assert path.exists(), f"缺少示例数据集 {relative}；请运行 uv run python tools/gen_datasets.py"
    return path


def _frame(relative: str) -> pd.DataFrame:
    return pd.read_csv(_csv(relative))


#: Label aliases accepted by the platform. Shared rather than rebuilt per test.
_MAPPING = {d.value: d for d in Decision}


class TestFilesExist:
    @pytest.mark.parametrize(
        "relative",
        [
            "fraud/seed_200.csv",
            "fraud/seed_2000.csv",
            "payment/seed_500.csv",
            "content/seed_300.csv",
            "edge_cases/minimal_8_rows.csv",
            "edge_cases/single_class_black_only.csv",
            "edge_cases/no_gray_label.csv",
            "edge_cases/heavy_missing.csv",
            "edge_cases/with_outliers.csv",
            "edge_cases/sensitive_fields.csv",
            "edge_cases/heavily_unbalanced.csv",
        ],
    )
    def test_present_and_non_empty(self, relative: str) -> None:
        frame = _frame(relative)
        assert not frame.empty
        assert frame.columns[0] != ""

    def test_readme_exists(self) -> None:
        assert (DATASET_DIR / "README.md").exists()


class TestLabelDetection:
    @pytest.mark.parametrize(
        "relative",
        ["fraud/seed_200.csv", "payment/seed_500.csv", "content/seed_300.csv"],
    )
    def test_label_column_is_found(self, relative: str) -> None:
        """Zero-data usability depends on the example files being loadable."""
        detection = detect_label_column(_frame(relative))
        assert detection.column == "label"
        assert detection.confidence in ("high", "medium")

    @pytest.mark.parametrize(
        "relative",
        ["fraud/seed_200.csv", "fraud/seed_2000.csv", "payment/seed_500.csv"],
    )
    def test_all_three_classes_present(self, relative: str) -> None:
        """A bundled example where gray is absent would make the class meaningless."""
        counts = label_counts(_frame(relative), "label", _MAPPING)
        assert counts is not None
        assert all(counts[d.value] > 0 for d in Decision)


class TestClassesCarrySignal:
    """A generator producing identical distributions would leave nothing to learn."""

    def test_black_and_white_accounts_differ(self) -> None:
        frame = _frame("fraud/seed_200.csv")
        black = frame[frame["label"] == "black"]["txn_amount_mean"].mean()
        white = frame[frame["label"] == "white"]["txn_amount_mean"].mean()
        assert black > white * 2, (black, white)

    def test_gray_sits_between_the_two(self) -> None:
        frame = _frame("fraud/seed_200.csv")
        means = {
            label: frame[frame["label"] == label]["txn_amount_mean"].mean()
            for label in ("black", "white", "gray")
        }
        assert means["white"] < means["gray"] < means["black"], means

    def test_ratio_columns_are_proportions(self) -> None:
        """Sampling a ratio column as integers would break any threshold on it."""
        frame = _frame("fraud/seed_200.csv")
        for column in ("night_txn_ratio", "cross_region_ratio", "new_device_ratio"):
            assert frame[column].between(0.0, 1.0).all(), column


class TestSensitiveFields:
    def test_fraud_files_carry_the_sensitive_columns(self) -> None:
        from son_data_pipeline import detect_sensitive_columns

        detected = detect_sensitive_columns(_frame("fraud/seed_200.csv"))
        assert {"id_card", "phone", "bank_card"} <= set(detected)

    def test_masking_is_not_reversible(self) -> None:
        from son_data_pipeline import apply_masking

        # Read as str: the id-like columns would otherwise arrive as ints and the
        # masker's shape rules would never see them.
        raw = pd.read_csv(_csv("edge_cases/sensitive_fields.csv"), dtype=str)
        masked, report = apply_masking(raw)
        assert {"id_card", "phone", "bank_card"} <= set(report.masked_fields)
        for original, hidden in zip(raw["phone"], masked["phone"], strict=True):
            assert original not in hidden

    def test_report_names_what_it_masked(self) -> None:
        from son_data_pipeline import apply_masking

        _, report = apply_masking(_frame("edge_cases/sensitive_fields.csv"))
        assert report.describe()
        assert report.skipped_fields == ()


class TestEdgeCases:
    def test_minimal_dataset_still_splits(self) -> None:
        """8 rows: the 15% share rounds to one test row."""
        frame = _frame("edge_cases/minimal_8_rows.csv")
        frame = frame.rename(columns={"label": "label"})
        frame["origin"] = DataOrigin.SEED.value
        result = stratified_split(frame)
        assert result.summary.test_rows >= 1
        assert result.summary.test_contains_synth is False

    def test_single_class_ratio_is_not_computable(self) -> None:
        """The whole point: must not report 'balanced' for a one-class dataset."""
        frame = _frame("edge_cases/single_class_black_only.csv")
        counts = label_counts(frame, "label", {d.value: d for d in Decision})
        assert counts is not None
        assert recommend_ratio(counts) is None

    def test_no_gray_is_detected(self) -> None:
        frame = _frame("edge_cases/no_gray_label.csv")
        counts = label_counts(frame, "label", {d.value: d for d in Decision})
        assert counts is not None
        assert counts[Decision.GRAY.value] == 0

    def test_missing_values_are_real_blanks(self) -> None:
        # The reader the API uses, so a blank stays a blank rather than becoming
        # NaN and disappearing from the missing-rate calculation.
        frame = pd.read_csv(_csv("edge_cases/heavy_missing.csv"), dtype=str, keep_default_na=False)
        blanks = int((frame == "").sum().sum())
        assert blanks > 0
        # A fraction worth reporting, not a stray typo.
        assert blanks / frame.size > 0.05

    def test_outliers_are_extreme(self) -> None:
        frame = _frame("edge_cases/with_outliers.csv")
        assert frame["txn_amount_max"].max() > 1_000_000

    def test_unbalanced_ratio_is_large(self) -> None:
        frame = _frame("edge_cases/heavily_unbalanced.csv")
        counts = label_counts(frame, "label", {d.value: d for d in Decision})
        assert counts is not None
        ratio = recommend_ratio(counts)
        assert ratio is not None and ratio > 40, ratio


class TestGeneratorIsReproducible:
    def test_check_mode_passes_on_a_clean_tree(self) -> None:
        """`--check` is the guard against a hand-edited fixture."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "tools/gen_datasets.py", "--check"],
            capture_output=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")

    def test_regenerating_gives_identical_bytes(self) -> None:
        import hashlib
        import subprocess
        import sys

        target = DATASET_DIR / "fraud" / "seed_200.csv"
        before = hashlib.sha256(target.read_bytes()).hexdigest()
        subprocess.run(
            [sys.executable, "tools/gen_datasets.py"],
            capture_output=True,
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        assert hashlib.sha256(target.read_bytes()).hexdigest() == before
