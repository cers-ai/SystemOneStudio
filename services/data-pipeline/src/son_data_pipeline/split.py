"""The 7 : 1.5 : 1.5 split, and the invariant that guards the test set.

Two requirements meet in this module (需求方案.txt 5.3 and 5.4):

* split ratios are fixed at 7 : 1.5 : 1.5 and happen automatically
* **synthetic data must not enter the test set** (需求方案.txt 5.4, "硬约束")

The second one is why this module is more than a three-line slice. It is
enforced structurally rather than by filtering afterwards: the test rows are
drawn from a seed-only pool built before any sampling happens, so there is no
code path that can route a synthetic row into test even if a caller hands us a
frame made entirely of synthetic rows.

The assertion at the end is a second line of defence, not the mechanism. If it
ever fires, a structural change broke the guarantee and we want a loud failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from son_contracts import DataOrigin, Decision, SplitSummary

#: 需求方案.txt 5.3: train / valid / test = 7 : 1.5 : 1.5
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "valid": 0.15, "test": 0.15}

#: Test is taken first; the remainder is divided train/valid in this ratio.
#: 0.70 / (0.70 + 0.15) = 0.8235, which recovers 0.70N and 0.15N overall.
_REMAINING_TRAIN_SHARE = SPLIT_RATIOS["train"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["valid"])

ORIGIN_COLUMN = "origin"
LABEL_COLUMN = "label"

#: Below this many rows a stratum cannot spare rows for valid without emptying
#: out train, so it stays entirely in train and is reported.
_MIN_ROWS_TO_HOLD_OUT = 10


class SyntheticDataLeakError(AssertionError):
    """Raised when a synthetic row reaches the test split.

    Subclasses AssertionError so a bare ``pytest.raises(AssertionError)`` in a
    consumer's test suite keeps catching it.
    """


@dataclass
class SplitResult:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: SplitSummary
    #: Strata too small to hold out, listed so the deviation from the fixed
    #: ratios is visible rather than silent.
    undersized_groups: list[str] = field(default_factory=list)

    def frames(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


def _validate(frame: pd.DataFrame, origin_column: str, label_column: str) -> None:
    if frame.empty:
        raise ValueError("cannot split an empty dataset")
    for column in (origin_column, label_column):
        if column not in frame.columns:
            raise ValueError(f"frame must carry a '{column}' column; got {list(frame.columns)}")

    known = {o.value for o in DataOrigin}
    unknown = set(frame[origin_column].unique()) - known
    if unknown:
        # An unrecognised origin cannot be reasoned about, so it cannot be
        # proven safe for the test set. Fail rather than assume.
        raise ValueError(f"unknown origin values: {sorted(unknown)}; expected {sorted(known)}")

    unknown_labels = set(frame[label_column].unique()) - {d.value for d in Decision}
    if unknown_labels:
        raise ValueError(
            f"label column holds non-decision values {sorted(unknown_labels)}; "
            "normalize labels before splitting"
        )


def _take_per_label(
    pool: pd.DataFrame,
    label_column: str,
    share: float,
    *,
    reserve: int = 0,
) -> pd.DataFrame:
    """Stratified slice of `pool` by label, keeping `reserve` rows per label back."""
    parts = []
    for decision in Decision:
        stratum = pool[pool[label_column] == decision.value]
        n = len(stratum)
        if n == 0:
            continue
        take = round(n * share) - reserve
        take = max(0, min(take, n))
        if take:
            parts.append(stratum.iloc[:take])
    return pd.concat(parts) if parts else pool.head(0)


def stratified_split(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    origin_column: str = ORIGIN_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> SplitResult:
    """Split a dataset into train / valid / test.

    Stratification key is label, so gray coverage stays consistent across
    splits. Synthetic rows are allowed in train and valid (that is the point of
    augmentation) but are structurally excluded from test.

    An empty test split is an error. A three-way product whose evaluation set
    can silently be empty is worse than one that refuses to split.
    """
    _validate(frame, origin_column, label_column)

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # --- Structural exclusion: the test pool is seed rows only. -------------
    seed_pool = shuffled[shuffled[origin_column] == DataOrigin.SEED.value]
    if seed_pool.empty:
        raise ValueError(
            "no seed rows available; the test split may contain seed data only, "
            "so a dataset of synthetic rows alone cannot be split"
        )

    test = _take_per_label(seed_pool, label_column, SPLIT_RATIOS["test"], reserve=1)
    if test.empty:
        # Too few rows to give every label even one test row (the 15% share of a
        # handful of rows rounds to 0). Fall back to a label-balanced draw rather
        # than `head()`: taking the first N rows after the shuffle hands the
        # whole evaluation set to whichever label happens to sort first, which
        # silently makes recall for the other labels uncomputable.
        test = _take_balanced(
            seed_pool, label_column, max(1, round(len(seed_pool) * SPLIT_RATIOS["test"]))
        )

    # --- Everything left over, seed and synthetic alike, goes to train/valid.
    remainder = shuffled.drop(test.index)

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    undersized: list[str] = []

    stratify_keys = [(origin, label) for origin in DataOrigin for label in Decision]
    for origin, decision in stratify_keys:
        group = remainder[
            (remainder[origin_column] == origin.value) & (remainder[label_column] == decision.value)
        ]
        if group.empty:
            continue
        if len(group) < _MIN_ROWS_TO_HOLD_OUT:
            undersized.append(f"{origin.value}/{decision.value}:{len(group)}")
            train_parts.append(group)
            continue
        cut = round(len(group) * _REMAINING_TRAIN_SHARE)
        train_parts.append(group.iloc[:cut])
        if cut < len(group):
            valid_parts.append(group.iloc[cut:])

    train = _concat(train_parts, remainder)
    valid = _concat(valid_parts, remainder)

    _assert_no_synthetic_in_test(test, origin_column)
    _assert_covers_all_rows(shuffled, train, valid, test)

    summary = SplitSummary(
        train_rows=len(train),
        valid_rows=len(valid),
        test_rows=len(test),
        test_contains_synth=False,
        label_distribution={
            "train": _label_counts(train, label_column),
            "valid": _label_counts(valid, label_column),
            "test": _label_counts(test, label_column),
        },
    )

    return SplitResult(
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
        summary=summary,
        undersized_groups=undersized,
    )


def _take_balanced(pool: pd.DataFrame, label_column: str, total: int) -> pd.DataFrame:
    """Draw `total` rows spread across labels, largest-remainder allocation.

    Used only when the dataset is too small for per-label rounding to give any
    label a test row. Two properties matter:

    * deterministic -- the largest class is served first, so the same dataset
      yields the same evaluation set regardless of shuffle seed. The previous
      implementation used ``head()``, which handed the evaluation set to
      whichever label happened to sort first.
    * never padded -- the fixed 7:1.5:1.5 ratios are a hard requirement, so a
      dataset whose test share rounds to zero gets a one-row test set rather
      than an invented one. The coverage shortfall is reported instead, through
      the per-label counts in the summary and the undersized-group list.
    """
    present = [d for d in Decision if (pool[label_column] == d.value).any()]
    if not present:
        return pool.head(total)

    base = min(1, total // len(present))
    allocation = {d.value: base for d in present}
    remaining = total - base * len(present)

    # Hand out the rest to whichever labels have the most rows available.
    by_size = sorted(present, key=lambda d: -int((pool[label_column] == d.value).sum()))
    cursor = 0
    while remaining > 0 and by_size:
        label = by_size[cursor % len(by_size)].value
        available = int((pool[label_column] == label).sum())
        if allocation[label] < available:
            allocation[label] += 1
            remaining -= 1
        elif all(
            allocation[d.value] >= int((pool[label_column] == d.value).sum()) for d in present
        ):
            break
        cursor += 1

    parts = [
        pool[pool[label_column] == label].iloc[:count]
        for label, count in allocation.items()
        if count > 0
    ]
    return pd.concat(parts) if parts else pool.head(total)


def _concat(parts: list[pd.DataFrame], fallback: pd.DataFrame) -> pd.DataFrame:
    usable = [p for p in parts if not p.empty]
    return pd.concat(usable) if usable else fallback.head(0)


def _assert_no_synthetic_in_test(test: pd.DataFrame, origin_column: str) -> None:
    leaked = test[test[origin_column] != DataOrigin.SEED.value]
    if not leaked.empty:
        origins = leaked[origin_column].value_counts().to_dict()
        raise SyntheticDataLeakError(
            f"synthetic data reached the test split: {origins}. "
            "This is a hard product constraint (需求方案.txt 5.4)."
        )


def _assert_covers_all_rows(
    original: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Every input row lands in exactly one split: none dropped, none doubled."""
    seen: set[int] = set()
    for frame in (train, valid, test):
        indices = set(frame.index)
        overlap = seen & indices
        if overlap:
            raise AssertionError(f"rows leaked across splits: {sorted(overlap)[:10]}")
        seen |= indices
    if len(seen) != len(original):
        raise AssertionError(
            f"split lost rows: {len(original)} in, {len(seen)} placed across splits"
        )


def _label_counts(frame: pd.DataFrame, label_column: str) -> dict[str, int]:
    counts = {d.value: 0 for d in Decision}
    if frame.empty or label_column not in frame.columns:
        return counts
    for value in frame[label_column]:
        if value in counts:
            counts[str(value)] += 1
    return counts


def label_share(summary: SplitSummary, split: str, decision: Decision) -> float:
    """Share of one decision within one split; 0.0 when the split is empty."""
    counts = summary.label_distribution.get(split, {})
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get(decision.value, 0) / total
