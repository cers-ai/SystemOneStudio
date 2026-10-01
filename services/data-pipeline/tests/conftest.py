"""Shared fixtures for the data pipeline tests."""

from __future__ import annotations

import pandas as pd
import pytest

from son_contracts import DataOrigin, Decision


def make_rows(
    *,
    black: int = 1000,
    white: int = 500,
    gray: int = 0,
    origin: str = DataOrigin.SEED.value,
    prefix: str = "",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, count in (
        (Decision.BLACK, black),
        (Decision.WHITE, white),
        (Decision.GRAY, gray),
    ):
        for i in range(count):
            # Account ids must be unique across labels: including the label
            # keeps the fixture honest about row identity.
            rows.append(
                {
                    "account": f"{prefix}{label.value[0].upper()}{i:06d}",
                    "amount": 1000 + (i * 37) % 90_000,
                    "device": ["ios", "android", "web"][i % 3],
                    "label": label.value,
                    "origin": origin,
                }
            )
    return rows


@pytest.fixture
def seed_frame() -> pd.DataFrame:
    return pd.DataFrame(make_rows(black=1000, white=500, gray=0))


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """Seed rows plus synthetic rows, which is the case that must be guarded."""
    seed = pd.DataFrame(make_rows(black=1000, white=500, prefix="seed"))
    synth = pd.DataFrame(
        make_rows(black=4000, white=4000, origin=DataOrigin.SYNTH.value, prefix="syn")
    )
    return pd.concat([seed, synth], ignore_index=True)


@pytest.fixture
def three_way_frame() -> pd.DataFrame:
    return pd.DataFrame(make_rows(black=600, white=400, gray=300))
