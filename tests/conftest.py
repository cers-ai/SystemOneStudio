"""Session-wide test isolation.

Without this, anything that calls ``create_app()`` writes ``./runtime/systemone.db``
into the repository and creates ``./runtime/workspace`` on disk. That pollutes the
working tree and, worse, leaks state between test modules -- a database written by
one suite is visible to the next.

Tests that need their own paths still override the environment in their fixture;
this only sets a floor so nothing escapes into the repo.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_runtime() -> Iterator[Path]:
    """Point SON_DB_PATH and SON_WORKSPACE at a temporary directory."""
    base = Path(tempfile.mkdtemp(prefix="son-tests-"))
    previous = {
        key: os.environ.get(key)
        for key in ("SON_DB_PATH", "SON_WORKSPACE", "SON_ASSISTANT_PROVIDER")
    }
    os.environ["SON_DB_PATH"] = str(base / "systemone.db")
    os.environ["SON_WORKSPACE"] = str(base / "workspace")
    yield base
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
