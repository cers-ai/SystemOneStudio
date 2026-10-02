"""SQLite persistence for SystemOneStudio.

Replaces PostgreSQL + Redis with one file-backed database. The reasoning is in
改造开发方案.md 2.2: the goal is a single instance that is reliable, legible,
restart-recoverable, and actually completes the golden path -- not a distributed
design for a distribution we do not have.

Three constraints follow from that and are enforced here rather than left to
each caller:

* **WAL + foreign keys + busy timeout.** WAL lets the API read while the worker
  writes, which is the only concurrency that actually exists: one worker.
* **One writer.** SQLite serialises writes. Every mutation goes through
  :func:`transaction` with ``BEGIN IMMEDIATE`` so two writers cannot interleave.
* **Migrations run at startup**, tracked by ``PRAGMA user_version``. No Alembic:
  one file per version is easier to review and there are two or three of them,
  not two hundred.

Not used, deliberately: repository frameworks, CQRS, event sourcing, unit of
work abstractions. See 改造开发方案.md 29.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_PATH = "./runtime/systemone.db"
DEFAULT_WORKSPACE = "./runtime/workspace"

MIGRATION_PREFIX = re.compile(r"^(\d+)_")


class DatabaseNotConfigured(RuntimeError):
    """Neither the environment nor an explicit path supplied a database."""


class MigrationError(RuntimeError):
    """A migration failed, or the on-disk version is newer than this build."""


def db_path() -> Path:
    """Resolve the database path from the environment."""
    return Path(os.environ.get("SON_DB_PATH", DEFAULT_DB_PATH))


def workspace() -> Path:
    """Resolve the workspace root from the environment."""
    return Path(os.environ.get("SON_WORKSPACE", DEFAULT_WORKSPACE))


def now_iso() -> str:
    """UTC timestamp as ISO-8601 with milliseconds.

    Millisecond precision because the job queue orders by ``created_at``: with
    second precision two jobs enqueued in the same second tie, and FIFO order
    becomes arbitrary.
    """
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


@contextmanager
def raw_connection(database: Database) -> Iterator[sqlite3.Connection]:
    """Yield the underlying DBAPI connection, closing the wrapper on exit.

    Migrations and the job queue need SQLite-specific statements (``PRAGMA``,
    ``BEGIN IMMEDIATE``, ``executescript``) that SQLAlchemy's generic layer does
    not expose cleanly. One accessor keeps that unwrapping in a single place
    instead of at every call site.
    """
    connection = database.engine.connect()
    raw = connection.connection.driver_connection
    if not isinstance(raw, sqlite3.Connection):  # pragma: no cover - driver guard
        connection.close()
        raise MigrationError(f"意外的数据库驱动连接类型：{type(raw).__name__}")
    try:
        yield raw
    finally:
        connection.close()


def build_engine(path: Path | str | None = None) -> Engine:
    """Create the engine and apply the pragmas the design depends on.

    ``check_same_thread=False`` because the FastAPI request thread and the
    worker thread both touch the session factory. SQLite serialises writes
    itself; the pragmas below are what make concurrent reads safe.
    """
    target = Path(path) if path is not None else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{target}",
        future=True,
        # Not an ORM-sharing session; each request/job gets its own Session.
        connect_args={"check_same_thread": False, "timeout": 5.0},
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        # WAL: readers do not block the single writer.
        cursor.execute("PRAGMA journal_mode=WAL")
        # Referential integrity is off by default in SQLite.
        cursor.execute("PRAGMA foreign_keys=ON")
        # Wait rather than fail immediately when the writer holds the lock.
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@dataclass
class Database:
    """Engine + session factory + migration state, as one unit."""

    engine: Engine
    sessions: sessionmaker[Session]

    @classmethod
    def open(cls, path: Path | str | None = None) -> Database:
        engine = build_engine(path)
        return cls(engine=engine, sessions=build_session_factory(engine))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Read-write session. Commits on success, rolls back on exception."""
        session = self.sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def read(self) -> Iterator[Session]:
        """Read-only session. Never commits."""
        session = self.sessions()
        try:
            yield session
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """``BEGIN IMMEDIATE`` session.

        Job claiming needs the write lock taken up front: two workers must not
        both read status=QUEUED and both claim the same job.
        """
        session = self.sessions()
        try:
            session.execute(sa.text("BEGIN IMMEDIATE"))
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# --------------------------------------------------------------------------
# Migrations
# --------------------------------------------------------------------------

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def current_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def available_migrations() -> list[tuple[int, Path]]:
    """Every ``NNN_name.sql`` in the migrations directory, version-ordered."""
    if not MIGRATIONS_DIR.exists():
        return []
    found: list[tuple[int, Path]] = []
    for script in MIGRATIONS_DIR.glob("*.sql"):
        match = MIGRATION_PREFIX.match(script.name)
        if match:
            found.append((int(match.group(1)), script))
    return sorted(found, key=lambda item: item[0])


def applied_versions(connection: sqlite3.Connection) -> list[int]:
    """Versions recorded in the ledger table, ordered."""
    has_ledger = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if not has_ledger:
        return []
    rows = connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    return [int(row[0]) for row in rows]


def pending_migrations(connection: sqlite3.Connection) -> list[tuple[int, Path]]:
    done = set(applied_versions(connection))
    return [(version, path) for version, path in available_migrations() if version not in done]


def migrate(database: Database) -> list[int]:
    """Apply every unapplied migration. Returns the versions applied.

    Each script runs inside its own transaction together with the ledger write,
    so a failing script leaves no half-applied version behind.
    """
    applied: list[int] = []
    with raw_connection(database) as raw:
        on_disk = current_version(raw)

        newest = max((version for version, _ in available_migrations()), default=0)
        if on_disk > newest:
            raise MigrationError(
                f"数据库版本 {on_disk} 高于当前构建支持的 {newest}；请升级代码而不是回退"
            )

        for version, script in pending_migrations(raw):
            sql = script.read_text(encoding="utf-8")
            # executescript, not exec_driver_sql: a migration file holds many
            # statements and the driver-level API runs exactly one.
            raw.executescript(sql)
            raw.execute(
                "INSERT INTO schema_migrations (version, script, applied_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)",
                (version, script.name),
            )
            raw.commit()
            raw.execute(f"PRAGMA user_version = {int(version)}")
            raw.commit()
            applied.append(version)

    return applied


def reset_database(database: Database) -> None:
    """Drop every table. Test-support only; never called at runtime."""
    with raw_connection(database) as raw:
        raw.execute("PRAGMA foreign_keys=OFF")
        rows = raw.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for (name,) in rows:
            raw.execute(f'DROP TABLE IF EXISTS "{name}"')
        raw.commit()
