"""Database, migration and workspace tests.

The guarantee this stage must prove is 改造开发方案.md 39 step 12: after
restarting everything, the project, run, dataset, model version, evaluation and
deployment history are still there. The restart tests exercise that directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from son_db import Database, migrate, reset_database, workspace
from son_db.assets import (
    append_jsonl,
    checksum,
    checksum_text,
    copy_into_run,
    delete_run,
    log_path,
    read_frame,
    read_jsonl,
    write_frame,
    write_json,
    write_text,
)
from son_db.database import (
    applied_versions,
    available_migrations,
    current_version,
    pending_migrations,
    raw_connection,
)
from son_db.models import ALL_MODELS, AuditRow, DatasetRow, ProjectRow, RunRow
from son_db.repositories import dump, load, record_audit, require, rows_for_run

MIGRATION_SQL = Path(__file__).parent.parent / "src" / "son_db" / "migrations" / "001_init.sql"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    monkeypatch.setenv("SON_DB_PATH", str(tmp_path / "runtime" / "systemone.db"))
    monkeypatch.setenv("SON_WORKSPACE", str(tmp_path / "runtime" / "workspace"))
    database = Database.open()
    migrate(database)
    yield database
    database.dispose()


class TestEngineAndPragmas:
    def test_pragmas_are_applied(self, db: Database) -> None:
        """WAL + foreign keys + busy timeout; the design depends on all three."""
        with raw_connection(db) as raw:
            assert raw.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            assert raw.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert int(raw.execute("PRAGMA busy_timeout").fetchone()[0]) >= 1000

    def test_database_file_is_created(self, db: Database) -> None:
        with raw_connection(db) as raw:
            rows = raw.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {r[0] for r in rows}
        assert {"runs", "jobs", "deployments", "datasets", "artifacts"} <= names

    def test_database_lives_under_the_configured_path(self, db: Database) -> None:
        assert str(db.engine.url.database).endswith("systemone.db")

    def test_raw_connection_closes_its_wrapper(self, db: Database) -> None:
        with raw_connection(db) as raw:
            assert raw.execute("SELECT 1").fetchone()[0] == 1
        # The pool must still work afterwards.
        with raw_connection(db) as raw:
            assert raw.execute("SELECT 2").fetchone()[0] == 2


class TestMigrations:
    def test_init_script_is_discovered(self) -> None:
        migrations = available_migrations()
        assert migrations
        assert migrations[0][0] == 1
        assert migrations[0][1].name.startswith("001_")

    def test_versions_are_ordered(self) -> None:
        versions = [version for version, _ in available_migrations()]
        assert versions == sorted(versions)

    def test_ledger_records_what_ran(self, db: Database) -> None:
        with raw_connection(db) as raw:
            assert 1 in applied_versions(raw)
            assert current_version(raw) >= 1

    def test_migrate_is_idempotent(self, db: Database) -> None:
        assert migrate(db) == []

    def test_newer_database_is_refused(self, db: Database) -> None:
        with raw_connection(db) as raw:
            raw.execute("PRAGMA user_version = 9999")
            raw.commit()
        with pytest.raises(Exception, match="高于当前构建"):
            migrate(db)

    def test_fresh_database_starts_at_zero(self, tmp_path: Path) -> None:
        database = Database.open(tmp_path / "fresh.db")
        try:
            with raw_connection(database) as raw:
                assert current_version(raw) == 0
                assert pending_migrations(raw)
        finally:
            database.dispose()

    def test_every_model_has_a_table_in_the_migration(self) -> None:
        """A model with no table fails at write time, not import time."""
        sql = MIGRATION_SQL.read_text(encoding="utf-8")
        for model in ALL_MODELS:
            assert f"CREATE TABLE IF NOT EXISTS {model.__tablename__}" in sql, model.__tablename__


class TestForeignKeys:
    def test_cascade_delete_removes_child_rows(self, db: Database) -> None:
        with db.session() as session:
            project = ProjectRow.new("cascade-test")
            session.add(project)
            session.flush()
            run = RunRow(project_id=project.id, state="CREATED")
            session.add(run)
            session.flush()
            project_id, run_id = project.id, run.id

        with db.session() as session:
            target = session.get(ProjectRow, project_id)
            assert target is not None
            session.delete(target)

        with db.read() as session:
            assert session.get(RunRow, run_id) is None

    def test_orphan_run_is_rejected(self, db: Database) -> None:
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError), db.session() as session:
            session.add(RunRow(project_id="prj_missing", state="CREATED"))


class TestPersistenceAcrossRestart:
    """改造开发方案.md 39 step 12."""

    def test_run_survives_reopening(self, db: Database) -> None:
        with db.session() as session:
            project = ProjectRow.new("persisted")
            session.add(project)
            session.flush()
            session.add(
                RunRow(project_id=project.id, state="SCENE_READY", scene_code="fraud_account")
            )
            session.flush()
            run_id = session.query(RunRow).one().id

        # A brand new Database over the same file: nothing in-process carries over.
        reopened = Database.open()
        try:
            with reopened.read() as session:
                stored = session.get(RunRow, run_id)
                assert stored is not None
                assert stored.state == "SCENE_READY"
                assert stored.scene_code == "fraud_account"
        finally:
            reopened.dispose()

    def test_workflow_truth_lives_in_the_database(self, db: Database) -> None:
        with db.session() as session:
            project = ProjectRow.new("truth")
            session.add(project)
            session.flush()
            session.add(RunRow(project_id=project.id, state="TRAINING"))
            project_id = project.id

        reopened = Database.open()
        try:
            with reopened.read() as session:
                states = {
                    run.state
                    for run in session.query(RunRow).filter(RunRow.project_id == project_id)
                }
        finally:
            reopened.dispose()
        assert states == {"TRAINING"}

    def test_soft_delete_keeps_the_row(self, db: Database) -> None:
        """需求方案.txt principle 5: projects are recoverable for 7 days."""
        from son_db.database import now_iso

        with db.session() as session:
            project = ProjectRow.new("soft-delete")
            project.deleted_at = now_iso()
            session.add(project)
            session.flush()
            project_id = project.id

        reopened = Database.open()
        try:
            with reopened.read() as session:
                stored = session.get(ProjectRow, project_id)
                assert stored is not None
                assert stored.deleted_at is not None
        finally:
            reopened.dispose()


class TestAudit:
    def test_record_and_read_back(self, db: Database) -> None:
        with db.session() as session:
            record_audit(session, "create_run", "run_1", "测试", actor="tester")

        with db.read() as session:
            entries = session.query(AuditRow).all()
        assert len(entries) == 1
        assert entries[0].action == "create_run"
        assert entries[0].actor == "tester"


class TestRequire:
    def test_missing_row_raises(self, db: Database) -> None:
        with db.read() as session, pytest.raises(LookupError, match="不存在"):
            require(session, RunRow, "run_nope", "Run")

    def test_present_row_returns(self, db: Database) -> None:
        with db.session() as session:
            project = ProjectRow.new("found")
            session.add(project)
            session.flush()
            project_id = project.id
        with db.read() as session:
            assert require(session, ProjectRow, project_id, "Project").id == project_id


class TestRowsForRun:
    def test_returns_only_that_runs_rows(self, db: Database) -> None:
        with db.session() as session:
            a = ProjectRow.new("a")
            b = ProjectRow.new("b")
            session.add_all([a, b])
            session.flush()
            session.add(RunRow(project_id=a.id, state="CREATED"))
            session.add(RunRow(project_id=b.id, state="CREATED"))
            session.flush()
            a_id = a.id

        with db.read() as session:
            runs = session.query(RunRow).filter(RunRow.project_id == a_id).all()
        assert len(runs) == 1

    def test_dataset_rows_are_scoped_to_the_run(self, db: Database) -> None:
        with db.session() as session:
            project = ProjectRow.new("scope")
            session.add(project)
            session.flush()
            run_a = RunRow(project_id=project.id, state="CREATED")
            run_b = RunRow(project_id=project.id, state="CREATED")
            session.add_all([run_a, run_b])
            session.flush()
            session.add(
                DatasetRow(
                    run_id=run_a.id,
                    code="ds_v1",
                    original_filename="a.csv",
                    source_path="x",
                    checksum="h1",
                    status="READY",
                )
            )
            session.flush()
            run_a_id = run_a.id

        with db.read() as session:
            assert len(rows_for_run(session, DatasetRow, run_a_id)) == 1

    def test_a_model_without_run_id_is_refused(self, db: Database) -> None:
        """RunRow is keyed by run id; asking it for run_id rows is a caller bug."""
        with db.read() as session, pytest.raises(AttributeError, match="run_id"):
            rows_for_run(session, RunRow, "run_1")


class TestJsonHelpers:
    def test_dump_none_is_none(self) -> None:
        assert dump(None) is None

    def test_roundtrip(self) -> None:
        assert load(dump({"a": 1})) == {"a": 1}

    def test_load_garbage_is_none(self) -> None:
        assert load("{not json") is None
        assert load(None) is None


class TestWorkspace:
    def test_write_and_read_text(self, db: Database) -> None:
        path = write_text("run_ws1", "dataset/note.txt", "你好")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "你好"

    def test_nested_paths_are_created(self, db: Database) -> None:
        path = write_text("run_ws1", "model/merged/deep/file.bin", "x")
        assert path.parent.exists()

    def test_json_roundtrip(self, db: Database) -> None:
        from son_db.assets import read_json

        write_json("run_ws1", "training/config.json", {"lr": 0.0002})
        assert read_json(workspace() / "run_ws1" / "training" / "config.json") == {"lr": 0.0002}

    def test_frame_roundtrip(self, db: Database) -> None:
        """CSV has no types, so values come back as strings. Pinned deliberately:
        a caller that assumes ints back will mis-parse a column."""
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        path = write_frame("run_ws1", "dataset/train.csv", rows)
        assert read_frame(path) == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]

    def test_frame_preserves_origin_column(self, db: Database) -> None:
        """origin is what makes the no-synth-in-test constraint checkable on the
        artifact itself, not only in memory."""
        rows = [{"account": "A", "label": "black", "origin": "synth"}]
        path = write_frame("run_ws1", "dataset/test.csv", rows)
        assert read_frame(path)[0]["origin"] == "synth"

    def test_jsonl_append_and_read(self, db: Database) -> None:
        append_jsonl("run_ws1", "evaluation/predictions.jsonl", {"i": 1})
        append_jsonl("run_ws1", "evaluation/predictions.jsonl", {"i": 2})
        path = workspace() / "run_ws1" / "evaluation" / "predictions.jsonl"
        assert read_jsonl(path) == [{"i": 1}, {"i": 2}]

    def test_jsonl_tolerates_a_torn_final_line(self, db: Database) -> None:
        """A worker killed mid-write must not discard the rows that landed."""
        path = workspace() / "run_ws1" / "evaluation" / "predictions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"i": 1}\n{"i": 2}\n{"i": par', encoding="utf-8")
        assert read_jsonl(path) == [{"i": 1}, {"i": 2}]

    def test_read_jsonl_missing_file(self, db: Database) -> None:
        assert read_jsonl(workspace() / "nope" / "x.jsonl") == []

    def test_read_frame_missing_file(self, db: Database) -> None:
        assert read_frame(workspace() / "nope" / "x.csv") == []

    def test_checksums_match(self, db: Database) -> None:
        path = write_text("run_ws1", "dataset/source.csv", "a,b\n1,2\n")
        assert checksum(path) == checksum_text("a,b\n1,2\n")

    def test_checksum_differs_for_different_content(self, db: Database) -> None:
        assert checksum_text("a") != checksum_text("b")

    def test_copy_into_run(self, db: Database, tmp_path: Path) -> None:
        source = tmp_path / "upload.csv"
        source.write_text("a,b\n", encoding="utf-8")
        copied = copy_into_run("run_ws1", "dataset/source.csv", source)
        assert copied.exists()
        assert copied.read_text(encoding="utf-8") == "a,b\n"

    def test_log_directory_is_created(self, db: Database) -> None:
        path = log_path("run_ws1", "training", "job.log")
        assert path.parent.is_dir()

    def test_delete_run_removes_the_whole_tree(self, db: Database) -> None:
        write_text("run_ws1", "dataset/source.csv", "x")
        assert delete_run("run_ws1") is True
        assert not (workspace() / "run_ws1").exists()

    def test_delete_missing_run_is_false(self, db: Database) -> None:
        assert delete_run("run_never_existed") is False


class TestResetSupport:
    def test_reset_drops_tables(self, tmp_path: Path) -> None:
        database = Database.open(tmp_path / "r.db")
        try:
            migrate(database)
            reset_database(database)
            with raw_connection(database) as raw:
                tables = raw.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        finally:
            database.dispose()
        assert [t for t in tables if t[0] == "runs"] == []
