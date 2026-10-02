"""Workspace layout for run assets.

改造开发方案.md section 8: one directory tree per run, so a run can be copied,
inspected, backed up or deleted as a unit.

```
runtime/workspace/run_000001/
  dataset/    source.csv train.csv valid.csv test.csv
  synth/      syn_v1.csv
  training/   config.json logs/ checkpoints/ adapter/
  model/      merged/ model-q4_k_m.gguf
  evaluation/ predictions.jsonl errors.jsonl report.json
  deployment/ llama-server.log
```

Every write goes through :func:`write_bytes` or :func:`write_text`, which create
parent directories and return the path. Nothing writes into the workspace by
constructing a path by hand -- that is how runs end up with assets outside their
own tree and nobody notices until a backup misses them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from son_db.database import workspace

DATASET = "dataset"
SYNTH = "synth"
TRAINING = "training"
MODEL = "model"
EVALUATION = "evaluation"
DEPLOYMENT = "deployment"


def root() -> Path:
    base = workspace()
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_dir(run_id: str) -> Path:
    path = root() / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def sub_dir(run_id: str, kind: str) -> Path:
    path = run_dir(run_id) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_bytes(run_id: str, relative: str, payload: bytes) -> Path:
    path = run_dir(run_id) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def write_text(run_id: str, relative: str, text: str) -> Path:
    return write_bytes(run_id, relative, text.encode("utf-8"))


def write_json(run_id: str, relative: str, payload: Any) -> Path:
    return write_text(run_id, relative, json.dumps(payload, ensure_ascii=False, indent=2))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(run_id: str, relative: str, record: Mapping[str, Any]) -> Path:
    """Append one JSON record. Used for predictions and error cases.

    Append rather than rewrite so a long inference run is resumable and a
    partial file still parses line by line.
    """
    path = run_dir(run_id) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # A torn final line from a killed process should not discard the
            # rows that did land.
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def write_frame(run_id: str, relative: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write a table to CSV.

    The origin column is always written. It is what makes the
    synthetic-rows-never-enter-test constraint checkable on the artifact itself,
    not only in memory.
    """
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    path = run_dir(run_id) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_frame(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def copy_into_run(run_id: str, relative: str, source: Path) -> Path:
    """Copy an external file into the run tree and return its new path.

    Uploaded files are copied rather than referenced in place: a run has to
    remain reproducible after the upload is deleted.
    """
    destination = run_dir(run_id) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def checksum(path: Path) -> str:
    """SHA-256 of a file. Recorded on every asset (Rule 7)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bytes_of(path: Path) -> int:
    return path.stat().st_size


def delete_run(run_id: str) -> bool:
    """Remove a run's whole asset tree. Returns whether anything was removed."""
    target = root() / run_id
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def log_path(run_id: str, kind: str, name: str) -> Path:
    """Log file path, creating its directory.

    Jobs write stdout here, so the directory must exist before the process
    starts or the first write fails inside the worker rather than at startup.
    """
    directory = sub_dir(run_id, kind) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def iter_run_dirs() -> Iterable[Path]:
    base = root()
    if not base.exists():
        return []
    return (path for path in base.iterdir() if path.is_dir())
