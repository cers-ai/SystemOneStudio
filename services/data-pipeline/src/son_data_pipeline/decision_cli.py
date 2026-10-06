"""CPU data preparation for the V3 Spike; no model or success simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from pydantic import TypeAdapter

from son_contracts.decision import DecisionSample, DecisionSchema, FieldMapping
from son_data_pipeline.decision_data import SplitMode, split_decision_seeds
from son_data_pipeline.decision_import import import_decision_csv


def prepare(
    dataset: Path,
    schema_path: Path,
    output: Path,
    *,
    mapping_path: Path | None = None,
    independent_rows: bool = False,
    mode: SplitMode = "independent",
    seed: int = 42,
) -> dict[str, object]:
    """Write a new immutable data directory atomically; never replace existing data."""
    if output.exists():
        raise ValueError("输出目录已存在，拒绝覆盖冻结数据")
    schema = DecisionSchema.model_validate_json(schema_path.read_bytes())
    dataset_bytes = dataset.read_bytes()
    if dataset.suffix.casefold() == ".csv":
        if mapping_path is None:
            raise ValueError("CSV导入必须提供字段映射文件")
        mappings = TypeAdapter(list[FieldMapping]).validate_json(mapping_path.read_bytes())
        samples = import_decision_csv(dataset_bytes, mappings, independent_rows=independent_rows)
    else:
        samples = []
        for number, line in enumerate(dataset_bytes.decode("utf-8-sig").splitlines(), start=1):
            if line.strip():
                try:
                    samples.append(DecisionSample.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"第{number}行JSONL样本违反契约") from exc
    split = split_decision_seeds(samples, mode=mode, seed=seed)
    split.verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".decision-prepare-", dir=output.parent))
    try:
        (staging / "schema.json").write_text(schema.model_dump_json(indent=2), encoding="utf-8")
        for name in ("train", "valid", "cal", "test"):
            rows = getattr(split, name)
            (staging / f"{name}.jsonl").write_text(
                "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
            )
        manifest = split.manifest() | {"manifest_hash": split.manifest_hash}
        (staging / "split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Trainer input descriptor deliberately has no test path.
        descriptor = {
            "schema": "schema.json",
            "train": "train.jsonl",
            "valid": "valid.jsonl",
            "cal": "cal.jsonl",
            "split_manifest_hash": split.manifest_hash,
            "schema_sha256": hashlib.sha256((staging / "schema.json").read_bytes()).hexdigest(),
        }
        (staging / "training_inputs.json").write_text(
            json.dumps(descriptor, indent=2), encoding="utf-8"
        )
        assets = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in staging.iterdir()
            if path.is_file()
        }
        (staging / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "files": assets,
                    "schema_version": schema.version,
                    "source_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        intended_parent = output.parent.resolve()
        if (
            staging.resolve().parent != intended_parent
            or output.resolve().parent != intended_parent
        ):
            raise ValueError("发布路径不在指定输出父目录，拒绝移动")
        if output.exists():
            raise ValueError("输出目录已存在，拒绝覆盖冻结数据")
        staging.rename(output)
    finally:
        if staging.exists():
            if staging.resolve().parent != output.parent.resolve() or not staging.name.startswith(
                ".decision-prepare-"
            ):
                raise RuntimeError("临时目录不在指定输出父目录，拒绝清理")
            shutil.rmtree(staging)
    return {
        "status": "DATA_PREPARED",
        "output": str(output),
        "counts": {name: len(getattr(split, name)) for name in ("train", "valid", "cal", "test")},
        "warnings": split.warnings,
        "manifest_hash": split.manifest_hash,
        "model_trained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 CPU数据准备；不执行训练")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--independent-rows", action="store_true")
    parser.add_argument(
        "--mode", choices=("independent", "shared_valid_cal"), default="independent"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = prepare(
            args.dataset,
            args.schema,
            args.output,
            mapping_path=args.mapping,
            independent_rows=args.independent_rows,
            mode=args.mode,
            seed=args.seed,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"数据准备失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
