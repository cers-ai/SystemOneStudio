"""Drive the whole pipeline against the bundled example dataset.

Run with the API up:

    uv run python tools/try_example_dataset.py

Uses the same HTTP calls a browser makes, so this proves the datasets work
through the real entry point rather than through a library call that bypasses
ingestion.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

BASE = os.environ.get("SON_API", "http://127.0.0.1:9969")
ROOT = Path(__file__).resolve().parent.parent


def show(label: str, value: Any) -> None:
    # The privacy report contains ✓ / ⚠, which the default Windows console
    # codepage cannot encode. Replace rather than crash: losing the tail of the
    # report because of a character is worse than an approximation.
    text = str(value)
    for pretty, plain in (("✓", "[ok]"), ("⚠", "[!]"), ("→", "->")):
        text = text.replace(pretty, plain)
    print(f"  {label:<18} {text}")


def main() -> int:
    dataset = ROOT / "dataset" / "fraud" / "seed_200.csv"
    if not dataset.exists():
        print(f"缺少示例数据集：{dataset}", file=sys.stderr)
        return 1

    client = httpx.Client(base_url=BASE, timeout=120.0)

    print(f"数据集：{dataset.relative_to(ROOT)}")
    print(f"接口：  {BASE}\n")

    project = client.post("/api/projects", json={"name": "示例数据验证"}).json()
    show("项目", project["name"])

    run = client.post(f"/api/projects/{project['id']}/runs").json()
    run_id = run["id"]
    show("Run", run_id)
    show("初始状态", run["state"])

    print("\n第 1 步 选择场景")
    run = client.patch(f"/api/runs/{run_id}/scene", json={"scene_code": "fraud_account"}).json()
    show("状态", run["state"])

    print("\n第 2 步 上传种子数据")
    with dataset.open("rb") as handle:
        run = client.post(
            f"/api/runs/{run_id}/dataset",
            files={"upload": (dataset.name, handle, "text/csv")},
        ).json()
    show("状态", run["state"])
    show("行数 × 列数", f"{run['dataset']['rows']} × {run['dataset']['cols']}")
    show("标签列", run["dataset"]["label_column"])
    show("校验和", run["dataset"]["checksum"][:16] + "…")

    print("\n第 3 步 数据治理与划分")
    run = client.post(f"/api/runs/{run_id}/prepare-data").json()
    split = run["split"]
    quality = split["quality"]
    show("状态", run["state"])
    show("质量评分", quality["score"])
    show(
        "标签分布",
        json.dumps(quality["label_distribution"], ensure_ascii=False),
    )
    show(
        "训练/验证/测试",
        f"{split['train_rows']} / {split['valid_rows']} / {split['test_rows']}",
    )
    show("测试集合成样本", f"{split['test_synth_rows']} 条（硬约束）")
    show("已脱敏字段", "、".join(quality["masked_fields"]) or "无")
    if quality["anomalies"]:
        show("异常检测", f"{len(quality['anomalies'])} 项")
    if quality["suggestions"]:
        show("系统建议", quality["suggestions"][0])

    print("\n第 4 步 数据合成")
    run = client.post(f"/api/runs/{run_id}/synth", json={"target_rows": 600}).json()
    synth = run["synth"]
    show("状态", run["state"])
    show("产物", synth["code"])
    show("合成行数", synth["rows"])
    show("保真度", f"{synth['fidelity_score']}（{synth['fidelity_verdict']}）")
    for line in synth["privacy"]["lines"]:
        show("隐私报告", line)

    print("\n第 5 步 选择基座模型")
    run = client.patch(
        f"/api/runs/{run_id}/model", json={"model_id": "qwen2.5-1.5b-instruct"}
    ).json()
    show("状态", run["state"])

    print("\n第 6 步 配置训练方法")
    run = client.patch(
        f"/api/runs/{run_id}/training-config", json={"method": "lora", "max_steps": 200}
    ).json()
    show("状态", run["state"])

    print("\n第 7 步 提交训练（API 不执行训练，只入队）")
    run = client.post(f"/api/runs/{run_id}/start").json()
    show("状态", run["state"])
    job = run["job"]
    show("任务类型", job["type"])
    show("任务状态", job["status"])
    show("进度", f"{job['progress']}%")

    print("\n血缘")
    for key, value in (run["lineage"] or {}).items():
        show(key, value)

    gpu_note = (
        "训练、量化、部署需要 GPU 节点，本机无法执行；"
        "在 GPU 机器上用 --profile worker 启动 worker 后本任务会被领取。"
    )
    print(f"\n{gpu_note}")
    print(f"\nRun 状态可查询：GET /api/runs/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
