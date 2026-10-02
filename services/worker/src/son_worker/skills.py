"""Worker skills: the actual pipeline execution.

One function per job type. Each one calls into the existing capability packages
and writes its result back through the job queue -- it does not reimplement
training, synthesis or evaluation (改造开发方案.md 12.1).

The GPU-dependent skills import torch and llama.cpp lazily. A CPU-only machine
can run the data-prep and synth skills; a training job reports the missing GPU as
a failed job with a clear message rather than the worker refusing to start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from son_db import workspace
from son_worker.prepare import (
    generate_synth,
    prepare_split,
    training_frame,
)

from son_contracts import QuantLevel, SynthMethod, TrainingMethod


class SkillUnavailable(RuntimeError):
    """This node cannot run the requested skill."""


def _run_dir(run_id: str) -> Path:
    return workspace() / run_id


def _dataset_dir(run_id: str) -> Path:
    return _run_dir(run_id) / "dataset"


def _require_gpu() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SkillUnavailable(
            f"训练依赖未安装：{exc}。请使用 GPU 镜像（deploy/gpu/Dockerfile.gpu）。"
        ) from exc
    if not torch.cuda.is_available():
        raise SkillUnavailable("未检测到 CUDA 设备，无法训练")


def synth_skill(context: Any) -> dict[str, Any]:
    """Step 4: generate ``syn_v1.csv`` from the training split."""
    run_id = context.job.run_id
    train_path = Path(context.payload["train_path"])
    target_rows = context.payload.get("target_rows")

    context.progress(20, stage="SYNTH", message="读取训练集")
    result = generate_synth(
        train_path,
        output_path=_run_dir(run_id) / "synth" / "syn_v1.csv",
        method=SynthMethod.DISTRIBUTION_FIT,
        target_rows=target_rows,
    )
    context.progress(100, stage="SYNTH", message="合成完成")

    return {
        "synth_path": str(result.path),
        "rows": result.rows,
        "checksum": result.checksum,
        "fidelity_score": result.fidelity_score,
        "fidelity_verdict": result.fidelity_verdict,
        "privacy": result.privacy,
        "label_counts": result.label_counts,
        "recommendation": result.recommendation,
    }


def prepare_skill(context: Any) -> dict[str, Any]:
    """Step 3 as a job, for callers that prefer the queue over a direct call."""
    run_id = context.job.run_id
    dataset_path = Path(context.payload["dataset_path"])

    context.progress(30, stage="PREPARE", message="脱敏与划分")
    result = prepare_split(dataset_path, run_id, dataset_dir=_dataset_dir(run_id))
    context.progress(100, stage="PREPARE", message="划分完成")

    return {
        "train_path": str(result.train_path),
        "valid_path": str(result.valid_path),
        "test_path": str(result.test_path),
        "train_rows": result.train_rows,
        "valid_rows": result.valid_rows,
        "test_rows": result.test_rows,
        "test_synth_rows": result.test_synth_rows,
        "quality": result.quality,
    }


def train_skill(context: Any) -> dict[str, Any]:
    """Step 7: train, merge and quantize, then report the artifact.

    Everything heavy happens here and every intermediate result is recorded, so
    a failure halfway through leaves a checkpoint and a log behind rather than
    an unexplained gap.
    """
    _require_gpu()

    from son_quantizer import QuantizeRequest, quantize_all
    from son_quantizer.peft_merge import LlamaCppConverter
    from son_trainer import (
        DatasetProfile,
        TorchRunConfig,
        TorchTrainingBackend,
        TrainRequest,
        TrainSample,
        run_training,
    )
    from son_trainer.torch_backend import TrainingUnavailable

    run_id = context.job.run_id
    payload = context.payload

    train_path = Path(payload["train_path"])
    synth_path = Path(payload["synth_path"]) if payload.get("synth_path") else None
    rows = training_frame(train_path, synth_path)

    context.progress(5, stage="TRAIN", message=f"准备 {len(rows)} 条训练样本")

    samples = tuple(
        TrainSample(
            prompt=_render_prompt(row),
            response=_render_target(row),
            origin=row.get("origin", "seed"),
        )
        for row in rows
    )

    profile = DatasetProfile(rows=len(samples), params_b=float(payload.get("params_b", 1.5)))
    request = TrainRequest(
        job_id=context.job.id,
        base_model=payload["base_model"],
        methods=(TrainingMethod.SFT,),
        samples=samples,
        profile=profile,
    )
    backend = TorchTrainingBackend(
        TorchRunConfig(
            base_model=payload["base_model"], output_dir=str(_run_dir(run_id) / "training")
        )
    )

    context.progress(15, stage="TRAIN", message="开始训练")
    try:
        run = run_training(request, backend)
    except TrainingUnavailable as exc:
        raise SkillUnavailable(str(exc)) from exc

    if not run.succeeded:
        raise SkillUnavailable(run.failure_reason or "训练未成功")

    adapter = run.final_adapter_path
    if not adapter or not Path(adapter).exists():
        # Rule 4: no real file, no artifact.
        raise SkillUnavailable(f"训练结束但没有产出适配器文件：{adapter}")

    context.progress(60, stage="MERGE", message="合并权重")
    tools = LlamaCppConverter()
    merged = tools.merge_lora(
        payload["base_model"], adapter, str(_run_dir(run_id) / "model" / "merged")
    )

    context.progress(80, stage="QUANTIZE", message="导出 Q4_K_M")
    result = quantize_all(
        QuantizeRequest(
            merged_path=merged,
            output_dir=str(_run_dir(run_id) / "model"),
            model_version=f"mv_{run_id[-4:]}",
            levels=(QuantLevel.Q4_K_M,),
        ),
        tools,
        include_native=False,
        hash_outputs=True,
    )

    gguf = result.recommended
    if gguf is None:
        raise SkillUnavailable("未产出 Q4_K_M 产物")

    context.progress(90, stage="QUANTIZE", message="校验产物")
    return {
        "adapter_path": adapter,
        "merged_path": merged,
        "gguf_path": gguf.path,
        "gguf_checksum": gguf.sha256,
        "gguf_bytes": gguf.size_bytes,
        "quant_level": gguf.quant.value if gguf.quant else None,
        "training": {
            "stages": [
                {"stage": s.stage, "status": s.status, "step": s.step, "metrics": s.metrics}
                for s in run.stages
            ],
            "hyperparams": run.hyperparams,
            "notes": list(run.notes),
        },
    }


def _render_prompt(row: dict[str, str]) -> str:
    """Build the user turn from every non-label, non-origin column."""
    parts = [
        f"{key}: {value}" for key, value in sorted(row.items()) if key not in ("label", "origin")
    ]
    return "\n".join(parts)


def _render_target(row: dict[str, str]) -> str:
    """The supervised target, in the JEV output shape.

    Trained in the same shape inference validates. Training on a bare label and
    validating a strict schema at inference is how format compliance ends up
    below its target.
    """
    import json

    from son_contracts import Decision

    label = str(row.get("label", "")).strip().lower()
    try:
        decision = Decision(label)
    except ValueError:
        decision = Decision.BLACK

    return json.dumps(
        {
            "decision": decision.value,
            "score": 1.0 if decision is Decision.BLACK else 0.0,
            "confidence": 0.9,
            "reason": "依据提供的特征综合判定",
        },
        ensure_ascii=False,
    )
