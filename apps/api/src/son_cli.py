"""Command line entry point for the GPU node.

One process that can serve the API, run a training job, quantize, serve
inference, or run a full end-to-end evaluation. Exists so the GPU node is
usable without the web client: the first thing worth checking on a new box is
whether the stack imports, sees the GPU, and can complete a round trip.

    python -m son_cli doctor            # environment check
    python -m son_cli train  --help
    python -m son_cli train  --dataset seed.csv --model qwen2.5-3b-instruct
    python -m son_cli quantize --adapter out/lora --model qwen2.5-3b-instruct
    python -m son_cli infer   --model models/mv_0017.Q4_K_M.gguf
    python -m son_cli serve
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from son_contracts import Decision


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report what this machine can actually do.

    Deliberately the default advice: the first question on a GPU node is whether
    the stack is real, and answering it by running a training job wastes an hour
    if the answer is no.
    """
    report: dict[str, Any] = {"node": "gpu", "checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        check("torch", True, torch.__version__)
        check("cuda_available", cuda, str(getattr(torch.version, "cuda", None)))
        if cuda:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                check(
                    f"gpu[{index}]",
                    True,
                    f"{props.name} {round(props.total_memory / 1024**3, 1)}GB "
                    f"bf16={torch.cuda.is_bf16_supported()}",
                )
        else:
            check("gpu", False, "torch.cuda.is_available() == False")
    except ImportError as exc:
        check("torch", False, str(exc))

    for module in ("transformers", "peft", "trl", "bitsandbytes", "accelerate"):
        try:
            mod = __import__(module)
            check(module, True, getattr(mod, "__version__", "?"))
        except ImportError as exc:
            check(module, False, str(exc))

    for binary in ("llama-server", "llama-quantize"):
        found = _which(binary)
        check(binary, found is not None, found or "不在 PATH 上")

    check(
        "convert_hf_to_gguf.py",
        Path("/usr/local/bin/convert_hf_to_gguf.py").exists(),
        "/usr/local/bin/convert_hf_to_gguf.py",
    )

    try:
        from son_inference.llamacpp_client import gpu_report

        report["gpu"] = gpu_report()
    except ImportError as exc:
        report["gpu"] = {"error": str(exc)}

    print(json.dumps(report, ensure_ascii=False, indent=2))

    blocking = [c for c in report["checks"] if not c["ok"] and c["name"] != "convert_hf_to_gguf.py"]
    if blocking:
        print(f"\n阻塞项 {len(blocking)} 个：", file=sys.stderr)
        for item in blocking:
            print(f"  - {item['name']}: {item['detail']}", file=sys.stderr)
        return 1
    print("\n环境就绪。", file=sys.stderr)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Run SFT (and DPO when preference pairs exist) from a CSV."""
    import pandas as pd
    from son_data_pipeline import detect_label_column, stratified_split
    from son_trainer import DatasetProfile, TrainRequest, TrainSample
    from son_trainer.torch_backend import (
        TorchRunConfig,
        TorchTrainingBackend,
        assert_training_available,
    )

    from son_contracts import DataOrigin, TrainingMethod

    assert_training_available()

    frame = pd.read_csv(args.dataset)
    detection = detect_label_column(frame)
    if not detection.column:
        print("未能识别标签列：请提供 black/white/gray 三分类标签", file=sys.stderr)
        return 2
    frame = frame.rename(columns={detection.column: "label"})
    frame["origin"] = DataOrigin.SEED.value

    split = stratified_split(frame, seed=args.seed)

    # Train only on train+valid; test is never read during training.
    train_frame = pd.concat([split.train, split.valid])
    prompt_column = args.prompt_column or next(
        (c for c in frame.columns if c not in {"label", "origin"}), None
    )
    if prompt_column is None:
        print("没有可用作 prompt 的字段", file=sys.stderr)
        return 2

    samples = tuple(
        TrainSample(
            prompt=str(row[prompt_column]),
            response=render_target(row, detection.mapping),
            origin=DataOrigin.SEED.value,
        )
        for _, row in train_frame.iterrows()
    )

    profile = DatasetProfile(rows=len(train_frame), params_b=args.params_b, quantized=args.qlora)
    methods = (TrainingMethod.SFT, TrainingMethod.DPO) if args.dpo else (TrainingMethod.SFT,)

    run = __import__("son_trainer").run_training(
        TrainRequest(
            job_id=args.job_id,
            base_model=args.model,
            methods=methods,
            samples=samples,
            profile=profile,
            seed=args.seed,
            jev_training_compat=True,
            n_per_prompt=args.n_per_prompt,
        ),
        TorchTrainingBackend(
            TorchRunConfig(
                base_model=args.model,
                output_dir=args.output_dir,
                quantized=args.qlora,
                seed=args.seed,
            )
        ),
    )

    print(json.dumps(run.describe(), ensure_ascii=False, indent=2))
    for stage in run.stages:
        print(f"  {stage.stage}: status={stage.status} step={stage.step} metrics={stage.metrics}")
    for note in run.notes:
        print(f"  note: {note}")
    if run.final_adapter_path:
        print(f"\n适配器输出：{run.final_adapter_path}")
    return 0 if run.succeeded else 1


def render_target(row: Any, mapping: dict[str, Decision]) -> str:
    """The supervised target: the decision, in the JEV output shape.

    Trained from the real schema so the model's output format matches what the
    inference path validates. Training on a loose label and validating a strict
    schema at inference is how format compliance ends up at 97%.
    """
    for raw, decision in mapping.items():
        if str(row.get("label")) == raw:
            return json.dumps(
                {
                    "decision": decision.value,
                    "score": 1.0 if decision.value == "black" else 0.0,
                    "confidence": 0.9,
                    "reason": "依据提供的特征综合判定",
                },
                ensure_ascii=False,
            )
    # An unmapped label means detection was wrong; fail loudly rather than
    # training on "{}".
    raise ValueError(f"标签 {row.get('label')!r} 不在映射表 {sorted(mapping)} 中；无法渲染训练目标")


def _cmd_quantize(args: argparse.Namespace) -> int:
    """Merge the adapter, then produce every GGUF level."""
    from son_quantizer import QuantizeRequest, quantize_all
    from son_quantizer.peft_merge import LlamaCppConverter

    tools = LlamaCppConverter()
    merged = tools.merge_lora(args.model, args.adapter, f"{args.output_dir}/merged")
    result = quantize_all(
        QuantizeRequest(
            merged_path=merged,
            output_dir=args.output_dir,
            model_version=args.job_id,
        ),
        tools,
        hash_outputs=True,
    )
    for line in result.describe():
        print(line)
    for note in result.notes:
        print(f"  note: {note}")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    """Start llama-server, send one request, report the timings, stop."""
    from son_inference.llamacpp_client import LlamaCppEngine, LlamaServerProcess
    from son_inference.server import ServerConfig

    config = ServerConfig(model_path=args.model, port=args.port, gpu_layers=args.gpu_layers)
    process = LlamaServerProcess(config)
    base_url = process.start()
    print(f"服务已启动：{base_url}")

    try:
        from son_jev import build_grammar, format_jev_prompt

        prompt = format_jev_prompt(json.loads(args.sample))
        engine = LlamaCppEngine(base_url=base_url)
        completion, total_ms, ttft_ms = engine.complete_timed(
            prompt, grammar=build_grammar().grammar
        )
        print(
            json.dumps(
                {
                    "completion": completion,
                    "ttft_ms": round(ttft_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "note": "单次请求，非压测；不足以支撑 P95 结论",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        process.stop()
        print("服务已停止")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("son_api.main:app", host="0.0.0.0", port=args.port, workers=1)
    return 0


def _which(binary: str) -> str | None:
    import shutil

    return shutil.which(binary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="son_cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="检查 GPU 与工具链是否就绪")
    doctor.set_defaults(func=_cmd_doctor)

    train = sub.add_parser("train", help="从 CSV 训练 LoRA / QLoRA")
    train.add_argument("--dataset", required=True)
    train.add_argument("--model", required=True, help="底座模型 id 或本地路径")
    train.add_argument("--output-dir", default="./runs")
    train.add_argument("--job-id", default="mv_local")
    train.add_argument("--prompt-column")
    train.add_argument("--params-b", type=float, default=3.0)
    train.add_argument("--qlora", action="store_true", help="4bit 量化底座")
    train.add_argument("--dpo", action="store_true", help="训练后追加偏好对齐")
    train.add_argument("--n-per-prompt", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=_cmd_train)

    quantize = sub.add_parser("quantize", help="合并 LoRA 并导出 GGUF")
    quantize.add_argument("--adapter", required=True)
    quantize.add_argument("--model", required=True)
    quantize.add_argument("--output-dir", default="./runs/gguf")
    quantize.add_argument("--job-id", default="mv_local")
    quantize.set_defaults(func=_cmd_quantize)

    infer = sub.add_parser("infer", help="启动推理服务并发一次请求")
    infer.add_argument("--model", required=True, help="GGUF 路径")
    infer.add_argument("--sample", default='{"account":"A12345","amount":50000}')
    infer.add_argument("--port", type=int, default=8080)
    infer.add_argument("--gpu-layers", type=int, default=0)
    infer.set_defaults(func=_cmd_infer)

    serve = sub.add_parser("serve", help="启动控制面 API")
    serve.add_argument("--port", type=int, default=9969)
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    code = args.func(args)
    elapsed = time.perf_counter() - started
    print(f"\n用时 {elapsed:.1f}s", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
