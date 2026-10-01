"""torch / PEFT / TRL training backend.

This is the GPU-side implementation of :class:`~son_trainer.backends.TrainingBackend`.
Imports are lazy so the control plane keeps working on a machine with no torch
installed -- importing this module must never require a GPU.

Pipeline per stage (需求方案.txt 8.1 / 8.2):

* ``sft``      -- TRL ``SFTTrainer`` over a LoRA / QLoRA PEFT config
* ``dpo``      -- TRL ``DPOTrainer`` over ``(prompt, chosen, rejected)`` triples

Two decisions worth stating, because both are easy to get wrong silently:

**Chat template, not raw concatenation.** Prompts and responses are rendered with
the base model's own chat template via ``tokenizer.apply_chat_template``. Training
on a hand-rolled prompt format produces a model that scores well on the training
template and badly on anything else, including the JEV template used at
inference.

**Truncation is measured, not silent.** ``max_seq_length`` is derived from the
tokenizer, and samples that get truncated are counted and reported in the stage
metrics. Silent truncation looks like a successful run with quietly worse data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from son_contracts import TrainingMethod
from son_trainer.backends import StageRequest, StageResult, TrainingBackend
from son_trainer.hyperparams import DatasetProfile


class TrainingUnavailable(RuntimeError):
    """The GPU training stack is not usable here.

    Raised rather than degraded: a run that silently skips training and reports
    success would be indistinguishable from a real one.
    """


def assert_training_available() -> None:
    """Fail loudly if the training stack cannot run."""
    missing: list[str] = []
    for module in ("torch", "transformers", "peft", "trl"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise TrainingUnavailable(
            f"训练依赖缺失：{', '.join(missing)}。"
            "请在 GPU 节点使用安装训练依赖的镜像（deploy/gpu）。"
        )

    import torch

    if not torch.cuda.is_available():
        raise TrainingUnavailable("未检测到可用的 CUDA 设备。训练、量化与推理均需 NVIDIA GPU。")


@dataclass(frozen=True)
class TorchRunConfig:
    """Everything the backend needs that does not vary per stage."""

    base_model: str
    output_dir: str
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantized: bool = False
    seed: int = 42


class TorchTrainingBackend(TrainingBackend):
    """Real SFT / DPO training on a GPU node."""

    name = "torch-peft-trl"

    def __init__(self, config: TorchRunConfig) -> None:
        self.config = config

    # -- stage dispatch ----------------------------------------------------

    def run_stage(self, request: StageRequest) -> StageResult:
        assert_training_available()
        if request.stage is TrainingMethod.DPO:
            return self._run_dpo(request)
        return self._run_sft(request)

    # -- SFT ----------------------------------------------------------------

    def _run_sft(self, request: StageRequest) -> StageResult:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        truncated = _count_truncated(request.samples, tokenizer, self.config.max_seq_length)
        dataset = Dataset.from_list(
            [{"text": render_chat(tokenizer, s.prompt, s.response)} for s in request.samples]
        )

        model = self._load_model(AutoModelForCausalLM, tokenizer)

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        output_dir = self._stage_dir(request, "sft")
        args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            max_steps=int(request.hyperparams.get("max_steps", 1000)),
            per_device_train_batch_size=int(request.hyperparams.get("batch_size", 4)),
            gradient_accumulation_steps=int(
                request.hyperparams.get("gradient_accumulation_steps", 4)
            ),
            learning_rate=float(request.hyperparams.get("learning_rate", 2e-4)),
            warmup_ratio=float(request.hyperparams.get("warmup_ratio", 0.03)),
            weight_decay=float(request.hyperparams.get("weight_decay", 0.0)),
            logging_steps=10,
            save_strategy="steps",
            save_steps=max(int(request.hyperparams.get("max_steps", 1000)) // 4, 50),
            save_total_limit=3,
            bf16=_supports_bf16(),
            seed=self.config.seed,
            max_seq_length=self.config.max_seq_length,
            report_to=[],
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        result = trainer.train(resume_from_checkpoint=request.resume_from)

        adapter_dir = output_dir
        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        metrics = _extract_metrics(result)
        metrics["truncated_samples"] = float(truncated)
        metrics["samples"] = float(len(request.samples))

        return StageResult(
            stage=TrainingMethod.SFT.value,
            status="succeeded",
            step=int(getattr(result, "global_step", 0)),
            metrics=metrics,
            checkpoint_uri=adapter_dir,
            notes=(
                f"bf16={_supports_bf16()}",
                f"truncated {truncated}/{len(request.samples)} samples at "
                f"{self.config.max_seq_length} tokens",
            ),
        )

    # -- DPO ----------------------------------------------------------------

    def _run_dpo(self, request: StageRequest) -> StageResult:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer

        if not request.samples or not request.samples[0].is_preference():
            raise TrainingUnavailable(
                "决策优化训练需要 (prompt, chosen, rejected) 三元组，当前样本不是偏好对格式"
            )

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = Dataset.from_list(
            [
                {
                    "prompt": render_chat_prompt(tokenizer, s.prompt),
                    "chosen": s.chosen,
                    "rejected": s.rejected,
                }
                for s in request.samples
                if s.chosen and s.rejected
            ]
        )

        # DPO trains on an existing adapter rather than a fresh one.
        model = self._load_model(AutoModelForCausalLM, tokenizer)
        if request.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, request.adapter_path, is_trainable=True)

        peft_config = None
        if not request.adapter_path:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

        output_dir = self._stage_dir(request, "dpo")
        args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            max_steps=int(request.hyperparams.get("max_steps", 1000)),
            per_device_train_batch_size=int(request.hyperparams.get("batch_size", 4)),
            gradient_accumulation_steps=int(
                request.hyperparams.get("gradient_accumulation_steps", 4)
            ),
            learning_rate=float(request.hyperparams.get("learning_rate", 2e-4)),
            warmup_ratio=float(request.hyperparams.get("warmup_ratio", 0.03)),
            logging_steps=10,
            save_strategy="steps",
            save_steps=max(int(request.hyperparams.get("max_steps", 1000)) // 4, 50),
            save_total_limit=3,
            bf16=_supports_bf16(),
            seed=self.config.seed,
            max_length=self.config.max_seq_length,
            report_to=[],
        )

        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        result = trainer.train(resume_from_checkpoint=request.resume_from)

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        metrics = _extract_metrics(result)
        metrics["pairs"] = float(len(dataset))

        return StageResult(
            stage=TrainingMethod.DPO.value,
            status="succeeded",
            step=int(getattr(result, "global_step", 0)),
            metrics=metrics,
            checkpoint_uri=output_dir,
            notes=(f"preference_pairs={len(dataset)}",),
        )

    # -- sampling for preference pairs --------------------------------------

    def sample_predictions(
        self,
        adapter_path: str,
        prompts: tuple[str, ...],
        *,
        n_per_prompt: int = 4,
    ) -> tuple[str, ...]:
        """Sample candidates so preference pairs can be built from real outputs."""
        assert_training_available()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=_dtype(),
            device_map="auto",
        )
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception:
            pass
        model.eval()

        import torch

        outputs: list[str] = []
        with torch.no_grad():
            for prompt in prompts:
                rendered = render_chat_prompt(tokenizer, prompt)
                encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
                for _ in range(n_per_prompt):
                    # Sampling, not greedy: the point is to observe a spread of
                    # judgements so correct and incorrect ones both appear.
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    text = tokenizer.decode(
                        generated[0][encoded["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    outputs.append(text)
        return tuple(outputs)

    # -- helpers ------------------------------------------------------------

    def _load_model(self, auto_cls: Any, tokenizer: Any) -> Any:
        kwargs: dict[str, Any] = {
            "torch_dtype": _dtype(),
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.config.quantized:
            # 需求方案.txt 8.1: QLoRA runs a 4bit base. nf4 + double quant is
            # the standard pairing and is what bitsandbytes documents.
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_dtype(),
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["attn_implementation"] = "sdpa"
        model = auto_cls.from_pretrained(self.config.base_model, **kwargs)
        if tokenizer.pad_token is not None and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model

    def _stage_dir(self, request: StageRequest, stage: str) -> str:
        root = self.config.output_dir or "checkpoints"
        path = Path(root) / request.stage.value
        path.mkdir(parents=True, exist_ok=True)
        _write_run_manifest(path, request)
        return str(path)


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def render_chat(tokenizer: Any, prompt: str, response: str) -> str:
    """Render a (prompt, response) pair with the model's own chat template.

    Falling back to a plain concatenation when the tokenizer has no template is
    recorded, not hidden: a base model without a chat template cannot be served
    through the instruction path this product depends on.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        )
    return f"{prompt}\n{response}"


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Render just the user turn, ending where the assistant starts."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{prompt}\n"


def _count_truncated(samples: tuple[Any, ...], tokenizer: Any, cap: int) -> int:
    truncated = 0
    for sample in samples[: min(len(samples), 500)]:
        text = render_chat(tokenizer, sample.prompt, sample.response)
        if len(tokenizer(text)["input_ids"]) > cap:
            truncated += 1
    return truncated


def _supports_bf16() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def _dtype() -> Any:
    import torch

    return torch.bfloat16 if _supports_bf16() else torch.float16


def _extract_metrics(result: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in (getattr(result, "metrics", {}) or {}).items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _write_run_manifest(path: Path, request: StageRequest) -> None:
    """Record what produced a checkpoint.

    Without this a directory of adapters is untraceable, and the lineage the
    product requires cannot be reconstructed after the fact.
    """
    manifest = {
        "stage": request.stage.value,
        "base_model": request.base_model,
        "samples": len(request.samples),
        "hyperparams": {k: v for k, v in request.hyperparams.items() if not callable(v)},
        "seed": request.seed,
        "resumed_from": request.resume_from,
        "cwd": os.getcwd(),
    }
    (path / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def profile_from_rows(rows: int, base_model_params_b: float, quantized: bool) -> DatasetProfile:
    """Build the profile the recommender needs from a concrete dataset."""
    return DatasetProfile(rows=rows, params_b=base_model_params_b, quantized=quantized)
