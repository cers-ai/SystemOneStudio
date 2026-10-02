"""Qwen 2.5 adapter: the one model family wired up this stage (改造开发方案.md 30).

Its whole purpose is to end the duplication 改造开发方案.md 14 complains about.
Before this, ``TorchTrainingBackend`` called ``AutoTokenizer`` and
``apply_chat_template`` directly, so ``BaseModelAdapter`` was an abstraction
nothing implemented and the family-specific knowledge lived in the trainer.

Everything model-specific now lives here: prompt rendering, the LoRA target
modules, the quantization bits, and the merge. The trainer asks the adapter and
does not branch on which model it is training.

Qwen 2.5 specifics:
* ``apply_chat_template`` with ``add_generation_prompt`` for the user turn; the
  assistant prefix differs per model and getting it wrong produces a model that
  scores well on the training template and badly on the JEV template.
* LoRA targets are ``q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj``. Projecting only attention works but measurably underperforms on
  the MLP, which is where most of a decoder's parameters live.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from son_model_registry import BaseModelAdapter, TrainConfig, register_adapter

#: Attention and MLP projections. See the module docstring for why all seven.
QWEN_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

MODEL_IDS: tuple[str, ...] = ("qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct")

SYSTEM_PROMPT = "你是一个专业的决策模型，负责对输入样本进行判定。"


class QwenAdapterUnavailable(RuntimeError):
    """The training stack cannot be used here."""


@register_adapter
class Qwen25Adapter(BaseModelAdapter):
    """Adapter for the Qwen 2.5 instruct family."""

    adapter_type: ClassVar[str] = "qwen25"

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    # -- loading ------------------------------------------------------------

    def load_tokenizer(self, model_path: str) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            # Qwen ships without a pad token; SFT needs one or batches misalign.
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self, *, quantized: bool = False) -> Any:
        """Load the causal LM, optionally 4-bit.

        Kept separate from ``load_tokenizer`` because the quantisation decision
        belongs to the training config, not to tokenisation.
        """
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "torch_dtype": self._dtype(),
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._dtype(),
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["attn_implementation"] = "sdpa"
        return AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)

    # -- prompts ------------------------------------------------------------

    def format_prompt(self, sample: dict[str, Any]) -> str:
        """Plain instruction format, used when JEV format compat is off."""
        tokenizer = self.load_tokenizer(self.model_path)
        return self.format_prompt_with(tokenizer, sample)

    def format_jev_prompt(self, sample: dict[str, Any]) -> str:
        """JEV format, using the model's own chat template."""
        from son_jev import format_jev_prompt

        from son_contracts import JevFlags

        return format_jev_prompt(sample, JevFlags())

    def format_prompt_with(self, tokenizer: Any, sample: dict[str, Any]) -> str:
        """Render the user turn, ending where the assistant begins."""
        features = render_features(sample)
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": features},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        # No chat template means the base model was not trained for this path.
        # Surfacing it beats silently producing a model that cannot follow turns.
        raise QwenAdapterUnavailable(f"{self.model_path} 没有对话模板，无法按指令方式构造输入")

    # -- training -----------------------------------------------------------

    def build_lora_config(self, train_config: TrainConfig) -> Any:
        from peft import LoraConfig

        return LoraConfig(
            r=int(getattr(train_config, "lora_r", 16)),
            lora_alpha=int(getattr(train_config, "lora_alpha", 32)),
            lora_dropout=float(getattr(train_config, "lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(QWEN_LORA_TARGETS),
        )

    def build_trainer(self, train_config: TrainConfig, dataset: Any) -> Any:
        """Build the SFT trainer for a prepared dataset."""
        from trl import SFTConfig, SFTTrainer

        model = self.load_model(quantized=bool(getattr(train_config, "quantized", False)))
        args = SFTConfig(
            output_dir=str(getattr(train_config, "output_dir", "./runs/lora")),
            num_train_epochs=1,
            max_steps=int(getattr(train_config, "max_steps", 1000)),
            per_device_train_batch_size=int(getattr(train_config, "batch_size", 4)),
            learning_rate=float(getattr(train_config, "learning_rate", 2e-4)),
            bf16=self._bf16(),
            seed=int(getattr(train_config, "seed", 42)),
            logging_steps=10,
            save_strategy="steps",
            save_steps=max(int(getattr(train_config, "max_steps", 1000)) // 4, 50),
            save_total_limit=2,
            report_to=[],
        )
        return SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            peft_config=self.build_lora_config(train_config),
            processing_class=self.load_tokenizer(self.model_path),
        )

    # -- artifacts ----------------------------------------------------------

    def merge_lora(self, base_path: str, lora_path: str, output_path: str) -> str:
        """Fuse the adapter into the base weights.

        Merges before conversion, always. Converting an unmerged base and an
        adapter separately produces a GGUF that has silently ignored training.
        """
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        base = Path(base_path)
        adapter = Path(lora_path)
        if not base.exists():
            raise QwenAdapterUnavailable(f"底座模型不存在：{base}")
        if not adapter.exists():
            raise QwenAdapterUnavailable(f"适配器不存在：{adapter}")

        # CPU merge: slower, but does not need the GPU free and avoids the peak
        # VRAM of a GPU merge on a 12-16GB card.
        model = AutoModelForCausalLM.from_pretrained(
            str(base), torch_dtype=torch.bfloat16, device_map="cpu"
        )
        model = PeftModel.from_pretrained(model, str(adapter))
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out), safe_serialization=True)

        AutoTokenizer.from_pretrained(str(base), trust_remote_code=True).save_pretrained(str(out))
        return str(out)

    def export_gguf(self, model_path: str, output_path: str, quant_level: Any = None) -> str:
        """Convert to f16 GGUF, then quantize.

        f16 first and quantize down: quantising straight from bf16 compounds the
        rounding error twice.
        """
        from son_quantizer.peft_merge import LlamaCppConverter

        from son_contracts import QuantLevel

        level = quant_level or QuantLevel.Q4_K_M
        source = Path(model_path)
        if not source.exists():
            raise QwenAdapterUnavailable(f"待转换的模型目录不存在：{model_path}")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        f16 = out.with_name(f"{out.stem}.f16.gguf")

        tools = LlamaCppConverter()
        tools.convert_hf_to_gguf(str(source), str(f16))
        tools.quantize(str(f16), str(out), level)
        return str(out)

    # -- inference ----------------------------------------------------------

    def infer(self, model_path: str, input_data: dict[str, Any]) -> dict[str, Any]:
        """Single prediction from a merged model. Used by tests and the CLI."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=self._dtype(), device_map="auto"
        )
        prompt = self.format_prompt_with(tokenizer, input_data)

        import torch

        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )

        from son_jev import parse_completion

        return parse_completion(text)

    # -- helpers ------------------------------------------------------------

    def _bf16(self) -> bool:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    def _dtype(self) -> Any:
        import torch

        return torch.bfloat16 if self._bf16() else torch.float16


def render_features(sample: dict[str, Any]) -> str:
    """Feature string for a sample, in stable key order."""
    skip = {"label", "origin"}
    return "\n".join(f"{key}: {sample[key]}" for key in sorted(sample) if key not in skip)


def resolve_model_path(model_id: str, *, cache_dir: str | None = None) -> str:
    """Map a registry id to a local path, downloading if needed.

    Raises rather than returning a bare id: passing a model id to
    ``from_pretrained`` looks like it works and then fails deep inside
    transformers with a confusing message.
    """
    import os

    if Path(model_id).exists():
        return model_id

    cache = cache_dir or os.environ.get("HF_HOME")
    if cache is None:
        raise QwenAdapterUnavailable(
            f"{model_id} 不是本地路径，且未设置 HF_HOME，无法下载；"
            "请先拉取底座模型或设置模型缓存目录"
        )
    from huggingface_hub import snapshot_download

    repo = f"Qwen/{model_id.replace('-instruct', '-Instruct')}"
    return snapshot_download(repo_id=repo, cache_dir=cache)


def supported_models() -> list[str]:
    return list(MODEL_IDS)
