"""Real LoRA merge using PEFT, plus llama.cpp conversion and quantization.

This is the GPU-side implementation of :class:`~son_quantizer.pipeline.LlamaCppTools`.
All three steps need a GPU node for the merge; conversion and quantization shell
out to llama.cpp binaries.

Ordering matters and is the thing most likely to be got wrong:

    base + adapter  --peft merge-->  merged/  --convert-->  f16.gguf
                                                      --quantize-->  Q4_K_M.gguf

The merge happens *before* conversion. Converting an unmerged base and an
adapter separately produces a GGUF that has silently ignored the training.

Nothing here fakes a file: if a binary is missing or a step fails, it raises with
the tool's own stderr. A quantized file that was not actually quantized would
corrupt every evaluation run that touched it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from son_contracts import QuantLevel
from son_quantizer.pipeline import LlamaCppTools, ToolchainError


class PeftLoRaMerger:
    """Merge a LoRA adapter into its base model weights."""

    def merge(
        self,
        base_path: str,
        adapter_path: str,
        destination: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cpu",
    ) -> str:
        """Merge and save to `destination`, returning that path.

        Merges on CPU by default. It is slower but does not need the GPU to be
        free, and the peak memory of a GPU merge is a common OOM source on the
        12-16GB cards the MVP targets.
        """
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ToolchainError(
                f"合并 LoRA 需要 torch / transformers / peft：{exc}。"
                "请使用安装 merge 依赖的 GPU 镜像（deploy/gpu）。"
            ) from exc

        base = Path(base_path)
        adapter = Path(adapter_path)
        if not base.exists():
            raise ToolchainError(f"底座模型路径不存在：{base_path}")
        if not adapter.exists():
            raise ToolchainError(f"适配器路径不存在：{adapter_path}")

        out = Path(destination)
        out.mkdir(parents=True, exist_ok=True)

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            str(base), torch_dtype=torch_dtype, device_map=device
        )
        model = PeftModel.from_pretrained(model, str(adapter))

        merged = model.merge_and_unload()
        merged.save_pretrained(str(out), safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
        tokenizer.save_pretrained(str(out))

        return str(out)


class LlamaCppConverter(LlamaCppTools):
    """llama.cpp conversion and quantization via subprocess.

    ``llama_merge_lora`` is used when no PEFT merge has happened, because it can
    fuse an adapter into an already-converted GGUF. Both paths are supported;
    the PEFT merge is preferred as it happens before conversion and therefore
    cannot leave a converted base behind an unapplied adapter.
    """

    def __init__(
        self,
        *,
        python_bin: str | None = None,
        convert_script: str | None = None,
        quantize_bin: str | None = None,
    ) -> None:
        self.python_bin = python_bin or "python"
        self.convert_script = convert_script or "convert_hf_to_gguf.py"
        self.quantize_bin = quantize_bin or "llama-quantize"
        self.merger = PeftLoRaMerger()

    # -- capability ---------------------------------------------------------

    def available(self) -> bool:
        return self.convert_script_exists() and shutil.which(self.quantize_bin) is not None

    def convert_script_exists(self) -> bool:
        return Path(self.convert_script).exists()

    def require(self) -> None:
        """Raise if the toolchain is not usable, naming what is missing."""
        missing: list[str] = []
        if not self.convert_script_exists():
            missing.append(f"{self.convert_script}（HF → GGUF 转换脚本）")
        if shutil.which(self.quantize_bin) is None:
            missing.append(f"{self.quantize_bin}（llama.cpp 量化工具）")
        if missing:
            raise ToolchainError(
                "llama.cpp 工具链不可用，缺少：" + "、".join(missing) + "。"
                "该步骤必须在安装了 llama.cpp 的 GPU 节点执行。"
            )

    # -- operations ---------------------------------------------------------

    def convert_hf_to_gguf(self, source: str, destination: str) -> None:
        self.require()
        argv = [
            self.python_bin,
            self.convert_script,
            source,
            "--outfile",
            destination,
            "--outtype",
            "f16",
        ]
        # f16 first, then quantize down: quantizing straight from bf16 weights
        # compounds the rounding error twice.
        _run(argv, "convert_hf_to_gguf.py")

    def quantize(self, source: str, destination: str, level: QuantLevel) -> None:
        self.require()
        if not Path(source).exists():
            raise ToolchainError(f"待量化的 GGUF 不存在：{source}")
        argv = [self.quantize_bin, source, destination, level.value]
        _run(argv, "llama-quantize")

    def merge_lora(self, base: str, adapter: str, destination: str) -> str:
        return self.merger.merge(base, adapter, destination)

    def merge_via_llama(self, base_gguf: str, adapter_gguf: str, destination: str) -> str:
        """Fuse an adapter into a converted GGUF with llama.cpp's own merger.

        Only needed when the PEFT merge was skipped. Kept separate so the
        preferred path stays the default.
        """
        binary = "llama-merge-lora"
        if shutil.which(binary) is None:
            raise ToolchainError(
                f"{binary} 不在 PATH 上。请先执行 PEFT 合并（推荐），"
                "或安装完整的 llama.cpp 工具集。"
            )
        argv = [binary, "--base", base_gguf, "--lora", adapter_gguf, "--out", destination]
        _run(argv, binary)
        return destination


def _run(argv: list[str], tool: str) -> None:
    try:
        completed = subprocess.run(argv, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ToolchainError(f"{tool} 不在 PATH 上：{argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace")[-1500:]
        stdout = exc.stdout.decode("utf-8", "replace")[-500:]
        raise ToolchainError(
            f"{tool} 执行失败（退出码 {exc.returncode}）\n"
            f"--- stderr ---\n{stderr}\n--- stdout ---\n{stdout}"
        ) from exc
    # Surface progress so a long quantize is not a silent wait.
    tail = completed.stdout.decode("utf-8", "replace")[-400:]
    if tail.strip():
        print(f"[{tool}] {tail.strip()}")
