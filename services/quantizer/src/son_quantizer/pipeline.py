"""LoRA merge and GGUF quantization orchestration (需求方案.txt 5.7.1).

Three steps, all requiring llama.cpp tooling and a GPU:

    merged safetensors -> GGUF convert -> llama-quantize to the target level

Q4_K_M is the default because the JEV eval baseline is defined at Q4_K_M
(``JEV_BASELINE_QUANT``), and evaluation numbers are only comparable when every
model is measured at the same level.

The subprocess boundary is the point of this module: it is what makes the
ordering, the default level and the artifact registration testable without a GPU
or a llama.cpp checkout. The commands themselves are assembled here and never
guessed at the call site, so a wrong flag is a visible string, not a runtime
surprise.

UNVERIFIED: this machine has no GPU and no llama.cpp. No file below has been
executed against a real model. See AGENTS.md.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from son_contracts import JEV_BASELINE_QUANT, ArtifactFormat, QuantLevel

#: Artifacts produced automatically after training (需求方案.txt 5.7.1).
#: Q4_K_M first because it is the recommended deployment format.
AUTO_ARTIFACT_LEVELS: tuple[QuantLevel, ...] = (
    QuantLevel.Q4_K_M,
    QuantLevel.Q5_K_M,
    QuantLevel.Q8_0,
)


class ToolchainError(RuntimeError):
    """A required external tool is missing or failed."""


@dataclass(frozen=True)
class QuantizeRequest:
    merged_path: str
    output_dir: str
    base_model: str | None = None
    levels: tuple[QuantLevel, ...] = AUTO_ARTIFACT_LEVELS
    model_version: str | None = None


@dataclass(frozen=True)
class Artifact:
    """A produced file, ready for registration.

    Frozen because this is a record of fact: mutating a produced artifact's
    path or format after the fact would corrupt the version lineage that
    需求方案.txt 5.7.1 requires.
    """

    path: str
    format: ArtifactFormat
    quant: QuantLevel | None
    size_bytes: int | None = None
    sha256: str | None = None
    is_recommended: bool = False

    def describe(self) -> str:
        if self.quant is None:
            return f"{self.path}（原生权重）"
        suffix = "（推荐部署格式）" if self.is_recommended else ""
        return f"{self.path}（{self.quant.value}）{suffix}"


@dataclass
class QuantizeResult:
    artifacts: list[Artifact] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    native_weights: Artifact | None = None
    notes: tuple[str, ...] = ()

    @property
    def recommended(self) -> Artifact | None:
        return next((a for a in self.artifacts if a.is_recommended), None)

    def describe(self) -> list[str]:
        lines = [a.describe() for a in self.artifacts]
        if self.native_weights is not None:
            lines.append(self.native_weights.describe())
        return lines


class LlamaCppTools(ABC):
    """The two llama.cpp binaries this pipeline drives."""

    @abstractmethod
    def convert_hf_to_gguf(self, source: str, destination: str) -> None: ...

    @abstractmethod
    def quantize(self, source: str, destination: str, level: QuantLevel) -> None: ...

    @abstractmethod
    def merge_lora(self, base: str, adapter: str, destination: str) -> None: ...

    def available(self) -> bool:
        return True


class SubprocessLlamaCppTools(LlamaCppTools):
    """Real implementation. Requires llama.cpp on PATH or explicit paths."""

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

    def available(self) -> bool:
        if shutil.which(self.quantize_bin) is None:
            return False
        if shutil.which(self.python_bin) is None:
            return False
        from pathlib import Path

        return Path(self.convert_script).exists()

    def convert_hf_to_gguf(self, source: str, destination: str) -> None:

        _run(
            [self.python_bin, self.convert_script, source, "--outfile", destination],
            "convert_hf_to_gguf.py",
        )

    def quantize(self, source: str, destination: str, level: QuantLevel) -> None:

        _run(
            [self.quantize_bin, source, destination, level.value],
            "llama-quantize",
        )

    def merge_lora(self, base: str, adapter: str, destination: str) -> None:
        raise ToolchainError(
            "合并 LoRA 权重需要目标底座的 PEFT 环境（torch/transformers/peft），"
            "属于 GPU 节点职责；本模块只编排外部命令，不在无 GPU 环境伪造结果"
        )


def _run(argv: list[str], tool: str) -> None:
    import subprocess

    try:
        subprocess.run(argv, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ToolchainError(f"{tool} 不在 PATH 上：{exc}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace")[-500:]
        raise ToolchainError(f"{tool} 失败：{stderr}") from exc


def build_gguf_filename(model_version: str | None, level: QuantLevel) -> str:
    stem = model_version or "model"
    return f"{stem}.{level.value}.gguf"


def quantize_all(
    request: QuantizeRequest,
    tools: LlamaCppTools,
    *,
    include_native: bool = True,
    hash_outputs: bool = False,
) -> QuantizeResult:
    """Produce every GGUF level plus the native weights.

    The conversion step runs once and each level quantizes from the converted
    file. Converting per level would redo the expensive part N times.
    """
    if not request.levels:
        raise ValueError("at least one quantization level is required")

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = QuantizeResult()

    if not tools.available():
        raise ToolchainError(
            "llama.cpp 工具链不可用（需要 convert_hf_to_gguf.py 与 llama-quantize）。"
            "该步骤必须在 GPU 节点执行；本机无 GPU 时不做本地验证。"
        )

    f16_path = output_dir / build_gguf_filename(request.model_version, QuantLevel.Q4_K_M).replace(
        f".{QuantLevel.Q4_K_M.value}.gguf", ".f16.gguf"
    )

    result.commands.append(f"convert_hf_to_gguf {request.merged_path} -> {f16_path}")
    tools.convert_hf_to_gguf(request.merged_path, str(f16_path))

    for level in request.levels:
        destination = output_dir / build_gguf_filename(request.model_version, level)
        result.commands.append(f"llama-quantize {f16_path} -> {destination} [{level.value}]")
        tools.quantize(str(f16_path), str(destination), level)
        result.artifacts.append(
            Artifact(
                path=str(destination),
                format=ArtifactFormat.GGUF,
                quant=level,
                size_bytes=destination.stat().st_size if destination.exists() else None,
                sha256=_sha256(destination) if hash_outputs and destination.exists() else None,
                is_recommended=level.value == JEV_BASELINE_QUANT,
            )
        )

    if include_native:
        merged = Path(request.merged_path)
        result.native_weights = Artifact(
            path=request.merged_path,
            format=ArtifactFormat.NATIVE,
            quant=None,
            size_bytes=merged.stat().st_size if merged.exists() else None,
        )

    notes: list[str] = []
    if result.recommended is None:
        notes.append(f"未产出 {JEV_BASELINE_QUANT}，评测结果将无法与 JEV 基线对比")
    if hash_outputs:
        notes.append("已记录校验和，可用于产物完整性核对")
    result.notes = tuple(notes)

    return result


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifacts_for_registration(result: QuantizeResult) -> list[Artifact]:
    """The artifacts to write to the database, native weights included."""
    return [*result.artifacts, *([result.native_weights] if result.native_weights else [])]
