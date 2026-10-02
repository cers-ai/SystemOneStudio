"""llama.cpp server client and process manager.

GPU-side implementation of :class:`~son_inference.predict.InferenceEngine`.
Talks to a ``llama-server`` over HTTP rather than embedding llama.cpp in-process,
so the inference runtime stays out of the Python dependency tree entirely.

Timing note: this client measures time-to-first-token and total separately by
streaming. Without streaming there is no way to tell them apart, and reporting
TTFT as the total is how a fast-looking model gets shipped against a latency
target it does not meet.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from son_inference.predict import InferenceEngine
from son_inference.server import ServerConfig, ServerError

# SIGKILL does not exist on Windows. Resolved once at import so `stop()` works on
# both platforms, and so mypy does not flag a name the platform may lack.
_SIGKILL = getattr(signal, "SIGKILL", 9)


class LlamaCppUnavailable(ServerError):
    """The llama.cpp server is not installed or not running."""


class LlamaCppEngine(InferenceEngine):
    """Streams from a running ``llama-server``."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)

    def is_ready(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/health", timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def complete(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        max_tokens: int = 256,
        stop: tuple[str, ...] = (),
    ) -> str:
        return "".join(self.stream(prompt, grammar=grammar, max_tokens=max_tokens, stop=stop))

    def stream(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        max_tokens: int = 256,
        stop: tuple[str, ...] = (),
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive.

        ``temperature=0`` by default: an evaluation run has to be reproducible,
        and sampling would make every measured latency and accuracy number noisy.
        Sampling is enabled explicitly by the preference-pair generator, which
        needs the spread.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "stream": True,
            "temperature": temperature,
            "top_p": 1.0 if temperature == 0.0 else 0.95,
            "cache_prompt": True,
        }
        if grammar:
            payload["grammar"] = grammar
        if stop:
            payload["stop"] = list(stop)

        with self._client.stream(
            "POST", f"{self.base_url}/completion", json=payload, timeout=self.timeout_s
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", "replace")[:500]
                raise LlamaCppUnavailable(f"llama-server 返回 {response.status_code}：{body}")
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = line[6:] if line.startswith("data: ") else line
                if chunk.strip() == "[DONE]":
                    return
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                piece = parsed.get("content")
                if piece:
                    yield piece
                if parsed.get("stop"):
                    return

    def complete_timed(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        max_tokens: int = 256,
    ) -> tuple[str, float, float]:
        """Return (completion, total_ms, ttft_ms) from one streaming call.

        TTFT is measured to the first content chunk, which is the number the
        requirement calls 首 Token 延迟.
        """
        start = time.perf_counter()
        ttft: float | None = None
        pieces: list[str] = []
        for piece in self.stream(prompt, grammar=grammar, max_tokens=max_tokens):
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            pieces.append(piece)
        total = (time.perf_counter() - start) * 1000
        return "".join(pieces), total, ttft if ttft is not None else total

    def close(self) -> None:
        self._client.close()


class LlamaServerProcess:
    """Starts and stops a ``llama-server`` as a child process."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path = Path(os.environ.get("SON_LLAMA_LOG", "/data/logs/llama-server.log"))

    def start(self, *, wait_s: float = 300.0) -> str:
        binary = shutil.which("llama-server")
        if binary is None:
            raise LlamaCppUnavailable(
                "llama-server 不在 PATH 上。该步骤必须在安装了 llama.cpp 的 GPU 节点执行；"
                "本机无 GPU 时不做本地验证。"
            )
        if not Path(self.config.model_path).exists():
            raise LlamaCppUnavailable(f"GGUF 模型文件不存在：{self.config.model_path}")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [binary, *self.config.argv()[1:]]

        with self.log_path.open("ab") as log:
            self.process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )

        base_url = self.base_url()
        deadline = time.monotonic() + wait_s
        engine = LlamaCppEngine(base_url=base_url)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = _tail(self.log_path)
                raise LlamaCppUnavailable(f"llama-server 启动即退出：\n{tail}")
            if engine.is_ready():
                return base_url
            time.sleep(1.0)

        self.stop()
        raise LlamaCppUnavailable(
            f"llama-server 在 {wait_s}s 内未就绪，日志：\n{_tail(self.log_path)}"
        )

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.config.port}"

    def stop(self, *, timeout: float = 15.0) -> None:
        """Stop the server, refusing to claim success it cannot verify.

        Used to swallow every signal failure and then set ``process = None``, so
        ``running`` reported False while the child still held port 8080 -- and
        the CLI printed "服务已停止" regardless.
        """
        if self.process is None:
            return
        process = self.process
        self.process = None
        self._signal(signal.SIGTERM, timeout)
        if process.poll() is None:
            self._signal(_SIGKILL, timeout)
        process.wait(timeout=timeout)
        if process.poll() is None:
            raise LlamaCppUnavailable(
                f"推理服务未在 {timeout}s 内退出（pid {process.pid}），"
                f"可能仍占用端口 {self.config.port}。请手动确认。"
            )

    def _signal(self, sig: int, timeout: float) -> None:
        """Signal the process group where the platform allows it.

        ``start_new_session=True`` puts the child in its own group, so signalling
        the group also reaps anything llama.cpp spawned. ``os.killpg`` and
        ``os.getpgid`` are POSIX-only and this module is imported on Windows for
        tests, so the direct signal is the fallback there.
        """
        process = self.process
        if process is None:
            return
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(process.pid), sig)  # type: ignore[attr-defined]
            elif sig == _SIGKILL:
                process.kill()
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


def _tail(path: Path, lines: int = 25) -> str:
    if not path.exists():
        return "(无日志)"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def gpu_report() -> dict[str, Any]:
    """What GPU is actually present, for the deployment report.

    Reported rather than assumed: the whole verification story depends on whether
    a number came from a real GPU or from a CPU fallback.
    """
    report: dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch
    except ImportError:
        report["reason"] = "torch 未安装"
        return report

    report["cuda_available"] = bool(torch.cuda.is_available())
    report["torch_version"] = torch.__version__
    report["cuda_version"] = getattr(torch.version, "cuda", None)
    if report["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 1),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    else:
        report["reason"] = "torch.cuda.is_available() 为 False"
    return report
