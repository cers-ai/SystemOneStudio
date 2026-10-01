"""llama.cpp server lifecycle (需求方案.txt 5.7.4).

Deployment options: HTTP API (recommended), Ollama local service, or plain file
export. llama.cpp is the default because the P95 target rules out heavier
stacks (技术方案.md 1.1).

NOT VERIFIED: no server has been started here. This machine has no GPU and no
llama.cpp checkout, so start/stop/load and the latency they produce are all
unexercised. See AGENTS.md.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any


class ServerError(RuntimeError):
    """The inference server could not be started or is not healthy."""


@dataclass(frozen=True)
class ServerConfig:
    """Launch parameters for ``llama-server``."""

    model_path: str
    port: int = 8080
    gpu_layers: int = 0
    context_size: int = 8192
    continuous_batching: bool = True
    #: Concurrency the UI asks for (需求方案.txt 5.7.4). Applied to the client
    #: side; llama-server's own parallelism is set separately.
    concurrency: int = 8

    def argv(self) -> list[str]:
        args = [
            "llama-server",
            "--model",
            self.model_path,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--n-gpu-layers",
            str(self.gpu_layers),
        ]
        if self.continuous_batching:
            # Required for throughput: without it, concurrent requests queue
            # behind each other and QPS collapses to 1/latency.
            args += ["--cont-batching"]
        args += ["--parallel", str(max(self.concurrency, 1))]
        return args


@dataclass
class ServerHandle:
    """A started server."""

    config: ServerConfig
    process: subprocess.Popen[bytes] | None = None
    started_at: float | None = None
    log: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.config.port}"

    def predict_path(self) -> str:
        """The public prediction endpoint (需求方案.txt 5.7.4)."""
        return f"{self.base_url()}/v1/predict"

    def uptime_s(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def stop(self, *, timeout: float = 10.0) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.log.append("terminate 超时，强制结束")
            self.process.kill()
        self.process = None


def llama_cpp_available(binary: str = "llama-server") -> bool:
    return shutil.which(binary) is not None


def start_server(
    config: ServerConfig,
    *,
    ready_timeout_s: float = 120.0,
    poll_interval_s: float = 0.5,
    dry_run: bool = False,
) -> ServerHandle:
    """Start ``llama-server`` and wait for it to accept requests.

    ``dry_run`` returns the handle without spawning, which is how the launch
    command is verified on a machine that has no GPU.
    """
    handle = ServerHandle(config=config)

    if not llama_cpp_available():
        raise ServerError(
            "llama-server 不在 PATH 上。该步骤必须在 GPU 节点执行；本机无 GPU 时不做本地验证。"
        )

    argv = config.argv()
    handle.log.append(" ".join(argv))

    if dry_run:
        handle.log.append("dry-run：未实际启动进程")
        return handle

    handle.process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    handle.started_at = time.monotonic()

    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            output = handle.process.stdout
            detail = ""
            if output is not None:
                detail = output.read().decode("utf-8", "replace")[-800:]
            raise ServerError(f"llama-server 启动即退出：{detail}")
        if _port_open(config.port):
            handle.log.append(f"服务就绪，用时 {handle.uptime_s():.1f}s")
            return handle
        time.sleep(poll_interval_s)

    handle.stop()
    raise ServerError(f"llama-server 在 {ready_timeout_s}s 内未就绪")


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def deployment_summary(config: ServerConfig, *, jev_format_compat: bool) -> dict[str, Any]:
    """The panel shown after 一键部署 (需求方案.txt 5.7.4)."""
    handle = ServerHandle(config=config)
    curl = (
        f"curl -X POST {handle.predict_path()} "
        f'-H "Content-Type: application/json" '
        f'-d \'{{"account":"A12345","amount":50000}}\''
    )
    return {
        "api_url": handle.predict_path(),
        "docs_url": f"{handle.base_url()}/docs",
        "curl": curl,
        "concurrency": config.concurrency,
        # Turning format compat off must be visible to the caller: they get the
        # same four fields but the response is flagged non-JEV.
        "jev_format_compat": jev_format_compat,
        "notes": (
            () if jev_format_compat else ("已关闭输出格式对齐：响应将标记 jev_compatible=false",)
        ),
    }
