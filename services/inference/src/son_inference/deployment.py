"""Deployment lifecycle (改造开发方案.md 20).

One runtime this stage: a real ``llama-server`` process, started, health-checked,
and stopped. Not a command the user is told to run.

``status`` only becomes ``SERVING`` after a successful health check. A process
that was launched but is not answering is ``STARTING``, not ``SERVING`` --
reporting it as serving is how a deployment looks healthy while every request
fails (Rule 6).
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class DeploymentStatus(StrEnum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    SERVING = "SERVING"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class DeploymentFailed(RuntimeError):
    """The server could not be started or did not become healthy."""


@dataclass
class Deployment:
    """A managed llama.cpp server."""

    gguf_path: str
    port: int = 8081
    host: str = "127.0.0.1"
    gpu_layers: int = 99
    context_size: int = 8192
    concurrency: int = 8

    status: DeploymentStatus = DeploymentStatus.PENDING
    pid: int | None = None
    log_path: str | None = None
    error: str | None = None
    started_at: float | None = None
    _process: Any = field(default=None, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def predict_path(self) -> str:
        return f"{self.base_url}/v1/predict"

    def argv(self) -> list[str]:
        """The exact command. Built here so a wrong flag is a visible string."""
        return [
            "llama-server",
            "--model",
            self.gguf_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--n-gpu-layers",
            str(self.gpu_layers),
            "--ctx-size",
            str(self.context_size),
            "--cont-batching",
            "--parallel",
            str(max(self.concurrency, 1)),
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "runtime": "llama_cpp",
            "status": self.status.value,
            "port": self.port,
            "pid": self.pid,
            "predict_url": self.predict_path,
            "docs_url": f"{self.base_url}/docs",
            "log_path": self.log_path,
            "error": self.error,
        }


def binary_available() -> bool:
    return shutil.which("llama-server") is not None


def gguf_exists(gguf_path: str) -> bool:
    return Path(gguf_path).exists()


def start(
    gguf_path: str,
    *,
    port: int = 8081,
    gpu_layers: int = 99,
    concurrency: int = 8,
    log_path: str | None = None,
    wait_s: float = 300.0,
    poll_s: float = 1.0,
    launcher: Any = None,
) -> Deployment:
    """Launch the server and block until it is healthy.

    ``launcher`` exists so a test can drive the lifecycle without spawning a real
    binary. The production path uses ``subprocess.Popen``.
    """
    if not gguf_exists(gguf_path):
        raise DeploymentFailed(f"GGUF 产物不存在：{gguf_path}")
    if launcher is None and not binary_available():
        raise DeploymentFailed(
            "llama-server 不在 PATH 上。该步骤需要在装有 llama.cpp 的 GPU 节点执行。"
        )

    deployment = Deployment(
        gguf_path=gguf_path,
        port=port,
        gpu_layers=gpu_layers,
        concurrency=concurrency,
        log_path=log_path,
        status=DeploymentStatus.STARTING,
    )

    if launcher is not None:
        deployment._process = launcher(deployment)
        deployment.pid = getattr(deployment._process, "pid", None)
    else:
        import subprocess

        Path(log_path or "llama-server.log").parent.mkdir(parents=True, exist_ok=True)
        # The log file is opened for the lifetime of the child, so it is closed
        # in this scope: the child inherits the descriptor and keeps it.
        with Path(log_path or "llama-server.log").open("ab") as handle:
            deployment._process = subprocess.Popen(
                deployment.argv(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        deployment.pid = deployment._process.pid

    deployment.started_at = time.monotonic()
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if health_check(deployment):
            deployment.status = DeploymentStatus.SERVING
            return deployment
        if _has_exited(deployment):
            deployment.status = DeploymentStatus.FAILED
            deployment.error = _log_tail(deployment.log_path)
            raise DeploymentFailed(f"推理服务启动即退出：{deployment.error}")
        time.sleep(poll_s)

    deployment.status = DeploymentStatus.FAILED
    deployment.error = f"{wait_s}s 内未通过健康检查"
    raise DeploymentFailed(deployment.error)


def health_check(deployment: Deployment, *, timeout_s: float = 3.0) -> bool:
    """Ask the server whether it is serving.

    Only this sets SERVING.
    """
    import httpx

    try:
        response = httpx.get(f"{deployment.base_url}/health", timeout=timeout_s)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def stop(deployment: Deployment, *, timeout: float = 15.0) -> Deployment:
    """Stop the server and verify it exited.

    Reporting a successful stop without checking is how a caller comes to believe
    a port is free while the process still holds it.
    """
    process = deployment._process
    if process is None:
        deployment.status = DeploymentStatus.STOPPED
        return deployment

    import contextlib

    with contextlib.suppress(ProcessLookupError, OSError):
        process.terminate()

    try:
        process.wait(timeout=timeout)
    except Exception:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()

    if not _has_exited(deployment):
        raise DeploymentFailed(
            f"推理服务未在 {timeout}s 内退出（pid {deployment.pid}），端口可能仍被占用"
        )

    deployment.status = DeploymentStatus.STOPPED
    deployment._process = None
    return deployment


def _has_exited(deployment: Deployment) -> bool:
    process = deployment._process
    if process is None:
        return True
    return process.poll() is not None


def _log_tail(log_path: str | None, lines: int = 20) -> str:
    if not log_path:
        return "(无日志路径)"
    path = Path(log_path)
    if not path.exists():
        return "(日志文件不存在)"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def curl_example(deployment: Deployment, sample: str = '{"account":"A12345"}') -> str:
    return (
        f"curl -X POST {deployment.predict_path} "
        f'-H "Content-Type: application/json" '
        f"-d '{sample}'"
    )
