"""GPU-side module tests.

What can be verified without a GPU: that the modules import without torch
installed, that they refuse loudly rather than degrading, that prompt rendering
falls back correctly, and that the CLI surface exists.

What cannot: anything that runs torch. Those paths are exercised on the GPU node
via `python -m son_cli doctor` and the runbook.
"""

from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from son_trainer.backends import StageRequest, TrainSample
from son_trainer.hyperparams import Hyperparams
from son_trainer.torch_backend import (
    TorchRunConfig,
    TorchTrainingBackend,
    TrainingUnavailable,
    assert_training_available,
    render_chat,
    render_chat_prompt,
)

from son_contracts import Decision, TrainingMethod


class TestModuleImportsWithoutTorch:
    """Importing a GPU module must never require a GPU.

    The control plane runs on machines with no CUDA, and it imports these
    modules for the protocol definitions.
    """

    @pytest.mark.parametrize(
        "module",
        [
            "son_trainer.torch_backend",
            "son_quantizer.peft_merge",
            "son_inference.llamacpp_client",
        ],
    )
    def test_imports(self, module: str) -> None:
        assert importlib.import_module(module) is not None


class TestUnavailableIsLoud:
    def test_assert_training_available_raises_without_torch_or_cuda(self) -> None:
        """A silent pass here would let a CPU-only box claim a training run."""
        with pytest.raises(TrainingUnavailable):
            assert_training_available()

    def test_message_names_the_missing_dependency_or_the_gpu(self) -> None:
        try:
            assert_training_available()
        except TrainingUnavailable as exc:
            assert ("torch" in str(exc)) or ("CUDA" in str(exc))
        else:  # pragma: no cover - would mean a GPU appeared mid-test
            pytest.fail("expected TrainingUnavailable on a GPU-less machine")

    def test_backend_refuses_rather_than_stubbing(self) -> None:
        backend = TorchTrainingBackend(
            TorchRunConfig(base_model="qwen2.5-3b-instruct", output_dir="/tmp/x")
        )
        request = StageRequest(
            stage=TrainingMethod.SFT,
            samples=(TrainSample(prompt="p", response="r"),),
            base_model="qwen2.5-3b-instruct",
            hyperparams=CAST_HYPERPARAMS,
        )
        with pytest.raises(TrainingUnavailable):
            backend.run_stage(request)


class TestRunConfig:
    def test_is_frozen(self) -> None:
        config = TorchRunConfig(base_model="m", output_dir="/tmp")
        with pytest.raises(FrozenInstanceError):
            config.base_model = "other"  # type: ignore[misc]

    def test_defaults_match_the_mvp_envelope(self) -> None:
        """LoRA rank 16 / alpha 32 is the conventional pair; seq length must
        hold a JEV prompt plus a 200-character reason."""
        config = TorchRunConfig(base_model="m", output_dir="/tmp")
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.max_seq_length >= 1024

    def test_qlora_is_opt_in(self) -> None:
        assert TorchRunConfig(base_model="m", output_dir="/tmp").quantized is False


class TestPromptRendering:
    """Rendering must use the model's own chat template.

    Training on a hand-rolled prompt format produces a model that scores well on
    the training template and badly on the JEV template used at inference.
    """

    class _Tokenizer:
        def __init__(self, template: str | None) -> None:
            self.chat_template = template
            self.calls: list[object] = []

        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            self.calls.append(kwargs)
            return "<tpl>" + str(messages) + "</tpl>"

    def test_uses_chat_template_when_available(self) -> None:
        tokenizer = self._Tokenizer("template")
        out = render_chat(tokenizer, "判定这条样本", '{"decision":"black"}')
        assert out.startswith("<tpl>")
        assert "add_generation_prompt" not in dict(tokenizer.calls[0] or {})  # type: ignore[call-overload]

    def test_prompt_render_adds_generation_marker(self) -> None:
        tokenizer = self._Tokenizer("template")
        render_chat_prompt(tokenizer, "hello")
        assert dict(tokenizer.calls[0] or {}) == {  # type: ignore[call-overload]
            "tokenize": False,
            "add_generation_prompt": True,
        }

    def test_falls_back_when_no_template(self) -> None:
        tokenizer = self._Tokenizer(None)
        out = render_chat(tokenizer, "hello", "world")
        assert out == "hello\nworld"

    def test_fallback_prompt_ends_with_newline(self) -> None:
        tokenizer = self._Tokenizer(None)
        assert render_chat_prompt(tokenizer, "hello") == "hello\n"


class TestQuantizerMergeContract:
    def test_merge_lora_declares_a_string_return(self) -> None:
        """The ABC says the merge returns a path; a None-typed override silently
        breaks the conversion step that consumes it."""
        from son_quantizer import LlamaCppTools

        hints = LlamaCppTools.merge_lora.__annotations__
        assert hints["return"] == "str"

    def test_gpu_merger_reports_missing_dependencies_first(self) -> None:
        """Dependency check precedes the path check.

        Without torch the merge is impossible regardless of whether the paths
        exist, and naming the missing dependency is the more useful error. The
        path guards are covered by the GPU node run instead, where the
        dependencies do exist.
        """
        from son_quantizer.peft_merge import LlamaCppConverter
        from son_quantizer.pipeline import ToolchainError

        tools = LlamaCppConverter()
        with pytest.raises(ToolchainError) as exc:
            tools.merge_lora("/nonexistent/base", "/nonexistent/adapter", "/tmp/out")
        message = str(exc.value)
        assert "torch" in message
        assert "deploy/gpu" in message

    def test_require_names_what_is_missing(self) -> None:
        from son_quantizer.peft_merge import LlamaCppConverter
        from son_quantizer.pipeline import ToolchainError

        tools = LlamaCppConverter(
            python_bin="python",
            convert_script="/nonexistent/convert_hf_to_gguf.py",
            quantize_bin="definitely-not-a-real-binary",
        )
        assert tools.available() is False
        with pytest.raises(ToolchainError) as exc:
            tools.require()
        assert "convert_hf_to_gguf.py" in str(exc.value)
        assert "definitely-not-a-real-binary" in str(exc.value)


class TestInferenceClientContract:
    def test_engine_is_ready_is_false_when_no_server(self) -> None:
        """A client that assumed readiness would time out instead of reporting."""
        from son_inference.llamacpp_client import LlamaCppEngine

        engine = LlamaCppEngine(base_url="http://127.0.0.1:59999", timeout_s=1.0)
        assert engine.is_ready() is False

    def test_gpu_report_handles_missing_torch(self) -> None:
        """The deployment report must degrade, not raise."""
        from son_inference.llamacpp_client import gpu_report

        report = gpu_report()
        assert "cuda_available" in report
        assert isinstance(report["cuda_available"], bool)

    def test_server_process_refuses_without_binary(self) -> None:
        from son_inference.llamacpp_client import (
            LlamaCppEngine,  # noqa: F401
            LlamaCppUnavailable,
            LlamaServerProcess,
        )
        from son_inference.server import ServerConfig

        if shutil_which("llama-server") is not None:  # pragma: no cover
            pytest.skip("llama-server installed; refusal path not exercised")
        process = LlamaServerProcess(ServerConfig(model_path="/tmp/x.gguf"))
        with pytest.raises(LlamaCppUnavailable, match="llama-server"):
            process.start()

    def test_base_url_is_derived_from_port(self) -> None:
        from son_inference.llamacpp_client import LlamaServerProcess
        from son_inference.server import ServerConfig

        process = LlamaServerProcess(ServerConfig(model_path="m.gguf", port=9090))
        assert process.base_url() == "http://127.0.0.1:9090"

    def test_stop_is_safe_when_nothing_started(self) -> None:
        from son_inference.llamacpp_client import LlamaServerProcess
        from son_inference.server import ServerConfig

        LlamaServerProcess(ServerConfig(model_path="m.gguf")).stop()


class TestCliSurface:
    """The CLI is what gets run on the GPU node, so its surface is a contract."""

    def test_subcommands_exist(self) -> None:
        from son_cli import build_parser

        parser = build_parser()
        actions = [
            action for action in parser._actions if hasattr(action, "choices") and action.choices
        ]
        assert actions, "no subcommand group found"
        assert set(actions[0].choices or ()) == {"doctor", "train", "quantize", "infer", "serve"}

    def test_train_accepts_the_documented_flags(self) -> None:
        from son_cli import build_parser

        args = build_parser().parse_args(
            [
                "train",
                "--dataset",
                "seed.csv",
                "--model",
                "qwen2.5-3b-instruct",
                "--qlora",
                "--dpo",
            ]
        )
        assert args.qlora is True
        assert args.dpo is True
        assert args.dataset == "seed.csv"

    def test_render_target_produces_the_jev_shape(self) -> None:
        from son_cli import render_target

        out = render_target({"label": "黑"}, {"黑": Decision.BLACK})
        assert '"decision": "black"' in out
        assert '"reason"' in out

    def test_render_target_refuses_an_unmapped_label(self) -> None:
        """Training on "{}" would produce a model that fails schema validation."""
        from son_cli import render_target

        with pytest.raises(ValueError, match="不在映射表"):
            render_target({"label": "unknown"}, {"黑": Decision.BLACK})


def shutil_which(binary: str) -> str | None:
    import shutil

    return shutil.which(binary)


#: The stage only reads the keys it needs; a partial mapping is realistic and
#: keeps the test focused on the refusal path rather than on hyperparameter setup.
CAST_HYPERPARAMS = cast(Hyperparams, {"max_steps": 1})
