"""Quantizer orchestration tests.

Everything here is about ordering, defaults and registration. No file is ever
produced by a real llama.cpp in this suite, because that needs a GPU node.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from son_quantizer import (
    AUTO_ARTIFACT_LEVELS,
    Artifact,
    LlamaCppTools,
    QuantizeRequest,
    ToolchainError,
    artifacts_for_registration,
    build_gguf_filename,
    quantize_all,
)

from son_contracts import JEV_BASELINE_QUANT, ArtifactFormat, QuantLevel


class RecordingTools(LlamaCppTools):
    """Writes placeholder files and records the call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def convert_hf_to_gguf(self, source: str, destination: str) -> None:
        self.calls.append(("convert", source, destination))
        Path(destination).write_bytes(b"f16-placeholder")

    def quantize(self, source: str, destination: str, level: QuantLevel) -> None:
        self.calls.append(("quantize", source, destination, level.value))
        Path(destination).write_bytes(b"gguf-placeholder")

    def merge_lora(self, base: str, adapter: str, destination: str) -> str:
        self.calls.append(("merge", base, adapter, destination))
        return destination


class UnavailableTools(RecordingTools):
    def available(self) -> bool:
        return False


def _request(tmp_path: Path, **overrides: object) -> QuantizeRequest:
    merged = tmp_path / "merged.safetensors"
    merged.write_bytes(b"weights")
    base: dict[str, object] = {
        "merged_path": str(merged),
        "output_dir": str(tmp_path / "out"),
        "model_version": "mv_0017",
    }
    base.update(overrides)
    return QuantizeRequest(**base)  # type: ignore[arg-type]


class TestDefaults:
    def test_q4_k_m_is_the_first_artifact(self) -> None:
        assert AUTO_ARTIFACT_LEVELS[0] is QuantLevel.Q4_K_M

    def test_q4_k_m_matches_the_jev_baseline(self) -> None:
        assert JEV_BASELINE_QUANT == "Q4_K_M"

    def test_three_gguf_levels_plus_native(self, tmp_path: Path) -> None:
        """需求方案.txt 5.7.1 lists exactly these outputs."""
        assert len(AUTO_ARTIFACT_LEVELS) == 3
        result = quantize_all(_request(tmp_path), RecordingTools())
        assert len(artifacts_for_registration(result)) == 4

    def test_empty_level_list_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            quantize_all(_request(tmp_path, levels=()), RecordingTools())


class TestOrdering:
    def test_conversion_happens_once(self, tmp_path: Path) -> None:
        """Converting per level would redo the expensive part three times."""
        tools = RecordingTools()
        quantize_all(_request(tmp_path), tools)
        conversions = [c for c in tools.calls if c[0] == "convert"]
        assert len(conversions) == 1

    def test_conversion_precedes_quantization(self, tmp_path: Path) -> None:
        tools = RecordingTools()
        quantize_all(_request(tmp_path), tools)
        assert tools.calls[0][0] == "convert"
        assert all(c[0] == "quantize" for c in tools.calls[1:])

    def test_each_level_quantizes_from_the_converted_file(self, tmp_path: Path) -> None:
        tools = RecordingTools()
        quantize_all(_request(tmp_path), tools)
        converted = tools.calls[0][2]
        assert all(call[1] == converted for call in tools.calls[1:])


class TestArtifacts:
    def test_filenames_carry_version_and_level(self) -> None:
        assert build_gguf_filename("mv_0017", QuantLevel.Q8_0) == "mv_0017.Q8_0.gguf"

    def test_falls_back_when_no_version(self) -> None:
        assert build_gguf_filename(None, QuantLevel.Q4_K_M) == "model.Q4_K_M.gguf"

    def test_baseline_level_is_flagged_recommended(self, tmp_path: Path) -> None:
        result = quantize_all(_request(tmp_path), RecordingTools())
        assert result.recommended is not None
        assert result.recommended.quant is QuantLevel.Q4_K_M
        assert result.recommended.is_recommended is True

    def test_only_one_artifact_is_recommended(self, tmp_path: Path) -> None:
        result = quantize_all(_request(tmp_path), RecordingTools())
        assert sum(a.is_recommended for a in result.artifacts) == 1

    def test_recommendation_is_explained_in_the_label(self, tmp_path: Path) -> None:
        result = quantize_all(_request(tmp_path), RecordingTools())
        assert result.recommended is not None
        assert "推荐部署格式" in result.recommended.describe()

    def test_native_weights_have_no_quant_level(self, tmp_path: Path) -> None:
        result = quantize_all(_request(tmp_path), RecordingTools())
        assert result.native_weights is not None
        assert result.native_weights.format is ArtifactFormat.NATIVE
        assert result.native_weights.quant is None

    def test_sizes_are_recorded(self, tmp_path: Path) -> None:
        result = quantize_all(_request(tmp_path), RecordingTools())
        assert all(a.size_bytes is not None and a.size_bytes > 0 for a in result.artifacts)

    def test_describe_lists_every_output(self, tmp_path: Path) -> None:
        result = quantize_all(_request(tmp_path), RecordingTools())
        lines = result.describe()
        assert len(lines) == 4
        assert any("原生权重" in line for line in lines)

    def test_hash_is_optional(self, tmp_path: Path) -> None:
        result = quantize_all(_request(tmp_path), RecordingTools(), hash_outputs=True)
        assert all(a.sha256 for a in result.artifacts)


class TestMissingBaseline:
    def test_absence_is_disclosed(self, tmp_path: Path) -> None:
        """Evaluating a model without Q4_K_M makes the numbers incomparable."""
        result = quantize_all(_request(tmp_path, levels=(QuantLevel.Q8_0,)), RecordingTools())
        assert result.recommended is None
        assert any("JEV 基线" in note for note in result.notes)


class TestToolchainAvailability:
    def test_missing_toolchain_is_an_explicit_error(self, tmp_path: Path) -> None:
        """Never fake a quantized file to keep a pipeline green."""
        with pytest.raises(ToolchainError, match=r"llama\.cpp"):
            quantize_all(_request(tmp_path), UnavailableTools())

    def test_error_names_the_gpu_requirement(self, tmp_path: Path) -> None:
        with pytest.raises(ToolchainError, match="GPU"):
            quantize_all(_request(tmp_path), UnavailableTools())

    def test_no_files_are_left_behind_on_failure(self, tmp_path: Path) -> None:
        with pytest.raises(ToolchainError):
            quantize_all(_request(tmp_path), UnavailableTools())
        assert not list((tmp_path / "out").glob("*.gguf"))

    def test_merge_refuses_rather_than_faking_weights(self) -> None:
        from son_quantizer import SubprocessLlamaCppTools

        with pytest.raises(ToolchainError, match="GPU"):
            SubprocessLlamaCppTools().merge_lora("base", "adapter", "out")


class TestRegistration:
    def test_registration_includes_native_last(self, tmp_path: Path) -> None:
        result = quantize_all(_request(tmp_path), RecordingTools())
        registered = artifacts_for_registration(result)
        assert registered[-1].format is ArtifactFormat.NATIVE

    def test_artifact_is_immutable_configuration(self) -> None:
        artifact = Artifact(path="x.gguf", format=ArtifactFormat.GGUF, quant=QuantLevel.Q4_K_M)
        with pytest.raises(AttributeError):
            artifact.path = "y.gguf"  # type: ignore[misc]
