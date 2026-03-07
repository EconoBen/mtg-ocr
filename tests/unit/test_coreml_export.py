"""Tests for CoreML export functionality.

coremltools may not be installed in all environments.
Tests skip gracefully when it is not available.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch

ct = pytest.importorskip("coremltools", reason="coremltools not installed")

_SKIP_COREML_EXPORT = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="coremltools BlobWriter is incompatible with Python 3.14+",
)

from mtg_ocr.export.coreml_export import CoreMLExporter, ExportResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTorchModel(torch.nn.Module):
    """Minimal torch model that mimics encode_image using convolutions."""

    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, embedding_dim, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.pool(out)
        return out.flatten(1)


def _make_mock_encoder(embedding_dim: int = 16) -> MagicMock:
    """Create a mock encoder with a real torch model for CoreML export."""
    encoder = MagicMock()
    model = _FakeTorchModel(embedding_dim=embedding_dim)
    model.eval()
    encoder.model = model
    encoder.embedding_dim = embedding_dim
    return encoder


# ---------------------------------------------------------------------------
# Tests: ExportResult
# ---------------------------------------------------------------------------


class TestExportResult:
    def test_creation(self, tmp_path):
        result = ExportResult(
            output_path=tmp_path / "model.mlpackage",
            model_size_mb=12.5,
            compute_units="ALL",
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 16),
        )
        assert result.model_size_mb == 12.5
        assert result.compute_units == "ALL"
        assert result.input_shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Tests: CoreMLExporter
# ---------------------------------------------------------------------------


class TestCoreMLExporter:
    @_SKIP_COREML_EXPORT
    def test_export_produces_mlpackage(self, tmp_path):
        encoder = _make_mock_encoder(embedding_dim=16)
        exporter = CoreMLExporter(encoder)
        output_path = tmp_path / "model.mlpackage"

        result = exporter.export(output_path)

        assert isinstance(result, ExportResult)
        assert output_path.exists()
        assert result.model_size_mb > 0
        assert result.compute_units == "ALL"
        assert result.input_shape == (1, 3, 224, 224)
        assert result.output_shape[1] == 16

    @_SKIP_COREML_EXPORT
    def test_export_creates_parent_directories(self, tmp_path):
        encoder = _make_mock_encoder()
        exporter = CoreMLExporter(encoder)
        output_path = tmp_path / "sub" / "dir" / "model.mlpackage"

        result = exporter.export(output_path)

        assert output_path.exists()
        assert result.output_path == output_path

    @_SKIP_COREML_EXPORT
    def test_export_cpu_only(self, tmp_path):
        encoder = _make_mock_encoder()
        exporter = CoreMLExporter(encoder)
        output_path = tmp_path / "model_cpu.mlpackage"

        result = exporter.export(output_path, compute_units="CPU_ONLY")

        assert result.compute_units == "CPU_ONLY"
        assert output_path.exists()

    @_SKIP_COREML_EXPORT
    def test_export_cpu_and_gpu(self, tmp_path):
        encoder = _make_mock_encoder()
        exporter = CoreMLExporter(encoder)
        output_path = tmp_path / "model_gpu.mlpackage"

        result = exporter.export(output_path, compute_units="CPU_AND_GPU")

        assert result.compute_units == "CPU_AND_GPU"
        assert output_path.exists()

    @_SKIP_COREML_EXPORT
    def test_export_from_onnx(self, tmp_path):
        from mtg_ocr.export.onnx_export import ONNXExporter

        encoder = _make_mock_encoder(embedding_dim=16)

        # First export to ONNX
        onnx_exporter = ONNXExporter(encoder)
        onnx_path = tmp_path / "model.onnx"
        onnx_exporter.export(onnx_path)

        # Then convert ONNX to CoreML
        coreml_exporter = CoreMLExporter(encoder)
        coreml_path = tmp_path / "model_from_onnx.mlpackage"
        result = coreml_exporter.export_from_onnx(onnx_path, coreml_path)

        assert isinstance(result, ExportResult)
        assert coreml_path.exists()
        assert result.model_size_mb > 0
        assert result.compute_units == "ALL"

    def test_invalid_compute_units_raises(self):
        encoder = _make_mock_encoder()
        exporter = CoreMLExporter(encoder)

        with pytest.raises(ValueError, match="Invalid compute_units"):
            exporter._resolve_compute_units(ct, "INVALID")

    def test_encoder_without_model_raises(self):
        encoder = MagicMock(spec=[])
        exporter = CoreMLExporter(encoder)

        with pytest.raises(AttributeError, match="does not expose a 'model' attribute"):
            exporter._get_torch_model()
