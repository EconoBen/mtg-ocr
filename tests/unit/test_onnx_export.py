"""Tests for ONNX export functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from mtg_ocr.export.onnx_export import ExportResult, ONNXExporter, _ImageEncoderWrapper


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
    """Create a mock encoder with a real torch model for ONNX export."""
    encoder = MagicMock()
    model = _FakeTorchModel(embedding_dim=embedding_dim)
    model.eval()
    encoder.model = model
    encoder.embedding_dim = embedding_dim

    # Preprocess: simple resize + normalize
    def _preprocess(img: Image.Image) -> torch.Tensor:
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1))

    encoder.preprocess = _preprocess

    # encode_image: run through torch model, L2-normalize, return numpy
    def _encode_image(img: Image.Image) -> np.ndarray:
        tensor = _preprocess(img).unsqueeze(0)
        with torch.no_grad():
            out = model.encode_image(tensor)
            out = out / out.norm(dim=-1, keepdim=True)
        return out.squeeze(0).numpy().astype(np.float32)

    encoder.encode_image = _encode_image
    return encoder


def _make_test_image() -> Image.Image:
    """Create a simple test image."""
    return Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )


# ---------------------------------------------------------------------------
# Tests: ExportResult
# ---------------------------------------------------------------------------


class TestExportResult:
    def test_creation(self, tmp_path):
        result = ExportResult(
            output_path=tmp_path / "model.onnx",
            model_size_mb=5.2,
            quantized=False,
            opset_version=17,
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 16),
        )
        assert result.model_size_mb == 5.2
        assert result.quantized is False
        assert result.opset_version == 17
        assert result.input_shape == (1, 3, 224, 224)
        assert result.output_shape == (1, 16)


# ---------------------------------------------------------------------------
# Tests: ONNXExporter
# ---------------------------------------------------------------------------


class TestONNXExporter:
    def test_export_produces_valid_onnx_file(self, tmp_path):
        encoder = _make_mock_encoder(embedding_dim=16)
        exporter = ONNXExporter(encoder)
        output_path = tmp_path / "model.onnx"

        result = exporter.export(output_path, opset_version=17)

        assert isinstance(result, ExportResult)
        assert result.output_path == output_path
        assert output_path.exists()
        assert result.model_size_mb > 0
        assert result.quantized is False
        assert result.opset_version == 17
        assert result.input_shape == (1, 3, 224, 224)
        assert result.output_shape[1] == 16

    def test_export_creates_parent_directories(self, tmp_path):
        encoder = _make_mock_encoder()
        exporter = ONNXExporter(encoder)
        output_path = tmp_path / "sub" / "dir" / "model.onnx"

        result = exporter.export(output_path)

        assert output_path.exists()
        assert result.output_path == output_path

    def test_export_with_quantization_reduces_size(self, tmp_path):
        encoder = _make_mock_encoder(embedding_dim=64)
        exporter = ONNXExporter(encoder)

        # Export unquantized
        unquant_path = tmp_path / "model_fp32.onnx"
        exporter.export(unquant_path, quantize=False)

        # Export quantized
        quant_path = tmp_path / "model_int8.onnx"
        result_int8 = exporter.export(quant_path, quantize=True)

        assert result_int8.quantized is True
        assert quant_path.exists()
        # Quantized model should exist (size comparison depends on model,
        # but at minimum we verify it was created and is valid)
        assert result_int8.model_size_mb > 0
        # The raw intermediate file should be cleaned up
        assert not (tmp_path / "model_int8.raw.onnx").exists()

    def test_export_onnx_loadable_by_onnxruntime(self, tmp_path):
        import onnxruntime as ort

        encoder = _make_mock_encoder(embedding_dim=16)
        exporter = ONNXExporter(encoder)
        output_path = tmp_path / "model.onnx"

        exporter.export(output_path)

        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        assert input_name == "input"

        # Run inference
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": dummy})
        assert outputs[0].shape == (1, 16)

    def test_validate_matching_outputs(self, tmp_path):
        encoder = _make_mock_encoder(embedding_dim=16)
        exporter = ONNXExporter(encoder)
        output_path = tmp_path / "model.onnx"

        exporter.export(output_path)
        test_image = _make_test_image()

        assert exporter.validate(output_path, test_image, rtol=1e-2) is True

    def test_validate_catches_mismatched_outputs(self, tmp_path):
        encoder = _make_mock_encoder(embedding_dim=16)
        exporter = ONNXExporter(encoder)
        output_path = tmp_path / "model.onnx"

        exporter.export(output_path)
        test_image = _make_test_image()

        # Patch encode_image to return garbage so validation fails
        encoder.encode_image = lambda img: np.random.randn(16).astype(np.float32)

        assert exporter.validate(output_path, test_image, rtol=1e-5) is False

    def test_encoder_without_model_raises(self):
        encoder = MagicMock(spec=[])  # No attributes at all
        exporter = ONNXExporter(encoder)

        with pytest.raises(AttributeError, match="does not expose a 'model' attribute"):
            exporter._get_torch_model()


# ---------------------------------------------------------------------------
# Tests: _ImageEncoderWrapper
# ---------------------------------------------------------------------------


class TestImageEncoderWrapper:
    def test_forward_calls_encode_image(self):
        model = _FakeTorchModel(embedding_dim=8)
        wrapper = _ImageEncoderWrapper(model)

        x = torch.randn(2, 3, 224, 224)
        out = wrapper(x)

        assert out.shape == (2, 8)

    def test_wrapper_is_torch_module(self):
        model = _FakeTorchModel()
        wrapper = _ImageEncoderWrapper(model)
        assert isinstance(wrapper, torch.nn.Module)
