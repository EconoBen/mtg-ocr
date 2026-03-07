"""Export visual encoder to ONNX format with optional INT8 quantization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mtg_ocr.encoder.base import VisualEncoder


@dataclass
class ExportResult:
    """Result of an ONNX export operation."""

    output_path: Path
    model_size_mb: float
    quantized: bool
    opset_version: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


class ONNXExporter:
    """Export visual encoder to ONNX format."""

    def __init__(self, encoder: VisualEncoder) -> None:
        self.encoder = encoder

    def _get_torch_model(self) -> torch.nn.Module:
        """Extract the underlying torch model from the encoder."""
        if hasattr(self.encoder, "model"):
            return self.encoder.model
        raise AttributeError(
            "Encoder does not expose a 'model' attribute. "
            "Cannot extract torch.nn.Module for ONNX export."
        )

    def _get_preprocess(self):
        """Extract the preprocessing transform from the encoder."""
        if hasattr(self.encoder, "preprocess"):
            return self.encoder.preprocess
        return None

    def export(
        self,
        output_path: Path,
        opset_version: int = 17,
        quantize: bool = False,
    ) -> ExportResult:
        """Export the encoder's image model to ONNX format.

        Args:
            output_path: Where to save the .onnx file.
            opset_version: ONNX opset version (default 17).
            quantize: If True, apply INT8 dynamic quantization after export.

        Returns:
            ExportResult with export metadata.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._get_torch_model()
        model.eval()

        input_shape = (1, 3, 224, 224)
        # Ensure dummy_input is created on the same device as the model to avoid
        # device mismatch errors when the model has been moved to CUDA/MPS.
        model_device = torch.device("cpu")
        first_param = next(model.parameters(), None)
        if first_param is not None:
            model_device = first_param.device
        else:
            first_buffer = next(model.buffers(), None)
            if first_buffer is not None:
                model_device = first_buffer.device
        dummy_input = torch.randn(*input_shape, device=model_device)

        # Export to ONNX — use the image encoder specifically
        if hasattr(model, "encode_image"):
            # Wrap encode_image as a forward-only module for tracing
            wrapper = _ImageEncoderWrapper(model)
        else:
            wrapper = model

        # Determine the raw export path (before potential quantization)
        if quantize:
            raw_path = output_path.with_suffix(".raw.onnx")
        else:
            raw_path = output_path

        wrapper.eval()

        export_kwargs: dict = {
            "input_names": ["input"],
            "output_names": ["embedding"],
        }
        # Dynamic axes cause shape inference issues during quantization,
        # so only enable them for non-quantized exports.
        if not quantize:
            export_kwargs["dynamic_axes"] = {
                "input": {0: "batch_size"},
                "embedding": {0: "batch_size"},
            }

        torch.onnx.export(
            wrapper,
            dummy_input,
            str(raw_path),
            opset_version=opset_version,
            **export_kwargs,
        )

        if quantize:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                str(raw_path),
                str(output_path),
                weight_type=QuantType.QInt8,
            )
            raw_path.unlink()

        # Get output shape by running the model
        with torch.no_grad():
            out = wrapper(dummy_input)
        output_shape = tuple(out.shape)

        model_size_mb = output_path.stat().st_size / (1024 * 1024)

        return ExportResult(
            output_path=output_path,
            model_size_mb=model_size_mb,
            quantized=quantize,
            opset_version=opset_version,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def validate(
        self,
        onnx_path: Path,
        test_image: Image.Image,
        rtol: float = 1e-3,
    ) -> bool:
        """Validate ONNX model output matches PyTorch output.

        Args:
            onnx_path: Path to the exported .onnx file.
            test_image: A PIL image to test with.
            rtol: Relative tolerance for comparison.

        Returns:
            True if outputs match within tolerance.
        """
        import onnxruntime as ort

        # Get PyTorch output
        pytorch_output = self.encoder.encode_image(test_image)

        # Get ONNX output
        preprocess = self._get_preprocess()
        if preprocess is None:
            raise AttributeError(
                "Encoder does not expose a 'preprocess' attribute. "
                "Cannot validate without matching preprocessing."
            )
        input_tensor = (
            preprocess(test_image).unsqueeze(0).detach().cpu().numpy()
        )

        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {"input": input_tensor})[0]

        # Normalize ONNX output the same way the encoder does (L2)
        onnx_norm = onnx_output / np.linalg.norm(onnx_output, axis=-1, keepdims=True)
        onnx_embedding = onnx_norm.squeeze(0)

        return bool(np.allclose(pytorch_output, onnx_embedding, rtol=rtol, atol=1e-5))


class _ImageEncoderWrapper(torch.nn.Module):
    """Wraps a model's encode_image method as a standard forward pass."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x)
