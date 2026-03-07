"""Export visual encoder to CoreML format for iOS deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from mtg_ocr.encoder.base import VisualEncoder
from mtg_ocr.export.onnx_export import _ImageEncoderWrapper


@dataclass
class ExportResult:
    """Result of a CoreML export operation."""

    output_path: Path
    model_size_mb: float
    compute_units: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


class CoreMLExporter:
    """Export visual encoder to CoreML format for iOS deployment."""

    def __init__(self, encoder: VisualEncoder) -> None:
        self.encoder = encoder

    def _get_torch_model(self) -> torch.nn.Module:
        """Extract the underlying torch model from the encoder."""
        if hasattr(self.encoder, "model"):
            return self.encoder.model
        raise AttributeError(
            "Encoder does not expose a 'model' attribute. "
            "Cannot extract torch.nn.Module for CoreML export."
        )

    def export(
        self,
        output_path: Path,
        compute_units: str = "ALL",
    ) -> ExportResult:
        """Export directly from PyTorch to CoreML.

        Args:
            output_path: Where to save the .mlpackage directory.
            compute_units: Target compute units — "ALL", "CPU_AND_GPU", or "CPU_ONLY".

        Returns:
            ExportResult with export metadata.
        """
        import coremltools as ct

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._get_torch_model()
        model.eval()

        if hasattr(model, "encode_image"):
            wrapper = _ImageEncoderWrapper(model)
        else:
            wrapper = model
        wrapper.eval()

        input_shape = (1, 3, 224, 224)
        dummy_input = torch.randn(*input_shape)

        traced = torch.jit.trace(wrapper, dummy_input)

        ct_compute_units = self._resolve_compute_units(ct, compute_units)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=input_shape)],
            compute_units=ct_compute_units,
        )

        mlmodel.save(str(output_path))

        # Get output shape
        with torch.no_grad():
            out = wrapper(dummy_input)
        output_shape = tuple(out.shape)

        model_size_mb = self._get_size_mb(output_path)

        return ExportResult(
            output_path=output_path,
            model_size_mb=model_size_mb,
            compute_units=compute_units,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def export_from_onnx(
        self,
        onnx_path: Path,
        output_path: Path,
        compute_units: str = "ALL",
    ) -> ExportResult:
        """Convert an existing ONNX model to CoreML.

        Args:
            onnx_path: Path to the source .onnx file.
            output_path: Where to save the .mlpackage directory.
            compute_units: Target compute units — "ALL", "CPU_AND_GPU", or "CPU_ONLY".

        Returns:
            ExportResult with export metadata.
        """
        import coremltools as ct

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ct_compute_units = self._resolve_compute_units(ct, compute_units)

        mlmodel = ct.converters.onnx.convert(
            model=str(onnx_path),
            compute_units=ct_compute_units,
        )

        mlmodel.save(str(output_path))

        # Infer shapes from the ONNX model
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        input_shape = tuple(input_info.shape)
        output_shape = tuple(output_info.shape)

        model_size_mb = self._get_size_mb(output_path)

        return ExportResult(
            output_path=output_path,
            model_size_mb=model_size_mb,
            compute_units=compute_units,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    @staticmethod
    def _resolve_compute_units(ct, compute_units: str):
        """Map string compute_units to coremltools enum."""
        mapping = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }
        if compute_units not in mapping:
            raise ValueError(
                f"Invalid compute_units '{compute_units}'. "
                f"Must be one of: {', '.join(mapping)}"
            )
        return mapping[compute_units]

    @staticmethod
    def _get_size_mb(path: Path) -> float:
        """Get total size in MB, handling both files and directories (.mlpackage)."""
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        # For directories (.mlpackage), sum all file sizes
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
