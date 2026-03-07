"""End-to-end card identification pipeline."""

from __future__ import annotations

import time

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mtg_ocr.detection.card_detector import CardDetector, ScanMode
from mtg_ocr.encoder.base import VisualEncoder
from mtg_ocr.models.card import IdentificationResult
from mtg_ocr.search.similarity import EmbeddingIndex


class CardIdentificationPipeline:
    """End-to-end card identification pipeline.

    photo -> detect card -> crop/normalize -> encode -> search -> CardMatch results
    """

    def __init__(
        self,
        encoder: VisualEncoder,
        index: EmbeddingIndex,
        detector: CardDetector,
    ) -> None:
        self.encoder = encoder
        self.index = index
        self.detector = detector

    def identify(
        self, image: np.ndarray | Image.Image, top_k: int = 5
    ) -> IdentificationResult:
        """Identify a card from an image.

        1. Detect card region
        2. Crop and normalize to 224x224
        3. Encode with visual encoder
        4. Search embedding index
        5. Return top-K matches with confidence and latency
        """
        start = time.perf_counter()

        bgr = self._to_bgr(image)
        detection = self.detector.detect(bgr)

        if detection is None:
            elapsed = (time.perf_counter() - start) * 1000
            return IdentificationResult(
                matches=[],
                latency_ms=elapsed,
                scan_mode=self.detector.scan_mode.value,
            )

        card_rgb = cv2.cvtColor(detection.card_image, cv2.COLOR_BGR2RGB)
        pil_card = Image.fromarray(card_rgb).resize((224, 224), Image.LANCZOS)

        embedding = self.encoder.encode_image(pil_card)
        matches = self.index.search(embedding, top_k=top_k)

        elapsed = (time.perf_counter() - start) * 1000
        return IdentificationResult(
            matches=matches,
            latency_ms=elapsed,
            scan_mode=self.detector.scan_mode.value,
        )

    def identify_batch(
        self, images: list[np.ndarray], top_k: int = 5
    ) -> list[IdentificationResult]:
        """Batch identification for rig mode."""
        return [self.identify(img, top_k=top_k) for img in images]

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Path,
        scan_mode: ScanMode = ScanMode.HANDHELD,
        encoder: VisualEncoder | None = None,
    ) -> CardIdentificationPipeline:
        """Load pipeline from a directory containing model + embeddings.

        Args:
            model_dir: Directory containing embeddings.npz (and optional encoder checkpoint).
            scan_mode: Detection mode (handheld or rig).
            encoder: Custom encoder instance (e.g., fine-tuned). If None, loads
                     default MobileCLIPEncoder. Pass your fine-tuned encoder here
                     to ensure query embeddings match the embedding database.
        """
        model_dir = Path(model_dir)

        if encoder is None:
            from mtg_ocr.encoder.mobileclip import MobileCLIPEncoder

            encoder = MobileCLIPEncoder()

        index = EmbeddingIndex()
        index.load(model_dir / "embeddings.npz")
        detector = CardDetector(scan_mode=scan_mode)

        return cls(encoder=encoder, index=index, detector=detector)

    @staticmethod
    def _to_bgr(image: np.ndarray | Image.Image) -> np.ndarray:
        """Convert input to BGR numpy array for OpenCV.

        PIL Images are converted from RGB to BGR automatically.
        numpy arrays are assumed to be BGR (OpenCV convention).
        If passing np.array(pil_image), convert to BGR first or pass the PIL Image directly.
        """
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return image
