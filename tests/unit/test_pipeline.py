"""Tests for the end-to-end card identification pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from mtg_ocr.data.models import CardInfo
from mtg_ocr.detection.card_detector import CardDetector, ScanMode
from mtg_ocr.models.card import IdentificationResult
from mtg_ocr.search.similarity import EmbeddingIndex


# ---------------------------------------------------------------------------
# Fixtures: mock encoder and helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 512


class MockEncoder:
    """A mock visual encoder that returns deterministic embeddings."""

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def encode_image(self, image: Image.Image) -> np.ndarray:
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        return np.stack([self.encode_image(img) for img in images])


def _build_test_index(n_cards: int = 10) -> EmbeddingIndex:
    """Build a small test index with random embeddings."""
    index = EmbeddingIndex()
    rng = np.random.default_rng(0)
    for i in range(n_cards):
        sid = f"card-{i:04d}"
        vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        info = CardInfo(
            scryfall_id=sid,
            name=f"Test Card {i}",
            set_code="TST",
            set_name="Test Set",
            collector_number=str(i),
            image_uris={"normal": f"https://example.com/{sid}.jpg"},
        )
        index.add(sid, vec, info)
    return index


def _make_card_image() -> np.ndarray:
    """Create a synthetic BGR image with a white card rectangle on dark background."""
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    # White rectangle roughly matching MTG card aspect ratio (63:88)
    x1, y1 = 80, 50
    x2, y2 = 320, 385  # ~240x335 ≈ 0.716 ratio
    img[y1:y2, x1:x2] = 255
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineInitialization:
    def test_pipeline_init_with_all_components(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        encoder = MockEncoder()
        index = _build_test_index()
        detector = CardDetector(scan_mode=ScanMode.HANDHELD)

        pipeline = CardIdentificationPipeline(
            encoder=encoder,
            index=index,
            detector=detector,
        )
        assert pipeline.encoder is encoder
        assert pipeline.index is index
        assert pipeline.detector is detector

    def test_pipeline_stores_scan_mode_from_detector(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        pipeline = CardIdentificationPipeline(
            encoder=MockEncoder(),
            index=_build_test_index(),
            detector=CardDetector(scan_mode=ScanMode.RIG),
        )
        assert pipeline.detector.scan_mode == ScanMode.RIG


class TestPipelineIdentify:
    def test_identify_returns_identification_result(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        pipeline = CardIdentificationPipeline(
            encoder=MockEncoder(),
            index=_build_test_index(),
            detector=CardDetector(),
        )
        image = _make_card_image()
        result = pipeline.identify(image, top_k=5)

        assert isinstance(result, IdentificationResult)
        assert result.scan_mode in ("handheld", "rig")
        assert result.latency_ms >= 0

    def test_identify_returns_matches_sorted_by_confidence(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        pipeline = CardIdentificationPipeline(
            encoder=MockEncoder(),
            index=_build_test_index(),
            detector=CardDetector(),
        )
        image = _make_card_image()
        result = pipeline.identify(image, top_k=5)

        assert len(result.matches) <= 5
        confidences = [m.confidence for m in result.matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_identify_accepts_pil_image(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        pipeline = CardIdentificationPipeline(
            encoder=MockEncoder(),
            index=_build_test_index(),
            detector=CardDetector(),
        )
        pil_image = Image.fromarray(_make_card_image()[:, :, ::-1])  # BGR->RGB for PIL
        result = pipeline.identify(pil_image, top_k=3)

        assert isinstance(result, IdentificationResult)
        assert len(result.matches) <= 3

    def test_identify_measures_latency(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        pipeline = CardIdentificationPipeline(
            encoder=MockEncoder(),
            index=_build_test_index(),
            detector=CardDetector(),
        )
        image = _make_card_image()
        result = pipeline.identify(image, top_k=3)

        assert result.latency_ms > 0


class TestPipelineNoCardDetected:
    def test_identify_no_card_returns_empty_matches(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        pipeline = CardIdentificationPipeline(
            encoder=MockEncoder(),
            index=_build_test_index(),
            detector=CardDetector(),
        )
        # Uniform gray image -- no card to detect
        no_card = np.full((400, 400, 3), 128, dtype=np.uint8)
        result = pipeline.identify(no_card, top_k=5)

        assert isinstance(result, IdentificationResult)
        assert len(result.matches) == 0
        assert result.latency_ms >= 0


class TestPipelineBatch:
    def test_identify_batch_returns_list(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        pipeline = CardIdentificationPipeline(
            encoder=MockEncoder(),
            index=_build_test_index(),
            detector=CardDetector(scan_mode=ScanMode.RIG),
        )
        images = [_make_card_image() for _ in range(3)]
        results = pipeline.identify_batch(images, top_k=3)

        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, IdentificationResult)

    def test_identify_batch_handles_mixed_detection(self):
        from mtg_ocr.pipeline import CardIdentificationPipeline

        pipeline = CardIdentificationPipeline(
            encoder=MockEncoder(),
            index=_build_test_index(),
            detector=CardDetector(scan_mode=ScanMode.RIG),
        )
        good_image = _make_card_image()
        bad_image = np.full((400, 400, 3), 128, dtype=np.uint8)
        results = pipeline.identify_batch([good_image, bad_image], top_k=3)

        assert len(results) == 2
        # First should have matches, second should be empty
        assert len(results[0].matches) > 0
        assert len(results[1].matches) == 0
