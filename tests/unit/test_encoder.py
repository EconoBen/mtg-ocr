"""Tests for the visual encoder abstraction and MobileCLIP wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mtg_ocr.encoder.base import VisualEncoder
from mtg_ocr.encoder.mobileclip import MobileCLIPEncoder


class TestVisualEncoderProtocol:
    """Test the VisualEncoder protocol interface."""

    def test_protocol_has_embedding_dim(self):
        """VisualEncoder protocol requires embedding_dim property."""
        assert hasattr(VisualEncoder, "embedding_dim")

    def test_protocol_has_encode_image(self):
        """VisualEncoder protocol requires encode_image method."""
        assert hasattr(VisualEncoder, "encode_image")

    def test_protocol_has_encode_images(self):
        """VisualEncoder protocol requires encode_images method."""
        assert hasattr(VisualEncoder, "encode_images")

    def test_concrete_class_satisfies_protocol(self):
        """MobileCLIPEncoder should satisfy the VisualEncoder protocol."""
        # Check structural subtyping - MobileCLIPEncoder has all required methods/properties
        assert callable(getattr(MobileCLIPEncoder, "encode_image", None))
        assert callable(getattr(MobileCLIPEncoder, "encode_images", None))
        assert isinstance(
            getattr(MobileCLIPEncoder, "embedding_dim", None), property
        )


def _make_test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a random test image."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestMobileCLIPEncoder:
    """Test MobileCLIP encoder functionality."""

    @pytest.fixture(scope="class")
    def encoder(self) -> MobileCLIPEncoder:
        """Create encoder without pretrained weights for testing."""
        return MobileCLIPEncoder(pretrained=None)

    def test_loads_model(self, encoder: MobileCLIPEncoder):
        """Encoder should load the model architecture."""
        assert encoder.model is not None
        assert encoder.preprocess is not None

    def test_embedding_dim_positive(self, encoder: MobileCLIPEncoder):
        """Embedding dimension should be a positive integer."""
        assert encoder.embedding_dim > 0

    def test_encode_image_returns_correct_shape(self, encoder: MobileCLIPEncoder):
        """encode_image should return a 1D array of shape (embedding_dim,)."""
        image = _make_test_image()
        embedding = encoder.encode_image(image)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (encoder.embedding_dim,)

    def test_encode_image_returns_float32(self, encoder: MobileCLIPEncoder):
        """Embeddings should be float32 numpy arrays."""
        image = _make_test_image()
        embedding = encoder.encode_image(image)
        assert embedding.dtype == np.float32

    def test_encode_images_batch_processing(self, encoder: MobileCLIPEncoder):
        """encode_images should process a batch and return (N, embedding_dim)."""
        images = [_make_test_image(width=200 + i, height=200 + i) for i in range(3)]
        embeddings = encoder.encode_images(images, batch_size=2)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, encoder.embedding_dim)

    def test_embedding_normalization(self, encoder: MobileCLIPEncoder):
        """Embeddings should be L2-normalized (unit vectors)."""
        image = _make_test_image()
        embedding = encoder.encode_image(image)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5, f"Embedding norm {norm} is not 1.0"

    def test_batch_embeddings_normalized(self, encoder: MobileCLIPEncoder):
        """All embeddings in a batch should be L2-normalized."""
        images = [_make_test_image(width=200 + i, height=200 + i) for i in range(3)]
        embeddings = encoder.encode_images(images, batch_size=2)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic_output(self, encoder: MobileCLIPEncoder):
        """Same image should produce the same embedding."""
        image = _make_test_image()
        emb1 = encoder.encode_image(image)
        emb2 = encoder.encode_image(image)
        np.testing.assert_allclose(emb1, emb2, atol=1e-6)
