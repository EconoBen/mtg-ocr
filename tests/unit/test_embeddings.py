"""Tests for embedding database builder."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from mtg_ocr.data.models import CardInfo
from mtg_ocr.embeddings.builder import EmbeddingBuilder, EmbeddingStats


def _make_card_info(scryfall_id: str, name: str = "Test Card") -> CardInfo:
    return CardInfo(
        scryfall_id=scryfall_id,
        name=name,
        set_code="tst",
        set_name="Test Set",
        collector_number="1",
        image_uris={"normal": f"https://example.com/{scryfall_id}.jpg"},
    )


class FakeEncoder:
    """Fake encoder that returns deterministic embeddings."""

    def __init__(self, dim: int = 512):
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_image(self, image):
        rng = np.random.RandomState(42)
        vec = rng.randn(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def encode_images(self, images, batch_size: int = 32):
        results = []
        for img in images:
            results.append(self.encode_image(img))
        return np.stack(results)


class TestEmbeddingBuilderBatchProcessing:
    """Test EmbeddingBuilder processes a batch of images."""

    def test_process_batch_returns_embeddings(self, tmp_path):
        encoder = FakeEncoder(dim=64)
        client = MagicMock()
        builder = EmbeddingBuilder(encoder=encoder, scryfall_client=client)

        # Create fake PIL images
        from PIL import Image

        images = [Image.new("RGB", (224, 224), color=(i * 30, 0, 0)) for i in range(5)]

        embeddings = builder.encode_batch(images, batch_size=2)

        assert embeddings.shape == (5, 64)

    def test_process_batch_empty_list(self, tmp_path):
        encoder = FakeEncoder(dim=64)
        client = MagicMock()
        builder = EmbeddingBuilder(encoder=encoder, scryfall_client=client)

        embeddings = builder.encode_batch([], batch_size=2)

        assert embeddings.shape == (0, 64)


class TestEmbeddingNormalization:
    """Test embeddings are normalized (unit length)."""

    def test_embeddings_are_unit_vectors(self):
        encoder = FakeEncoder(dim=64)
        client = MagicMock()
        builder = EmbeddingBuilder(encoder=encoder, scryfall_client=client)

        from PIL import Image

        images = [Image.new("RGB", (224, 224)) for _ in range(3)]
        embeddings = builder.encode_batch(images)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestSaveLoadRoundtrip:
    """Test save/load roundtrip with FP16 quantization."""

    def test_save_and_load_preserves_data(self, tmp_path):
        encoder = FakeEncoder(dim=64)
        client = MagicMock()
        builder = EmbeddingBuilder(encoder=encoder, scryfall_client=client)

        # Build some test data
        card_infos = {
            f"id-{i}": _make_card_info(f"id-{i}", f"Card {i}") for i in range(5)
        }
        rng = np.random.RandomState(123)
        embeddings = rng.randn(5, 64).astype(np.float32)
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        card_ids = list(card_infos.keys())

        output_path = tmp_path / "test_embeddings.npz"
        builder.save_embeddings(
            embeddings=embeddings,
            card_ids=card_ids,
            metadata=card_infos,
            output_path=output_path,
        )

        assert output_path.exists()

        loaded_embeddings, loaded_card_ids, loaded_metadata = builder.load_embeddings(
            output_path
        )

        assert loaded_embeddings.shape == (5, 64)
        assert loaded_card_ids == card_ids
        assert set(loaded_metadata.keys()) == set(card_infos.keys())
        # FP16 quantization means we lose some precision
        np.testing.assert_allclose(
            loaded_embeddings.astype(np.float32),
            embeddings,
            atol=1e-2,
        )

    def test_save_creates_parent_directories(self, tmp_path):
        encoder = FakeEncoder(dim=64)
        client = MagicMock()
        builder = EmbeddingBuilder(encoder=encoder, scryfall_client=client)

        output_path = tmp_path / "nested" / "dir" / "embeddings.npz"
        embeddings = np.random.randn(2, 64).astype(np.float32)
        card_ids = ["a", "b"]
        metadata = {k: _make_card_info(k) for k in card_ids}

        builder.save_embeddings(
            embeddings=embeddings,
            card_ids=card_ids,
            metadata=metadata,
            output_path=output_path,
        )

        assert output_path.exists()


class TestIncrementalUpdate:
    """Test incremental update (add new cards to existing database)."""

    def test_update_adds_new_cards(self, tmp_path):
        encoder = FakeEncoder(dim=64)
        client = MagicMock()
        builder = EmbeddingBuilder(encoder=encoder, scryfall_client=client)

        # Create initial embeddings
        rng = np.random.RandomState(42)
        initial_embeddings = rng.randn(3, 64).astype(np.float32)
        initial_embeddings = initial_embeddings / np.linalg.norm(
            initial_embeddings, axis=1, keepdims=True
        )
        initial_ids = ["card-0", "card-1", "card-2"]
        initial_meta = {k: _make_card_info(k) for k in initial_ids}

        initial_path = tmp_path / "initial.npz"
        builder.save_embeddings(
            embeddings=initial_embeddings,
            card_ids=initial_ids,
            metadata=initial_meta,
            output_path=initial_path,
        )

        # Add new cards
        new_embeddings = rng.randn(2, 64).astype(np.float32)
        new_embeddings = new_embeddings / np.linalg.norm(
            new_embeddings, axis=1, keepdims=True
        )
        new_ids = ["card-3", "card-4"]
        new_meta = {k: _make_card_info(k) for k in new_ids}

        output_path = tmp_path / "updated.npz"
        builder.merge_embeddings(
            existing_path=initial_path,
            new_embeddings=new_embeddings,
            new_card_ids=new_ids,
            new_metadata=new_meta,
            output_path=output_path,
        )

        loaded_emb, loaded_ids, loaded_meta = builder.load_embeddings(output_path)

        assert loaded_emb.shape == (5, 64)
        assert len(loaded_ids) == 5
        assert "card-0" in loaded_ids
        assert "card-4" in loaded_ids
        assert len(loaded_meta) == 5

    def test_update_skips_existing_cards(self, tmp_path):
        encoder = FakeEncoder(dim=64)
        client = MagicMock()
        builder = EmbeddingBuilder(encoder=encoder, scryfall_client=client)

        rng = np.random.RandomState(42)
        initial_embeddings = rng.randn(3, 64).astype(np.float32)
        initial_ids = ["card-0", "card-1", "card-2"]
        initial_meta = {k: _make_card_info(k) for k in initial_ids}

        initial_path = tmp_path / "initial.npz"
        builder.save_embeddings(
            embeddings=initial_embeddings,
            card_ids=initial_ids,
            metadata=initial_meta,
            output_path=initial_path,
        )

        # Try to add with one duplicate
        new_embeddings = rng.randn(2, 64).astype(np.float32)
        new_ids = ["card-2", "card-3"]  # card-2 is a duplicate
        new_meta = {k: _make_card_info(k) for k in new_ids}

        output_path = tmp_path / "updated.npz"
        builder.merge_embeddings(
            existing_path=initial_path,
            new_embeddings=new_embeddings,
            new_card_ids=new_ids,
            new_metadata=new_meta,
            output_path=output_path,
        )

        loaded_emb, loaded_ids, _ = builder.load_embeddings(output_path)

        # Should have 4, not 5 (card-2 was deduplicated)
        assert loaded_emb.shape[0] == 4
        assert len(loaded_ids) == 4


class TestEmbeddingStats:
    """Test EmbeddingStats dataclass."""

    def test_stats_fields(self):
        stats = EmbeddingStats(
            total_cards=100,
            new_cards=50,
            skipped_cards=50,
            embedding_dim=512,
            file_size_mb=15.5,
        )
        assert stats.total_cards == 100
        assert stats.new_cards == 50
        assert stats.skipped_cards == 50
        assert stats.embedding_dim == 512
        assert stats.file_size_mb == 15.5
