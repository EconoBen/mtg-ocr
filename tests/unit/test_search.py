"""Tests for embedding similarity search."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from mtg_ocr.data.models import CardInfo
from mtg_ocr.models.card import CardMatch
from mtg_ocr.search.similarity import EmbeddingIndex


def _make_card_info(scryfall_id: str, name: str = "Test Card") -> CardInfo:
    return CardInfo(
        scryfall_id=scryfall_id,
        name=name,
        set_code="TST",
        set_name="Test Set",
        collector_number="1",
        image_uris={"normal": f"https://example.com/{scryfall_id}.jpg"},
    )


def _random_unit_vector(dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestEmbeddingIndexLoad:
    def test_load_from_npz(self, tmp_path: Path) -> None:
        dim = 512
        n = 10
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((n, dim)).astype(np.float16)
        card_ids = [f"id-{i}" for i in range(n)]
        np.savez(tmp_path / "index.npz", embeddings=embeddings, card_ids=card_ids)

        index = EmbeddingIndex()
        index.load(tmp_path / "index.npz")

        assert index.embeddings is not None
        assert index.embeddings.shape == (n, dim)
        assert len(index.card_ids) == n

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        index = EmbeddingIndex()
        with pytest.raises(FileNotFoundError):
            index.load(tmp_path / "missing.npz")


class TestEmbeddingIndexSearch:
    def test_dot_product_returns_top_k(self) -> None:
        dim = 512
        rng = np.random.default_rng(42)
        n = 100

        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = EmbeddingIndex()
        card_ids = [f"id-{i}" for i in range(n)]
        metadata = {cid: _make_card_info(cid, f"Card {i}") for i, cid in enumerate(card_ids)}
        index.build_from_arrays(embeddings, card_ids, metadata)

        query = embeddings[7]
        results = index.search(query, top_k=5)

        assert len(results) == 5
        assert all(isinstance(r, CardMatch) for r in results)
        assert results[0].scryfall_id == "id-7"
        assert results[0].confidence == pytest.approx(1.0, abs=1e-3)

    def test_search_results_sorted_by_confidence(self) -> None:
        dim = 512
        rng = np.random.default_rng(99)
        n = 50

        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = EmbeddingIndex()
        card_ids = [f"id-{i}" for i in range(n)]
        metadata = {cid: _make_card_info(cid) for cid in card_ids}
        index.build_from_arrays(embeddings, card_ids, metadata)

        query = _random_unit_vector(dim, rng)
        results = index.search(query, top_k=10)

        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_cosine_similarity_with_normalized_vectors(self) -> None:
        embeddings = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0, 0]],
            dtype=np.float32,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = EmbeddingIndex()
        card_ids = ["a", "b", "c"]
        metadata = {cid: _make_card_info(cid) for cid in card_ids}
        index.build_from_arrays(embeddings, card_ids, metadata)

        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = index.search(query, top_k=3)

        assert results[0].scryfall_id == "a"
        assert results[0].confidence == pytest.approx(1.0, abs=1e-5)
        assert results[1].scryfall_id == "c"


class TestFP16Quantization:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        dim = 512
        n = 20
        rng = np.random.default_rng(7)

        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = EmbeddingIndex()
        card_ids = [f"id-{i}" for i in range(n)]
        metadata = {cid: _make_card_info(cid) for cid in card_ids}
        index.build_from_arrays(embeddings, card_ids, metadata)

        save_path = tmp_path / "test.npz"
        index.save(save_path)

        loaded = EmbeddingIndex()
        loaded.load(save_path)

        assert loaded.embeddings is not None
        assert loaded.embeddings.dtype == np.float16
        assert loaded.embeddings.shape == (n, dim)
        assert loaded.card_ids == card_ids
        np.testing.assert_allclose(
            loaded.embeddings.astype(np.float32),
            embeddings,
            atol=1e-2,
        )


class TestEmptyDatabase:
    def test_search_empty_index_returns_empty(self) -> None:
        index = EmbeddingIndex()
        query = _random_unit_vector(512)
        results = index.search(query, top_k=5)
        assert results == []

    def test_add_single_embedding(self) -> None:
        index = EmbeddingIndex()
        embedding = _random_unit_vector(512)
        info = _make_card_info("card-1", "Lightning Bolt")
        index.add("card-1", embedding, info)

        results = index.search(embedding, top_k=1)
        assert len(results) == 1
        assert results[0].scryfall_id == "card-1"
        assert results[0].card_name == "Lightning Bolt"


class TestSearchLatency:
    def test_search_30k_under_10ms(self) -> None:
        dim = 512
        n = 30_000
        rng = np.random.default_rng(123)

        embeddings = rng.standard_normal((n, dim)).astype(np.float16)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / norms).astype(np.float16)

        index = EmbeddingIndex()
        index.embeddings = embeddings
        index.card_ids = [f"id-{i}" for i in range(n)]
        index.metadata = {
            cid: _make_card_info(cid) for cid in index.card_ids
        }

        query = _random_unit_vector(dim, rng)

        # Warmup
        index.search(query, top_k=5)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            index.search(query, top_k=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        mean_ms = sum(times) / len(times)
        assert mean_ms < 10.0, f"Search took {mean_ms:.1f}ms, expected < 10ms"
