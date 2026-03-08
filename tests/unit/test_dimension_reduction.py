"""Tests for embedding dimension reduction."""

from __future__ import annotations

import numpy as np
import pytest

from mtg_ocr.embeddings.quantize import DimensionReducer, DimensionReductionReport


class TestTruncation:
    """Tests for truncation-based dimension reduction."""

    def test_truncation_preserves_first_n_dimensions(self):
        """Truncation should keep exactly the first N dimensions."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 512)).astype(np.float32)

        reducer = DimensionReducer(method="truncation", target_dim=256)
        reducer.fit(embeddings)
        reduced = reducer.transform(embeddings)

        np.testing.assert_array_equal(reduced, embeddings[:, :256])

    def test_truncation_output_shape(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 512)).astype(np.float32)

        reducer = DimensionReducer(method="truncation", target_dim=128)
        reduced = reducer.fit_transform(embeddings)

        assert reduced.shape == (50, 128)

    def test_truncation_fit_transform_equals_fit_then_transform(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 512)).astype(np.float32)

        reducer = DimensionReducer(method="truncation", target_dim=256)
        result_combined = reducer.fit_transform(embeddings)

        reducer2 = DimensionReducer(method="truncation", target_dim=256)
        reducer2.fit(embeddings)
        result_separate = reducer2.transform(embeddings)

        np.testing.assert_array_equal(result_combined, result_separate)


class TestPCA:
    """Tests for PCA-based dimension reduction."""

    def test_pca_output_shape(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((300, 512)).astype(np.float32)

        reducer = DimensionReducer(method="pca", target_dim=256)
        reduced = reducer.fit_transform(embeddings)

        assert reduced.shape == (300, 256)

    def test_pca_components_are_orthogonal(self):
        """PCA components should be orthogonal to each other."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((200, 64)).astype(np.float32)

        reducer = DimensionReducer(method="pca", target_dim=32)
        reducer.fit(embeddings)

        # Components should be orthogonal: V^T @ V ≈ I
        components = reducer._components  # (target_dim, original_dim)
        gram = components @ components.T
        np.testing.assert_allclose(gram, np.eye(32), atol=1e-5)

    def test_pca_preserves_similarity_ordering(self):
        """Top-K nearest neighbors should be preserved after PCA reduction."""
        rng = np.random.default_rng(42)
        # Create embeddings with clear cluster structure
        cluster1 = rng.standard_normal((50, 128)).astype(np.float32) + 5.0
        cluster2 = rng.standard_normal((50, 128)).astype(np.float32) - 5.0
        embeddings = np.concatenate([cluster1, cluster2], axis=0)

        # Compute similarities in original space
        query = embeddings[0:1]
        orig_sims = (query @ embeddings.T).flatten()
        orig_top5 = np.argsort(-orig_sims)[:5]

        # Reduce dimensions
        reducer = DimensionReducer(method="pca", target_dim=64)
        reduced = reducer.fit_transform(embeddings)
        reduced_query = reduced[0:1]
        reduced_sims = (reduced_query @ reduced.T).flatten()
        reduced_top5 = np.argsort(-reduced_sims)[:5]

        # Top-5 should overlap significantly (at least 4 of 5)
        overlap = len(set(orig_top5) & set(reduced_top5))
        assert overlap >= 4, f"Only {overlap}/5 top matches preserved"

    def test_pca_transform_without_fit_raises(self):
        reducer = DimensionReducer(method="pca", target_dim=256)
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((10, 512)).astype(np.float32)

        with pytest.raises(RuntimeError, match="fit"):
            reducer.transform(embeddings)

    def test_pca_fit_transform_equals_fit_then_transform(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((300, 512)).astype(np.float32)

        reducer = DimensionReducer(method="pca", target_dim=256)
        result_combined = reducer.fit_transform(embeddings)

        reducer2 = DimensionReducer(method="pca", target_dim=256)
        reducer2.fit(embeddings)
        result_separate = reducer2.transform(embeddings)

        np.testing.assert_allclose(result_combined, result_separate, atol=1e-5)


class TestSaveLoad:
    """Tests for save/load roundtrip."""

    def test_truncation_save_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 512)).astype(np.float32)

        reducer = DimensionReducer(method="truncation", target_dim=256)
        reducer.fit(embeddings)
        original_result = reducer.transform(embeddings)

        save_path = tmp_path / "reducer.npz"
        reducer.save(save_path)

        loaded = DimensionReducer.load(save_path)
        loaded_result = loaded.transform(embeddings)

        np.testing.assert_array_equal(original_result, loaded_result)

    def test_pca_save_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((300, 512)).astype(np.float32)

        reducer = DimensionReducer(method="pca", target_dim=256)
        reducer.fit(embeddings)
        original_result = reducer.transform(embeddings)

        save_path = tmp_path / "reducer.npz"
        reducer.save(save_path)

        loaded = DimensionReducer.load(save_path)
        loaded_result = loaded.transform(embeddings)

        np.testing.assert_allclose(original_result, loaded_result, atol=1e-5)

    def test_save_without_fit_raises(self, tmp_path):
        reducer = DimensionReducer(method="pca", target_dim=256)
        with pytest.raises(RuntimeError, match="fit"):
            reducer.save(tmp_path / "reducer.npz")


class TestDimensionReductionReport:
    """Tests for the reduction report dataclass."""

    def test_report_fields(self):
        report = DimensionReductionReport(
            original_dim=512,
            target_dim=256,
            method="pca",
            variance_retained=0.95,
            file_size_reduction_pct=50.0,
        )
        assert report.original_dim == 512
        assert report.target_dim == 256
        assert report.method == "pca"
        assert report.variance_retained == 0.95
        assert report.file_size_reduction_pct == 50.0

    def test_report_from_reducer(self):
        """DimensionReducer.report() should compute correct stats."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((300, 512)).astype(np.float32)

        reducer = DimensionReducer(method="pca", target_dim=256)
        reducer.fit(embeddings)
        report = reducer.report()

        assert report.original_dim == 512
        assert report.target_dim == 256
        assert report.method == "pca"
        assert 0.0 < report.variance_retained <= 1.0
        assert 0.0 < report.file_size_reduction_pct < 100.0


class TestInvalidInputs:
    """Edge cases and invalid inputs."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            DimensionReducer(method="invalid", target_dim=256)

    def test_target_dim_larger_than_input_raises(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((10, 64)).astype(np.float32)

        reducer = DimensionReducer(method="pca", target_dim=128)
        with pytest.raises(ValueError, match="target_dim"):
            reducer.fit(embeddings)
