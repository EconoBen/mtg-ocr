"""Embedding dimension reduction via PCA or truncation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


VALID_METHODS = ("pca", "truncation")


@dataclass
class DimensionReductionReport:
    """Statistics from a dimension reduction operation."""

    original_dim: int
    target_dim: int
    method: str
    variance_retained: float
    file_size_reduction_pct: float


class DimensionReducer:
    """Reduce embedding dimensions via PCA or truncation."""

    def __init__(self, method: str = "pca", target_dim: int = 256) -> None:
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS}, got {method!r}"
            )
        self.method = method
        self.target_dim = target_dim
        self._fitted = False
        self._components: np.ndarray | None = None  # PCA: (target_dim, original_dim)
        self._mean: np.ndarray | None = None  # PCA: (original_dim,)
        self._original_dim: int | None = None
        self._variance_retained: float | None = None

    def fit(self, embeddings: np.ndarray) -> DimensionReducer:
        """Fit the reducer on training embeddings.

        Args:
            embeddings: (N, D) array of embeddings.

        Returns:
            self for chaining.
        """
        n_samples, original_dim = embeddings.shape
        if self.target_dim > original_dim:
            raise ValueError(
                f"target_dim ({self.target_dim}) must be <= input dim ({original_dim})"
            )
        self._original_dim = original_dim

        if self.method == "truncation":
            self._variance_retained = self.target_dim / original_dim
            self._fitted = True
            return self

        if self.target_dim > n_samples:
            raise ValueError(
                f"target_dim ({self.target_dim}) must be <= number of samples ({n_samples}) for PCA"
            )

        # PCA via numpy SVD (no sklearn dependency)
        self._mean = embeddings.mean(axis=0)
        centered = embeddings - self._mean
        # Economy SVD
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        self._components = vt[: self.target_dim]  # (target_dim, original_dim)

        # Variance retained
        total_var = np.sum(s**2)
        retained_var = np.sum(s[: self.target_dim] ** 2)
        self._variance_retained = float(retained_var / total_var) if total_var > 0 else 1.0

        self._fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to reduced dimensions.

        Args:
            embeddings: (N, D) array.

        Returns:
            (N, target_dim) reduced embeddings.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        if self.method == "truncation":
            return embeddings[:, : self.target_dim]

        # PCA projection
        centered = embeddings - self._mean
        return (centered @ self._components.T).astype(embeddings.dtype)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(embeddings)
        return self.transform(embeddings)

    def report(self) -> DimensionReductionReport:
        """Generate a report of the reduction statistics."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before report()")

        file_size_reduction = (1.0 - self.target_dim / self._original_dim) * 100.0

        return DimensionReductionReport(
            original_dim=self._original_dim,
            target_dim=self.target_dim,
            method=self.method,
            variance_retained=self._variance_retained,
            file_size_reduction_pct=file_size_reduction,
        )

    def save(self, path: Path) -> None:
        """Save the fitted reducer to disk."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before save()")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "method": np.array([self.method]),
            "target_dim": np.array([self.target_dim]),
            "original_dim": np.array([self._original_dim]),
            "variance_retained": np.array([self._variance_retained]),
        }

        if self.method == "pca":
            save_dict["components"] = self._components
            save_dict["mean"] = self._mean

        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: Path) -> DimensionReducer:
        """Load a fitted reducer from disk."""
        data = np.load(path, allow_pickle=False)

        method = str(data["method"][0])
        target_dim = int(data["target_dim"][0])

        reducer = cls(method=method, target_dim=target_dim)
        reducer._original_dim = int(data["original_dim"][0])
        reducer._variance_retained = float(data["variance_retained"][0])
        reducer._fitted = True

        if method == "pca":
            reducer._components = data["components"]
            reducer._mean = data["mean"]

        return reducer
