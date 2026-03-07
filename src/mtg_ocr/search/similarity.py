"""Exact nearest-neighbor search via dot product."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mtg_ocr.data.models import CardInfo
from mtg_ocr.models.card import CardMatch


def _ensure_npz_suffix(path: Path) -> Path:
    """Ensure path ends with .npz (np.savez auto-appends it otherwise)."""
    if not str(path).endswith(".npz"):
        return Path(str(path) + ".npz")
    return path


def _meta_path(npz_path: Path) -> Path:
    """Derive the metadata sidecar path from an .npz path."""
    return Path(str(npz_path).removesuffix(".npz") + ".meta.json")


class EmbeddingIndex:
    """Exact nearest-neighbor search via dot product.

    Stores pre-computed card embeddings as FP16 numpy array.
    For 30K cards x 512 dims = ~30MB (FP16).
    Search is simple matrix multiplication -- fast on CPU.
    """

    def __init__(self) -> None:
        self.embeddings: np.ndarray | None = None  # (N, D) FP16
        self._embeddings_f32: np.ndarray | None = None  # cached FP32 for search
        self.card_ids: list[str] = []
        self.metadata: dict[str, CardInfo] = {}

    def load(self, path: Path) -> None:
        """Load pre-computed embeddings from .npz file and optional metadata.

        Loads embeddings and card_ids from the .npz file. Also attempts to load
        metadata from a corresponding .meta.json sidecar file. If the sidecar
        is missing, self.metadata remains empty.
        """
        import json

        path = _ensure_npz_suffix(Path(path))
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        data = np.load(path, allow_pickle=False)
        self.embeddings = data["embeddings"].astype(np.float16)
        self._embeddings_f32 = None  # invalidate cache
        self.card_ids = [s for s in data["card_ids"]]
        self.metadata = {}  # clear stale metadata before loading

        # Attempt to load metadata sidecar if it exists
        meta_path = _meta_path(path)
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            for cid, info_dict in raw.items():
                self.metadata[cid] = CardInfo(**info_dict)

    def save(self, path: Path) -> None:
        """Save embeddings to .npz file with FP16 quantization and metadata sidecar."""
        import json

        if self.embeddings is None:
            raise ValueError("No embeddings to save")
        path = _ensure_npz_suffix(Path(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self.embeddings.astype(np.float16),
            card_ids=np.array(self.card_ids),
        )

        # Save metadata sidecar so load() roundtrip is lossless
        if self.metadata:
            meta = {}
            for cid, info in self.metadata.items():
                meta[cid] = {
                    "scryfall_id": info.scryfall_id,
                    "name": info.name,
                    "set_code": info.set_code,
                    "set_name": info.set_name,
                    "collector_number": info.collector_number,
                    "image_uris": info.image_uris,
                }
            _meta_path(path).write_text(json.dumps(meta))

    def add(self, scryfall_id: str, embedding: np.ndarray, card_info: CardInfo) -> None:
        """Add a single card embedding."""
        embedding = embedding.astype(np.float32).reshape(1, -1)
        if self.embeddings is None:
            self.embeddings = embedding.astype(np.float16)
        else:
            self.embeddings = np.concatenate(
                [self.embeddings.astype(np.float32), embedding], axis=0
            ).astype(np.float16)
        self.card_ids.append(scryfall_id)
        self.metadata[scryfall_id] = card_info

    def search(self, query: np.ndarray, top_k: int = 5) -> list[CardMatch]:
        """Find top-K most similar cards by cosine similarity (dot product on normalized vectors)."""
        if self.embeddings is None or len(self.card_ids) == 0:
            return []
        if top_k <= 0:
            return []

        query = query.astype(np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        # Cache FP32 conversion to avoid O(N*D) allocation per query
        if self._embeddings_f32 is None or self._embeddings_f32.shape != self.embeddings.shape:
            self._embeddings_f32 = self.embeddings.astype(np.float32)

        scores = self._embeddings_f32 @ query  # (N,)

        k = min(top_k, len(self.card_ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            card_id = self.card_ids[idx]
            info = self.metadata.get(card_id)
            results.append(
                CardMatch(
                    scryfall_id=card_id,
                    card_name=info.name if info else card_id,
                    set_code=info.set_code if info else "",
                    set_name=info.set_name if info else "",
                    confidence=float(scores[idx]),
                    image_uri=info.image_uris.get("normal") if info else None,
                )
            )
        return results

    def build_from_arrays(
        self,
        embeddings: np.ndarray,
        card_ids: list[str],
        metadata: dict[str, CardInfo],
    ) -> None:
        """Build index from pre-computed arrays."""
        self.embeddings = embeddings.astype(np.float16)
        self._embeddings_f32 = None  # invalidate cache
        self.card_ids = list(card_ids)
        self.metadata = dict(metadata)
