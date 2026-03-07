"""Build card embedding database from Scryfall images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mtg_ocr.data.models import CardInfo
from mtg_ocr.encoder.base import VisualEncoder


@dataclass
class EmbeddingStats:
    """Statistics from an embedding build or update run."""

    total_cards: int
    new_cards: int
    skipped_cards: int
    embedding_dim: int
    file_size_mb: float


class EmbeddingBuilder:
    """Build card embedding database from Scryfall images.

    Downloads card images, runs them through the encoder, and saves
    the embedding database as a quantized numpy file.
    """

    def __init__(self, encoder: VisualEncoder, scryfall_client) -> None:
        self.encoder = encoder
        self.scryfall_client = scryfall_client

    def encode_batch(
        self, images: list[Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        """Encode a batch of images into normalized embeddings.

        Returns (N, D) array of unit-length embeddings, or (0, D) if empty.
        """
        if not images:
            return np.empty((0, self.encoder.embedding_dim), dtype=np.float32)

        embeddings = self.encoder.encode_images(images, batch_size=batch_size)

        # Ensure normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        card_ids: list[str],
        metadata: dict[str, CardInfo],
        output_path: Path,
    ) -> None:
        """Save embeddings to .npz file with FP16 quantization and metadata JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_path,
            embeddings=embeddings.astype(np.float16),
            card_ids=np.array(card_ids),
        )

        # Save metadata as JSON sidecar
        meta_path = output_path.with_suffix(".meta.json")
        meta_dict = {}
        for cid, info in metadata.items():
            meta_dict[cid] = {
                "scryfall_id": info.scryfall_id,
                "name": info.name,
                "set_code": info.set_code,
                "set_name": info.set_name,
                "collector_number": info.collector_number,
                "image_uris": info.image_uris,
            }
        meta_path.write_text(json.dumps(meta_dict))

    def load_embeddings(
        self, path: Path
    ) -> tuple[np.ndarray, list[str], dict[str, CardInfo]]:
        """Load embeddings from .npz file and metadata from JSON sidecar."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        embeddings = data["embeddings"]
        card_ids = [s for s in data["card_ids"]]

        meta_path = path.with_suffix("").with_suffix(".meta.json")
        metadata: dict[str, CardInfo] = {}
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            for cid, info_dict in raw.items():
                metadata[cid] = CardInfo(**info_dict)

        return embeddings, card_ids, metadata

    def merge_embeddings(
        self,
        existing_path: Path,
        new_embeddings: np.ndarray,
        new_card_ids: list[str],
        new_metadata: dict[str, CardInfo],
        output_path: Path,
    ) -> EmbeddingStats:
        """Merge new embeddings into an existing database, skipping duplicates."""
        old_emb, old_ids, old_meta = self.load_embeddings(existing_path)
        existing_set = set(old_ids)

        # Filter out duplicates
        keep_indices = [
            i for i, cid in enumerate(new_card_ids) if cid not in existing_set
        ]
        skipped = len(new_card_ids) - len(keep_indices)

        if keep_indices:
            filtered_emb = new_embeddings[keep_indices]
            filtered_ids = [new_card_ids[i] for i in keep_indices]
            filtered_meta = {cid: new_metadata[cid] for cid in filtered_ids}

            merged_emb = np.concatenate(
                [old_emb.astype(np.float32), filtered_emb.astype(np.float32)], axis=0
            )
            merged_ids = old_ids + filtered_ids
            merged_meta = {**old_meta, **filtered_meta}
        else:
            merged_emb = old_emb.astype(np.float32)
            merged_ids = old_ids
            merged_meta = old_meta

        self.save_embeddings(
            embeddings=merged_emb,
            card_ids=merged_ids,
            metadata=merged_meta,
            output_path=output_path,
        )

        output_path = Path(output_path)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        return EmbeddingStats(
            total_cards=len(merged_ids),
            new_cards=len(keep_indices),
            skipped_cards=skipped,
            embedding_dim=merged_emb.shape[1],
            file_size_mb=round(file_size_mb, 2),
        )

    async def build(
        self, output_path: Path, batch_size: int = 64
    ) -> EmbeddingStats:
        """Build complete embedding database.

        1. Get all card image URIs from Scryfall
        2. Download images (with rate limiting)
        3. Encode in batches
        4. Save as FP16 .npz
        """
        bulk_path = self.scryfall_client.download_bulk_data()
        cards = self.scryfall_client.build_card_dictionary(bulk_path)
        image_uris = self.scryfall_client.get_image_uris(cards)

        all_embeddings = []
        all_ids = []

        for i in range(0, len(image_uris), batch_size):
            batch = image_uris[i : i + batch_size]
            images = []
            batch_ids = []
            for scryfall_id, uri in batch:
                try:
                    import httpx
                    import time

                    time.sleep(0.075)
                    resp = httpx.get(uri, timeout=30)
                    resp.raise_for_status()
                    from io import BytesIO

                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    images.append(img)
                    batch_ids.append(scryfall_id)
                except Exception:
                    continue

            if images:
                emb = self.encode_batch(images, batch_size=batch_size)
                all_embeddings.append(emb)
                all_ids.extend(batch_ids)

        if all_embeddings:
            embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            embeddings = np.empty(
                (0, self.encoder.embedding_dim), dtype=np.float32
            )

        self.save_embeddings(
            embeddings=embeddings,
            card_ids=all_ids,
            metadata={cid: cards[cid] for cid in all_ids if cid in cards},
            output_path=output_path,
        )

        output_path = Path(output_path)
        file_size_mb = (
            output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
        )

        return EmbeddingStats(
            total_cards=len(all_ids),
            new_cards=len(all_ids),
            skipped_cards=len(image_uris) - len(all_ids),
            embedding_dim=self.encoder.embedding_dim,
            file_size_mb=round(file_size_mb, 2),
        )

    async def update(
        self, existing_path: Path, output_path: Path
    ) -> EmbeddingStats:
        """Incrementally update embeddings with new cards only."""
        _, existing_ids, _ = self.load_embeddings(existing_path)
        existing_set = set(existing_ids)

        bulk_path = self.scryfall_client.download_bulk_data()
        cards = self.scryfall_client.build_card_dictionary(bulk_path)
        image_uris = self.scryfall_client.get_image_uris(cards)

        # Filter to only new cards
        new_uris = [
            (sid, uri) for sid, uri in image_uris if sid not in existing_set
        ]

        new_embeddings_list = []
        new_ids = []

        for scryfall_id, uri in new_uris:
            try:
                import httpx
                import time

                time.sleep(0.075)
                resp = httpx.get(uri, timeout=30)
                resp.raise_for_status()
                from io import BytesIO

                img = Image.open(BytesIO(resp.content)).convert("RGB")
                emb = self.encode_batch([img])
                new_embeddings_list.append(emb)
                new_ids.append(scryfall_id)
            except Exception:
                continue

        if new_embeddings_list:
            new_emb = np.concatenate(new_embeddings_list, axis=0)
            new_meta = {cid: cards[cid] for cid in new_ids if cid in cards}
            return self.merge_embeddings(
                existing_path=existing_path,
                new_embeddings=new_emb,
                new_card_ids=new_ids,
                new_metadata=new_meta,
                output_path=output_path,
            )

        # No new cards — just copy existing
        import shutil

        shutil.copy2(existing_path, output_path)
        meta_src = existing_path.with_suffix("").with_suffix(".meta.json")
        if meta_src.exists():
            shutil.copy2(meta_src, output_path.with_suffix("").with_suffix(".meta.json"))

        return EmbeddingStats(
            total_cards=len(existing_ids),
            new_cards=0,
            skipped_cards=0,
            embedding_dim=self.encoder.embedding_dim,
            file_size_mb=round(
                existing_path.stat().st_size / (1024 * 1024), 2
            ),
        )
