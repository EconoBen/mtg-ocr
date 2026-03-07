"""Compute card embeddings on RunPod GPU.

Usage:
    python scripts/compute_embeddings.py --output data/embeddings.npz --batch-size 128
    python scripts/compute_embeddings.py --update data/embeddings.npz --output data/embeddings_v2.npz

Requires GPU for efficient batch processing of ~30K images.
Estimated time: ~15-30 min on A40, cost ~$1-2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MTG card embeddings from Scryfall images."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for embeddings .npz file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding (default: 128)",
    )
    parser.add_argument(
        "--update",
        type=Path,
        default=None,
        help="Path to existing embeddings for incremental update",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/scryfall"),
        help="Directory for cached Scryfall data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from mtg_ocr.data.scryfall import ScryfallClient
    from mtg_ocr.embeddings.builder import EmbeddingBuilder
    from mtg_ocr.encoder.mobileclip import MobileCLIPEncoder

    print("Initializing encoder...")
    encoder = MobileCLIPEncoder()
    print(f"Encoder ready: dim={encoder.embedding_dim}")

    client = ScryfallClient(cache_dir=args.cache_dir)
    builder = EmbeddingBuilder(encoder=encoder, scryfall_client=client)

    if args.update:
        print(f"Incremental update from {args.update}")
        stats = builder.update(
            existing_path=args.update,
            output_path=args.output,
        )
    else:
        print("Building full embedding database...")
        stats = builder.build(
            output_path=args.output,
            batch_size=args.batch_size,
        )

    print(f"Done! Stats:")
    print(f"  Total cards: {stats.total_cards}")
    print(f"  New cards:   {stats.new_cards}")
    print(f"  Skipped:     {stats.skipped_cards}")
    print(f"  Dimensions:  {stats.embedding_dim}")
    print(f"  File size:   {stats.file_size_mb:.1f} MB")
    print(f"  Output:      {args.output}")


if __name__ == "__main__":
    main()
