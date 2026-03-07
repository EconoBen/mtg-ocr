"""Prepare training data for contrastive fine-tuning.

Downloads card images from Scryfall, applies augmentation, and generates
triplet training data (anchor, positive, negative) for contrastive learning.

Usage:
    python scripts/prepare_training_data.py --output data/training/ --num-triplets 10000
    python scripts/prepare_training_data.py --output data/training/ --severity heavy --batch-size 64

Requires: Scryfall bulk data (downloaded automatically), card images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare augmented training triplets from Scryfall card images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing card images (organized as <scryfall_id>.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for training data",
    )
    parser.add_argument(
        "--num-triplets",
        type=int,
        default=10000,
        help="Number of triplets to generate (default: 10000)",
    )
    parser.add_argument(
        "--severity",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Augmentation severity (default: medium)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def load_card_images(image_dir: Path) -> dict[str, list[np.ndarray]]:
    """Load card images from directory, grouped by card ID."""
    card_images: dict[str, list[np.ndarray]] = {}
    for ext in ("*.jpg", "*.png"):
        for img_path in sorted(image_dir.glob(ext)):
            card_id = img_path.stem
            with Image.open(img_path) as img:
                img_array = np.array(img.convert("RGB"))
            if card_id not in card_images:
                card_images[card_id] = []
            card_images[card_id].append(img_array)

    return card_images


def main() -> None:
    args = parse_args()

    from mtg_ocr.training.augmentation import CardAugmentation
    from mtg_ocr.training.dataset import CardTripletDataset

    print(f"Loading card images from {args.image_dir}...")
    card_images = load_card_images(args.image_dir)
    if len(card_images) < 2:
        print(f"Error: Need at least 2 cards, found {len(card_images)}", file=sys.stderr)
        sys.exit(1)

    total_images = sum(len(imgs) for imgs in card_images.values())
    print(f"Loaded {total_images} images for {len(card_images)} cards")

    rng = np.random.RandomState(args.seed)
    augmentation = CardAugmentation(severity=args.severity, rng=rng)
    dataset = CardTripletDataset(card_images=card_images, augmentation=augmentation, rng=rng)

    args.output.mkdir(parents=True, exist_ok=True)
    anchors_dir = args.output / "anchors"
    positives_dir = args.output / "positives"
    negatives_dir = args.output / "negatives"
    anchors_dir.mkdir(exist_ok=True)
    positives_dir.mkdir(exist_ok=True)
    negatives_dir.mkdir(exist_ok=True)

    manifest = []
    print(f"Generating {args.num_triplets} triplets with {args.severity} augmentation...")

    for i in range(args.num_triplets):
        idx = rng.randint(0, len(dataset))
        anchor, positive, negative = dataset[idx]

        anchor_path = anchors_dir / f"{i:06d}.jpg"
        positive_path = positives_dir / f"{i:06d}.jpg"
        negative_path = negatives_dir / f"{i:06d}.jpg"

        Image.fromarray(anchor).save(anchor_path, quality=95)
        Image.fromarray(positive).save(positive_path, quality=95)
        Image.fromarray(negative).save(negative_path, quality=95)

        manifest.append({
            "index": i,
            "anchor_card": dataset.get_card_id(idx),
            "negative_card": dataset.last_negative_card_id,
        })

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{args.num_triplets} triplets")

    manifest_path = args.output / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done! Training data saved to {args.output}")
    print(f"  Anchors:   {anchors_dir}")
    print(f"  Positives: {positives_dir}")
    print(f"  Negatives: {negatives_dir}")
    print(f"  Manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
