"""Generate training triplets for contrastive learning.

Each triplet: (anchor_image, positive_variant, negative_card)
- anchor: original card image
- positive: same card, different augmentation (or different printing)
- negative: different card entirely
"""

from __future__ import annotations

import numpy as np

from mtg_ocr.training.augmentation import CardAugmentation


class CardTripletDataset:
    """Generate training triplets for contrastive learning.

    Each triplet: (anchor_image, positive_variant, negative_card)
    - anchor: original card image
    - positive: same card, different augmentation (or different printing)
    - negative: different card entirely
    """

    def __init__(
        self,
        card_images: dict[str, list[np.ndarray]],
        augmentation: CardAugmentation | None = None,
        rng: np.random.RandomState | None = None,
    ):
        self._card_images = card_images
        self._augmentation = augmentation
        self._rng = rng or np.random.RandomState()
        self.last_negative_card_id: str | None = None

        # Build flat index: (card_id, image_index) for each sample
        self._index: list[tuple[str, int]] = []
        for card_id, images in card_images.items():
            for img_idx in range(len(images)):
                self._index.append((card_id, img_idx))

        # Validate no empty image lists
        empty_cards = [cid for cid, imgs in card_images.items() if len(imgs) == 0]
        if empty_cards:
            raise ValueError(
                f"Cards with empty image lists: {empty_cards[:5]}"
            )

        self._card_ids = list(card_images.keys())

    def __len__(self) -> int:
        return len(self._index)

    def get_card_id(self, idx: int) -> str:
        return self._index[idx][0]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (anchor, positive, negative) triplet.

        - anchor: the image at idx (with optional augmentation)
        - positive: another image of the same card (or augmented version)
        - negative: image from a different card
        """
        card_id, img_idx = self._index[idx]

        if len(self._card_ids) < 2:
            raise ValueError("Need at least 2 different cards to generate triplets")

        # Anchor
        anchor = self._card_images[card_id][img_idx].copy()

        # Positive: pick another image of the same card, or augment the anchor
        same_card_images = self._card_images[card_id]
        if len(same_card_images) > 1:
            other_indices = [i for i in range(len(same_card_images)) if i != img_idx]
            pos_idx = self._rng.choice(other_indices)
            positive = same_card_images[pos_idx].copy()
        else:
            positive = anchor.copy()

        # Apply augmentation to positive (always) to create variation
        if self._augmentation is not None:
            positive = self._augmentation(positive)
            anchor = self._augmentation(anchor)

        # Negative: random image from a different card
        neg_card_id = card_id
        while neg_card_id == card_id:
            neg_card_id = self._rng.choice(self._card_ids)
        self.last_negative_card_id = neg_card_id

        neg_images = self._card_images[neg_card_id]
        neg_idx = self._rng.randint(0, len(neg_images))
        negative = neg_images[neg_idx].copy()
        if self._augmentation is not None:
            negative = self._augmentation(negative)

        return anchor, positive, negative
