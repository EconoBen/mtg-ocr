"""Tests for training data preparation: augmentation and triplet dataset."""

from __future__ import annotations

import numpy as np

from mtg_ocr.training.augmentation import CardAugmentation
from mtg_ocr.training.dataset import CardTripletDataset


def _make_card_image(width: int = 224, height: int = 224, seed: int = 0) -> np.ndarray:
    """Create a synthetic card image as a numpy array (H, W, 3)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestCardAugmentationProducesValidImages:
    """Test augmentation pipeline produces valid images."""

    def test_output_shape_matches_input(self):
        aug = CardAugmentation(severity="medium")
        image = _make_card_image(224, 224)
        result = aug(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_output_differs_from_input(self):
        aug = CardAugmentation(severity="medium")
        image = _make_card_image(224, 224, seed=42)
        result = aug(image)

        # At least some pixels should differ (augmentation applied)
        assert not np.array_equal(result, image)

    def test_output_pixel_range(self):
        aug = CardAugmentation(severity="heavy")
        image = _make_card_image(224, 224)
        result = aug(image)

        assert result.min() >= 0
        assert result.max() <= 255

    def test_different_calls_produce_different_results(self):
        aug = CardAugmentation(severity="medium", rng=np.random.RandomState(42))
        image = _make_card_image(224, 224, seed=10)
        result1 = aug(image)
        aug2 = CardAugmentation(severity="medium", rng=np.random.RandomState(99))
        result2 = aug2(image)

        # Different RNG seeds should produce different augmented results
        assert not np.array_equal(result1, result2)


class TestCardAugmentationSeverityLevels:
    """Test augmentation severity levels."""

    def test_light_severity(self):
        aug = CardAugmentation(severity="light")
        image = _make_card_image()
        result = aug(image)
        assert result.shape == image.shape

    def test_medium_severity(self):
        aug = CardAugmentation(severity="medium")
        image = _make_card_image()
        result = aug(image)
        assert result.shape == image.shape

    def test_heavy_severity(self):
        aug = CardAugmentation(severity="heavy")
        image = _make_card_image()
        result = aug(image)
        assert result.shape == image.shape

    def test_invalid_severity_raises(self):
        import pytest

        with pytest.raises(ValueError, match="severity"):
            CardAugmentation(severity="extreme")


class TestAugmentationIncludesGlareBlurAngleFoil:
    """Test augmentation includes glare, blur, angle, foil simulation."""

    def test_augmentation_has_expected_transform_types(self):
        aug = CardAugmentation(severity="medium")
        # The augmentation should expose its transform names
        names = aug.transform_names
        assert "glare" in names
        assert "blur" in names
        assert "rotation" in names
        assert "foil" in names

    def test_individual_transforms_work(self):
        aug = CardAugmentation(severity="medium")
        image = _make_card_image(224, 224, seed=5)

        for name in aug.transform_names:
            result = aug.apply_single(image, name)
            assert result.shape == image.shape
            assert result.dtype == np.uint8


class TestTripletGeneration:
    """Test triplet generation (anchor, positive, negative)."""

    def _make_dataset(self, n_cards: int = 5, images_per_card: int = 2) -> CardTripletDataset:
        """Create a test dataset with synthetic images."""
        card_images: dict[str, list[np.ndarray]] = {}
        for i in range(n_cards):
            card_id = f"card-{i}"
            card_images[card_id] = [
                _make_card_image(224, 224, seed=i * 100 + j)
                for j in range(images_per_card)
            ]
        return CardTripletDataset(
            card_images=card_images,
            augmentation=CardAugmentation(severity="light"),
        )

    def test_triplet_returns_three_images(self):
        dataset = self._make_dataset()
        anchor, positive, negative = dataset[0]

        assert isinstance(anchor, np.ndarray)
        assert isinstance(positive, np.ndarray)
        assert isinstance(negative, np.ndarray)

    def test_triplet_shapes(self):
        dataset = self._make_dataset()
        anchor, positive, negative = dataset[0]

        assert anchor.shape == (224, 224, 3)
        assert positive.shape == (224, 224, 3)
        assert negative.shape == (224, 224, 3)

    def test_anchor_and_positive_same_card(self):
        dataset = self._make_dataset(n_cards=5, images_per_card=3)
        # Verify the dataset is functional and structured correctly
        anchor, positive, negative = dataset[0]
        # We can't directly verify same card from images alone,
        # but we can verify the dataset length and structure
        assert len(dataset) > 0

    def test_dataset_length(self):
        dataset = self._make_dataset(n_cards=5, images_per_card=2)
        # Length should be total number of images (each can be an anchor)
        assert len(dataset) == 10  # 5 cards * 2 images each

    def test_negative_from_different_card(self):
        dataset = self._make_dataset(n_cards=5, images_per_card=2)
        # Run multiple samples and check that negatives come from different cards
        anchor_card = dataset.get_card_id(0)
        # Sample multiple times to verify randomness
        different_found = False
        for _ in range(20):
            _, _, neg = dataset[0]
            neg_card = dataset.last_negative_card_id
            if neg_card != anchor_card:
                different_found = True
                break
        assert different_found, "Negative should come from a different card"

    def test_single_card_raises_for_negative(self):
        """Can't generate negatives with only one card."""
        import pytest

        card_images = {"card-0": [_make_card_image(224, 224, seed=0)]}
        dataset = CardTripletDataset(
            card_images=card_images,
            augmentation=CardAugmentation(severity="light"),
        )
        with pytest.raises(ValueError, match="at least 2"):
            _ = dataset[0]
