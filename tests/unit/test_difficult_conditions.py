"""Tests for difficult card identification conditions.

Verifies the augmentation pipeline handles each difficult condition
without errors and produces valid images (correct shape, dtype, value range).
"""

from __future__ import annotations

import numpy as np
import pytest

from mtg_ocr.benchmark.difficult import (
    ConditionResult,
    DifficultConditionsReport,
)
from mtg_ocr.training.augmentation import CardAugmentation

from tests.fixtures.difficult.generate import (
    create_card_on_background,
    create_fullart_card,
    create_old_border_card,
    create_standard_card,
)


@pytest.fixture
def standard_image() -> np.ndarray:
    return create_card_on_background(create_standard_card())


@pytest.fixture
def old_border_image() -> np.ndarray:
    return create_card_on_background(create_old_border_card())


@pytest.fixture
def fullart_image() -> np.ndarray:
    return create_card_on_background(create_fullart_card())


@pytest.fixture
def heavy_aug() -> CardAugmentation:
    return CardAugmentation(severity="heavy", rng=np.random.RandomState(42))


def _assert_valid_image(img: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    """Assert image has correct shape, dtype, and value range."""
    assert img.shape == expected_shape, f"Shape mismatch: {img.shape} != {expected_shape}"
    assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"
    assert img.min() >= 0, f"Values below 0: {img.min()}"
    assert img.max() <= 255, f"Values above 255: {img.max()}"


class TestFoilGlare:
    """Foil cards with high glare simulation."""

    def test_glare_augmentation_produces_valid_image(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "glare")
        _assert_valid_image(result, standard_image.shape)

    def test_foil_augmentation_produces_valid_image(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "foil")
        _assert_valid_image(result, standard_image.shape)

    def test_combined_foil_and_glare(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "foil")
        result = heavy_aug.apply_single(result, "glare")
        _assert_valid_image(result, standard_image.shape)


class TestHeavyBlur:
    """Motion blur and out of focus."""

    def test_blur_produces_valid_image(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "blur")
        _assert_valid_image(result, standard_image.shape)

    def test_heavy_blur_reduces_high_frequency(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "blur")
        # Blurred image should have lower variance in local gradients
        orig_grad = np.abs(np.diff(standard_image.astype(float), axis=1)).mean()
        blur_grad = np.abs(np.diff(result.astype(float), axis=1)).mean()
        assert blur_grad < orig_grad, "Blur should reduce gradient magnitude"


class TestExtremeRotation:
    """Rotation > 20 degrees."""

    def test_rotation_produces_valid_image(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "rotation")
        _assert_valid_image(result, standard_image.shape)

    def test_rotation_changes_pixel_content(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "rotation")
        # Rotated image should differ from original
        assert not np.array_equal(result, standard_image)


class TestPoorLighting:
    """Very dark and very bright conditions."""

    def test_brightness_produces_valid_image(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "brightness")
        _assert_valid_image(result, standard_image.shape)

    def test_very_dark_image(self, standard_image: np.ndarray) -> None:
        """Simulate very dark conditions by reducing brightness."""
        dark = np.clip(standard_image.astype(np.float32) - 100, 0, 255).astype(np.uint8)
        aug = CardAugmentation(severity="heavy", rng=np.random.RandomState(123))
        result = aug(dark)
        _assert_valid_image(result, dark.shape)

    def test_very_bright_image(self, standard_image: np.ndarray) -> None:
        """Simulate very bright / overexposed conditions."""
        bright = np.clip(standard_image.astype(np.float32) + 100, 0, 255).astype(np.uint8)
        aug = CardAugmentation(severity="heavy", rng=np.random.RandomState(456))
        result = aug(bright)
        _assert_valid_image(result, bright.shape)


class TestPartialOcclusion:
    """Finger over card edge or partial obstruction."""

    def test_occluded_card_augments_without_error(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        """Simulate partial occlusion by blacking out a strip on one edge."""
        occluded = standard_image.copy()
        h, w = occluded.shape[:2]
        # Black strip on right edge (simulating finger)
        occluded[:, w - 60 :] = 30
        result = heavy_aug(occluded)
        _assert_valid_image(result, occluded.shape)

    def test_corner_occlusion(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        """Simulate corner occlusion."""
        occluded = standard_image.copy()
        h, w = occluded.shape[:2]
        # Black triangle in bottom-right corner
        for y in range(h - 80, h):
            x_start = w - (y - (h - 80))
            occluded[y, max(0, x_start) :] = 30
        result = heavy_aug(occluded)
        _assert_valid_image(result, occluded.shape)


class TestOldBorderCards:
    """Old-border cards with different aspect ratios."""

    def test_old_border_augments_without_error(
        self, old_border_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug(old_border_image)
        _assert_valid_image(result, old_border_image.shape)

    def test_old_border_all_individual_transforms(
        self, old_border_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        for transform in heavy_aug.transform_names:
            result = heavy_aug.apply_single(old_border_image, transform)
            _assert_valid_image(result, old_border_image.shape)


class TestFullArtCards:
    """Full-art / borderless cards."""

    def test_fullart_augments_without_error(
        self, fullart_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug(fullart_image)
        _assert_valid_image(result, fullart_image.shape)

    def test_fullart_all_individual_transforms(
        self, fullart_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        for transform in heavy_aug.transform_names:
            result = heavy_aug.apply_single(fullart_image, transform)
            _assert_valid_image(result, fullart_image.shape)


class TestAllConditionsCombined:
    """Verify heavy augmentation on all card types produces valid images."""

    @pytest.mark.parametrize("card_factory", [
        create_standard_card,
        create_old_border_card,
        create_fullart_card,
    ])
    def test_full_augmentation_pipeline(
        self, card_factory: callable, heavy_aug: CardAugmentation
    ) -> None:
        card = create_card_on_background(card_factory())
        result = heavy_aug(card)
        _assert_valid_image(result, card.shape)

    def test_repeated_augmentation_stays_valid(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        """Multiple rounds of augmentation should still produce valid images."""
        result = standard_image.copy()
        for _ in range(3):
            result = heavy_aug(result)
        _assert_valid_image(result, standard_image.shape)


class TestDifficultConditionsReport:
    """Tests for the DifficultConditionsReport dataclass."""

    def test_empty_report(self) -> None:
        report = DifficultConditionsReport()
        assert report.total_images == 0
        assert report.conditions == []
        assert report.overall_top_1_accuracy == 0.0

    def test_add_single_condition(self) -> None:
        report = DifficultConditionsReport()
        report.add_condition(ConditionResult(
            condition="foil",
            total_images=10,
            correct_top_1=8,
            correct_top_5=10,
            top_1_accuracy=0.8,
            top_5_accuracy=1.0,
            mean_latency_ms=50.0,
        ))
        assert report.total_images == 10
        assert report.overall_top_1_accuracy == 0.8
        assert report.overall_top_5_accuracy == 1.0
        assert report.worst_condition == "foil"

    def test_add_multiple_conditions(self) -> None:
        report = DifficultConditionsReport()
        report.add_condition(ConditionResult(
            condition="foil",
            total_images=10,
            correct_top_1=8,
            correct_top_5=10,
            top_1_accuracy=0.8,
            top_5_accuracy=1.0,
            mean_latency_ms=50.0,
        ))
        report.add_condition(ConditionResult(
            condition="blur",
            total_images=10,
            correct_top_1=6,
            correct_top_5=9,
            top_1_accuracy=0.6,
            top_5_accuracy=0.9,
            mean_latency_ms=55.0,
        ))
        assert report.total_images == 20
        assert report.overall_top_1_accuracy == 14 / 20
        assert report.overall_top_5_accuracy == 19 / 20
        assert report.worst_condition == "blur"

    def test_to_json_roundtrip(self) -> None:
        report = DifficultConditionsReport()
        report.add_condition(ConditionResult(
            condition="rotation",
            total_images=5,
            correct_top_1=4,
            correct_top_5=5,
            top_1_accuracy=0.8,
            top_5_accuracy=1.0,
            mean_latency_ms=45.0,
        ))
        json_str = report.to_json()
        assert "rotation" in json_str
        data = report.to_dict()
        assert data["total_images"] == 5
        assert len(data["conditions"]) == 1

    def test_perspective_augmentation_valid(
        self, standard_image: np.ndarray, heavy_aug: CardAugmentation
    ) -> None:
        result = heavy_aug.apply_single(standard_image, "perspective")
        _assert_valid_image(result, standard_image.shape)
