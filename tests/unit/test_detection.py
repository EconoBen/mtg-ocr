"""Tests for card detection module."""

from __future__ import annotations

import numpy as np

from mtg_ocr.detection.card_detector import CardDetector, DetectionResult, ScanMode


def make_synthetic_card_image(
    card_w: int = 252,
    card_h: int = 352,
    bg_size: tuple[int, int] = (800, 600),
    angle: float = 0.0,
    bg_color: int = 40,
    card_color: int = 220,
) -> np.ndarray:
    """Create a synthetic image with a white rectangle (card) on a dark background.

    MTG card ratio is 63:88 ~= 0.716. Default card_w/card_h = 252/352 = 0.716.
    """
    img = np.full((bg_size[1], bg_size[0], 3), bg_color, dtype=np.uint8)

    cx, cy = bg_size[0] // 2, bg_size[1] // 2

    if angle == 0.0:
        x1 = cx - card_w // 2
        y1 = cy - card_h // 2
        x2 = x1 + card_w
        y2 = y1 + card_h
        img[y1:y2, x1:x2] = card_color
    else:
        import cv2

        half_w, half_h = card_w // 2, card_h // 2
        corners = np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=np.float32,
        )
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        rotated = (corners @ rot.T + np.array([cx, cy])).astype(np.int32)
        cv2.fillConvexPoly(img, rotated, (card_color, card_color, card_color))

    return img


class TestScanMode:
    def test_handheld_value(self):
        assert ScanMode.HANDHELD == "handheld"

    def test_rig_value(self):
        assert ScanMode.RIG == "rig"

    def test_is_str_enum(self):
        assert isinstance(ScanMode.HANDHELD, str)


class TestDetectionResult:
    def test_creation(self):
        card_img = np.zeros((352, 252, 3), dtype=np.uint8)
        result = DetectionResult(
            card_image=card_img,
            confidence=0.95,
            bounding_box=(100, 50, 252, 352),
            scan_mode=ScanMode.HANDHELD,
        )
        assert result.confidence == 0.95
        assert result.bounding_box == (100, 50, 252, 352)
        assert result.scan_mode == ScanMode.HANDHELD
        assert result.card_image.shape == (352, 252, 3)


class TestCardDetectorHandheld:
    def test_detect_card_on_synthetic_image(self):
        img = make_synthetic_card_image()
        detector = CardDetector(scan_mode=ScanMode.HANDHELD)
        result = detector.detect(img)

        assert result is not None
        assert isinstance(result, DetectionResult)
        assert result.card_image.shape[0] > 0
        assert result.card_image.shape[1] > 0
        assert result.confidence > 0.0
        assert result.scan_mode == ScanMode.HANDHELD

    def test_perspective_correction_produces_upright_rectangle(self):
        img = make_synthetic_card_image(angle=15.0)
        detector = CardDetector(scan_mode=ScanMode.HANDHELD)
        result = detector.detect(img)

        assert result is not None
        h, w = result.card_image.shape[:2]
        aspect = w / h
        # MTG card ratio ~0.716; after correction should be close
        assert 0.5 < aspect < 0.9, f"Aspect ratio {aspect} not in expected range"

    def test_no_card_returns_none(self):
        # Uniform dark image with no card
        img = np.full((600, 800, 3), 40, dtype=np.uint8)
        detector = CardDetector(scan_mode=ScanMode.HANDHELD)
        result = detector.detect(img)

        assert result is None

    def test_multiple_cards_returns_largest(self):
        """When multiple card-like rectangles exist, return the largest."""
        img = np.full((600, 800, 3), 40, dtype=np.uint8)

        # Small card
        img[50:150, 50:122] = 220  # 100x72, ratio 0.72

        # Large card (should be detected)
        img[150:502, 300:552] = 220  # 352x252, ratio 0.716

        detector = CardDetector(scan_mode=ScanMode.HANDHELD)
        result = detector.detect(img)

        assert result is not None
        # The detected card should be closer to the large card's size
        h, w = result.card_image.shape[:2]
        assert h > 200, f"Expected large card, got height {h}"


class TestCardDetectorRig:
    def test_detect_rig_mode(self):
        img = make_synthetic_card_image()
        detector = CardDetector(scan_mode=ScanMode.RIG)
        result = detector.detect(img)

        assert result is not None
        assert result.scan_mode == ScanMode.RIG

    def test_rig_no_card_returns_none(self):
        img = np.full((600, 800, 3), 40, dtype=np.uint8)
        detector = CardDetector(scan_mode=ScanMode.RIG)
        result = detector.detect(img)

        assert result is None
