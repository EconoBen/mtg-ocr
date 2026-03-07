"""Card detection using OpenCV contour detection with perspective correction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

# MTG card dimensions: 63mm x 88mm, aspect ratio ~0.716
MTG_CARD_ASPECT_RATIO = 63.0 / 88.0
ASPECT_RATIO_TOLERANCE = 0.15
MIN_CARD_AREA_FRACTION = 0.02  # Card must be at least 2% of image area
MAX_CARD_AREA_FRACTION = 0.85  # Card cannot be more than 85% of image area
OUTPUT_WIDTH = 252
OUTPUT_HEIGHT = 352


class ScanMode(StrEnum):
    HANDHELD = "handheld"
    RIG = "rig"


@dataclass
class DetectionResult:
    card_image: np.ndarray  # Cropped, perspective-corrected card image
    confidence: float
    bounding_box: tuple[int, int, int, int]  # x, y, w, h in original image
    scan_mode: ScanMode


class CardDetector:
    """Detect MTG card rectangle in an image using OpenCV contour detection.

    Handheld mode: adaptive thresholding, perspective correction, handles variable lighting.
    Rig mode: simpler detection assuming consistent positioning and lighting.
    """

    def __init__(self, scan_mode: ScanMode = ScanMode.HANDHELD):
        self.scan_mode = scan_mode

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        """Find the card in the image. Returns crop + metadata or None."""
        if self.scan_mode == ScanMode.HANDHELD:
            return self._detect_handheld(image)
        return self._detect_rig(image)

    def _detect_handheld(self, image: np.ndarray) -> DetectionResult | None:
        """Adaptive detection for variable conditions.

        Tries multiple thresholding strategies and returns the best card match.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        all_contours: list = []

        # Strategy 1: Canny edge detection (best for high-contrast synthetic images)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

        # Strategy 2: Otsu threshold
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

        # Strategy 3: Adaptive threshold (for variable lighting)
        thresh_adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        contours, _ = cv2.findContours(
            thresh_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours)

        return self._find_best_card(all_contours, image)

    def _detect_rig(self, image: np.ndarray) -> DetectionResult | None:
        """Simplified detection for controlled rig environment."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Simple Otsu threshold works well with consistent lighting
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return self._find_best_card(contours, image)

    def _find_best_card(
        self, contours: list, image: np.ndarray
    ) -> DetectionResult | None:
        """Find the largest contour matching MTG card aspect ratio."""
        img_h, img_w = image.shape[:2]
        total_area = img_h * img_w
        min_area = total_area * MIN_CARD_AREA_FRACTION
        max_area = total_area * MAX_CARD_AREA_FRACTION

        candidates: list[tuple[float, np.ndarray]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            if len(approx) != 4:
                continue

            # Check aspect ratio using the bounding rect
            rect = cv2.minAreaRect(contour)
            (_, (w, h), _) = rect
            if w == 0 or h == 0:
                continue

            # Ensure width < height (portrait orientation)
            short_side = min(w, h)
            long_side = max(w, h)
            aspect = short_side / long_side

            if abs(aspect - MTG_CARD_ASPECT_RATIO) <= ASPECT_RATIO_TOLERANCE:
                candidates.append((area, approx))

        if not candidates:
            return None

        # Pick the largest qualifying contour
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_area, best_approx = candidates[0]

        return self._extract_card(best_approx, image, best_area, img_h * img_w)

    def _extract_card(
        self,
        approx: np.ndarray,
        image: np.ndarray,
        card_area: float,
        img_area: int,
    ) -> DetectionResult:
        """Apply perspective correction and extract the card image."""
        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2).astype(np.float32)
        ordered = self._order_points(pts)

        dst = np.array(
            [
                [0, 0],
                [OUTPUT_WIDTH - 1, 0],
                [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
                [0, OUTPUT_HEIGHT - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        x, y, w, h = cv2.boundingRect(approx)
        confidence = min(1.0, card_area / img_area * 10)

        return DetectionResult(
            card_image=warped,
            confidence=confidence,
            bounding_box=(x, y, w, h),
            scan_mode=self.scan_mode,
        )

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]  # top-right has smallest difference
        rect[3] = pts[np.argmax(d)]  # bottom-left has largest difference

        return rect
