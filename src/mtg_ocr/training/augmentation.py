"""Image augmentation pipeline for training data.

Simulates real-world conditions: glare, blur, rotation,
perspective distortion, lighting variation, foil reflections.
Uses OpenCV and numpy (no albumentations dependency required).
"""

from __future__ import annotations

import cv2
import numpy as np

_VALID_SEVERITIES = ("light", "medium", "heavy")

_SEVERITY_PARAMS: dict[str, dict[str, dict]] = {
    "light": {
        "glare": {"intensity_range": (0.05, 0.15), "radius_frac": (0.1, 0.25)},
        "blur": {"ksize_range": (3, 5)},
        "rotation": {"angle_range": (-5, 5)},
        "brightness": {"delta_range": (-20, 20)},
        "foil": {"intensity_range": (0.02, 0.08)},
        "perspective": {"warp_frac": 0.02},
    },
    "medium": {
        "glare": {"intensity_range": (0.1, 0.3), "radius_frac": (0.15, 0.35)},
        "blur": {"ksize_range": (3, 9)},
        "rotation": {"angle_range": (-15, 15)},
        "brightness": {"delta_range": (-40, 40)},
        "foil": {"intensity_range": (0.05, 0.15)},
        "perspective": {"warp_frac": 0.05},
    },
    "heavy": {
        "glare": {"intensity_range": (0.2, 0.5), "radius_frac": (0.2, 0.45)},
        "blur": {"ksize_range": (5, 15)},
        "rotation": {"angle_range": (-30, 30)},
        "brightness": {"delta_range": (-60, 60)},
        "foil": {"intensity_range": (0.1, 0.25)},
        "perspective": {"warp_frac": 0.08},
    },
}


class CardAugmentation:
    """Image augmentation pipeline for training data.

    Simulates real-world conditions: glare, blur, rotation,
    perspective distortion, lighting variation, foil reflections.
    """

    _TRANSFORMS = ("glare", "blur", "rotation", "brightness", "foil", "perspective")

    def __init__(self, severity: str = "medium", rng: np.random.RandomState | None = None):
        if severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {_VALID_SEVERITIES}, got {severity!r}"
            )
        self.severity = severity
        self._params = _SEVERITY_PARAMS[severity]
        self._rng = rng or np.random.RandomState()

    @property
    def transform_names(self) -> list[str]:
        return list(self._TRANSFORMS)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply a random subset of augmentations to a card image."""
        result = image.copy()
        # Randomly select 2-4 transforms to apply
        n_transforms = self._rng.randint(2, min(5, len(self._TRANSFORMS) + 1))
        chosen = self._rng.choice(list(self._TRANSFORMS), size=n_transforms, replace=False)
        for name in chosen:
            result = self.apply_single(result, name)
        return result

    _TRANSFORMS = ("glare", "blur", "rotation", "brightness", "foil", "perspective")

    def apply_single(self, image: np.ndarray, transform_name: str) -> np.ndarray:
        """Apply a single named transform."""
        transforms = {
            "glare": self._apply_glare,
            "blur": self._apply_blur,
            "rotation": self._apply_rotation,
            "brightness": self._apply_brightness,
            "foil": self._apply_foil,
            "perspective": self._apply_perspective,
        }
        if transform_name not in transforms:
            raise ValueError(
                f"Unknown transform '{transform_name}'. Valid: {self._TRANSFORMS}"
            )
        return transforms[transform_name](image)

    def _apply_glare(self, image: np.ndarray) -> np.ndarray:
        """Simulate light glare as a bright elliptical spot."""
        h, w = image.shape[:2]
        p = self._params["glare"]
        intensity = self._rng.uniform(*p["intensity_range"])
        r_frac = self._rng.uniform(*p["radius_frac"])

        cx = self._rng.randint(0, w)
        cy = self._rng.randint(0, h)
        radius = int(max(h, w) * r_frac)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(
            mask,
            (cx, cy),
            (radius, int(radius * self._rng.uniform(0.5, 1.5))),
            self._rng.uniform(0, 360),
            0, 360, 1.0, -1,
        )
        mask = cv2.GaussianBlur(mask, (0, 0), radius * 0.4)
        if mask.max() > 0:
            mask = mask / mask.max() * intensity

        result = image.astype(np.float32)
        result = result + mask[..., np.newaxis] * 255
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        p = self._params["blur"]
        ksize = self._rng.randint(p["ksize_range"][0] // 2, p["ksize_range"][1] // 2 + 1) * 2 + 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """Apply rotation with border replication."""
        h, w = image.shape[:2]
        p = self._params["rotation"]
        angle = self._rng.uniform(*p["angle_range"])
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _apply_brightness(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness by a random delta."""
        p = self._params["brightness"]
        delta = self._rng.uniform(*p["delta_range"])
        result = image.astype(np.float32) + delta
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_foil(self, image: np.ndarray) -> np.ndarray:
        """Simulate holographic foil with color-shifted noise bands."""
        h, w = image.shape[:2]
        p = self._params["foil"]
        intensity = self._rng.uniform(*p["intensity_range"])

        # Create diagonal rainbow bands
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        freq = self._rng.uniform(0.02, 0.08)
        phase = self._rng.uniform(0, 2 * np.pi)
        wave = np.sin(freq * (xx + yy) + phase)

        # Map to HSV-like color shift
        foil_layer = np.zeros_like(image, dtype=np.float32)
        foil_layer[..., 0] = wave * 128 + 128  # R
        foil_layer[..., 1] = np.sin(wave * np.pi + 1) * 128 + 128  # G
        foil_layer[..., 2] = np.cos(wave * np.pi + 2) * 128 + 128  # B

        result = image.astype(np.float32) * (1 - intensity) + foil_layer * intensity
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_perspective(self, image: np.ndarray) -> np.ndarray:
        """Apply slight perspective warp."""
        h, w = image.shape[:2]
        warp = self._params["perspective"]["warp_frac"]

        # Random corner offsets
        offsets = self._rng.uniform(-warp, warp, (4, 2)).astype(np.float32)
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst = src + offsets * np.array([w, h], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
