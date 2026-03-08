"""Generate synthetic card images for difficult condition testing.

Creates colored rectangles with text overlays to simulate card-like images
under various conditions (different borders, aspect ratios, etc.).
No real card images needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _draw_text(img: np.ndarray, text: str, position: tuple[int, int], scale: float = 0.5) -> None:
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def create_standard_card(w: int = 252, h: int = 352) -> np.ndarray:
    """Standard modern-border card (63:88 ratio)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    # Border
    cv2.rectangle(img, (5, 5), (w - 6, h - 6), (50, 50, 50), 2)
    # Title area
    cv2.rectangle(img, (10, 10), (w - 11, 40), (180, 200, 220), -1)
    _draw_text(img, "Card Name", (15, 30))
    # Art area
    cv2.rectangle(img, (10, 45), (w - 11, 200), (100, 150, 120), -1)
    # Text area
    cv2.rectangle(img, (10, 205), (w - 11, h - 40), (220, 215, 200), -1)
    _draw_text(img, "Card text here", (15, 230), 0.35)
    return img


def create_old_border_card(w: int = 240, h: int = 340) -> np.ndarray:
    """Old-border card with slightly different aspect ratio and thicker borders."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    # Thick outer border
    cv2.rectangle(img, (3, 3), (w - 4, h - 4), (80, 60, 40), 3)
    # Inner border
    cv2.rectangle(img, (12, 12), (w - 13, h - 13), (100, 80, 50), 2)
    # Title
    cv2.rectangle(img, (18, 18), (w - 19, 45), (190, 180, 150), -1)
    _draw_text(img, "Old Card", (22, 38))
    # Art
    cv2.rectangle(img, (18, 50), (w - 19, 190), (120, 130, 100), -1)
    # Text
    cv2.rectangle(img, (18, 195), (w - 19, h - 50), (200, 195, 170), -1)
    return img


def create_fullart_card(w: int = 252, h: int = 352) -> np.ndarray:
    """Full-art / borderless card — art extends to edges."""
    img = np.full((h, w, 3), dtype=np.uint8, fill_value=0)
    # Gradient art fill
    for y in range(h):
        r = int(80 + 120 * y / h)
        g = int(60 + 80 * (1 - y / h))
        b = int(100 + 100 * abs(0.5 - y / h) * 2)
        img[y, :] = [b, g, r]
    # Semi-transparent text overlay at bottom
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    _draw_text(img, "Fullart Card", (10, h - 50), 0.4)
    return img


def create_card_on_background(card: np.ndarray, bg_w: int = 640, bg_h: int = 480) -> np.ndarray:
    """Place a card image centered on a dark background."""
    bg = np.full((bg_h, bg_w, 3), 40, dtype=np.uint8)
    ch, cw = card.shape[:2]
    y_off = (bg_h - ch) // 2
    x_off = (bg_w - cw) // 2
    bg[y_off:y_off + ch, x_off:x_off + cw] = card
    return bg


def generate_all(output_dir: Path) -> dict[str, Path]:
    """Generate all synthetic test images and return name->path mapping."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path] = {}

    # Standard card
    standard = create_card_on_background(create_standard_card())
    path = output_dir / "standard_card.png"
    cv2.imwrite(str(path), standard)
    images["standard"] = path

    # Old border card
    old_border = create_card_on_background(create_old_border_card())
    path = output_dir / "old_border_card.png"
    cv2.imwrite(str(path), old_border)
    images["old_border"] = path

    # Full-art card
    fullart = create_card_on_background(create_fullart_card())
    path = output_dir / "fullart_card.png"
    cv2.imwrite(str(path), fullart)
    images["fullart"] = path

    return images


if __name__ == "__main__":
    out = Path(__file__).parent
    result = generate_all(out)
    for name, p in result.items():
        print(f"Created {name}: {p}")
