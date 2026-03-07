"""Generate synthetic card test images for unit testing.

Run this script to regenerate the test fixture:
    python tests/fixtures/create_synthetic_card.py
"""

from pathlib import Path

import cv2
import numpy as np


def create_card_on_dark_bg(
    output_path: Path,
    card_w: int = 252,
    card_h: int = 352,
    bg_w: int = 800,
    bg_h: int = 600,
) -> None:
    """Create a white rectangle (MTG card ratio 63:88) on dark background."""
    img = np.full((bg_h, bg_w, 3), 40, dtype=np.uint8)

    cx, cy = bg_w // 2, bg_h // 2
    x1 = cx - card_w // 2
    y1 = cy - card_h // 2

    img[y1 : y1 + card_h, x1 : x1 + card_w] = 220

    cv2.imwrite(str(output_path), img)


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent
    create_card_on_dark_bg(fixtures_dir / "synthetic_card.png")
    print("Created synthetic_card.png")
