"""Visual encoder protocol for card image embedding."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class VisualEncoder(Protocol):
    """Protocol for visual encoding models.

    Any encoder must produce L2-normalized embeddings from PIL images.
    """

    @property
    def embedding_dim(self) -> int: ...

    def encode_image(self, image: Image.Image) -> np.ndarray: ...

    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray: ...
