"""MobileCLIP-S0 visual encoder wrapper."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class MobileCLIPEncoder:
    """MobileCLIP-S0 visual encoder wrapper.

    Uses Apple's ml-mobileclip for mobile-optimized image embeddings.
    Model: MobileCLIP-S0, ~50-80MB, 3-15ms inference on iPhone.
    """

    def __init__(
        self,
        model_name: str = "mobileclip_s0",
        pretrained: str | None = "datacompdr",
    ):
        import mobileclip

        self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model.eval()
        self._embedding_dim: int = self._detect_embedding_dim()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _detect_embedding_dim(self) -> int:
        """Detect embedding dimensionality by running a dummy forward pass."""
        dummy = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            out = self.model.encode_image(dummy)
        return out.shape[-1]

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        return next(self.model.parameters()).device

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode single image to normalized embedding vector."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Batch encode images. Returns (N, embedding_dim) array of normalized embeddings."""
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_image(tensors)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeddings, axis=0)
