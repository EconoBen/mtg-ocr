from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import httpx

from mtg_ocr.data.models import CardInfo

SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"
RATE_LIMIT_SECONDS = 0.075


class ScryfallClient:
    """Download and cache Scryfall bulk data for card names, set codes, image URIs."""

    def __init__(self, cache_dir: Path = Path(".cache/scryfall")):
        self.cache_dir = cache_dir

    def get_bulk_data_url(self, data_type: str = "default_cards") -> str:
        """Fetch the download URL for a specific bulk data type."""
        response = httpx.get(SCRYFALL_BULK_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        for entry in data["data"]:
            if entry["type"] == data_type:
                return entry["download_uri"]

        raise ValueError(f"Bulk data type '{data_type}' not found")

    def download_bulk_data(self, data_type: str = "default_cards") -> Path:
        """Download bulk data JSON file. Cache locally."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        url = self.get_bulk_data_url(data_type)

        filename = url.rsplit("/", 1)[-1]
        output_path = self.cache_dir / filename

        if output_path.exists():
            return output_path

        time.sleep(RATE_LIMIT_SECONDS)

        # Write to a unique temp file, then rename atomically to avoid
        # corrupt cache if download is interrupted or concurrent runs clash
        fd, tmp_str = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_str)
        try:
            with httpx.stream("GET", url, timeout=120) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            tmp_path.rename(output_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        return output_path

    def build_card_dictionary(self, bulk_data_path: Path) -> dict[str, CardInfo]:
        """Build scryfall_id -> CardInfo mapping from bulk data."""
        with open(bulk_data_path) as f:
            cards_data = json.load(f)

        result: dict[str, CardInfo] = {}
        for card in cards_data:
            scryfall_id = card["id"]
            image_uris = card.get("image_uris", {})

            # For double-faced cards, use front face image URIs
            if not image_uris and "card_faces" in card:
                faces = card["card_faces"]
                if faces and "image_uris" in faces[0]:
                    image_uris = faces[0]["image_uris"]

            result[scryfall_id] = CardInfo(
                scryfall_id=scryfall_id,
                name=card["name"],
                set_code=card["set"],
                set_name=card["set_name"],
                collector_number=card["collector_number"],
                image_uris=image_uris,
            )

        return result

    def get_image_uris(self, cards: dict[str, CardInfo]) -> list[tuple[str, str]]:
        """Return list of (scryfall_id, normal_image_uri) for embedding computation."""
        result: list[tuple[str, str]] = []
        for scryfall_id, card in cards.items():
            uri = card.image_uris.get("normal") or card.image_uris.get("large")
            if uri:
                result.append((scryfall_id, uri))
        return result
