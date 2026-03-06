from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CardInfo:
    """Card metadata from Scryfall bulk data."""

    scryfall_id: str
    name: str
    set_code: str
    set_name: str
    collector_number: str
    image_uris: dict[str, str]
