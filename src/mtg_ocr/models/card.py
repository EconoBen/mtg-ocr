from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CardMatch(BaseModel):
    scryfall_id: str
    card_name: str
    set_code: str
    set_name: str
    confidence: float
    image_uri: str | None = None


class IdentificationResult(BaseModel):
    matches: list[CardMatch]
    latency_ms: float
    scan_mode: Literal["handheld", "rig"]


class EmbeddingRecord(BaseModel):
    scryfall_id: str
    card_name: str
    set_code: str
    embedding: list[float]
