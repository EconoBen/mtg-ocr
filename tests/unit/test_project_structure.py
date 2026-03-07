from __future__ import annotations


def test_version():
    from mtg_ocr import __version__

    assert __version__ == "0.1.0"


def test_models_import():
    from mtg_ocr.models.card import CardMatch, EmbeddingRecord, IdentificationResult

    assert CardMatch is not None
    assert IdentificationResult is not None
    assert EmbeddingRecord is not None


def test_card_match_model():
    from mtg_ocr.models.card import CardMatch

    match = CardMatch(
        scryfall_id="abc-123",
        card_name="Lightning Bolt",
        set_code="lea",
        set_name="Limited Edition Alpha",
        confidence=0.95,
        image_uri="https://example.com/image.jpg",
    )
    assert match.scryfall_id == "abc-123"
    assert match.card_name == "Lightning Bolt"
    assert match.confidence == 0.95


def test_identification_result_model():
    from mtg_ocr.models.card import CardMatch, IdentificationResult

    result = IdentificationResult(
        matches=[
            CardMatch(
                scryfall_id="abc-123",
                card_name="Lightning Bolt",
                set_code="lea",
                set_name="Limited Edition Alpha",
                confidence=0.95,
            )
        ],
        latency_ms=12.5,
        scan_mode="handheld",
    )
    assert len(result.matches) == 1
    assert result.latency_ms == 12.5
    assert result.scan_mode == "handheld"


def test_embedding_record_model():
    from mtg_ocr.models.card import EmbeddingRecord

    record = EmbeddingRecord(
        scryfall_id="abc-123",
        card_name="Lightning Bolt",
        set_code="lea",
        embedding=[0.1, 0.2, 0.3],
    )
    assert len(record.embedding) == 3


def test_subpackages_importable():
    import mtg_ocr.encoder
    import mtg_ocr.search
    import mtg_ocr.detection
    import mtg_ocr.data
    import mtg_ocr.training
    import mtg_ocr.embeddings
    import mtg_ocr.export
    import mtg_ocr.benchmark
    import mtg_ocr.scanning

    assert mtg_ocr.encoder is not None
    assert mtg_ocr.search is not None
    assert mtg_ocr.detection is not None
    assert mtg_ocr.data is not None
    assert mtg_ocr.training is not None
    assert mtg_ocr.embeddings is not None
    assert mtg_ocr.export is not None
    assert mtg_ocr.benchmark is not None
    assert mtg_ocr.scanning is not None
