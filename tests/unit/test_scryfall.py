from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mtg_ocr.data.models import CardInfo
from mtg_ocr.data.scryfall import SCRYFALL_BULK_URL, ScryfallClient


@pytest.fixture
def tmp_cache(tmp_path):
    return tmp_path / "scryfall_cache"


@pytest.fixture
def client(tmp_cache):
    return ScryfallClient(cache_dir=tmp_cache)


SAMPLE_BULK_DATA_RESPONSE = {
    "object": "list",
    "has_more": False,
    "data": [
        {
            "object": "bulk_data",
            "id": "abc123",
            "type": "default_cards",
            "name": "Default Cards",
            "download_uri": "https://data.scryfall.io/default-cards/default-cards-20240101.json",
            "size": 123456,
        },
        {
            "object": "bulk_data",
            "id": "def456",
            "type": "oracle_cards",
            "name": "Oracle Cards",
            "download_uri": "https://data.scryfall.io/oracle-cards/oracle-cards-20240101.json",
            "size": 654321,
        },
    ],
}

SAMPLE_CARD_DATA = [
    {
        "id": "card-001",
        "name": "Lightning Bolt",
        "set": "lea",
        "set_name": "Limited Edition Alpha",
        "collector_number": "161",
        "image_uris": {
            "small": "https://cards.scryfall.io/small/front/card-001.jpg",
            "normal": "https://cards.scryfall.io/normal/front/card-001.jpg",
            "large": "https://cards.scryfall.io/large/front/card-001.jpg",
        },
    },
    {
        "id": "card-002",
        "name": "Black Lotus",
        "set": "lea",
        "set_name": "Limited Edition Alpha",
        "collector_number": "232",
        "image_uris": {
            "small": "https://cards.scryfall.io/small/front/card-002.jpg",
            "normal": "https://cards.scryfall.io/normal/front/card-002.jpg",
            "large": "https://cards.scryfall.io/large/front/card-002.jpg",
        },
    },
    {
        "id": "card-003",
        "name": "Counterspell",
        "set": "lea",
        "set_name": "Limited Edition Alpha",
        "collector_number": "54",
        "image_uris": {
            "small": "https://cards.scryfall.io/small/front/card-003.jpg",
            "normal": "https://cards.scryfall.io/normal/front/card-003.jpg",
        },
    },
    # Card with no image_uris (e.g., double-faced card with card_faces)
    {
        "id": "card-004",
        "name": "Delver of Secrets // Insectile Aberration",
        "set": "isd",
        "set_name": "Innistrad",
        "collector_number": "51",
        "card_faces": [
            {
                "name": "Delver of Secrets",
                "image_uris": {
                    "normal": "https://cards.scryfall.io/normal/front/card-004-front.jpg",
                },
            },
            {
                "name": "Insectile Aberration",
                "image_uris": {
                    "normal": "https://cards.scryfall.io/normal/back/card-004-back.jpg",
                },
            },
        ],
    },
]


class TestBulkDataFetch:
    """Test bulk data URL fetch returns valid JSON structure."""

    def test_fetch_bulk_data_url_returns_download_uri(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_BULK_DATA_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            url = client.get_bulk_data_url(data_type="default_cards")

        assert url == "https://data.scryfall.io/default-cards/default-cards-20240101.json"
        mock_get.assert_called_once_with(SCRYFALL_BULK_URL, timeout=30)

    def test_fetch_bulk_data_url_raises_for_unknown_type(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_BULK_DATA_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(ValueError, match="not found"):
                client.get_bulk_data_url(data_type="nonexistent_type")


class TestCardDictionary:
    """Test card dictionary building from bulk data."""

    def test_build_card_dictionary_returns_all_cards(self, client, tmp_cache):
        bulk_path = tmp_cache / "bulk.json"
        bulk_path.parent.mkdir(parents=True, exist_ok=True)
        bulk_path.write_text(json.dumps(SAMPLE_CARD_DATA))

        cards = client.build_card_dictionary(bulk_path)

        assert len(cards) == 4
        assert "card-001" in cards
        assert "card-002" in cards

    def test_card_info_fields_populated(self, client, tmp_cache):
        bulk_path = tmp_cache / "bulk.json"
        bulk_path.parent.mkdir(parents=True, exist_ok=True)
        bulk_path.write_text(json.dumps(SAMPLE_CARD_DATA))

        cards = client.build_card_dictionary(bulk_path)
        bolt = cards["card-001"]

        assert isinstance(bolt, CardInfo)
        assert bolt.scryfall_id == "card-001"
        assert bolt.name == "Lightning Bolt"
        assert bolt.set_code == "lea"
        assert bolt.set_name == "Limited Edition Alpha"
        assert bolt.collector_number == "161"
        assert "normal" in bolt.image_uris

    def test_double_faced_card_uses_front_face_image(self, client, tmp_cache):
        bulk_path = tmp_cache / "bulk.json"
        bulk_path.parent.mkdir(parents=True, exist_ok=True)
        bulk_path.write_text(json.dumps(SAMPLE_CARD_DATA))

        cards = client.build_card_dictionary(bulk_path)
        delver = cards["card-004"]

        assert delver.name == "Delver of Secrets // Insectile Aberration"
        assert "normal" in delver.image_uris


class TestSetCodeMapping:
    """Test set code mapping from card data."""

    def test_cards_have_correct_set_codes(self, client, tmp_cache):
        bulk_path = tmp_cache / "bulk.json"
        bulk_path.parent.mkdir(parents=True, exist_ok=True)
        bulk_path.write_text(json.dumps(SAMPLE_CARD_DATA))

        cards = client.build_card_dictionary(bulk_path)

        assert cards["card-001"].set_code == "lea"
        assert cards["card-002"].set_code == "lea"
        assert cards["card-004"].set_code == "isd"


class TestImageURIExtraction:
    """Test image URI extraction."""

    def test_get_image_uris_returns_tuples(self, client, tmp_cache):
        bulk_path = tmp_cache / "bulk.json"
        bulk_path.parent.mkdir(parents=True, exist_ok=True)
        bulk_path.write_text(json.dumps(SAMPLE_CARD_DATA))

        cards = client.build_card_dictionary(bulk_path)
        uris = client.get_image_uris(cards)

        assert len(uris) == 4
        ids = [scryfall_id for scryfall_id, _ in uris]
        assert "card-001" in ids
        assert "card-002" in ids

    def test_image_uris_prefer_normal_size(self, client, tmp_cache):
        bulk_path = tmp_cache / "bulk.json"
        bulk_path.parent.mkdir(parents=True, exist_ok=True)
        bulk_path.write_text(json.dumps(SAMPLE_CARD_DATA))

        cards = client.build_card_dictionary(bulk_path)
        uris = client.get_image_uris(cards)

        uri_dict = dict(uris)
        assert "normal" in uri_dict["card-001"]

    def test_double_faced_card_image_uri_extracted(self, client, tmp_cache):
        bulk_path = tmp_cache / "bulk.json"
        bulk_path.parent.mkdir(parents=True, exist_ok=True)
        bulk_path.write_text(json.dumps(SAMPLE_CARD_DATA))

        cards = client.build_card_dictionary(bulk_path)
        uris = client.get_image_uris(cards)

        uri_dict = dict(uris)
        assert "card-004" in uri_dict
        assert "front" in uri_dict["card-004"]


class TestRateLimiting:
    """Test rate limiting behavior."""

    def test_download_bulk_data_respects_rate_limit(self, client, tmp_cache):
        tmp_cache.mkdir(parents=True, exist_ok=True)

        bulk_url_response = MagicMock()
        bulk_url_response.json.return_value = SAMPLE_BULK_DATA_RESPONSE
        bulk_url_response.raise_for_status = MagicMock()

        bulk_data_content = json.dumps(SAMPLE_CARD_DATA).encode()

        def mock_stream_context(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.iter_bytes = MagicMock(return_value=iter([bulk_data_content]))
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        sleep_calls = []

        with patch("httpx.get", return_value=bulk_url_response):
            with patch("httpx.stream", side_effect=mock_stream_context):
                with patch("mtg_ocr.data.scryfall.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
                    path = client.download_bulk_data(data_type="default_cards")

        assert path.exists()
        # Verify rate limit sleep was called with correct value
        assert len(sleep_calls) >= 1
        assert all(s == 0.075 for s in sleep_calls)

    def test_client_has_rate_limit_configured(self, client):
        from mtg_ocr.data.scryfall import RATE_LIMIT_SECONDS

        assert RATE_LIMIT_SECONDS == 0.075
