"""Tests for the batch scanning pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from mtg_ocr.models.card import CardMatch, ScanReport, ScanResult
from mtg_ocr.scanning.batch import BatchScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pipeline() -> MagicMock:
    """Create a mock CardIdentificationPipeline that returns deterministic results."""
    from mtg_ocr.models.card import IdentificationResult

    pipeline = MagicMock()

    def _identify(image, top_k=5):
        return IdentificationResult(
            matches=[
                CardMatch(
                    scryfall_id="abc-123",
                    card_name="Lightning Bolt",
                    set_code="lea",
                    set_name="Limited Edition Alpha",
                    confidence=0.95,
                )
            ],
            latency_ms=10.0,
            scan_mode="rig",
        )

    pipeline.identify.side_effect = _identify
    return pipeline


def _create_test_images(tmp_path: Path, count: int = 3) -> list[Path]:
    """Create synthetic test images in a temp directory."""
    paths = []
    for i in range(count):
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 75, 3), dtype=np.uint8)
        )
        p = tmp_path / f"card_{i:03d}.png"
        img.save(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Tests: ScanResult / ScanReport models
# ---------------------------------------------------------------------------


class TestScanModels:
    def test_scan_result_creation(self):
        result = ScanResult(
            image_path="/tmp/card.png",
            matches=[
                CardMatch(
                    scryfall_id="abc",
                    card_name="Sol Ring",
                    set_code="cmd",
                    set_name="Commander",
                    confidence=0.9,
                )
            ],
            latency_ms=15.5,
        )
        assert result.image_path == "/tmp/card.png"
        assert len(result.matches) == 1
        assert result.latency_ms == 15.5

    def test_scan_report_creation(self):
        report = ScanReport(
            results=[],
            total_cards=0,
            avg_latency_ms=0.0,
            cards_per_minute=0.0,
            elapsed_seconds=0.0,
        )
        assert report.total_cards == 0

    def test_scan_report_serialization(self):
        result = ScanResult(
            image_path="/tmp/card.png",
            matches=[],
            latency_ms=10.0,
        )
        report = ScanReport(
            results=[result],
            total_cards=1,
            avg_latency_ms=10.0,
            cards_per_minute=120.0,
            elapsed_seconds=0.5,
        )
        data = json.loads(report.model_dump_json())
        assert data["total_cards"] == 1
        assert data["cards_per_minute"] == 120.0
        assert len(data["results"]) == 1


# ---------------------------------------------------------------------------
# Tests: BatchScanner
# ---------------------------------------------------------------------------


class TestBatchScanner:
    def test_scan_images_empty_list(self):
        pipeline = _make_mock_pipeline()
        scanner = BatchScanner(pipeline=pipeline, workers=2)
        report = scanner.scan_images([])
        assert report.total_cards == 0
        assert report.avg_latency_ms == 0.0
        assert report.cards_per_minute == 0.0
        assert report.elapsed_seconds == 0.0
        pipeline.identify.assert_not_called()

    def test_scan_images_returns_report(self, tmp_path):
        pipeline = _make_mock_pipeline()
        paths = _create_test_images(tmp_path, count=3)
        scanner = BatchScanner(pipeline=pipeline, workers=2, top_k=3)

        report = scanner.scan_images(paths)

        assert isinstance(report, ScanReport)
        assert report.total_cards == 3
        assert len(report.results) == 3
        assert report.avg_latency_ms == 10.0
        assert report.elapsed_seconds > 0
        assert report.cards_per_minute > 0
        assert pipeline.identify.call_count == 3

    def test_scan_images_preserves_order(self, tmp_path):
        pipeline = _make_mock_pipeline()
        paths = _create_test_images(tmp_path, count=5)
        scanner = BatchScanner(pipeline=pipeline, workers=2)

        report = scanner.scan_images(paths)

        for i, result in enumerate(report.results):
            assert result.image_path == str(paths[i])

    def test_scan_images_passes_top_k(self, tmp_path):
        pipeline = _make_mock_pipeline()
        paths = _create_test_images(tmp_path, count=1)
        scanner = BatchScanner(pipeline=pipeline, workers=1, top_k=10)

        scanner.scan_images(paths)

        pipeline.identify.assert_called_once()
        _, kwargs = pipeline.identify.call_args
        assert kwargs["top_k"] == 10

    def test_scan_directory(self, tmp_path):
        pipeline = _make_mock_pipeline()
        _create_test_images(tmp_path, count=4)
        scanner = BatchScanner(pipeline=pipeline, workers=2)

        report = scanner.scan_directory(tmp_path)

        assert report.total_cards == 4
        assert pipeline.identify.call_count == 4

    def test_scan_directory_ignores_non_images(self, tmp_path):
        pipeline = _make_mock_pipeline()
        _create_test_images(tmp_path, count=2)
        # Create non-image files
        (tmp_path / "readme.txt").write_text("not an image")
        (tmp_path / "data.csv").write_text("a,b,c")

        scanner = BatchScanner(pipeline=pipeline, workers=1)
        report = scanner.scan_directory(tmp_path)

        assert report.total_cards == 2

    def test_scan_directory_writes_json_report(self, tmp_path):
        pipeline = _make_mock_pipeline()
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=2)
        output_path = tmp_path / "output" / "report.json"

        scanner = BatchScanner(pipeline=pipeline, workers=1)
        scanner.scan_directory(img_dir, output_path=output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["total_cards"] == 2
        assert len(data["results"]) == 2

    def test_parallel_loading(self, tmp_path):
        """Verify that image loading happens in parallel via ThreadPoolExecutor."""
        pipeline = _make_mock_pipeline()
        paths = _create_test_images(tmp_path, count=6)
        scanner = BatchScanner(pipeline=pipeline, workers=4)

        with patch(
            "mtg_ocr.scanning.batch.ThreadPoolExecutor"
        ) as mock_executor_cls:
            # Set up the mock to actually work
            from concurrent.futures import ThreadPoolExecutor

            real_executor = ThreadPoolExecutor(max_workers=4)
            mock_executor_cls.return_value.__enter__ = MagicMock(
                return_value=real_executor
            )
            mock_executor_cls.return_value.__exit__ = MagicMock(
                return_value=False
            )

            scanner.scan_images(paths)

            mock_executor_cls.assert_called_once_with(max_workers=4)
            real_executor.shutdown(wait=False)

    def test_report_statistics_accuracy(self, tmp_path):
        """Verify that report statistics are computed correctly."""
        from mtg_ocr.models.card import IdentificationResult

        pipeline = MagicMock()
        latencies = [10.0, 20.0, 30.0]
        call_idx = 0

        def _identify(image, top_k=5):
            nonlocal call_idx
            lat = latencies[call_idx]
            call_idx += 1
            return IdentificationResult(
                matches=[], latency_ms=lat, scan_mode="rig"
            )

        pipeline.identify.side_effect = _identify

        paths = _create_test_images(tmp_path, count=3)
        scanner = BatchScanner(pipeline=pipeline, workers=1)
        report = scanner.scan_images(paths)

        assert report.total_cards == 3
        assert report.avg_latency_ms == pytest.approx(20.0)
        assert report.elapsed_seconds > 0
        assert report.cards_per_minute > 0

    def test_scan_result_contains_matches(self, tmp_path):
        """Verify each ScanResult contains the pipeline's matches."""
        pipeline = _make_mock_pipeline()
        paths = _create_test_images(tmp_path, count=1)
        scanner = BatchScanner(pipeline=pipeline, workers=1)

        report = scanner.scan_images(paths)

        assert len(report.results) == 1
        result = report.results[0]
        assert len(result.matches) == 1
        assert result.matches[0].card_name == "Lightning Bolt"
        assert result.matches[0].confidence == 0.95
