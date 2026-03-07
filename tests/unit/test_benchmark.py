"""Tests for benchmarking framework."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mtg_ocr.benchmark.runner import BenchmarkResult, BenchmarkRunner
from mtg_ocr.models.card import CardMatch, IdentificationResult


class FakePipeline:
    """Fake pipeline that returns predetermined results for testing."""

    def __init__(self, results_map: dict[str, str]) -> None:
        """results_map: filename -> scryfall_id that will be returned as top match."""
        self.results_map = results_map

    def identify(self, image: np.ndarray | Image.Image, top_k: int = 5) -> IdentificationResult:
        # Use a hash of the image to look up the predetermined result
        # In practice, we pass the expected scryfall_id via the image's info dict
        if isinstance(image, Image.Image):
            scryfall_id = image.info.get("scryfall_id", "unknown")
        else:
            scryfall_id = "unknown"

        predicted_id = self.results_map.get(scryfall_id, "wrong-id")
        matches = [
            CardMatch(
                scryfall_id=predicted_id,
                card_name=f"Card {predicted_id}",
                set_code="TST",
                set_name="Test Set",
                confidence=0.95,
            ),
        ]
        # Add some lower-confidence matches for top-5 testing
        for i in range(1, top_k):
            matches.append(
                CardMatch(
                    scryfall_id=f"other-{i}",
                    card_name=f"Other Card {i}",
                    set_code="TST",
                    set_name="Test Set",
                    confidence=0.95 - i * 0.1,
                )
            )
        return IdentificationResult(
            matches=matches,
            latency_ms=5.0,
            scan_mode="handheld",
        )


def _setup_corpus(tmp_path: Path, entries: list[dict]) -> Path:
    """Create a test corpus directory with images and ground truth."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    ground_truth = {}
    for entry in entries:
        filename = entry["filename"]
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.save(corpus_dir / filename)
        ground_truth[filename] = {
            "scryfall_id": entry["scryfall_id"],
            "card_name": entry["card_name"],
            "set_code": entry["set_code"],
        }

    with open(corpus_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f)

    return corpus_dir


class TestBenchmarkRunnerLoadsCorpus:
    def test_loads_ground_truth_and_images(self, tmp_path: Path) -> None:
        entries = [
            {"filename": "card1.png", "scryfall_id": "id-1", "card_name": "Lightning Bolt", "set_code": "M21"},
            {"filename": "card2.png", "scryfall_id": "id-2", "card_name": "Counterspell", "set_code": "MH2"},
        ]
        corpus_dir = _setup_corpus(tmp_path, entries)

        pipeline = FakePipeline({})
        runner = BenchmarkRunner(pipeline=pipeline, corpus_dir=corpus_dir)

        assert runner.total_images == 2

    def test_missing_ground_truth_raises(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "empty_corpus"
        corpus_dir.mkdir()

        pipeline = FakePipeline({})
        with pytest.raises(FileNotFoundError):
            BenchmarkRunner(pipeline=pipeline, corpus_dir=corpus_dir)


class TestAccuracyComputation:
    def test_top_1_accuracy_all_correct(self, tmp_path: Path) -> None:
        entries = [
            {"filename": "card1.png", "scryfall_id": "id-1", "card_name": "Lightning Bolt", "set_code": "M21"},
            {"filename": "card2.png", "scryfall_id": "id-2", "card_name": "Counterspell", "set_code": "MH2"},
        ]
        corpus_dir = _setup_corpus(tmp_path, entries)

        # Pipeline returns correct IDs for all cards
        pipeline = FakePipeline({"id-1": "id-1", "id-2": "id-2"})
        runner = BenchmarkRunner(pipeline=pipeline, corpus_dir=corpus_dir)
        result = runner.run()

        assert isinstance(result, BenchmarkResult)
        assert result.top_1_accuracy == 1.0
        assert result.correct_top_1 == 2
        assert result.total_images == 2

    def test_top_1_accuracy_partial(self, tmp_path: Path) -> None:
        entries = [
            {"filename": "card1.png", "scryfall_id": "id-1", "card_name": "Lightning Bolt", "set_code": "M21"},
            {"filename": "card2.png", "scryfall_id": "id-2", "card_name": "Counterspell", "set_code": "MH2"},
            {"filename": "card3.png", "scryfall_id": "id-3", "card_name": "Dark Ritual", "set_code": "A25"},
            {"filename": "card4.png", "scryfall_id": "id-4", "card_name": "Swords to Plowshares", "set_code": "2XM"},
        ]
        corpus_dir = _setup_corpus(tmp_path, entries)

        # Pipeline only gets 2 of 4 correct
        pipeline = FakePipeline({"id-1": "id-1", "id-2": "wrong", "id-3": "id-3", "id-4": "wrong"})
        runner = BenchmarkRunner(pipeline=pipeline, corpus_dir=corpus_dir)
        result = runner.run()

        assert result.top_1_accuracy == pytest.approx(0.5)
        assert result.correct_top_1 == 2
        assert len(result.failures) == 2

    def test_top_k_accuracy(self, tmp_path: Path) -> None:
        entries = [
            {"filename": "card1.png", "scryfall_id": "id-1", "card_name": "Lightning Bolt", "set_code": "M21"},
        ]
        corpus_dir = _setup_corpus(tmp_path, entries)

        # Top-1 is wrong, but we'll set up a pipeline where top-5 includes correct
        pipeline = FakePipelineWithTop5(correct_in_top_5={"id-1"})
        runner = BenchmarkRunner(pipeline=pipeline, corpus_dir=corpus_dir)
        result = runner.run()

        assert result.top_k_accuracy == 1.0
        assert result.top_1_accuracy == 0.0


class FakePipelineWithTop5:
    """Pipeline where top-1 is wrong but correct ID appears in top-5."""

    def __init__(self, correct_in_top_5: set[str]) -> None:
        self.correct_in_top_5 = correct_in_top_5

    def identify(self, image: np.ndarray | Image.Image, top_k: int = 5) -> IdentificationResult:
        if isinstance(image, Image.Image):
            scryfall_id = image.info.get("scryfall_id", "unknown")
        else:
            scryfall_id = "unknown"

        matches = [
            CardMatch(scryfall_id="wrong-top1", card_name="Wrong", set_code="X", set_name="X", confidence=0.9),
            CardMatch(scryfall_id="wrong-2", card_name="Wrong 2", set_code="X", set_name="X", confidence=0.8),
        ]
        if scryfall_id in self.correct_in_top_5:
            matches.append(
                CardMatch(scryfall_id=scryfall_id, card_name="Correct", set_code="X", set_name="X", confidence=0.7)
            )
        return IdentificationResult(matches=matches, latency_ms=5.0, scan_mode="handheld")


class TestLatencyMeasurement:
    def test_mean_and_p95_latency(self, tmp_path: Path) -> None:
        entries = [
            {"filename": f"card{i}.png", "scryfall_id": f"id-{i}", "card_name": f"Card {i}", "set_code": "TST"}
            for i in range(10)
        ]
        corpus_dir = _setup_corpus(tmp_path, entries)

        pipeline = FakePipeline({f"id-{i}": f"id-{i}" for i in range(10)})
        runner = BenchmarkRunner(pipeline=pipeline, corpus_dir=corpus_dir)
        result = runner.run()

        assert result.mean_latency_ms > 0
        assert result.p95_latency_ms > 0
        assert result.p95_latency_ms >= result.mean_latency_ms

    def test_run_latency_standalone(self, tmp_path: Path) -> None:
        entries = [
            {"filename": "card1.png", "scryfall_id": "id-1", "card_name": "Lightning Bolt", "set_code": "M21"},
        ]
        corpus_dir = _setup_corpus(tmp_path, entries)

        pipeline = FakePipeline({"id-1": "id-1"})
        runner = BenchmarkRunner(pipeline=pipeline, corpus_dir=corpus_dir)
        latency_report = runner.run_latency(n_iterations=5)

        assert "mean_latency_ms" in latency_report
        assert "p95_latency_ms" in latency_report
        assert "n_iterations" in latency_report
        assert latency_report["n_iterations"] == 5


class TestReportGeneration:
    def test_result_to_dict(self, tmp_path: Path) -> None:
        entries = [
            {"filename": "card1.png", "scryfall_id": "id-1", "card_name": "Lightning Bolt", "set_code": "M21"},
        ]
        corpus_dir = _setup_corpus(tmp_path, entries)

        pipeline = FakePipeline({"id-1": "id-1"})
        runner = BenchmarkRunner(pipeline=pipeline, corpus_dir=corpus_dir)
        result = runner.run()

        report = result.to_dict()
        assert isinstance(report, dict)
        assert "top_1_accuracy" in report
        assert "top_k_accuracy" in report
        assert "mean_latency_ms" in report
        assert "p95_latency_ms" in report
        assert "total_images" in report
        assert "failures" in report

    def test_result_to_json(self, tmp_path: Path) -> None:
        result = BenchmarkResult(
            top_1_accuracy=0.85,
            top_k_accuracy=0.95,
            mean_latency_ms=12.3,
            p95_latency_ms=18.7,
            total_images=100,
            correct_top_1=85,
            correct_top_k=95,
            failures=[],
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["top_1_accuracy"] == 0.85
        assert parsed["total_images"] == 100
