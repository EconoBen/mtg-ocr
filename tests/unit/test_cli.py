"""Tests for the mtg-ocr CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from mtg_ocr.__main__ import cli


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
class TestTopLevel:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MTG card visual identification" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "mtg-ocr" in result.output


# ---------------------------------------------------------------------------
# Subcommand --help
# ---------------------------------------------------------------------------
class TestSubcommandHelp:
    @pytest.mark.parametrize(
        "subcmd",
        [
            ["scan"],
            ["benchmark"],
            ["data"],
            ["data", "download"],
            ["embeddings"],
            ["embeddings", "build"],
            ["embeddings", "update"],
            ["embeddings", "reduce"],
            ["export"],
            ["export", "onnx"],
            ["export", "coreml"],
        ],
    )
    def test_help_works(self, runner, subcmd):
        result = runner.invoke(cli, [*subcmd, "--help"])
        assert result.exit_code == 0
        assert "--help" in result.output or "Usage" in result.output


# ---------------------------------------------------------------------------
# scan --image
# ---------------------------------------------------------------------------
class TestScanImage:
    def test_scan_image_with_mock_pipeline(self, runner, tmp_path):
        from mtg_ocr.models.card import CardMatch, IdentificationResult
        from PIL import Image

        img = Image.new("RGB", (224, 224), color="red")
        img_path = tmp_path / "card.png"
        img.save(img_path)

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_result = IdentificationResult(
            matches=[
                CardMatch(
                    scryfall_id="abc-123",
                    card_name="Lightning Bolt",
                    set_code="LEA",
                    set_name="Limited Edition Alpha",
                    confidence=0.95,
                ),
                CardMatch(
                    scryfall_id="def-456",
                    card_name="Chain Lightning",
                    set_code="LEG",
                    set_name="Legends",
                    confidence=0.82,
                ),
            ],
            latency_ms=12.5,
            scan_mode="handheld",
        )

        mock_pipeline = MagicMock()
        mock_pipeline.identify.return_value = mock_result

        with patch(
            "mtg_ocr.pipeline.CardIdentificationPipeline.from_pretrained",
            return_value=mock_pipeline,
        ):
            result = runner.invoke(
                cli, ["scan", "--image", str(img_path), "--model-dir", str(model_dir)]
            )

        assert result.exit_code == 0, result.output
        assert "Lightning Bolt" in result.output
        assert "Chain Lightning" in result.output
        assert "12.5" in result.output

    def test_scan_requires_image_or_dir(self, runner, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = runner.invoke(cli, ["scan", "--model-dir", str(model_dir)])
        assert result.exit_code != 0
        assert "Provide either --image or --dir" in result.output


# ---------------------------------------------------------------------------
# scan --dir
# ---------------------------------------------------------------------------
class TestScanDir:
    def test_scan_dir_with_mock_pipeline(self, runner, tmp_path):
        from mtg_ocr.models.card import ScanReport
        from PIL import Image

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = Image.new("RGB", (224, 224), color="blue")
        img.save(img_dir / "card1.png")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_report = ScanReport(
            results=[],
            total_cards=1,
            avg_latency_ms=15.0,
            cards_per_minute=240.0,
            elapsed_seconds=0.25,
        )

        mock_pipeline = MagicMock()
        mock_scanner = MagicMock()
        mock_scanner.scan_directory.return_value = mock_report

        with (
            patch(
                "mtg_ocr.pipeline.CardIdentificationPipeline.from_pretrained",
                return_value=mock_pipeline,
            ),
            patch(
                "mtg_ocr.scanning.batch.BatchScanner",
                return_value=mock_scanner,
            ),
        ):
            result = runner.invoke(
                cli,
                [
                    "scan",
                    "--dir", str(img_dir),
                    "--model-dir", str(model_dir),
                    "--workers", "2",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "1 cards" in result.output
        assert "240" in result.output


# ---------------------------------------------------------------------------
# data download
# ---------------------------------------------------------------------------
class TestDataDownload:
    def test_download_with_mock_client(self, runner, tmp_path):
        cache_dir = tmp_path / "cache"

        mock_client = MagicMock()
        mock_client.download_bulk_data.return_value = Path("/fake/bulk.json")

        with patch(
            "mtg_ocr.data.scryfall.ScryfallClient",
            return_value=mock_client,
        ):
            result = runner.invoke(
                cli, ["data", "download", "--cache-dir", str(cache_dir)]
            )

        assert result.exit_code == 0, result.output
        assert "Downloaded" in result.output
        mock_client.download_bulk_data.assert_called_once()


# ---------------------------------------------------------------------------
# embeddings reduce
# ---------------------------------------------------------------------------
class TestEmbeddingsReduce:
    def test_reduce_with_synthetic_data(self, runner, tmp_path):
        rng = np.random.default_rng(42)
        # Need more samples than target_dim for PCA
        embeddings = rng.standard_normal((200, 512)).astype(np.float32)
        card_ids = np.array([f"card-{i}" for i in range(200)])

        input_path = tmp_path / "embeddings.npz"
        np.savez(input_path, embeddings=embeddings, card_ids=card_ids)

        output_path = tmp_path / "reduced.npz"

        result = runner.invoke(
            cli,
            [
                "embeddings", "reduce",
                "--input", str(input_path),
                "--output", str(output_path),
                "--dim", "128",
                "--method", "pca",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "512D -> 128D" in result.output
        assert "Variance retained" in result.output

        out_data = np.load(output_path, allow_pickle=False)
        assert out_data["embeddings"].shape == (200, 128)
        assert len(out_data["card_ids"]) == 200

    def test_reduce_truncation(self, runner, tmp_path):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((20, 512)).astype(np.float32)

        input_path = tmp_path / "embeddings.npz"
        np.savez(input_path, embeddings=embeddings)

        output_path = tmp_path / "reduced.npz"

        result = runner.invoke(
            cli,
            [
                "embeddings", "reduce",
                "--input", str(input_path),
                "--output", str(output_path),
                "--dim", "256",
                "--method", "truncation",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "512D -> 256D" in result.output
        assert "truncation" in result.output


# ---------------------------------------------------------------------------
# export onnx --help / coreml --help
# ---------------------------------------------------------------------------
class TestExportHelp:
    def test_export_onnx_help(self, runner):
        result = runner.invoke(cli, ["export", "onnx", "--help"])
        assert result.exit_code == 0
        assert "--quantize" in result.output

    def test_export_coreml_help(self, runner):
        result = runner.invoke(cli, ["export", "coreml", "--help"])
        assert result.exit_code == 0
        assert "--compute-units" in result.output
