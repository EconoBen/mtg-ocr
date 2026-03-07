"""Batch card scanning for rig-mounted setups."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from mtg_ocr.models.card import ScanReport, ScanResult
from mtg_ocr.pipeline import CardIdentificationPipeline


def _load_image(path: Path) -> tuple[Path, Image.Image]:
    """Load a single image from disk (I/O bound, suitable for threading)."""
    with Image.open(path) as im:
        img = im.convert("RGB").copy()
    return path, img


class BatchScanner:
    """Batch card scanning for rig-mounted setups.

    Processes a directory of card images or a list of image paths,
    identifies each card, and produces a scan report.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        pipeline: CardIdentificationPipeline,
        workers: int = 4,
        top_k: int = 5,
    ) -> None:
        self.pipeline = pipeline
        self.workers = workers
        self.top_k = top_k

    def scan_directory(
        self, input_dir: Path, output_path: Path | None = None
    ) -> ScanReport:
        """Scan all images in a directory.

        Args:
            input_dir: Directory containing card images.
            output_path: Optional path to write JSON report.

        Returns:
            ScanReport with results and aggregate statistics.
        """
        input_dir = Path(input_dir)
        image_paths = sorted(
            p
            for p in input_dir.iterdir()
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        report = self.scan_images(image_paths)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report.model_dump_json(indent=2))

        return report

    def scan_images(self, images: list[Path]) -> ScanReport:
        """Scan a list of image paths.

        Uses ThreadPoolExecutor for parallel image loading (I/O bound),
        then sequential pipeline inference.

        Args:
            images: List of image file paths.

        Returns:
            ScanReport with per-image results and aggregate stats.
        """
        if not images:
            return ScanReport(
                results=[],
                total_cards=0,
                avg_latency_ms=0.0,
                cards_per_minute=0.0,
                elapsed_seconds=0.0,
            )

        wall_start = time.perf_counter()

        # Parallel image loading
        loaded: dict[Path, Image.Image] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for path, img in executor.map(_load_image, images):
                loaded[path] = img

        # Sequential pipeline inference
        results: list[ScanResult] = []
        for path in images:
            img = loaded[path]
            result = self.pipeline.identify(img, top_k=self.top_k)
            results.append(
                ScanResult(
                    image_path=str(path),
                    matches=result.matches,
                    latency_ms=result.latency_ms,
                )
            )

        wall_elapsed = time.perf_counter() - wall_start

        total_cards = len(results)
        avg_latency = (
            sum(r.latency_ms for r in results) / total_cards
            if total_cards > 0
            else 0.0
        )
        cards_per_minute = (
            (total_cards / wall_elapsed) * 60.0 if wall_elapsed > 0 else 0.0
        )

        return ScanReport(
            results=results,
            total_cards=total_cards,
            avg_latency_ms=avg_latency,
            cards_per_minute=cards_per_minute,
            elapsed_seconds=wall_elapsed,
        )
