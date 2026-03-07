"""Benchmarking framework for card identification accuracy and latency."""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class BenchmarkResult:
    top_1_accuracy: float
    top_5_accuracy: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_images: int
    correct_top_1: int
    correct_top_5: int
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BenchmarkRunner:
    """Run accuracy and latency benchmarks on a test corpus.

    Test corpus format: directory of images with ground_truth.json
    mapping filename -> {scryfall_id, card_name, set_code}
    """

    def __init__(self, pipeline: Any, corpus_dir: Path) -> None:
        self.pipeline = pipeline
        self.corpus_dir = Path(corpus_dir)

        gt_path = self.corpus_dir / "ground_truth.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        with open(gt_path) as f:
            self.ground_truth: dict[str, dict[str, str]] = json.load(f)

        self._image_files = [
            fname for fname in self.ground_truth
            if (self.corpus_dir / fname).exists()
        ]

    @property
    def total_images(self) -> int:
        return len(self._image_files)

    def run(self, top_k: int = 5) -> BenchmarkResult:
        """Run full benchmark on corpus.

        Args:
            top_k: Number of top matches to retrieve for accuracy evaluation.
        """
        correct_top_1 = 0
        correct_top_5 = 0
        latencies: list[float] = []
        failures: list[dict[str, Any]] = []

        for filename in self._image_files:
            gt = self.ground_truth[filename]
            expected_id = gt["scryfall_id"]

            with Image.open(self.corpus_dir / filename) as img:
                img.info["scryfall_id"] = expected_id

                start = time.perf_counter()
                result = self.pipeline.identify(img, top_k=top_k)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

                top_ids = [m.scryfall_id for m in result.matches]

                if top_ids and top_ids[0] == expected_id:
                    correct_top_1 += 1

                if expected_id in top_ids:
                    correct_top_5 += 1
                else:
                    failures.append({
                        "image_path": filename,
                        "expected": expected_id,
                        "predicted": top_ids[0] if top_ids else None,
                        "confidence": result.matches[0].confidence if result.matches else 0.0,
                    })

        total = self.total_images
        latency_arr = np.array(latencies) if latencies else np.array([0.0])

        return BenchmarkResult(
            top_1_accuracy=correct_top_1 / total if total > 0 else 0.0,
            top_5_accuracy=correct_top_5 / total if total > 0 else 0.0,
            mean_latency_ms=float(np.mean(latency_arr)),
            p95_latency_ms=float(np.percentile(latency_arr, 95)),
            total_images=total,
            correct_top_1=correct_top_1,
            correct_top_5=correct_top_5,
            failures=failures,
        )

    def run_latency(self, n_iterations: int = 100, top_k: int = 5) -> dict[str, Any]:
        """Run latency-only benchmark."""
        if not self._image_files or n_iterations <= 0:
            return {"mean_latency_ms": 0.0, "p95_latency_ms": 0.0, "n_iterations": 0}

        # Use first image for repeated latency measurement
        filename = self._image_files[0]
        gt = self.ground_truth[filename]
        with Image.open(self.corpus_dir / filename) as img:
            img.info["scryfall_id"] = gt["scryfall_id"]
            # Load image data before entering measurement loop
            img.load()

            latencies: list[float] = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                self.pipeline.identify(img, top_k=top_k)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

        latency_arr = np.array(latencies)
        return {
            "mean_latency_ms": float(np.mean(latency_arr)),
            "p95_latency_ms": float(np.percentile(latency_arr, 95)),
            "n_iterations": n_iterations,
        }
