"""MTG OCR CLI — card identification, batch scanning, benchmarking, and model export."""

from __future__ import annotations

from pathlib import Path

import click

from mtg_ocr import __version__


@click.group()
@click.version_option(version=__version__, prog_name="mtg-ocr")
def cli() -> None:
    """MTG card visual identification using MobileCLIP embeddings."""


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--image", "image_path", type=click.Path(exists=True, path_type=Path), help="Path to a single card image.")
@click.option("--dir", "dir_path", type=click.Path(exists=True, file_okay=False, path_type=Path), help="Directory of card images (rig mode).")
@click.option("--top-k", default=5, show_default=True, help="Number of top matches to return.")
@click.option("--mode", type=click.Choice(["handheld", "rig"]), default="handheld", show_default=True, help="Detection mode.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Output JSON report path (for --dir).")
@click.option("--workers", default=4, show_default=True, help="Parallel image-loading workers (for --dir).")
@click.option("--model-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing embeddings.npz and model data.")
def scan(image_path, dir_path, top_k, mode, output, workers, model_dir) -> None:
    """Identify cards from images or scan a directory."""
    if not image_path and not dir_path:
        raise click.UsageError("Provide either --image or --dir.")

    from mtg_ocr.detection.card_detector import ScanMode
    from mtg_ocr.pipeline import CardIdentificationPipeline

    scan_mode = ScanMode.RIG if mode == "rig" else ScanMode.HANDHELD
    pipeline = CardIdentificationPipeline.from_pretrained(model_dir, scan_mode=scan_mode)

    if image_path:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        result = pipeline.identify(img, top_k=top_k)

        if not result.matches:
            click.echo("No card detected.")
            return

        click.echo(f"{'Rank':<6}{'Confidence':<12}{'Card Name':<40}{'Set':<10}")
        click.echo("-" * 68)
        for i, m in enumerate(result.matches, 1):
            click.echo(f"{i:<6}{m.confidence:<12.4f}{m.card_name:<40}{m.set_code:<10}")
        click.echo(f"\nLatency: {result.latency_ms:.1f} ms")

    elif dir_path:
        from mtg_ocr.scanning.batch import BatchScanner

        scanner = BatchScanner(pipeline, workers=workers, top_k=top_k)
        report = scanner.scan_directory(dir_path, output_path=output)

        click.echo(f"Scanned {report.total_cards} cards in {report.elapsed_seconds:.1f}s")
        click.echo(f"Avg latency: {report.avg_latency_ms:.1f} ms")
        click.echo(f"Throughput: {report.cards_per_minute:.0f} cards/min")
        if output:
            click.echo(f"Report saved to {output}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--corpus", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Test corpus directory with ground_truth.json.")
@click.option("--top-k", default=5, show_default=True, help="Top-K for accuracy measurement.")
@click.option("--model-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing embeddings.npz and model data.")
def benchmark(corpus, top_k, model_dir) -> None:
    """Run accuracy and latency benchmarks on a test corpus."""
    from mtg_ocr.benchmark.runner import BenchmarkRunner
    from mtg_ocr.pipeline import CardIdentificationPipeline

    pipeline = CardIdentificationPipeline.from_pretrained(model_dir)
    runner = BenchmarkRunner(pipeline, corpus)
    result = runner.run()

    click.echo(f"Images: {result.total_images}")
    click.echo(f"Top-1 accuracy: {result.top_1_accuracy:.2%}")
    click.echo(f"Top-5 accuracy: {result.top_5_accuracy:.2%}")
    click.echo(f"Mean latency: {result.mean_latency_ms:.1f} ms")
    click.echo(f"P95 latency: {result.p95_latency_ms:.1f} ms")

    if result.failures:
        click.echo(f"\n{len(result.failures)} failures:")
        for f in result.failures[:10]:
            click.echo(f"  {f['image_path']}: expected={f['expected']}, got={f['predicted']}")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@cli.group()
def data() -> None:
    """Manage Scryfall card data."""


@data.command()
@click.option("--cache-dir", type=click.Path(path_type=Path), default=".cache/scryfall", show_default=True, help="Local cache directory.")
def download(cache_dir) -> None:
    """Download Scryfall bulk data for card identification."""
    from mtg_ocr.data.scryfall import ScryfallClient

    client = ScryfallClient(cache_dir=cache_dir)
    click.echo("Downloading Scryfall bulk data...")
    path = client.download_bulk_data()
    click.echo(f"Downloaded to {path}")


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------
@cli.group()
def embeddings() -> None:
    """Build and manage card embedding databases."""


@embeddings.command()
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Output path for embeddings .npz file.")
@click.option("--batch-size", default=64, show_default=True, help="Batch size for encoding.")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=".cache/scryfall", show_default=True, help="Scryfall cache directory.")
def build(output, batch_size, cache_dir) -> None:
    """Build a complete embedding database from Scryfall card images."""
    from mtg_ocr.data.scryfall import ScryfallClient
    from mtg_ocr.embeddings.builder import EmbeddingBuilder
    from mtg_ocr.encoder.mobileclip import MobileCLIPEncoder

    encoder = MobileCLIPEncoder()
    client = ScryfallClient(cache_dir=cache_dir)
    builder = EmbeddingBuilder(encoder, client)

    click.echo("Building embedding database (this may take a while)...")
    stats = builder.build(output, batch_size=batch_size)

    click.echo(f"Built {stats.total_cards} card embeddings ({stats.embedding_dim}D)")
    click.echo(f"File size: {stats.file_size_mb:.1f} MB")


@embeddings.command()
@click.option("--existing", type=click.Path(exists=True, path_type=Path), required=True, help="Existing embeddings .npz file.")
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Output path for updated embeddings.")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=".cache/scryfall", show_default=True, help="Scryfall cache directory.")
def update(existing, output, cache_dir) -> None:
    """Incrementally update embeddings with new cards."""
    from mtg_ocr.data.scryfall import ScryfallClient
    from mtg_ocr.embeddings.builder import EmbeddingBuilder
    from mtg_ocr.encoder.mobileclip import MobileCLIPEncoder

    encoder = MobileCLIPEncoder()
    client = ScryfallClient(cache_dir=cache_dir)
    builder = EmbeddingBuilder(encoder, client)

    click.echo("Updating embedding database...")
    stats = builder.update(existing, output)

    click.echo(f"Total: {stats.total_cards} cards, {stats.new_cards} new, {stats.skipped_cards} skipped")
    click.echo(f"File size: {stats.file_size_mb:.1f} MB")


@embeddings.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True, help="Input embeddings .npz file.")
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Output path for reduced embeddings.")
@click.option("--dim", type=int, required=True, help="Target embedding dimension.")
@click.option("--method", type=click.Choice(["pca", "truncation"]), default="pca", show_default=True, help="Reduction method.")
def reduce(input_path, output, dim, method) -> None:
    """Reduce embedding dimensions (e.g. 512 -> 256)."""
    import numpy as np

    from mtg_ocr.embeddings.quantize import DimensionReducer

    click.echo(f"Loading embeddings from {input_path}...")
    data = np.load(input_path, allow_pickle=False)
    source_embeddings = data["embeddings"].astype(np.float32)

    reducer = DimensionReducer(method=method, target_dim=dim)
    reduced = reducer.fit_transform(source_embeddings)
    report = reducer.report()

    # Save reduced embeddings, preserving card_ids
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"embeddings": reduced}
    if "card_ids" in data:
        save_dict["card_ids"] = data["card_ids"]
    np.savez(output, **save_dict)

    click.echo(f"Reduced {report.original_dim}D -> {report.target_dim}D ({report.method})")
    click.echo(f"Variance retained: {report.variance_retained:.2%}")
    click.echo(f"File size reduction: {report.file_size_reduction_pct:.1f}%")
    click.echo(f"Saved to {output}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
@cli.group()
def export() -> None:
    """Export models to ONNX or CoreML formats."""


@export.command(name="onnx")
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Output .onnx file path.")
@click.option("--quantize", is_flag=True, help="Apply INT8 dynamic quantization.")
@click.option("--opset", default=17, show_default=True, help="ONNX opset version.")
def export_onnx(output, quantize, opset) -> None:
    """Export visual encoder to ONNX format."""
    from mtg_ocr.encoder.mobileclip import MobileCLIPEncoder
    from mtg_ocr.export.onnx_export import ONNXExporter

    click.echo("Loading encoder...")
    encoder = MobileCLIPEncoder()
    exporter = ONNXExporter(encoder)

    click.echo(f"Exporting to ONNX (opset={opset}, quantize={quantize})...")
    result = exporter.export(output, opset_version=opset, quantize=quantize)

    click.echo(f"Exported to {result.output_path}")
    click.echo(f"Model size: {result.model_size_mb:.1f} MB")
    click.echo(f"Input shape: {result.input_shape}")
    click.echo(f"Output shape: {result.output_shape}")


@export.command(name="coreml")
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Output .mlpackage path.")
@click.option("--compute-units", type=click.Choice(["ALL", "CPU_AND_GPU", "CPU_ONLY"]), default="ALL", show_default=True, help="Target compute units.")
def export_coreml(output, compute_units) -> None:
    """Export visual encoder to CoreML format for iOS."""
    from mtg_ocr.encoder.mobileclip import MobileCLIPEncoder
    from mtg_ocr.export.coreml_export import CoreMLExporter

    click.echo("Loading encoder...")
    encoder = MobileCLIPEncoder()
    exporter = CoreMLExporter(encoder)

    click.echo(f"Exporting to CoreML (compute_units={compute_units})...")
    result = exporter.export(output, compute_units=compute_units)

    click.echo(f"Exported to {result.output_path}")
    click.echo(f"Model size: {result.model_size_mb:.1f} MB")
    click.echo(f"Input shape: {result.input_shape}")
    click.echo(f"Output shape: {result.output_shape}")


if __name__ == "__main__":
    cli()
