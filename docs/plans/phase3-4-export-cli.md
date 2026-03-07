# Phase 3-4: Export, Batch Scanning, CLI & Test Suite

## Context

Phase 1-2 foundation is complete (PR #1): MobileCLIP encoder, embedding search, card detection, pipeline, augmentation, Scryfall client, benchmark runner. This phase builds the export layer (ONNX/CoreML for on-device inference), batch scanning for rig mode, dimension reduction experiments, the CLI tool, and a difficult conditions test suite.

All code lives in the `mtg_ocr` package at `src/mtg_ocr/`. Tests are in `tests/`. Use TDD — write failing tests first, then implement.

**Important**: The base MobileCLIP model is sufficient for export tasks. Fine-tuning (T12) will happen on RunPod separately. Export code should work with ANY encoder that implements the `VisualEncoder` protocol.

## Validation Commands

- `uv run pytest tests/ -q`
- `uv run ruff check src/ tests/`

## Phase 1: Local Infrastructure

### Task 1: Difficult Conditions Test Suite (T15)

Create test fixtures and test cases for difficult card identification scenarios.

- [x] Create `tests/fixtures/difficult/` directory with synthetic test images generated programmatically (no real card images needed — use colored rectangles with text overlays to simulate card-like images under various conditions)
- [x] Create `tests/unit/test_difficult_conditions.py` with test cases for:
  - Foil cards (high glare simulation via augmentation)
  - Heavy blur (motion blur, out of focus)
  - Extreme rotation (>20 degrees)
  - Poor lighting (very dark, very bright)
  - Partial occlusion (finger over card edge)
  - Old border cards vs new border cards (different aspect ratios)
  - Full-art / borderless cards
- [x] Each test should use `CardAugmentation` with "heavy" severity to generate augmented versions of synthetic card images
- [x] Create a `DifficultConditionsReport` dataclass in `src/mtg_ocr/benchmark/difficult.py` that collects per-condition accuracy when run against real data
- [x] Tests should verify the augmentation pipeline handles each condition without errors and produces valid images (correct shape, dtype, value range)
- [x] Run: `uv run pytest tests/unit/test_difficult_conditions.py -v` — all tests pass

### Task 2: Batch Scanning Pipeline — Rig Mode (T18)

Build the batch scanning orchestrator for the Card Slinger rig.

- [x] Create `src/mtg_ocr/scanning/__init__.py`
- [x] Create `src/mtg_ocr/scanning/batch.py` with `BatchScanner` class:
  ```python
  class BatchScanner:
      """Batch card scanning for rig-mounted setups.

      Processes a directory of card images or a video stream,
      identifies each card, and produces a scan report.
      """
      def __init__(self, pipeline: CardIdentificationPipeline, workers: int = 4): ...
      def scan_directory(self, input_dir: Path, output_path: Path | None = None) -> ScanReport: ...
      def scan_images(self, images: list[Path]) -> ScanReport: ...
  ```
- [x] Create `ScanReport` dataclass in `src/mtg_ocr/models/card.py`:
  ```python
  class ScanResult(BaseModel):
      image_path: str
      matches: list[CardMatch]
      latency_ms: float

  class ScanReport(BaseModel):
      results: list[ScanResult]
      total_cards: int
      avg_latency_ms: float
      cards_per_minute: float
      elapsed_seconds: float
  ```
- [x] Use `concurrent.futures.ThreadPoolExecutor` for parallel image loading (I/O bound), sequential pipeline inference
- [x] Write tests in `tests/unit/test_batch_scanner.py` — mock the pipeline, verify parallel loading, report stats
- [x] Run: `uv run pytest tests/unit/test_batch_scanner.py -v` — all tests pass

### Task 3: Embedding Dimension Reduction (T10)

Implement and test dimension reduction for embeddings (512 -> 256 -> 128).

- [x] Create `src/mtg_ocr/embeddings/quantize.py` with:
  ```python
  class DimensionReducer:
      """Reduce embedding dimensions via PCA or truncation."""
      def __init__(self, method: str = "pca", target_dim: int = 256): ...
      def fit(self, embeddings: np.ndarray) -> DimensionReducer: ...
      def transform(self, embeddings: np.ndarray) -> np.ndarray: ...
      def fit_transform(self, embeddings: np.ndarray) -> np.ndarray: ...
      def save(self, path: Path) -> None: ...
      @classmethod
      def load(cls, path: Path) -> DimensionReducer: ...
  ```
- [x] Support two methods: `"truncation"` (just slice first N dims) and `"pca"` (sklearn-free PCA using numpy SVD)
- [x] Create `DimensionReductionReport` dataclass with fields: `original_dim`, `target_dim`, `method`, `variance_retained`, `file_size_reduction_pct`
- [x] Write tests in `tests/unit/test_dimension_reduction.py`:
  - Truncation preserves first N dimensions exactly
  - PCA output has correct shape and is orthogonal
  - Reduced embeddings maintain relative similarity ordering (top-K matches preserved)
  - Save/load roundtrip preserves the transform
  - Report stats are computed correctly
- [x] Run: `uv run pytest tests/unit/test_dimension_reduction.py -v` — all tests pass

## Phase 2: Export Pipeline

### Task 4: ONNX Export + INT8 Quantization (T16)

Export the MobileCLIP image encoder to ONNX format.

- [x] Create `src/mtg_ocr/export/__init__.py`
- [x] Create `src/mtg_ocr/export/onnx_export.py`:
  ```python
  class ONNXExporter:
      """Export visual encoder to ONNX format."""
      def __init__(self, encoder: VisualEncoder): ...
      def export(self, output_path: Path, opset_version: int = 17,
                 quantize: bool = False) -> ExportResult: ...
      def validate(self, onnx_path: Path, test_image: Image.Image,
                   rtol: float = 1e-3) -> bool: ...
  ```
- [x] `ExportResult` dataclass: `output_path`, `model_size_mb`, `quantized`, `opset_version`, `input_shape`, `output_shape`
- [x] Export flow: get a dummy input tensor (1, 3, 224, 224), trace the model with `torch.onnx.export`, optionally quantize with `onnxruntime.quantization`
- [x] Validation compares PyTorch output vs ONNX Runtime output on same input image (within rtol)
- [x] Write tests in `tests/unit/test_onnx_export.py`:
  - Mock the encoder's model to avoid downloading MobileCLIP in CI
  - Test export produces valid .onnx file
  - Test quantization reduces file size
  - Test validation catches mismatched outputs
- [x] Run: `uv run pytest tests/unit/test_onnx_export.py -v` — all tests pass

### Task 5: CoreML Export for iOS (T17)

Export the visual encoder to CoreML format for on-device inference.

- [x] Create `src/mtg_ocr/export/coreml_export.py`:
  ```python
  class CoreMLExporter:
      """Export visual encoder to CoreML format for iOS deployment."""
      def __init__(self, encoder: VisualEncoder): ...
      def export(self, output_path: Path,
                 compute_units: str = "ALL") -> ExportResult: ...
      def export_from_onnx(self, onnx_path: Path, output_path: Path,
                           compute_units: str = "ALL") -> ExportResult: ...
  ```
- [x] Two export paths: direct PyTorch → CoreML via `coremltools`, or ONNX → CoreML as fallback
- [x] `compute_units` options: "ALL" (CPU+GPU+ANE), "CPU_AND_GPU", "CPU_ONLY"
- [x] Write tests in `tests/unit/test_coreml_export.py`:
  - Mock the encoder to avoid model download
  - Test export produces .mlpackage or .mlmodel file
  - Test both export paths (direct and from ONNX)
  - Test compute_units parameter is passed through
- [x] Note: coremltools may not be installed in all environments — tests should skip gracefully with `pytest.importorskip("coremltools")`
- [x] Run: `uv run pytest tests/unit/test_coreml_export.py -v` — all tests pass

## Phase 3: CLI & Integration

### Task 6: CLI Tool (T21)

Build the command-line interface for all major operations.

- [ ] Update `src/mtg_ocr/__main__.py` to use `click` (add to pyproject.toml dependencies) with subcommands:
  ```
  mtg-ocr scan --image <path> [--top-k 5] [--mode handheld|rig]
  mtg-ocr scan --dir <path> [--output report.json] [--workers 4]
  mtg-ocr benchmark --corpus <dir> [--top-k 5]
  mtg-ocr data download [--cache-dir .cache/scryfall]
  mtg-ocr embeddings build --output <path> [--batch-size 64]
  mtg-ocr embeddings update --existing <path> --output <path>
  mtg-ocr embeddings reduce --input <path> --output <path> --dim 256 [--method pca]
  mtg-ocr export onnx --output <path> [--quantize]
  mtg-ocr export coreml --output <path> [--compute-units ALL]
  ```
- [ ] Each subcommand should have `--help` with clear descriptions
- [ ] `scan --image` uses `CardIdentificationPipeline.from_pretrained()` and prints results as formatted table
- [ ] `scan --dir` uses `BatchScanner` and writes JSON report
- [ ] Add `click` to pyproject.toml dependencies and add `[project.scripts]` entry: `mtg-ocr = "mtg_ocr.__main__:cli"`
- [ ] Write tests in `tests/unit/test_cli.py` using `click.testing.CliRunner`:
  - Test `--help` works for all subcommands
  - Test `scan --image` with a mock pipeline
  - Test `data download` with a mock Scryfall client
  - Test `embeddings reduce` with synthetic data
- [ ] Run: `uv run pytest tests/unit/test_cli.py -v` — all tests pass

### Task 7: Full Regression & PR

- [ ] `uv run pytest tests/ -v --tb=short` — ALL tests pass (existing + new)
- [ ] `uv run ruff check src/ tests/` — no lint errors
- [ ] Verify package installs cleanly: `uv pip install -e .`
- [ ] Verify CLI entry point works: `uv run mtg-ocr --help`
- [ ] Create PR: `gh pr create --title "feat: Phase 3-4 — export pipeline, batch scanning, CLI, test suite" --body "$(cat <<'EOF'
## Context

Phase 1-2 established the core visual identification pipeline (MobileCLIP encoder, embedding search, card detection). This PR adds the deployment and usability layers: ONNX/CoreML export for on-device inference, batch scanning for rig mode, embedding dimension reduction, a comprehensive CLI, and difficult conditions test suite.

## What changed

- **export/onnx_export.py**: ONNX export with optional INT8 quantization and validation against PyTorch output
- **export/coreml_export.py**: CoreML export for iOS (direct and ONNX-based paths)
- **scanning/batch.py**: BatchScanner for rig-mode parallel card processing with throughput reporting
- **embeddings/quantize.py**: Dimension reduction via PCA or truncation (512→256→128) with variance reporting
- **benchmark/difficult.py**: Difficult conditions framework (foil, blur, rotation, occlusion, lighting)
- **__main__.py**: Full CLI with scan, benchmark, data, embeddings, and export subcommands
- **tests/**: Comprehensive test coverage for all new modules

## Test plan

- [x] All new module tests pass
- [x] Full regression (existing + new tests) green
- [x] Ruff lint clean
- [x] CLI entry point verified

EOF
)"
- [ ] Assign PR for review
