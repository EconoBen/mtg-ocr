# MTG Card Visual Identification — Phase 1 & 2: Foundation + Embedding Pipeline

## Context

We're building a visual embedding system for MTG card identification using MobileCLIP-S0 (Apple's mobile-optimized CLIP). Instead of OCR, we encode card images into embeddings and match against a pre-computed database of all ~30K MTG cards via dot product search. Target: ~15-20ms per card on iPhone.

This plan covers Phase 1 (project structure, data pipeline, encoder, search, benchmarking) and Phase 2 code (card detection, embedding computation scripts, end-to-end pipeline). RunPod GPU execution is manual — this plan creates all the scripts and infrastructure.

Reference project: `/Users/blabaschin/Documents/GitHub/mtg-vault/` uses Python 3.12, uv, pytest, httpx, pydantic, DuckDB.

## Validation Commands
- `uv run pytest tests/ -q`
- `uv run ruff check src/ tests/`
- `uv run python -c "import mtg_ocr; print(mtg_ocr.__version__)"`

## Phase 1: Foundation

### Task 1: Project Structure & Configuration (T1, bead mtg_ocr-dxo)

Write tests first (TDD), then implement.

- [x] Create `pyproject.toml` with:
  - Package name: `mtg-ocr`
  - Python requires: `>=3.12`
  - Dependencies: `torch`, `open-clip-torch`, `mobileclip`, `onnxruntime`, `httpx`, `pydantic>=2.6`, `Pillow`, `numpy`, `imagehash`
  - Optional deps group `[training]`: `albumentations`, `paddle2onnx`
  - Optional deps group `[export]`: `coremltools`, `paddle2onnx`
  - Dev deps: `pytest>=8.0`, `pytest-cov>=5.0`, `ruff>=0.8`
  - Entry point: `mtg-ocr = "mtg_ocr.__main__:main"`
  - Note: for `mobileclip`, install from Apple's repo: `pip install git+https://github.com/apple/ml-mobileclip.git`
- [x] Create package directory structure:
  ```
  src/mtg_ocr/__init__.py          # __version__ = "0.1.0"
  src/mtg_ocr/__main__.py          # CLI placeholder with argparse
  src/mtg_ocr/models/              # Pydantic models
  src/mtg_ocr/encoder/             # Visual encoder abstraction
  src/mtg_ocr/search/              # Embedding similarity search
  src/mtg_ocr/detection/           # Card region detection
  src/mtg_ocr/data/                # Scryfall data pipeline
  src/mtg_ocr/training/            # Training pipeline
  src/mtg_ocr/embeddings/          # Embedding database management
  src/mtg_ocr/export/              # ONNX/CoreML export
  src/mtg_ocr/benchmark/           # Benchmarking framework
  src/mtg_ocr/scanning/            # Scan mode orchestration
  tests/unit/__init__.py
  tests/integration/__init__.py
  tests/fixtures/                  # Test card images
  configs/                         # Training configs
  scripts/                         # RunPod scripts
  ```
- [x] Create `.gitignore` with Python defaults, model files (*.pt, *.onnx, *.mlpackage), data cache dirs, .env
- [x] Create `src/mtg_ocr/models/card.py` with Pydantic models:
  ```python
  class CardMatch(BaseModel):
      scryfall_id: str
      card_name: str
      set_code: str
      set_name: str
      confidence: float  # cosine similarity score
      image_uri: str | None = None

  class IdentificationResult(BaseModel):
      matches: list[CardMatch]
      latency_ms: float
      scan_mode: Literal["handheld", "rig"]

  class EmbeddingRecord(BaseModel):
      scryfall_id: str
      card_name: str
      set_code: str
      embedding: list[float]  # will be stored as numpy array in practice
  ```
- [x] Run: `uv venv --python python3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"` — package installs
- [x] Run: `uv run python -c "import mtg_ocr; print(mtg_ocr.__version__)"` — prints 0.1.0
- [x] Run: `uv run ruff check src/ tests/` — no lint errors
- [x] Run: `uv run pytest tests/ -q` — tests collected (even if empty)
- [x] Update bead: `bd update mtg_ocr-dxo --status closed --json`

### Task 2: Scryfall Data Pipeline (T2, bead mtg_ocr-7ut)

Write tests first (TDD), then implement. Use httpx with rate limiting (75ms between requests). Reference `/Users/blabaschin/Documents/GitHub/mtg-vault/src/mtg_vault/sources/scryfall.py` for patterns.

- [x] Write failing tests in `tests/unit/test_scryfall.py`:
  - Test bulk data URL fetch returns valid JSON structure
  - Test card dictionary building from bulk data
  - Test set code mapping
  - Test image URI extraction
  - Test rate limiting behavior (mock httpx)
- [x] Implement `src/mtg_ocr/data/scryfall.py`:
  ```python
  SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"
  RATE_LIMIT_SECONDS = 0.075

  class ScryfallClient:
      """Download and cache Scryfall bulk data for card names, set codes, image URIs."""

      def __init__(self, cache_dir: Path = Path(".cache/scryfall")):
          ...

      async def download_bulk_data(self, data_type: str = "default_cards") -> Path:
          """Download bulk data JSON file. Cache locally."""
          ...

      def build_card_dictionary(self, bulk_data_path: Path) -> dict[str, CardInfo]:
          """Build card name -> CardInfo mapping from bulk data.
          CardInfo includes: scryfall_id, name, set_code, set_name, image_uris, collector_number
          """
          ...

      def get_image_uris(self, cards: dict) -> list[tuple[str, str]]:
          """Return list of (scryfall_id, normal_image_uri) for embedding computation."""
          ...
  ```
- [x] Create `src/mtg_ocr/data/models.py` with `CardInfo` dataclass
- [x] Run: `uv run pytest tests/unit/test_scryfall.py -v` — all tests pass
- [x] Update bead: `bd update mtg_ocr-7ut --status closed --json`

### Task 3: Visual Encoder Abstraction + MobileCLIP Wrapper (T3, bead mtg_ocr-xxq)

Write tests first (TDD), then implement.

- [x] Write failing tests in `tests/unit/test_encoder.py`:
  - Test encoder protocol interface
  - Test MobileCLIP encoder loads model
  - Test encode_image returns correct shape tensor
  - Test encode_images batch processing
  - Test embedding normalization (unit vectors)
- [x] Implement `src/mtg_ocr/encoder/base.py`:
  ```python
  from typing import Protocol
  import numpy as np
  from PIL import Image

  class VisualEncoder(Protocol):
      """Protocol for visual encoding models."""
      @property
      def embedding_dim(self) -> int: ...
      def encode_image(self, image: Image.Image) -> np.ndarray: ...
      def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray: ...
  ```
- [x] Implement `src/mtg_ocr/encoder/mobileclip.py`:
  ```python
  class MobileCLIPEncoder:
      """MobileCLIP-S0 visual encoder wrapper.

      Uses Apple's ml-mobileclip for mobile-optimized image embeddings.
      Model: MobileCLIP-S0, ~50-80MB, 3-15ms inference on iPhone.
      """

      def __init__(self, model_name: str = "mobileclip_s0", pretrained: str = "datacompdr"):
          # Load model via open_clip or mobileclip package
          ...

      def encode_image(self, image: Image.Image) -> np.ndarray:
          """Encode single image to normalized embedding vector."""
          ...

      def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
          """Batch encode images. Returns (N, embedding_dim) array."""
          ...
  ```
  Note: MobileCLIP may need to be installed from Apple's GitHub repo. If `mobileclip` package is not available via pip, use `open_clip_torch` with a compatible architecture, or download and load the checkpoint directly. Check https://github.com/apple/ml-mobileclip for the latest installation instructions. The key is to get a working image encoder that produces embeddings — if MobileCLIP-S0 specifically is hard to install, fall back to `open_clip` with `ViT-B-32` as a temporary placeholder and document the fallback.
- [x] Run: `uv run pytest tests/unit/test_encoder.py -v` — all tests pass
- [x] Update bead: `bd update mtg_ocr-xxq --status closed --json`

### Task 4: Embedding Similarity Search (T4, bead mtg_ocr-z27)

Write tests first (TDD), then implement.

- [x] Write failing tests in `tests/unit/test_search.py`:
  - Test loading embeddings from file
  - Test dot product search returns correct top-K
  - Test FP16 quantization roundtrip
  - Test search with normalized vectors (cosine similarity)
  - Test empty database handling
  - Test search latency under 10ms for 30K embeddings (synthetic data)
- [x] Implement `src/mtg_ocr/search/similarity.py`:
  ```python
  class EmbeddingIndex:
      """Exact nearest-neighbor search via dot product.

      Stores pre-computed card embeddings as FP16 numpy array.
      For 30K cards x 512 dims = ~30MB (FP16).
      Search is simple matrix multiplication — fast on CPU.
      """

      def __init__(self):
          self.embeddings: np.ndarray | None = None  # (N, D) FP16
          self.card_ids: list[str] = []  # scryfall_ids, aligned with embeddings
          self.metadata: dict[str, CardInfo] = {}  # scryfall_id -> card info

      def load(self, path: Path) -> None:
          """Load pre-computed embeddings from .npz file."""
          ...

      def save(self, path: Path) -> None:
          """Save embeddings to .npz file with FP16 quantization."""
          ...

      def add(self, scryfall_id: str, embedding: np.ndarray, card_info: CardInfo) -> None:
          """Add a single card embedding."""
          ...

      def search(self, query: np.ndarray, top_k: int = 5) -> list[CardMatch]:
          """Find top-K most similar cards by cosine similarity (dot product on normalized vectors)."""
          # scores = query @ self.embeddings.T  # (D,) @ (D, N) = (N,)
          ...

      def build_from_arrays(self, embeddings: np.ndarray, card_ids: list[str], metadata: dict) -> None:
          """Build index from pre-computed arrays."""
          ...
  ```
- [x] Run: `uv run pytest tests/unit/test_search.py -v` — all tests pass
- [x] Update bead: `bd update mtg_ocr-z27 --status closed --json`

### Task 5: Benchmarking Framework (T5, bead mtg_ocr-xrp)

Write tests first (TDD), then implement.

- [x] Write failing tests in `tests/unit/test_benchmark.py`:
  - Test benchmark runner loads test corpus
  - Test accuracy computation (top-1, top-5)
  - Test latency measurement (mean, P95)
  - Test report generation (dict/JSON output)
- [x] Implement `src/mtg_ocr/benchmark/runner.py`:
  ```python
  @dataclass
  class BenchmarkResult:
      top_1_accuracy: float
      top_5_accuracy: float
      mean_latency_ms: float
      p95_latency_ms: float
      total_images: int
      correct_top_1: int
      correct_top_5: int
      failures: list[dict]  # {image_path, expected, predicted, confidence}

  class BenchmarkRunner:
      """Run accuracy and latency benchmarks on a test corpus.

      Test corpus format: directory of images with ground_truth.json
      mapping filename -> {scryfall_id, card_name, set_code}
      """

      def __init__(self, pipeline, corpus_dir: Path):
          ...

      def run(self) -> BenchmarkResult:
          """Run full benchmark on corpus."""
          ...

      def run_latency(self, n_iterations: int = 100) -> dict:
          """Run latency-only benchmark."""
          ...
  ```
- [x] Create `tests/fixtures/ground_truth.json` with a small sample (use 5-10 placeholder entries — actual card images will be added manually)
- [x] Run: `uv run pytest tests/unit/test_benchmark.py -v` — all tests pass
- [x] Update bead: `bd update mtg_ocr-xrp --status closed --json`

## Phase 2: Embedding Database & Pipeline

### Task 6: Card Detection — OpenCV Contours + Perspective Correction (T7, bead mtg_ocr-4g7)

Write tests first (TDD), then implement. This is T7 in the plan but can run in parallel with T6.

- [x] Write failing tests in `tests/unit/test_detection.py`:
  - Test card contour detection on a synthetic test image (white rectangle on dark background)
  - Test perspective correction produces upright rectangle
  - Test ScanMode enum (handheld, rig)
  - Test no card found returns None
  - Test multiple cards detected returns largest
- [x] Implement `src/mtg_ocr/detection/card_detector.py`:
  ```python
  from enum import StrEnum

  class ScanMode(StrEnum):
      HANDHELD = "handheld"
      RIG = "rig"

  class CardDetector:
      """Detect MTG card rectangle in an image using OpenCV contour detection.

      Handheld mode: adaptive thresholding, perspective correction, handles variable lighting.
      Rig mode: simpler detection assuming consistent positioning and lighting.
      """

      def __init__(self, scan_mode: ScanMode = ScanMode.HANDHELD):
          self.scan_mode = scan_mode

      def detect(self, image: np.ndarray) -> DetectionResult | None:
          """Find the card in the image. Returns crop + metadata or None."""
          ...

      def _detect_handheld(self, image: np.ndarray) -> DetectionResult | None:
          """Adaptive detection for variable conditions."""
          # 1. Convert to grayscale
          # 2. Gaussian blur
          # 3. Adaptive threshold or Canny edge detection
          # 4. Find contours, filter by area and aspect ratio (MTG cards are 63x88mm, ratio ~0.716)
          # 5. Find largest qualifying contour
          # 6. Perspective transform to upright rectangle
          ...

      def _detect_rig(self, image: np.ndarray) -> DetectionResult | None:
          """Simplified detection for controlled rig environment."""
          # Fixed crop region or simple threshold — lighting is consistent
          ...

  @dataclass
  class DetectionResult:
      card_image: np.ndarray  # Cropped, perspective-corrected card image
      confidence: float
      bounding_box: tuple[int, int, int, int]  # x, y, w, h in original image
      scan_mode: ScanMode
  ```
- [x] Create a synthetic test image in `tests/fixtures/` — a white rectangle (63x88 ratio) on dark background, for unit testing
- [x] Run: `uv run pytest tests/unit/test_detection.py -v` — all tests pass
- [x] Update bead: `bd update mtg_ocr-4g7 --status closed --json`

### Task 7: Embedding Computation Script for RunPod (T6, bead mtg_ocr-llx)

Write the script and infrastructure for computing embeddings on RunPod GPU. The actual RunPod execution is manual.

- [ ] Write failing tests in `tests/unit/test_embeddings.py`:
  - Test EmbeddingBuilder processes a batch of images
  - Test embeddings are normalized (unit length)
  - Test save/load roundtrip with FP16 quantization
  - Test incremental update (add new cards to existing database)
- [ ] Implement `src/mtg_ocr/embeddings/builder.py`:
  ```python
  class EmbeddingBuilder:
      """Build card embedding database from Scryfall images.

      Downloads card images, runs them through the encoder, and saves
      the embedding database as a quantized numpy file.
      """

      def __init__(self, encoder: VisualEncoder, scryfall_client: ScryfallClient):
          ...

      async def build(self, output_path: Path, batch_size: int = 64) -> EmbeddingStats:
          """Build complete embedding database.
          1. Get all card image URIs from Scryfall
          2. Download images (with rate limiting)
          3. Encode in batches
          4. Save as FP16 .npz
          """
          ...

      async def update(self, existing_path: Path, output_path: Path) -> EmbeddingStats:
          """Incrementally update embeddings with new cards only."""
          ...
  ```
- [ ] Create `scripts/compute_embeddings.py` — standalone RunPod script:
  ```python
  """Compute card embeddings on RunPod GPU.

  Usage:
      python scripts/compute_embeddings.py --output data/embeddings.npz --batch-size 128

  Requires GPU for efficient batch processing of ~30K images.
  Estimated time: ~15-30 min on A40, cost ~$1-2.
  """
  ```
- [ ] Run: `uv run pytest tests/unit/test_embeddings.py -v` — all tests pass
- [ ] Update bead: `bd update mtg_ocr-llx --status closed --json`

### Task 8: End-to-End Pipeline (T8, bead mtg_ocr-bhe)

Write tests first (TDD), then implement. This wires everything together.

- [ ] Write failing tests in `tests/unit/test_pipeline.py`:
  - Test pipeline initialization with all components
  - Test identify() with a mock encoder and pre-built index
  - Test identify() returns CardMatch list sorted by confidence
  - Test identify_batch() for rig mode
  - Test pipeline handles "no card detected" gracefully
- [ ] Implement `src/mtg_ocr/pipeline.py`:
  ```python
  class CardIdentificationPipeline:
      """End-to-end card identification pipeline.

      photo → detect card → crop/normalize → encode → search → CardMatch results
      """

      def __init__(
          self,
          encoder: VisualEncoder,
          index: EmbeddingIndex,
          detector: CardDetector,
      ):
          ...

      def identify(self, image: np.ndarray | Image.Image, top_k: int = 5) -> IdentificationResult:
          """Identify a card from an image.
          1. Detect card region
          2. Crop and normalize to 224x224
          3. Encode with MobileCLIP
          4. Search embedding index
          5. Return top-K matches with confidence and latency
          """
          ...

      def identify_batch(self, images: list[np.ndarray], top_k: int = 5) -> list[IdentificationResult]:
          """Batch identification for rig mode."""
          ...

      @classmethod
      def from_pretrained(cls, model_dir: Path, scan_mode: ScanMode = ScanMode.HANDHELD) -> "CardIdentificationPipeline":
          """Load pipeline from a directory containing model + embeddings."""
          ...
  ```
- [ ] Run: `uv run pytest tests/unit/test_pipeline.py -v` — all tests pass
- [ ] Run: `uv run pytest tests/ -q` — ALL tests pass (full suite regression)
- [ ] Run: `uv run ruff check src/ tests/` — no lint errors
- [ ] Update bead: `bd update mtg_ocr-bhe --status closed --json`

### Task 9: Training Data Preparation Script (T11, bead mtg_ocr-qwk)

Write the augmentation pipeline and training data preparation. This creates the infrastructure for Phase 3 fine-tuning.

- [ ] Write failing tests in `tests/unit/test_training.py`:
  - Test augmentation pipeline produces valid images
  - Test triplet generation (anchor, positive, negative)
  - Test augmentation includes glare, blur, angle, foil simulation
- [ ] Implement `src/mtg_ocr/training/augmentation.py`:
  ```python
  class CardAugmentation:
      """Image augmentation pipeline for training data.

      Simulates real-world conditions: glare, blur, rotation,
      perspective distortion, lighting variation, foil reflections.
      """

      def __init__(self, severity: str = "medium"):
          # severity: "light", "medium", "heavy"
          ...

      def __call__(self, image: np.ndarray) -> np.ndarray:
          """Apply random augmentation to a card image."""
          ...
  ```
- [ ] Implement `src/mtg_ocr/training/dataset.py`:
  ```python
  class CardTripletDataset:
      """Generate training triplets for contrastive learning.

      Each triplet: (anchor_image, positive_variant, negative_card)
      - anchor: original card image
      - positive: same card, different augmentation (or different printing)
      - negative: different card entirely
      """
      ...
  ```
- [ ] Create `scripts/prepare_training_data.py` — downloads images, generates augmented triplets
- [ ] Run: `uv run pytest tests/unit/test_training.py -v` — all tests pass
- [ ] Update bead: `bd update mtg_ocr-qwk --status closed --json`

### Task 10: Final Validation & PR

- [ ] Run: `uv run pytest tests/ -v --tb=short` — all tests pass
- [ ] Run: `uv run ruff check src/ tests/` — no lint errors
- [ ] Run: `uv run python -c "import mtg_ocr; print(mtg_ocr.__version__)"` — prints 0.1.0
- [ ] Create PR: `gh pr create --title "feat: Phase 1-2 foundation — visual encoder, embedding pipeline, card detection" --body "## Context\n\nFoundation for MTG card visual identification system. Implements MobileCLIP-S0 encoder, embedding similarity search, card detection, Scryfall data pipeline, and end-to-end identification pipeline.\n\n## What changed\n\n- Project structure with full package layout\n- Scryfall bulk data pipeline for card dictionary\n- MobileCLIP-S0 visual encoder wrapper\n- Embedding similarity search (exact NN, FP16)\n- OpenCV card detection with perspective correction\n- End-to-end identification pipeline\n- Benchmarking framework\n- Training data augmentation pipeline\n- RunPod embedding computation script\n\n## Test plan\n\n- [x] All unit tests pass\n- [x] Ruff lint clean\n- [x] Package installs and imports\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"`
