# **Toy Project: Prism - Pipeline for Research Imaging & Scalable ML**
Goal: learn about challenges in image ML pipeline to support both research and scalable production deployment

```
┌─────────────────────────────────────────────────────────────┐
│                     CONTROL (Streamlit)                     │
│            (Dashboard + Experiment Tracking)                │
│         Optional — built last as integration layer          │
└───────────┬─────────────────────────────────────┬───────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────────┐               ┌───────────────────────┐
│    PROCESSING LAYER   │               │     TRAINING LAYER    │
│    (Local Dask +      │─────────────▶│    (PyTorch +         │
│     S3 Integration)   │  (Optimized   │     Profiler)         │
└───────────────────────┘  Data Format  └───────────────────────┘
         │                           │
         ▼                           ▼
┌───────────────────────┐     ┌───────────────────────┐
│    STORAGE (S3)       │     │   MODEL REGISTRY      │
│    - Raw: landing/    │     │   (Local MLflow +     │
│    - Processed: lmdb/ │     │    S3 checkpoints)    │
└───────────────────────┘     └───────────────────────┘
```

---

## **Project Philosophy**

Build the smallest useful system that touches every part of the problem: **ingest → process → store → train → profile**.
The goal is not a production platform but a deep understanding.

---

## **Foundation & S3 Ingestion**

### **Goal:** Get data into our system and understand the cloud storage layer.

**Configuration & secrets**
- Centralize in `prism/config.py`: S3 buckets, region, MLflow tracking URI, AWS profile
- Document required env vars in `.env.example` (e.g. `AWS_PROFILE`, `MLFLOW_TRACKING_URI`)
- Use `.env` (gitignored) for local overrides

**Environment & S3 Setup**
- Create AWS account (or use localstack for mocking)
- Set up project structure with Poetry/uv
- Create S3 buckets via boto3 script: `prism-landing/`, `prism-processed/`, `prism-model-collection/`

**S3 Utilities**
- `prism/storage/s3_client.py` — Wrapper around boto3 with retries, progress bars
- `prism/storage/download.py` — Download prefix to local cache with concurrency
- `prism/storage/upload.py` — Upload with multipart for large files  
  **Use:** Push processed outputs to `prism-processed/`, checkpoints to `prism-model-collection/`
- **Test:** Download 100+ images from a public dataset (COCO, ImageNet) to verify

**Dataset Scanner & Metadata**
- `prism/dataset/scanner.py` — Recursively scan S3 prefix, collect metadata
  - Image dimensions, format, EXIF if present
  - Store as JSONL in `prism-processed/metadata/`
- **Simple validation:** Detect corrupted images, unexpected formats
- **Deliverable:** Script that prints "Found 1,234 images, 3 corrupted, avg size 2.3MB"

**Checkpoint:** We can point at any S3 folder and get a report of what's inside.

---

## **Distributed Processing with Dask**

### **Goal:** Scale image processing across multiple cores and understand parallel bottlenecks.

Dask is chosen for DAG visibility and scaling insight; for small local runs a `ProcessPoolExecutor` could suffice, but we use Dask to learn bottlenecks and visualize stages.

**Dask Local Cluster Setup**
- `prism/processing/cluster.py` — Create local Dask cluster with configurable workers
- `prism/processing/tasks.py` — Define core operations:
  - `resize_image` (maintain aspect ratio, pad if needed)
  - `normalize` (convert to float, standardize range)
  - `basic_augment` (flip, rotate, color jitter)
- **Test:** Process 1,000 images, measure throughput

**Augmentation Pipeline**
- `prism/processing/pipeline.py` — Chain operations together
- Implement **photorealistic camera effects**:
  - Gaussian/Poisson noise models
  - Lens distortion simulation
  - Vignetting
  - Bayer pattern simulation (optional)
- **Benchmark:** Compare per-image vs batch processing speeds

**Integration with S3**
- `prism/processing/workflow.py` — Pull from S3, process, push results
- **Checkpointing:** Save intermediate results to local disk, resume on failure
- **Failure handling:** On crash, resume from last checkpoint; document recovery steps
- **Profiling:** Add timing decorators to track slow stages
- **Deliverable:** DAG visualization of processing stages with timing breakdown

**Checkpoint:** We can take raw images from S3, apply camera-accurate augmentations at scale, and see exactly where time is spent.

---

## **Storage Optimization for PyTorch**

### **Goal:** Understand why format matters and implement the fastest possible training data access.

**LMDB Deep Dive**
- `prism/storage/lmdb_writer.py` — Write images and metadata to LMDB
  - Key design: Use image hash as key, store (tensor, metadata) as value
  - **Chunking:** For datasets too large for one LMDB, write multiple envs; maintain a **manifest** (e.g. JSON) listing chunk DB paths and key ranges
- `prism/storage/lmdb_reader.py` — Random access by key, iteration  
  - **Chunked mode:** Discover chunks via manifest; iterate over combined set transparently

**Format Benchmarking**
- `prism/benchmark/format_comparison.py`
  - Compare: raw files on disk, LMDB, HDF5, Parquet
  - Metrics: read speed (sequential), random access speed, memory usage, disk usage
  - **Plot results:** "LMDB is ??x faster for random access during training"
  - **Narrative:** Frame LMDB as good enough and simpler to adopt; FFCV (Deep Dives) can be added later—document when each format wins
- Write up findings as markdown

**Conversion Pipeline**
- `prism/convert.py` (or `scripts/convert.py`) — End-to-end: S3 → Dask processing → LMDB
- Add metadata sidecar: which S3 source, processing params, version hash
- **Failure handling:** Prefer atomic writes (write to temp, then rename) or append-only with validation; document recovery if conversion is interrupted
- **Deliverable:** One command converts any S3 prefix to a training-ready LMDB dataset with full provenance

**Checkpoint:** We have a repeatable process to turn raw S3 data into optimized training datasets, with benchmarks proving our format choices.

---

## **PyTorch Integration & Profiling**

### **Goal:** Build the fastest possible DataLoader and prove it with metrics.

**Custom PyTorch Dataset**
- `prism/torch/lmdb_dataset.py` — `torch.utils.data.Dataset` over LMDB
  - Implement `__len__` and `__getitem__`
  - Add caching for frequently accessed samples
  - Support for transforms (compose with torchvision)
  - Support chunked LMDB via manifest

**DataLoader Profiler**
- `prism/torch/profiler.py` — Wrapper that measures:
  - Time to fetch batch
  - GPU idle time (if GPU available)
  - CPU utilization during loading
  - Memory usage
- **Visualization:** Simple plots showing bottleneck
- **Auto-detection:** "num_workers=4: GPU idle 45% | num_workers=12: GPU idle 12%"

**Auto-tuner**
- `prism/torch/tuner.py` — Grid search over:
  - `num_workers` (2, 4, 8, 16, 32)
  - `prefetch_factor` (2, 4, 8)
  - `pin_memory` (True/False)
  - `persistent_workers` (True/False)
- Output: Recommended config for our hardware/dataset
- **Deliverable:** "For this dataset on an r5.2xlarge, optimal config: num_workers=12, prefetch_factor=4, pin_memory=True yields 8,500 images/sec"

**GPU-native Data Pipeline (NVIDIA DALI)**
- Add a DALI-based input pipeline as an alternative to the standard PyTorch `DataLoader`
- Start simple: read images from LMDB or raw files, apply GPU-side decode/augment, and feed into PyTorch via `nvidia.dali.pytorch`
- Reuse `prism/torch/profiler.py` to compare:
  - GPU idle time (PyTorch DataLoader vs DALI)
  - Images/sec throughput and CPU utilization
- **Deliverable:** "On this dataset/hardware, DALI reduces GPU idle from X% → Y% and increases images/sec from A → B"

**Minimal Training Script**
- `scripts/train_demo.py` — One small model (e.g. ResNet classifier on a tiny dataset), one or a few epochs
  - Logs to MLflow (dataset version, DataLoader config, metrics)
  - Used by profiler and tuner to measure throughput and GPU utilization
- **Deliverable:** One command runs training with tuned DataLoader and logs metrics; closes the loop from data → training → tracking

**Checkpoint:** We can train any PyTorch model on our LMDB datasets with an optimally tuned DataLoader, and prove it.

---

## **Synthetic Camera Data & Dashboard**

### **Goal:** Play with synthetic data generation and showcase workflow expansion

**Camera Sensor Simulator**
- `prism/synthetic/sensor.py` — Generate realistic camera data
  - RAW-like Bayer patterns (RGGB)
  - Noise models: photon shot noise (Poisson), read noise (Gaussian)
  - Lens effects: vignetting, radial distortion
  - Color space conversions (RAW → RGB → sRGB)
- **Value:** Generate perfectly annotated data for scenarios where real data is scarce (low light, unusual lenses)
- **Integration:** Synthetic outputs are written to the **same LMDB format** and consumed by the **same DataLoader**; one path for real and synthetic data

**Streamlit Dashboard**
- `dashboard.py` — Simple UI (build incrementally):
  - **v1 (smallest useful):** Dataset browser (thumbnails from LMDB) + one benchmark view (format comparison or DataLoader tuning)
  - **Later:** Processing pipeline status, training run tracker (linked to MLflow)

**MLflow Integration & Polish**
- `prism/tracking/mlflow_logger.py` — Log:
  - Dataset version (hash of source + processing params)
  - DataLoader config
  - Model architecture summary
  - Final metrics
- Add `--version` flag to all scripts for reproducibility
- Write comprehensive README with architecture diagram
- **Deliverable:** End-to-end demo video (3–5 minutes) showing the whole pipeline

**Checkpoint:** A polished project with clear documentation and a compelling demo.

---

## **Deep Dives**

### **Goal:** Handle overflows

**Catch-up / Deepen**
- **FFCV integration:** Implement the ultra-fast FFCV format for comparison; document when FFCV vs LMDB wins (e.g. very large batches, GPU decoding)
- **Distributed Dask:** Try a multi-node cluster on EC2
- **More camera physics:** Add rolling shutter simulation, thermal noise
- **WebDataset / Sharded format:** Implement sharded dataset format
- **Ray + DALI training experiment:** Use Ray (e.g. Ray Train/Ray AIR) to orchestrate multi-GPU training jobs where each worker uses the DALI pipeline; compare end-to-end throughput and GPU utilization against the tuned PyTorch DataLoader baseline, and document where Ray-style orchestration is preferable to Dask in this project.

**Shard experiment (tiny-imagenet-200)**  
Use the local dataset `data/tiny-imagenet-200` (100k images) to validate a WebDataset-style sharding path: group images into 20–50 `.tar` shards (~2k images each) via Dask `bag → map_partitions → TarWriter`, so each Dask task has enough work to amortize scheduling overhead. Then load shards with `dask_image.imread` or `db.from_sequence(shard_paths).map(process_shard)`. To make this reusable we add: a small **sharding** module (path listing, partitioning, TarWriter per partition), a **script** (e.g. `scripts/shard_dataset.py`) for CLI-driven shard creation, optional **config** for paths and shard size, and **reader helpers** for Dask-based consumption. Detail plan: `sharding.md`.

---

## **Testing**

- **Foundation:** `tests/test_storage.py` (S3 client, download, upload), `tests/test_dataset.py` (scanner, metadata)
- **Processing:** `tests/test_processing.py` (tasks, pipeline, workflow checkpointing)
- **Torch:** `tests/test_torch_dataset.py` (LMDB dataset, DataLoader over small LMDB)
- Keep tests fast; use small fixtures and mocks for S3 where appropriate

---

## **Project Structure (Final)**

```
prism/
├── README.md
├── pyproject.toml
├── .env.example
├── docs/
│   ├── architecture.md
│   └── benchmarks.md
├── prism/
│   ├── __init__.py
│   ├── config.py
│   ├── storage/
│   │   ├── s3_client.py
│   │   ├── download.py
│   │   ├── upload.py
│   │   ├── lmdb_writer.py
│   │   └── lmdb_reader.py
│   ├── dataset/
│   │   └── scanner.py
│   ├── benchmark/
│   │   └── format_comparison.py
│   ├── processing/
│   │   ├── cluster.py
│   │   ├── tasks.py
│   │   ├── pipeline.py
│   │   └── workflow.py
│   ├── synthetic/
│   │   └── sensor.py
│   ├── torch/
│   │   ├── lmdb_dataset.py
│   │   ├── profiler.py
│   │   └── tuner.py
│   └── tracking/
│       └── mlflow_logger.py
├── scripts/
│   ├── ingest.py
│   ├── convert.py
│   ├── benchmark_formats.py
│   └── train_demo.py
├── dashboard.py
└── tests/
    ├── test_storage.py
    ├── test_dataset.py
    ├── test_processing.py
    └── test_torch_dataset.py
```

---

## **Success Criteria**

1. **Run one command** to take any S3 folder of images and produce an optimized LMDB dataset
2. **Prove** our format choice with benchmarks (LMDB vs alternatives)
3. **Auto-tune** a PyTorch DataLoader for maximum throughput
4. **Generate** realistic synthetic camera data with physics-based noise; feed it through the same LMDB + DataLoader path
5. **Explain** every design decision and trade-off
6. **Demo** the whole thing in under 5 minutes
