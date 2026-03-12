# Prism
Toy project to learn about challenges in image ML data pipeline to support both research and scalable production deployment.

**ingest → process → store → train → profile**

## Vision and plan

See [vision.md](vision.md) for the full roadmap (LMDB, PyTorch DataLoader, synthetic data, dashboard, etc.). Sharding experiment detail: [sharding.md](sharding.md).

## Setup

```bash
cp .env.example .env   # edit with your AWS profile or keys
uv sync                # or: poetry install
```

## Implemented

### S3 and ingestion

- **Buckets:** `uv run python scripts/create_buckets.py`
- **Scan a prefix:** `uv run python scripts/ingest.py s3://your-bucket/prefix/`  
  Output: image count, corrupted count, average size; metadata JSONL in `prism-processed/metadata/`.
- **Storage:** `prism/storage/` — S3 client, download (single and prefix), upload (single and bytes).
- **Metadata registry:** `prism/dataset/registry.py` — append each scan to a small JSON registry in S3 so you can see recent runs.

### Processing workflow

Two modes, selected automatically by source layout or via flags:

- **Flat:** List image keys under an S3 prefix → download → run pipeline → upload per image. Checkpoint: `downloaded_keys`, `processed_keys`.
- **Sharded:** Source is WebDataset-style: `metadata/*.parquet` + `.tar` shards. Workflow loads the Parquet manifest, processes only needed samples from each shard, writes processed images into **one output .tar per shard** and uploads that tar (one PUT per shard). Checkpoint: `processed_keys` by `__key__`. Supports **Dask**: one task per shard; workers download their shard, process, build the tar, upload. Progress bar with configurable chunked updates (`progress_chunk_size`, default 100).

**Run:**

```bash
# Flat source (many .jpg under prefix)
uv run python scripts/run_workflow.py s3://prism-landing/my-dataset/

# Sharded source (metadata + .tar; auto-detected)
uv run python scripts/run_workflow.py s3://prism-processed/sharded/train/

# Force sharded or flat
uv run python scripts/run_workflow.py s3://... --sharded
uv run python scripts/run_workflow.py s3://... --no-sharded

# Sequential (no Dask), custom pipeline
uv run python scripts/run_workflow.py s3://... --no-dask --pipeline config/pipeline.yaml
```

**Resume:** Re-run the same bucket/prefix; progress is resumed from the checkpoint. To run again from scratch, remove the checkpoint (or the whole job dir) under the cache — see [Checkpoints](#checkpoints) below.

**Modules:**

- `prism/processing/workflow.py` — Flat workflow and single entry point `run_workflow()`.
- `prism/processing/workflow_sharded.py` — Sharded workflow: metadata load, one task per shard (sequential or Dask), output one .tar per shard.
- `prism/processing/pipeline.py` — Pipeline (resize, normalize, camera noise, lens distortion, vignetting); YAML config via `pipeline_from_yaml`.
- `prism/processing/cluster.py` — `create_local_cluster()` for Dask.

### Sharding (create sharded datasets)

- **Paths:** `prism/sharding/paths.py` — list image paths (local or S3), with labels.
- **Writer:** `prism/sharding/writer.py` — write WebDataset `.tar` shards from paths or S3 keys; partition and multi-shard helpers.
- **Metadata:** `prism/sharding/metadata.py` — write Parquet manifest (`shard`, `sample_idx`, `__key__`, `cls`, `source`) for Dask/consumers.
- **Reader:** `prism/sharding/reader.py` — list shards, paths for Dask.

- **CLI:** `scripts/shard_dataset.py` — shard a local directory or S3 prefix into WebDataset `.tar` files, optionally using Dask and uploading shards + metadata back to S3. See `config/shard.example.yaml` for configuration.

See [sharding.md](sharding.md) for the full sharding plan and configuration details.

### LMDB datasets

- **Writer:** `prism/storage/lmdb_writer.py` — write an LMDB dataset from `(image_path, class_index)` samples, plus a JSON `manifest.json` with counts, label mapping, and provenance.
- **Reader:** `prism/storage/lmdb_reader.py` — random-access reader over the LMDB dataset, with manifest loading and simple iteration.
- **CLI conversion:** `scripts/convert.py` — convert an ImageFolder-style directory into an LMDB dataset:

  ```bash
  uv run python scripts/convert.py --input-dir data/tiny-imagenet-200/train --output-dir data/lmdb/train
  ```

  The manifest includes `num_classes` and a stable `label_to_index` mapping.

### PyTorch + DALI training demo

- **PyTorch Dataset over LMDB:** `prism/torch/lmdb_dataset.py` — `LMDBDataset` that decodes images from LMDB and applies transforms, exposing `(image, label)` for `DataLoader`.
- **DALI WebDataset pipeline:** `prism/torch/dali_pipeline.py` — single-GPU DALI pipeline and `create_webdataset_dali_iterator()` over WebDataset `.tar` shards with ImageNet-style augmentation and normalization.
- **Training script:** `scripts/train_demo.py` — minimal ResNet18 training loop with three loader modes:
  - `pytorch`: `torchvision.datasets.ImageFolder` over a directory.
  - `lmdb`: `LMDBDataset` created via `scripts/convert.py`.
  - `dali-webdataset`: DALI iterator over WebDataset shards (CUDA only).
  Reports images/sec, loss, and optional GPU utilization/memory CSV for profiling.

### Checkpoints

- **Location:** `{cache_dir}/workflow/{job_id}/workflow_checkpoint.json`.  
  `cache_dir` defaults to `PRISM_CACHE_DIR` or `.prism_cache` in the current directory. `job_id` is derived from bucket and prefix (e.g. `my-bucket_sharded_train`).
- **To re-run from scratch:** delete the checkpoint file or the whole job dir:  
  `rm {cache_dir}/workflow/{job_id}/workflow_checkpoint.json` or `rm -rf {cache_dir}/workflow/{job_id}`.

### Tests

- `tests/test_storage.py` — S3 client, upload, download.
- `tests/test_dataset.py` — Scanner, metadata.
- `tests/test_processing.py` — Pipeline, workflow checkpointing.
- `tests/test_sharding.py` — Paths, partitioning, write one/many shards, metadata Parquet.
- `tests/integration/test_process_sharded.py` — Sharded workflow (sequential and Dask) against moto S3.

```bash
uv run pytest tests/ -v
```
