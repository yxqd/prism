## Fastest Path to Ray + NVIDIA DALI

This note summarizes the quickest way, given the current `prism` codebase, to get to a working **Ray + NVIDIA DALI** training setup, and what gaps need to be filled.

---

## 1. Current State (Relevant Pieces)

From the existing code, you already have:

- **Data acquisition & layout**
  - `scripts/download_imagenet.py` downloads Tiny-ImageNet or arbitrary URL lists to local disk.
  - `scripts/ingest.py` + `prism/dataset/scanner.py` scan S3 prefixes and write image metadata.

- **Sharding into WebDataset format**
  - `scripts/shard_dataset.py` + `prism/sharding/*` turn a local directory or S3 prefix into **WebDataset-style `.tar` shards + Parquet metadata**:
    - Shards: `shard_00000.tar`, `shard_00001.tar`, ...
    - Manifest: `metadata/part.0.parquet` with `(shard, sample_idx, __key__, cls, source)`.
  - `prism/processing/workflow_sharded.py` can process sharded datasets at scale (Dask/local), writing new `.tar` shards.

- **Image processing pipeline**
  - `prism/processing/pipeline.py` and `prism/processing/tasks.py` implement camera-like augmentations:
    - Resize, normalize, flips, rotations, color jitter.
    - Camera noise, vignetting, lens distortion.

What’s **not** implemented yet (but specified in `vision.md`):

- **PyTorch + LMDB stack (placeholders)**
  - `prism/storage/lmdb_writer.py`, `prism/storage/lmdb_reader.py`.
  - `prism/torch/lmdb_dataset.py`, `prism/torch/profiler.py`, `prism/torch/tuner.py`.
  - `scripts/train_demo.py`, `scripts/benchmark_formats.py`, `scripts/convert.py`.

The key observation: you already have a **good path to WebDataset shards**, which are a natural fit for **DALI’s `readers.webdataset`**. You don’t need LMDB or the full auto-tuner to get to a first Ray + DALI result.

---

## 2. Overall Strategy

**Goal:** Maximize GPU utilization quickly using **Ray + NVIDIA DALI**, while reusing as much of the existing pipeline as possible.

**Strategy:**  
Skip LMDB for the initial spike and:

1. Use your existing tooling to build a **canonical sharded dataset** (WebDataset + Parquet).
2. Implement a **minimal PyTorch training script** as a baseline (no DALI, no Ray).
3. Add a **single-GPU DALI pipeline** that reads those shards and plugs into PyTorch.
4. Optionally add light **profiling glue**.
5. Wrap the DALI pipeline in **Ray** to do multi-GPU training and measure scaling.

This gives you a clear “before vs after” story:

- Baseline: PyTorch `DataLoader` with CPU transforms.
- Improved: Single-GPU DALI.
- Scaled: Ray + DALI across multiple GPUs.

---

## 3. Step-by-Step: Fastest Route to Ray + DALI

### Step 0 – Environment Setup

- **Gaps:**
  - DALI and Ray are not yet in the project dependencies.

- **Actions:**
  - Pick a GPU machine with CUDA available.
  - Install NVIDIA DALI matching your CUDA version, e.g.:
    - `nvidia-dali-cuda11` / `nvidia-dali-cuda12` (depending on environment). (updated pyproject.toml and ran uv sync)
  - Install Ray (for Ray Train / AIR):
    - `ray[default]` or `ray[train]`.

No code changes needed in the repo for this step; just ensure the environment can import `nvidia.dali` and `ray`.

---

### Step 1 – Create a Small Sharded Benchmark Dataset

Use existing tools to generate a **canonical benchmark dataset** that all experiments will share.

1. **Download Tiny-ImageNet locally**
   - Use:
     - `python scripts/download_imagenet.py tiny-imagenet --out-dir data/tiny-imagenet-200`
   - This creates a standard Tiny-ImageNet folder structure under `data/tiny-imagenet-200`.

2. **Shard it into WebDataset format**
   - Use:
     - `python scripts/shard_dataset.py --input data/tiny-imagenet-200/train --output data/sharded/train --shard-size 2000`
   - Output:
     - `data/sharded/train/shard_00000.tar`, `shard_00001.tar`, ...
     - `data/sharded/train/metadata/part.0.parquet`

This dataset:

- Is small enough to iterate quickly.
- Matches the “sharding deep dive” you already planned.
- Is directly consumable by **DALI’s `readers.webdataset`** for GPU-side decoding and augmentation.

---

### Step 2 – Minimal PyTorch Baseline Training Script

**Gap:** `scripts/train_demo.py` is currently a placeholder.

For a fast DALI/Ray spike, you don’t need the full LMDB + tuner stack; you just need a **simple, repeatable baseline** to compare against.

**Minimal baseline design:**

- Implement a small training script (either in `scripts/train_demo.py` or a new script) that:
  - Uses **either**:
    - `torchvision.datasets.ImageFolder` pointing at `data/tiny-imagenet-200/train`, **or**
    - A simple `webdataset`-based `IterableDataset` over `data/sharded/train/*.tar`.
  - Constructs a standard `torch.utils.data.DataLoader` with:
    - CPU-based transforms (resize, crop, normalize).
    - Reasonable `num_workers`, `pin_memory`, etc.
  - Defines a small CNN / ResNet-style model.
  - Runs for a small fixed number of iterations/epochs (e.g. a few hundred steps).
  - Prints:
    - **Effective images/sec**.
    - Optionally, running loss (to confirm training isn’t totally broken).

This gives you:

- A **CPU-preprocessing PyTorch baseline**.
- A place to plug in your DALI iterator next (same model, same loss, same optimizer, only the data loader changes).

---

### Step 3 – Single-GPU DALI Pipeline on WebDataset Shards

**New core component:** a DALI-based input pipeline that reads **your existing WebDataset shards**.

Conceptually, create a small module (e.g. `prism/torch/dali_pipeline.py`) that:

- Uses `nvidia.dali.fn.readers.webdataset` to:
  - Read from the set of tar files under `data/sharded/train/`.
  - Respect shard shuffling / sample shuffling as needed.
- Applies GPU-side decode and augmentations:
  - Decode JPEGs on GPU.
  - Resize / crop / normalize.
  - Optionally simple flips or brightness/contrast adjustments.
- Wraps the pipeline in a PyTorch-compatible iterator, such as:
  - `nvidia.dali.plugin.pytorch.DALIClassificationIterator` (or the newer PyTorch integration).

