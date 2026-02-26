# Sharding experiment: tiny-imagenet-200 → WebDataset-style .tar shards

## 1. Goal and context

**Goal:** Shard the tiny-imagenet-200 dataset (see `data/tiny-imagenet-200`) into 20–50 files.  
Don't process 100,000 files individually. Instead, group them into ~50 shards of ~2,000 images each so each Dask task has enough work to amortize scheduling overhead.

**Preferred format:** WebDataset (`.tar`). Images stored sequentially in tarballs; Dask can read as a stream.

**Experiment workflow (conceptual):**
- **Read:** Use `dask.bag` to list all local JPEG paths under the dataset root.
- **Batch:** Use `.map_partitions` to group filenames into chunks of 2,000.
- **Write shards:** In each partition, use `webdataset.TarWriter` to write one `.tar` file per chunk.

**Loading shards in Dask:**
- **dask-image:** `dask_image.imread.imread('shards/*.tar')` to get a Dask Array.
- **Dask Bag:** `db.from_sequence(shard_paths).map(process_shard)` to process 2,000 images per task.

---

## 2. What we have vs what’s left

So that sharding works for tiny-imagenet-200 and any other image directory:

| Layer | Status | Purpose |
|-------|--------|--------|
| **Module: path listing** | **Implemented** (`prism/sharding/paths.py`) | Collect image paths from a root (local dir or S3 prefix); filter by extension; optionally respect train/val split layout (e.g. class subdirs). |
| **Module: partition + write** | **Implemented** (`prism/sharding/writer.py`) | Given a sequence of paths and shard size, partition into chunks and write each chunk with `TarWriter` to one `.tar`. |
| **Module: shard reader** | **Implemented** (`prism/sharding/reader.py`) | Helpers to load shards: list shard paths from an output directory and return paths for `dask.bag.from_sequence(...).map(process_shard)`. |
| **Metadata manifest** | **Implemented** (`prism/sharding/metadata.py`) | Write Parquet manifest (`shard`, `sample_idx`, `__key__`, `cls`, `source`) for downstream consumers (including `workflow_sharded`). |
| **Script** | **Planned** (`scripts/shard_dataset.py`) | Single entrypoint to drive sharding from CLI: input path/URI, output dir, shard size (or num shards), optional Dask workers; calls the modules above. |
| **Config** | **Example present** (`config/shard.example.yaml`) | Example shard config: input root, output dir, shard size, extensions, splits, optional upload URI. CLI script should read this when implemented. |
| **Tests** | **Implemented** (`tests/test_sharding.py`) | Unit tests for path listing, partition sizing, single/multi-shard write, and listing shards. |

---

## 3. Detailed design and remaining TODOs

### 3.1 Path listing

- **Input:** Root path (e.g. `data/tiny-imagenet-200/train` or `data/tiny-imagenet-200` with subdirs).
- **Output:** Ordered list of image file paths (or generator), optionally with labels/metadata if we parse class from path (e.g. `train/n02106662/...`).
- **Design choices to decide:**
  - Recursive glob vs explicit train/val; include or exclude `val_annotations.txt` / `*_boxes.txt`.
  - Extensions: default `.jpg`/`.jpeg`/`.png`; make configurable.
  - Whether to emit (path, label) or path only; label needed for WebDataset sample keys (e.g. `__key__`, `cls`).

### 3.2 Partitioning and TarWriter

- **Input:** Sequence of (path, optional metadata) and `shard_size` (e.g. 2,000).
- **Output:** For each chunk, one `.tar` file (WebDataset convention: e.g. `000000.tar`, `000001.tar`, …).
- **Design choices:**
  - In Dask: one partition = one shard (simplest), or one partition = many paths that get split into multiple shards in one task.
  - Naming: `shard_00000.tar` vs `{split}_00000.tar` if we have train/val.
  - What to write inside each sample: at least image bytes; optionally sidecar (label, metadata) as separate keys in the same sample.

### 3.3 Script: `scripts/shard_dataset.py` (planned)

- **CLI (proposed):**
  - `--input` (required): root directory (e.g. `data/tiny-imagenet-200` or `data/tiny-imagenet-200/train`).
  - `--output` (required): directory where `.tar` shards will be written.
  - `--shard-size`: images per shard (default 2,000).
  - `--num-shards`: alternative to `--shard-size`; compute shard size from total count.
  - `--split`: optional, e.g. `train` or `val`; only list that subdir.
  - `--extensions`: default `jpg,jpeg,png`.
  - `--use-dask` / `--workers`: use Dask with N workers for parallel shard writing.
- **Flow:** List paths → partition → write shards (sequential or via Dask); print summary (num shards, total images, paths written).

### 3.4 Shard reader helpers

- **List shards:** Given output dir, return sorted list of `.tar` paths.
- **Dask Bag usage:** Document or wrap `db.from_sequence(shard_paths).map(process_shard)` with a simple `process_shard(tar_path)` that opens the tar and runs a user function on each sample.
- **dask-image:** Document whether `dask_image.imread.imread('shards/*.tar')` works as-is or needs a small adapter; if adapter needed, add a thin helper.

### 3.5 Config (example exists, wiring TODO)

- Optional: add a `shard` section to pipeline config (or `config/shard.example.yaml`) with:
  - `input_root`, `output_dir`, `shard_size`, `extensions`, `splits` (e.g. `[train, val]`).
- Script reads config when provided (e.g. `--config shard.yaml`) and overrides with CLI flags.

### 3.6 Tests (mostly done)

- **Unit:**
  - Path lister: mock dir with a few class subdirs and images; assert count and extensions.
  - Partitioning: given 5,500 paths and shard_size 2,000, expect 3 shards (2000, 2000, 1500).
  - Single-shard write: write one small .tar from 10 paths; read back and assert sample count and key format.
- **Integration (optional):** Run script on `data/tiny-imagenet-200/train` with a tiny shard size (e.g. 100), then list shards and open first shard to sanity-check.

### 3.7 Dependencies

- Add `webdataset` (and optionally `tensorboard` if we ever log; not required for this plan) to `pyproject.toml` if not already present.

### 3.8 Order of remaining implementation

1. Path listing (pure function + tests).
2. Single-shard write (TarWriter on a list of paths; tests).
3. Partitioning + multi-shard write (sequential first).
4. CLI script wiring and optional Dask path.
5. Reader helpers and docs.
6. Config and integration test with tiny-imagenet-200.

---

## 4. Implementation summary (current state)

- **`prism/sharding/paths.py`** — `list_image_paths(root, extensions, split, with_labels)` for local dirs; `list_image_keys_from_s3(client, bucket, prefix, ...)` for S3 prefixes. Both exclude `*_boxes.txt` and `val_annotations.txt`; labels are derived from path/key.
- **`prism/sharding/writer.py`** — `write_one_shard(samples, out_path)` (local paths), `write_one_shard_from_s3(client, bucket, samples, out_path)` (S3 keys, streamed); `partition_paths` / `partition_keys`; `write_shards` / `write_shards_from_s3`; `write_shards_from_root`. WebDataset samples use `__key__`, `jpg`, and optional `cls`.
- **`prism/sharding/metadata.py`** — `write_metadata_parquet(samples_with_labels, shard_size, output_dir, ...)` writes `output_dir/metadata/part.0.parquet` with columns `shard`, `shard_idx`, `sample_idx`, `__key__`, `cls`, `source`. These manifests are what `prism/processing/workflow_sharded.py` expects when reading from S3.
- **`prism/sharding/reader.py`** — `list_shards(output_dir)`, `shard_paths_for_dask(output_dir)` for `dask.bag.from_sequence(...).map(process_shard)`.
- **`config/shard.example.yaml`** — Example config for a future `scripts/shard_dataset.py`: `input_root` (local or `s3://...`), `output_dir`, `shard_size`, `extensions`, `splits`, optional `upload_uri`.
- **`tests/test_sharding.py`** — Unit tests for path listing, partitioning, single-shard write, multi-shard write, and listing shards.

**Not yet implemented:** `scripts/shard_dataset.py` CLI and its `--upload` flow. The example commands at the bottom are design sketches, not working commands today.