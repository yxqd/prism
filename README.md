# Prism

Pipeline for Research Imaging & Scalable ML: **ingest → process → store → train → profile**.

## Setup

```bash
cp .env.example .env   # edit with your AWS profile or keys
uv sync                # or: poetry install
```

## Foundation: S3 Ingestion

- **Create buckets:** `uv run python scripts/create_buckets.py`
- **Scan a prefix:** `uv run python scripts/ingest.py s3://your-bucket/prefix/`
- Output: report of image count, corrupted count, average size; metadata JSONL in `prism-processed/metadata/`.

See [vision.md](vision.md) for the full plan.
