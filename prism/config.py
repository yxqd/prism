"""Centralized configuration: S3 buckets, region, MLflow, AWS profile."""

import os
from pathlib import Path

# Load .env if present (no-op if python-dotenv not used or file missing)
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_env)
except ImportError:
    pass

# AWS
AWS_PROFILE = os.environ.get("AWS_PROFILE", "default")
REGION = os.environ.get("PRISM_REGION", "us-east-1")

# S3 buckets
BUCKET_LANDING = os.environ.get("PRISM_BUCKET_LANDING", "prism-landing")
BUCKET_PROCESSED = os.environ.get("PRISM_BUCKET_PROCESSED", "prism-processed")
BUCKET_MODELS = os.environ.get("PRISM_BUCKET_MODELS", "prism-model-collection")

# MLflow (for later)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

# Optional: local cache root for downloads
CACHE_DIR = Path(os.environ.get("PRISM_CACHE_DIR", Path.cwd() / ".prism_cache"))
