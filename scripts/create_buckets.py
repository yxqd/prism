#!/usr/bin/env python3
"""Create S3 buckets: prism-landing, prism-processed, prism-model-collection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism import config
from prism.storage.s3_client import get_s3_client


def main() -> int:
    client = get_s3_client()
    buckets = [config.BUCKET_LANDING, config.BUCKET_PROCESSED, config.BUCKET_MODELS]
    for name in buckets:
        kwargs: dict = {"Bucket": name}
        if config.REGION != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": config.REGION}
        try:
            client.create_bucket(**kwargs)
            print(f"Created bucket: {name}")
        except client.exceptions.BucketAlreadyOwnedByYou:
            print(f"Bucket already exists: {name}")
        except Exception as e:
            print(f"Failed to create {name}: {e}", file=sys.stderr)
            return 1

        # Enable versioning for the models bucket
        if name == config.BUCKET_MODELS:
            try:
                client.put_bucket_versioning(
                    Bucket=name,
                    VersioningConfiguration={"Status": "Enabled"},
                )
                print(f"Enabled versioning on bucket: {name}")
            except Exception as e:
                print(f"Failed to enable versioning on {name}: {e}", file=sys.stderr)
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
