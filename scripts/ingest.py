#!/usr/bin/env python3
"""Scan an S3 prefix and report image count, corrupted count, average size.

Usage:
  python scripts/ingest.py s3://bucket/prefix/
  python scripts/ingest.py --bucket prism-landing --prefix my-dataset/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root so prism is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism import config
from prism.dataset.registry import append_scan, get_registry
from prism.dataset.scanner import (
    scan_s3_prefix,
    upload_metadata_to_s3,
    write_metadata_jsonl,
)
from prism.storage.s3_client import get_s3_client


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, prefix) from s3://bucket/prefix/."""
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with s3://")
    rest = uri[5:].strip("/")
    if "/" not in rest:
        return rest, ""
    bucket, prefix = rest.split("/", 1)
    return bucket, prefix + "/" if not rest.endswith("/") else rest


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan S3 prefix and report image metadata.")
    parser.add_argument(
        "uri",
        nargs="?",
        help="S3 URI, e.g. s3://prism-landing/my-dataset/",
    )
    parser.add_argument("--bucket", default=None, help="S3 bucket (overrides URI)")
    parser.add_argument("--prefix", default="", help="S3 prefix (overrides URI)")
    parser.add_argument(
        "--write-jsonl",
        metavar="PATH",
        default=None,
        help="Write metadata JSONL to this local path",
    )
    parser.add_argument(
        "--upload-metadata",
        action="store_true",
        help="Upload metadata JSONL to prism-processed/metadata/",
    )
    parser.add_argument(
        "--metadata-key",
        default=None,
        help="S3 key for metadata JSONL (default: metadata/<prefix_safe>.jsonl)",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrency for scanning")
    parser.add_argument(
        "--list-scans",
        action="store_true",
        help="List recent scans from the S3 registry and exit",
    )
    args = parser.parse_args()

    if args.list_scans:
        client = get_s3_client()
        registry = get_registry(config.BUCKET_PROCESSED, client=client)
        entries = registry.get("entries", [])
        if not entries:
            print("No scans recorded yet.")
            return 0
        # Print table: source (bucket/prefix), metadata_key, at, total
        print(f"{'Source (bucket/prefix)':<45} {'Metadata key':<40} {'At':<28} {'Total':>8}")
        print("-" * 125)
        for e in reversed(entries):
            src = (e.get("source_bucket", "") or "") + "/" + (e.get("source_prefix", "") or "")
            key = e.get("metadata_key", "")
            at = e.get("at", "")
            total = e.get("total", "")
            print(f"{src:<45} {key:<40} {at:<28} {total:>8}")
        return 0

    if args.uri:
        bucket, prefix = parse_s3_uri(args.uri)
    elif args.bucket:
        bucket = args.bucket
        prefix = args.prefix or ""
    else:
        parser.error("Provide either uri (e.g. s3://bucket/prefix/) or --bucket")

    result = scan_s3_prefix(bucket, prefix, max_workers=args.max_workers)

    # Deliverable: "Found 1,234 images, 3 corrupted, avg size 2.3MB"
    avg_mb = result.avg_size_mb
    print(
        f"Found {result.total} images, {result.corrupted} corrupted, "
        f"{result.unexpected_format} unexpected format, avg size {avg_mb:.1f}MB"
    )

    if args.write_jsonl:
        write_metadata_jsonl(result, args.write_jsonl)
        print(f"Wrote metadata to {args.write_jsonl}")

    if args.upload_metadata:
        key = args.metadata_key or f"metadata/{prefix.replace('/', '_').strip('_') or 'scan'}.jsonl"
        upload_metadata_to_s3(result, key, bucket=config.BUCKET_PROCESSED)
        print(f"Uploaded metadata to s3://{config.BUCKET_PROCESSED}/{key}")
        entry = {
            "source_bucket": bucket,
            "source_prefix": prefix,
            "metadata_key": key,
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total": result.total,
            "corrupted": result.corrupted,
        }
        client = get_s3_client()
        append_scan(config.BUCKET_PROCESSED, entry, client=client)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
