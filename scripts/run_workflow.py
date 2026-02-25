#!/usr/bin/env python3
"""Run S3 pull -> process -> push workflow. Optional Dask for distributed processing.

Usage:
  python scripts/run_workflow.py s3://prism-landing/my-dataset/
  python scripts/run_workflow.py --bucket prism-landing --prefix my-dataset/ --no-dask
  python scripts/run_workflow.py s3://prism-landing/my-dataset/ --pipeline config/pipeline.yaml

Resume: Re-run the same bucket/prefix to resume from checkpoint (download missing, process remaining).
Start fresh: Delete the checkpoint under PRISM_CACHE_DIR/workflow/<job_id>/workflow_checkpoint.json
Dashboard: When using Dask, open http://localhost:8787 for DAG and task stream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism import config
from prism.processing import create_local_cluster, run_workflow, print_timing_summary
from prism.processing.pipeline import Pipeline
from prism.processing.pipeline_config import pipeline_from_yaml


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with s3://")
    rest = uri[5:].strip("/")
    if "/" not in rest:
        return rest, ""
    bucket, prefix = rest.split("/", 1)
    return bucket, prefix + "/" if not rest.endswith("/") else rest


def main() -> int:
    parser = argparse.ArgumentParser(description="S3 pull -> process -> push workflow.")
    parser.add_argument("uri", nargs="?", help="S3 URI, e.g. s3://prism-landing/my-dataset/")
    parser.add_argument("--bucket", default=None, help="S3 bucket (overrides URI)")
    parser.add_argument("--prefix", default="", help="S3 prefix (overrides URI)")
    parser.add_argument("--no-dask", action="store_true", help="Run processing sequentially (no Dask)")
    parser.add_argument("--pipeline", metavar="YAML", default=None, help="Path to YAML pipeline config (default: resize + normalize)")
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"), help="Resize target when not using --pipeline (default: 224 224)")
    args = parser.parse_args()

    if args.uri:
        bucket, prefix = parse_s3_uri(args.uri)
    elif args.bucket:
        bucket = args.bucket
        prefix = args.prefix or ""
    else:
        parser.error("Provide either uri (e.g. s3://bucket/prefix/) or --bucket")

    if args.pipeline:
        pipeline = pipeline_from_yaml(args.pipeline)
    else:
        pipeline = Pipeline().resize(tuple(args.target_size)).normalize()

    if args.no_dask:
        timings = run_workflow(
            source_bucket=bucket,
            source_prefix=prefix,
            pipeline=pipeline,
            use_dask=False,
            show_progress=True,
        )
    else:
        with create_local_cluster() as (client, _cluster):
            timings = run_workflow(
                source_bucket=bucket,
                source_prefix=prefix,
                pipeline=pipeline,
                use_dask=True,
                dask_client=client,
                show_progress=True,
            )

    print_timing_summary(timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
