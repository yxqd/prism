#!/usr/bin/env python3
"""Shard an image directory or S3 prefix into WebDataset-style .tar files.

Usage:
  python scripts/shard_dataset.py --input data/tiny-imagenet-200/train --output data/sharded
  python scripts/shard_dataset.py --input s3://prism-landing/tiny-imagenet-200/ --output data/sharded --split train
  python scripts/shard_dataset.py --config config/shard.example.yaml --shard-size 2000

Input may be a local directory or an S3 URI (s3://bucket/prefix/). Config may set input_root, output_dir, etc.
CLI flags override config. Use either --shard-size or --num-shards (computed from total image count).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from prism.sharding import (
    list_image_keys_from_s3,
    list_image_paths,
    partition_keys,
    partition_paths,
    write_metadata_parquet,
    write_one_shard,
    write_one_shard_from_s3,
    write_shards,
    write_shards_from_s3,
)
from prism.storage.s3_client import get_s3_client
from prism.storage.upload import upload_file


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, prefix) from s3://bucket/prefix/. Prefix has trailing slash."""
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with s3://")
    rest = uri[5:].strip("/")
    if "/" not in rest:
        return rest, ""
    bucket, prefix = rest.split("/", 1)
    return bucket, prefix + "/" if not rest.endswith("/") else rest


def load_shard_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Shard image directory into WebDataset .tar files.")
    parser.add_argument("--config", metavar="YAML", help="Path to shard config YAML (optional)")
    parser.add_argument("--input", metavar="DIR_OR_URI", help="Local directory or S3 URI (s3://bucket/prefix/) of images (or use config input_root)")
    parser.add_argument("--output", metavar="DIR", help="Output directory for .tar shards (or use config output_dir)")
    parser.add_argument("--shard-size", type=int, default=None, help="Images per shard (default: 2000)")
    parser.add_argument("--num-shards", type=int, default=None, help="Alternative: number of shards (shard_size computed from total)")
    parser.add_argument("--split", default=None, help="Only list images under root/split (e.g. train or val)")
    parser.add_argument("--extensions", default="jpg,jpeg,png", help="Comma-separated extensions (default: jpg,jpeg,png)")
    parser.add_argument("--use-dask", action="store_true", help="Write shards in parallel with Dask")
    parser.add_argument("--workers", type=int, default=4, help="Dask workers when --use-dask (default: 4)")
    parser.add_argument("--upload", metavar="s3://bucket/prefix/", default=None, help="After writing shards, upload each .tar to this S3 location (key = prefix + filename)")
    parser.add_argument("--upload-progress", action="store_true", help="Show progress bar per file when uploading to S3")
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        p = Path(args.config)
        if not p.is_file():
            print(f"Config not found: {p}", file=sys.stderr)
            return 1
        cfg = load_shard_config(p)

    upload_uri = args.upload or cfg.get("upload_uri")
    input_spec = args.input or cfg.get("input_root")
    output_dir = args.output or cfg.get("output_dir")
    if not input_spec or not output_dir:
        parser.error("Provide --input and --output, or --config with input_root and output_dir")
    output_dir = Path(output_dir)
    extensions = set(e.strip() for e in (args.extensions or cfg.get("extensions", "jpg,jpeg,png")).split(","))
    split = args.split or (cfg.get("splits", [None])[0] if cfg.get("splits") else None)
    shard_size_arg = args.shard_size if args.shard_size is not None else cfg.get("shard_size")
    num_shards_arg = args.num_shards
    shard_prefix = "shard"
    split_name = split

    if isinstance(input_spec, str) and input_spec.startswith("s3://"):
        bucket, prefix = parse_s3_uri(input_spec)
        client = get_s3_client()
        keys_with_labels = list_image_keys_from_s3(
            client, bucket, prefix, extensions=extensions, split=split, with_labels=True
        )
        samples_for_metadata = keys_with_labels
        total = len(keys_with_labels)
        if total == 0:
            print("No images found under S3 prefix.", file=sys.stderr)
            return 1
        if num_shards_arg is not None:
            shard_size = max(1, math.ceil(total / num_shards_arg))
        else:
            shard_size = shard_size_arg if shard_size_arg is not None else 2000
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.use_dask:
            import dask.bag as db
            from dask.distributed import Client as DaskClient, LocalCluster

            chunks = list(partition_keys(keys_with_labels, shard_size))
            cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
            dask_client = DaskClient(cluster)
            try:
                def write_s3_chunk(item: tuple) -> str:
                    idx, chunk = item
                    out_path = output_dir / f"{split_name or shard_prefix}_{idx:05d}.tar"
                    s3_client = get_s3_client()
                    write_one_shard_from_s3(s3_client, bucket, chunk, out_path)
                    return str(out_path)

                bag = db.from_sequence(list(enumerate(chunks)))
                written_paths = list(bag.map(write_s3_chunk).compute())
            finally:
                dask_client.close()
                cluster.close()
        else:
            written_paths = [
                str(p) for p in write_shards_from_s3(
                    client, bucket, keys_with_labels, output_dir, shard_size,
                    shard_prefix=shard_prefix, split_name=split_name,
                )
            ]
    else:
        input_root = Path(input_spec)
        if not input_root.is_dir():
            print(f"Input directory not found: {input_root}", file=sys.stderr)
            return 1
        paths_with_labels = list_image_paths(input_root, extensions=extensions, split=split, with_labels=True)
        samples_for_metadata = paths_with_labels
        total = len(paths_with_labels)
        if total == 0:
            print("No images found.", file=sys.stderr)
            return 1

        # Map string class labels to stable integer indices for WebDataset `cls`.
        from typing import Dict

        label_to_index: Dict[str, int] = {}
        next_idx = 0

        def _to_index(lbl: Optional[str]) -> Optional[int]:
            nonlocal next_idx
            if lbl is None:
                return None
            if lbl not in label_to_index:
                label_to_index[lbl] = next_idx
                next_idx += 1
            return label_to_index[lbl]

        indexed_paths = [(p, _to_index(lbl)) for p, lbl in paths_with_labels]

        if num_shards_arg is not None:
            shard_size = max(1, math.ceil(total / num_shards_arg))
        else:
            shard_size = shard_size_arg if shard_size_arg is not None else 2000

        output_dir.mkdir(parents=True, exist_ok=True)

        if args.use_dask:
            import dask.bag as db
            from dask.distributed import Client as DaskClient, LocalCluster

            chunks = list(partition_paths(indexed_paths, shard_size))
            cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
            dask_client = DaskClient(cluster)
            try:
                def write_chunk(item: tuple) -> tuple:
                    idx, chunk = item
                    out_path = output_dir / f"{split_name or shard_prefix}_{idx:05d}.tar"
                    n = write_one_shard(chunk, out_path)
                    return (str(out_path), n)

                bag = db.from_sequence(list(enumerate(chunks)))
                results = bag.map(write_chunk).compute()
                written_paths = [r[0] for r in results]
            finally:
                dask_client.close()
                cluster.close()
        else:
            written_paths = [
                str(p) for p in write_shards(
                    indexed_paths,
                    output_dir,
                    shard_size,
                    shard_prefix=shard_prefix,
                    split_name=split_name,
                )
            ]

    n_shards = len(written_paths)
    meta_path = write_metadata_parquet(
        samples_for_metadata, shard_size, output_dir,
        shard_prefix=shard_prefix, split_name=split_name,
    )
    print(f"Wrote {n_shards} shard(s), {total} images total -> {output_dir}")
    print(f"Wrote metadata -> {meta_path}")
    if upload_uri:
        up_bucket, up_prefix = parse_s3_uri(upload_uri)
        for path_str in written_paths:
            p = Path(path_str)
            key = up_prefix + p.name
            upload_file(p, up_bucket, key, show_progress=args.upload_progress)
        # Upload metadata so dd.read_parquet("s3://bucket/prefix/metadata/") works
        meta_key = up_prefix + "metadata/part.0.parquet"
        upload_file(meta_path, up_bucket, meta_key, show_progress=args.upload_progress)
        print(f"Uploaded {n_shards} shard(s) + metadata to {upload_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
