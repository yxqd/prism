#!/usr/bin/env python3
"""S3 → Dask processing → LMDB conversion (implemented in Storage Optimization phase).

One command: convert a local image directory (e.g. ImageFolder layout) into a
training-ready LMDB dataset with manifest and provenance metadata.

Usage:
  uv run python scripts/convert.py --input-dir data/tiny-imagenet-200/train --output-dir data/lmdb/train
  uv run python scripts/convert.py --input-dir data/tiny-imagenet-200 --output-dir data/lmdb/train --split train
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism.sharding.paths import list_image_paths
from prism.storage.lmdb_writer import write_lmdb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert image directory to LMDB dataset with manifest.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing images (ImageFolder layout: optional split/class/img.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for LMDB environment and manifest.json.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Subdir under --input-dir to use (e.g. train or val). If omitted, scan --input-dir recursively.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1024**4,
        help="LMDB map size in bytes (default 1TB).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        return 1

    paths_with_labels = list_image_paths(
        input_dir,
        split=args.split,
        with_labels=True,
    )
    if not paths_with_labels:
        print("Error: no images found.", file=sys.stderr)
        return 1

    # Stable integer indices for class labels (sorted for reproducibility)
    label_to_index: dict[str, int] = {}
    for _, label in paths_with_labels:
        if label is not None and label not in label_to_index:
            label_to_index[label] = len(label_to_index)
    # Sort so that ordering is deterministic
    sorted_labels = sorted(label_to_index.keys())
    label_to_index = {lbl: i for i, lbl in enumerate(sorted_labels)}

    samples: list[tuple[Path, int]] = []
    for path, label in paths_with_labels:
        idx = label_to_index.get(label, 0) if label is not None else 0
        samples.append((path, idx))

    # Provenance metadata for manifest
    source_str = str(input_dir)
    if args.split:
        source_str += f"/{args.split}"
    version_hash = hashlib.sha256(source_str.encode()).hexdigest()[:12]
    metadata = {
        "source": source_str,
        "version_hash": version_hash,
        "num_classes": len(label_to_index),
        "label_to_index": label_to_index,
    }

    manifest = write_lmdb(
        samples,
        args.output_dir.resolve(),
        map_size=args.map_size,
        metadata=metadata,
    )
    print(f"Wrote {manifest['count']} samples to {manifest['path']}")
    print(f"  manifest: {Path(manifest['path']) / 'manifest.json'}")
    print(f"  num_classes: {metadata['num_classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
