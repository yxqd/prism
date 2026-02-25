"""Write image paths into WebDataset-style .tar shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, List, Optional, Set, Tuple

import webdataset as wds

from prism.sharding.paths import list_image_paths

# Default image extension key for WebDataset (bytes stored as .jpg in tar)
IMAGE_KEY = "jpg"


def _sample_from_path(path: Path, label: Optional[str], index: int) -> dict:
    """Build a WebDataset sample dict from a single image path."""
    key = path.stem if path.stem else f"{index:06d}"
    sample: dict = {
        "__key__": key,
        IMAGE_KEY: path.read_bytes(),
    }
    if label is not None:
        sample["cls"] = label
    return sample


def _sample_from_s3_key(body: bytes, key_str: str, label: Optional[str], index: int) -> dict:
    """Build a WebDataset sample dict from S3 object body and key/label."""
    stem = key_str.rsplit("/", 1)[-1].rsplit(".", 1)[0] if "." in key_str else key_str.rsplit("/", 1)[-1]
    sample_key = stem if stem else f"{index:06d}"
    sample: dict = {"__key__": sample_key, IMAGE_KEY: body}
    if label is not None:
        sample["cls"] = label
    return sample


def write_one_shard(
    samples: List[Tuple[Path, Optional[str]]],
    out_path: Path,
) -> int:
    """Write one .tar shard from a list of (path, label) pairs. Returns number of samples written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with wds.TarWriter(str(out_path)) as sink:
        for i, (path, label) in enumerate(samples):
            if not path.is_file():
                continue
            sink.write(_sample_from_path(path, label, i))
            count += 1
    return count


def write_one_shard_from_s3(
    client: Any,
    bucket: str,
    samples: List[Tuple[str, Optional[str]]],
    out_path: Path,
) -> int:
    """Write one .tar shard from (s3_key, label) pairs by streaming from S3. Returns number of samples written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with wds.TarWriter(str(out_path)) as sink:
        for i, (key, label) in enumerate(samples):
            resp = client.get_object(Bucket=bucket, Key=key)
            body = resp["Body"].read()
            sink.write(_sample_from_s3_key(body, key, label, i))
            count += 1
    return count


def partition_paths(
    paths_with_labels: List[Tuple[Path, Optional[str]]],
    shard_size: int,
) -> Iterator[List[Tuple[Path, Optional[str]]]]:
    """Yield chunks of (path, label) of size shard_size (last chunk may be smaller)."""
    for i in range(0, len(paths_with_labels), shard_size):
        yield paths_with_labels[i : i + shard_size]


def partition_keys(
    keys_with_labels: List[Tuple[str, Optional[str]]],
    shard_size: int,
) -> Iterator[List[Tuple[str, Optional[str]]]]:
    """Yield chunks of (key, label) of size shard_size (last chunk may be smaller)."""
    for i in range(0, len(keys_with_labels), shard_size):
        yield keys_with_labels[i : i + shard_size]


def write_shards(
    paths_with_labels: List[Tuple[Path, Optional[str]]],
    output_dir: Path,
    shard_size: int,
    *,
    shard_prefix: str = "shard",
    split_name: Optional[str] = None,
) -> List[Path]:
    """Write all shards sequentially. Returns list of written .tar paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = split_name or shard_prefix
    written: List[Path] = []
    for shard_idx, chunk in enumerate(partition_paths(paths_with_labels, shard_size)):
        name = f"{prefix}_{shard_idx:05d}.tar"
        out_path = output_dir / name
        write_one_shard(chunk, out_path)
        written.append(out_path)
    return written


def write_shards_from_s3(
    client: Any,
    bucket: str,
    keys_with_labels: List[Tuple[str, Optional[str]]],
    output_dir: Path,
    shard_size: int,
    *,
    shard_prefix: str = "shard",
    split_name: Optional[str] = None,
) -> List[Path]:
    """Write all shards from S3 keys sequentially (streaming per key). Returns list of written .tar paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = split_name or shard_prefix
    written: List[Path] = []
    for shard_idx, chunk in enumerate(partition_keys(keys_with_labels, shard_size)):
        name = f"{prefix}_{shard_idx:05d}.tar"
        out_path = output_dir / name
        write_one_shard_from_s3(client, bucket, chunk, out_path)
        written.append(out_path)
    return written


def write_shards_from_root(
    root: Path,
    output_dir: Path,
    shard_size: int,
    *,
    extensions: Optional[Set[str]] = None,
    split: Optional[str] = None,
    shard_prefix: str = "shard",
) -> Tuple[List[Path], int]:
    """List images under root (optionally for one split), then write shards. Returns (shard paths, total images)."""
    paths_with_labels = list_image_paths(root, extensions=extensions, split=split, with_labels=True)
    if not paths_with_labels:
        return [], 0
    split_name = split if split else None
    written = write_shards(
        paths_with_labels,
        output_dir,
        shard_size,
        shard_prefix=shard_prefix,
        split_name=split_name,
    )
    return written, len(paths_with_labels)
