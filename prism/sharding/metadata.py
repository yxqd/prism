"""Write Parquet metadata manifest for sharded image datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq


def _sample_key_from_source(source: Union[Path, str]) -> str:
    """Derive __key__ (sample id) from path or S3 key."""
    if isinstance(source, Path):
        return source.stem or ""
    key_str = str(source)
    if "." in key_str:
        return key_str.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return key_str.rsplit("/", 1)[-1] if "/" in key_str else key_str


def write_metadata_parquet(
    samples_with_labels: List[Tuple[Union[Path, str], Optional[str]]],
    shard_size: int,
    output_dir: Path,
    *,
    shard_prefix: str = "shard",
    split_name: Optional[str] = None,
) -> Path:
    """Write a Parquet manifest of (shard, sample_idx, __key__, cls, source) for Dask/pandas.

    Writes to output_dir/metadata/part.0.parquet so that:
      dd.read_parquet("s3://bucket/prefix/metadata/")  # after uploading output_dir to prefix
    loads the full manifest.

    Returns the path to the written Parquet file.
    """
    output_dir = Path(output_dir)
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "part.0.parquet"

    prefix = split_name or shard_prefix
    shard_names: List[str] = []
    sample_idxs: List[int] = []
    keys: List[str] = []
    classes: List[Optional[str]] = []
    sources: List[str] = []

    for i, (path_or_key, label) in enumerate(samples_with_labels):
        shard_idx = i // shard_size
        sample_idx = i % shard_size
        shard_names.append(f"{prefix}_{shard_idx:05d}.tar")
        sample_idxs.append(sample_idx)
        keys.append(_sample_key_from_source(path_or_key))
        classes.append(label)
        sources.append(str(path_or_key))

    table = pa.table({
        "shard": shard_names,
        "shard_idx": pa.array([i // shard_size for i in range(len(samples_with_labels))]),
        "sample_idx": pa.array([i % shard_size for i in range(len(samples_with_labels))]),
        "__key__": keys,
        "cls": classes,
        "source": sources,
    })
    pq.write_table(table, out_path)
    return out_path
