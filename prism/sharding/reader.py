"""Helpers to discover and iterate over WebDataset shards."""

from __future__ import annotations

from pathlib import Path
from typing import List


def list_shards(output_dir: Path, *, pattern: str = "*.tar") -> List[Path]:
    """Return sorted list of .tar shard paths under output_dir."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    return sorted(output_dir.glob(pattern))


def shard_paths_for_dask(output_dir: Path) -> List[str]:
    """Return sorted list of absolute path strings for use with dask.bag.from_sequence(...).map(process_shard)."""
    return [str(p.resolve()) for p in list_shards(output_dir)]
