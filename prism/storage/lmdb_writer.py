"""LMDB writer for training datasets (implemented in Storage Optimization phase)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lmdb

# Default max size for one LMDB env (1 TB); can be overridden for huge datasets
DEFAULT_MAP_SIZE = 1024 * 1024 * 1024 * 1024  # 1 TB


def _encode_key(index: int, key_width: int = 10) -> bytes:
    """Encode integer index as fixed-width string key for ordering."""
    return f"{index:0{key_width}d}".encode("ascii")


def write_lmdb(
    samples: List[Tuple[Path, int]],
    out_path: Path,
    *,
    map_size: int = DEFAULT_MAP_SIZE,
    metadata: Optional[Dict[str, Any]] = None,
    key_width: int = 10,
) -> Dict[str, Any]:
    """Write image paths and integer labels to an LMDB environment.

    Each value is stored as pickle({"image": bytes, "label": int}). Keys are
    zero-padded indices "0000000000", "0000000001", ... for stable ordering.

    Args:
        samples: List of (image_path, class_index) in desired order.
        out_path: Directory path for the LMDB environment (created if missing).
        map_size: LMDB map size in bytes.
        metadata: Optional dict to store in manifest (source, version, etc.).
        key_width: Width of numeric key string.

    Returns:
        Manifest dict with count, key_width, metadata, and path.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(out_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        create=True,
    )

    manifest_meta = dict(metadata or {})
    manifest_meta["key_width"] = key_width

    try:
        write_idx = 0
        with env.begin(write=True) as txn:
            for path, label in samples:
                if not path.is_file():
                    continue
                image_bytes = path.read_bytes()
                value = pickle.dumps({"image": image_bytes, "label": int(label)})
                txn.put(_encode_key(write_idx, key_width), value)
                write_idx += 1
        count = write_idx
    finally:
        env.close()

    manifest = {
        "count": count,
        "version": 1,
        "path": str(out_path.resolve()),
        **manifest_meta,
    }
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
