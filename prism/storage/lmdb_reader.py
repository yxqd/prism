"""LMDB reader for training datasets (implemented in Storage Optimization phase)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import lmdb


def load_manifest(env_path: Path) -> Dict[str, Any]:
    """Load manifest.json from an LMDB directory. Raises if missing."""
    manifest_path = Path(env_path) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)


class LMDBReader:
    """Random access and iteration over an LMDB dataset written by lmdb_writer."""

    def __init__(
        self,
        env_path: Path | str,
        *,
        readonly: bool = True,
        lock: bool = False,
    ):
        self.env_path = Path(env_path)
        self._env = lmdb.open(
            str(self.env_path),
            readonly=readonly,
            lock=lock,
            subdir=True,
            create=False,
        )
        self._manifest: Optional[Dict[str, Any]] = None

    @property
    def manifest(self) -> Dict[str, Any]:
        if self._manifest is None:
            self._manifest = load_manifest(self.env_path)
        return self._manifest

    @property
    def key_width(self) -> int:
        return int(self.manifest.get("key_width", 10))

    def __len__(self) -> int:
        return int(self.manifest["count"])

    def _key_at(self, index: int) -> bytes:
        return f"{index:0{self.key_width}d}".encode("ascii")

    def get(self, index: int) -> Tuple[bytes, int]:
        """Return (image_bytes, label) for the given index. Raises KeyError if missing."""
        key = self._key_at(index)
        with self._env.begin() as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(f"Key not found: {index}")
        record = pickle.loads(value)
        return record["image"], record["label"]

    def keys(self) -> List[int]:
        """Return sorted list of all key indices."""
        return list(range(len(self)))

    def __iter__(self) -> Iterator[Tuple[int, bytes, int]]:
        """Yield (index, image_bytes, label) in order."""
        for i in range(len(self)):
            img_bytes, label = self.get(i)
            yield i, img_bytes, label

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> "LMDBReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
