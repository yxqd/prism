"""PyTorch Dataset over LMDB (implemented in PyTorch Integration phase)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from prism.storage.lmdb_reader import LMDBReader


class LMDBDataset(Dataset[Tuple[Any, int]]):
    """torch.utils.data.Dataset over an LMDB dataset produced by lmdb_writer.

    Each __getitem__ returns (image, label) where image is the result of applying
    transform to the decoded PIL Image (or the raw PIL Image if transform is None).
    """

    def __init__(
        self,
        env_path: Path | str,
        *,
        transform: Optional[Callable[[Image.Image], Any]] = None,
    ):
        self.reader = LMDBReader(env_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        image_bytes, label = self.reader.get(index)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def manifest(self) -> dict:
        return self.reader.manifest

    def __del__(self) -> None:
        if hasattr(self, "reader") and self.reader is not None:
            self.reader.close()
