"""Single-GPU DALI pipeline over WebDataset shards for Tiny-ImageNet-style data.

This is the Step 3 building block from `towards_ray+dali.md`:
it reads `.tar` shards produced by `scripts/shard_dataset.py`, performs
GPU-side decode and basic augment/normalize, and exposes a PyTorch-compatible
iterator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from nvidia.dali import fn, types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy


# ImageNet normalization (channel-wise, in [0,1] space).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _shard_paths_from_dir(root: Path) -> List[str]:
    """Return sorted list of WebDataset shard paths (*.tar) under root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"WebDataset shard directory not found: {root}")
    paths = sorted(p for p in root.glob("*.tar") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No .tar shards found under {root}")
    return [str(p) for p in paths]


@pipeline_def
def _webdataset_pipeline(
    paths: Sequence[str],
    image_size: int = 224,
    random_mirror: bool = True,
):
    """DALI pipeline: WebDataset reader -> GPU decode -> resize/crop/normalize."""
    # `ext` names match WebDataset keys written by prism.sharding.writer (jpg + numeric cls).
    # Labels are stored as fixed-width int32 NumPy arrays, so we can ask DALI
    # to interpret them as INT32 tensors directly.
    jpegs, labels = fn.readers.webdataset(
        paths=paths,
        ext=["jpg", "cls"],
        dtypes=[types.UINT8, types.INT32],
        random_shuffle=True,
        shard_id=0,
        num_shards=1,
        pad_last_batch=True,
        name="Reader",
    )

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    # Standard "resize shorter" then random crop to image_size x image_size
    images = fn.resize(images, resize_shorter=256)
    images = fn.random_resized_crop(
        images,
        size=(image_size, image_size),
        random_area=[0.8, 1.0],
        random_aspect_ratio=[3.0 / 4.0, 4.0 / 3.0],
    )

    mirror = fn.random.coin_flip(probability=0.5) if random_mirror else 0

    # Convert to NCHW float32 on GPU and apply ImageNet normalization.
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mirror=mirror,
        mean=[m * 255.0 for m in IMAGENET_MEAN],
        std=[s * 255.0 for s in IMAGENET_STD],
    )

    # Cast labels to int64 and squeeze singleton dimension so PyTorch sees (N,)
    labels = fn.cast(labels, dtype=types.INT64)
    labels = fn.squeeze(labels, axes=[1])
    return images, labels


def create_webdataset_dali_iterator(
    shards_root: Path | str,
    batch_size: int,
    device_id: int = 0,
    num_threads: int = 4,
    image_size: int = 224,
) -> DALIClassificationIterator:
    """Create a single-GPU DALI iterator over WebDataset shards.

    Args:
        shards_root: Directory containing *.tar shards (e.g. data/sharded/train).
        batch_size: Global batch size for this GPU.
        device_id: CUDA device index.
        num_threads: DALI worker threads.
        image_size: Final square crop size.

    Returns:
        A DALIClassificationIterator that yields batches compatible with PyTorch:
        [{'data': images, 'label': labels}], where `data` is float32 (N, C, H, W)
        on GPU and `label` is int64 (N, 1).
    """
    shard_paths = _shard_paths_from_dir(Path(shards_root))

    pipe = _webdataset_pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        paths=shard_paths,
        image_size=image_size,
    )
    pipe.build()

    iterator = DALIClassificationIterator(
        pipelines=pipe,
        reader_name="Reader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )
    return iterator

