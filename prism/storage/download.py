"""Download S3 prefix to local cache with concurrency and progress."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

from prism.storage.s3_client import get_s3_client


def list_objects_under_prefix(
    client: Any,
    bucket: str,
    prefix: str,
    extensions: Optional[set] = None,
) -> List[Tuple[str, int]]:
    """List object keys and sizes under prefix. Returns [(key, size_bytes), ...]."""
    extensions = extensions or set()
    out: List[Tuple[str, int]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            size = obj.get("Size", 0)
            if extensions and not any(key.lower().endswith(ext) for ext in extensions):
                continue
            out.append((key, size))
    return out


def download_one(
    client: Any,
    bucket: str,
    key: str,
    local_path: Path,
    show_progress: bool = False,
) -> Path:
    """Download a single object to local_path. Returns path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if show_progress:
        # boto3 transfer doesn't give per-file progress easily; we use tqdm in download_prefix
        client.download_file(bucket, key, str(local_path))
    else:
        client.download_file(bucket, key, str(local_path))
    return local_path


def download_prefix(
    bucket: str,
    prefix: str,
    local_dir: Path,
    client: Optional[Any] = None,
    max_workers: int = 8,
    image_extensions_only: bool = False,
    show_progress: bool = True,
) -> List[Path]:
    """Download all objects under prefix to local_dir with concurrent workers.

    If image_extensions_only is True, only keys ending in common image extensions are downloaded.
    Returns list of local paths.
    """
    client = client or get_s3_client()
    image_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
    extensions = set(image_ext) if image_extensions_only else None
    keys_sizes = list_objects_under_prefix(client, bucket, prefix, extensions=extensions)
    if not keys_sizes:
        return []

    local_dir = Path(local_dir)
    downloaded: List[Path] = []

    def task(item: Tuple[str, int]) -> Path:
        key, _ = item
        # Preserve key structure under local_dir
        local_path = local_dir / key
        return download_one(client, bucket, key, local_path, show_progress=False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, item): item for item in keys_sizes}
        with tqdm(total=len(keys_sizes), desc="Downloading", unit="file", disable=not show_progress) as pbar:
            for fut in as_completed(futures):
                try:
                    path = fut.result()
                    downloaded.append(path)
                except Exception:
                    raise
                pbar.update(1)

    return downloaded
