"""Recursively scan S3 prefix, collect image metadata, validate, output JSONL."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from PIL import Image
from tqdm import tqdm

from prism import config
from prism.storage.s3_client import get_s3_client

# Common image extensions we treat as images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
# Formats we consider expected (Pillow format names)
EXPECTED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP", "TIFF"}


@dataclass
class ImageMeta:
    """Metadata for one image object."""
    key: str
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    corrupted: bool = False
    unexpected_format: bool = False
    exif_present: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "key": self.key,
            "size_bytes": self.size_bytes,
        }
        if self.width is not None:
            d["width"] = self.width
        if self.height is not None:
            d["height"] = self.height
        if self.format is not None:
            d["format"] = self.format
        d["corrupted"] = self.corrupted
        d["unexpected_format"] = self.unexpected_format
        d["exif_present"] = self.exif_present
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class ScanResult:
    """Aggregate result of scanning an S3 prefix."""
    total: int = 0
    corrupted: int = 0
    unexpected_format: int = 0
    total_size_bytes: int = 0
    entries: List[ImageMeta] = field(default_factory=list)

    @property
    def avg_size_mb(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.total_size_bytes / self.total) / (1024 * 1024)


def _is_image_key(key: str) -> bool:
    key_lower = key.lower()
    return any(key_lower.endswith(ext) for ext in IMAGE_EXTENSIONS)


def _inspect_image(client: Any, bucket: str, key: str, size_bytes: int) -> ImageMeta:
    """Fetch object, open with Pillow; return metadata or mark corrupted."""
    meta = ImageMeta(key=key, size_bytes=size_bytes)
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
    except Exception as e:
        meta.corrupted = True
        meta.error = str(e)
        return meta

    try:
        img = Image.open(io.BytesIO(body))
        img.load()  # force read to detect truncated/corrupt
    except Exception as e:
        meta.corrupted = True
        meta.error = str(e)
        return meta

    meta.width = img.width
    meta.height = img.height
    meta.format = img.format
    if meta.format and meta.format.upper() not in EXPECTED_FORMATS:
        meta.unexpected_format = True
    meta.exif_present = bool(img.getexif())
    return meta


def scan_s3_prefix(
    bucket: str,
    prefix: str,
    client: Optional[Any] = None,
    max_workers: int = 8,
    show_progress: bool = True,
) -> ScanResult:
    """List image objects under prefix, fetch each to get dimensions and validate.

    Returns ScanResult with total count, corrupted count, unexpected format count,
    total size, and list of ImageMeta entries.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = client or get_s3_client()
    keys_sizes: List[tuple] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if not _is_image_key(key):
                continue
            size = obj.get("Size", 0)
            keys_sizes.append((key, size))

    result = ScanResult(total=len(keys_sizes))
    result.total_size_bytes = sum(s for _, s in keys_sizes)

    if not keys_sizes:
        return result

    def task(key: str, size: int) -> ImageMeta:
        return _inspect_image(client, bucket, key, size)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, k, s): (k, s) for k, s in keys_sizes}
        with tqdm(total=len(keys_sizes), desc="Scanning", unit="image", disable=not show_progress) as pbar:
            for fut in as_completed(futures):
                meta = fut.result()
                result.entries.append(meta)
                if meta.corrupted:
                    result.corrupted += 1
                if meta.unexpected_format:
                    result.unexpected_format += 1
                pbar.update(1)

    return result


def write_metadata_jsonl(result: ScanResult, path: str) -> None:
    """Write result entries as JSONL to a local file."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in result.entries:
            f.write(json.dumps(entry.to_dict()) + "\n")


def upload_metadata_to_s3(
    result: ScanResult,
    metadata_key: str,
    bucket: Optional[str] = None,
    client: Optional[Any] = None,
) -> str:
    """Write result entries as JSONL and upload to S3. Returns the key used."""
    bucket = bucket or config.BUCKET_PROCESSED
    client = client or get_s3_client()
    lines = [json.dumps(entry.to_dict()) + "\n" for entry in result.entries]
    body = "".join(lines).encode("utf-8")
    client.put_object(Bucket=bucket, Key=metadata_key, Body=body, ContentType="application/x-ndjson")
    return metadata_key
