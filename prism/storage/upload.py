"""Upload files to S3 with multipart for large files and optional progress."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from prism.storage.s3_client import get_s3_client

# boto3 uses multipart for uploads > 8MB by default (Configurable in TransferConfig)
MB = 1024 * 1024
MULTIPART_THRESHOLD = 8 * MB


def upload_file(
    local_path: Path,
    bucket: str,
    key: str,
    client: Optional[Any] = None,
    show_progress: bool = True,
) -> str:
    """Upload a file to S3. Uses multipart for large files (boto3 default).

    Returns the key (for convenience).
    """
    client = client or get_s3_client()
    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    file_size = path.stat().st_size
    if show_progress and file_size > 0:
        with tqdm(total=file_size, desc=path.name, unit="B", unit_scale=True) as pbar:
            def _progress(bytes_transferred: int) -> None:
                pbar.update(bytes_transferred - pbar.n)
            client.upload_file(
                str(path),
                bucket,
                key,
                Callback=_progress,
            )
    else:
        client.upload_file(str(path), bucket, key)

    return key


def upload_bytes(
    data: bytes,
    bucket: str,
    key: str,
    client: Optional[Any] = None,
) -> str:
    """Upload bytes to S3 (e.g. JSONL content). No multipart for small payloads."""
    client = client or get_s3_client()
    client.put_object(Bucket=bucket, Key=key, Body=data)
    return key
