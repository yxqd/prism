"""Tests for dataset scanner."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

try:
    from moto import mock_aws
except ImportError:
    mock_aws = None

from PIL import Image

from prism.dataset.scanner import (
    _is_image_key,
    _inspect_image,
    scan_s3_prefix,
    ScanResult,
    ImageMeta,
    write_metadata_jsonl,
)

pytestmark = pytest.mark.skipif(mock_aws is None, reason="moto not installed (pip install moto[s3])")


def test_is_image_key() -> None:
    assert _is_image_key("a.jpg") is True
    assert _is_image_key("a.JPEG") is True
    assert _is_image_key("a.png") is True
    assert _is_image_key("a.txt") is False
    assert _is_image_key("a") is False


def test_inspect_image_valid(tmp_path: Path) -> None:
    """Create a real small PNG and verify _inspect_image returns metadata."""
    from prism.storage.s3_client import get_s3_client

    img_path = tmp_path / "tiny.png"
    img = Image.new("RGB", (10, 20), color="red")
    img.save(img_path, format="PNG")
    data = img_path.read_bytes()

    with mock_aws():
        client = get_s3_client()
        client.create_bucket(Bucket="test-bucket")
        client.put_object(Bucket="test-bucket", Key="tiny.png", Body=data)
        meta = _inspect_image(client, "test-bucket", "tiny.png", len(data))

    assert meta.width == 10
    assert meta.height == 20
    assert meta.format == "PNG"
    assert meta.corrupted is False
    assert meta.size_bytes == len(data)


def test_inspect_image_corrupted() -> None:
    from prism.storage.s3_client import get_s3_client

    with mock_aws():
        client = get_s3_client()
        client.create_bucket(Bucket="test-bucket")
        client.put_object(Bucket="test-bucket", Key="bad.png", Body=b"not an image")
        meta = _inspect_image(client, "test-bucket", "bad.png", 12)

    assert meta.corrupted is True
    assert meta.error is not None


@mock_aws
def test_scan_s3_prefix_empty() -> None:
    from prism.storage.s3_client import get_s3_client

    client = get_s3_client()
    client.create_bucket(Bucket="test-bucket")
    result = scan_s3_prefix("test-bucket", "prefix/", client=client)
    assert result.total == 0
    assert result.corrupted == 0
    assert result.avg_size_mb == 0.0


@mock_aws
def test_scan_s3_prefix_two_images() -> None:
    from prism.storage.s3_client import get_s3_client

    client = get_s3_client()
    client.create_bucket(Bucket="test-bucket")
    img = Image.new("RGB", (5, 5), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    client.put_object(Bucket="test-bucket", Key="p/a.png", Body=data)
    client.put_object(Bucket="test-bucket", Key="p/b.png", Body=data)
    client.put_object(Bucket="test-bucket", Key="p/readme.txt", Body=b"text")  # ignored

    result = scan_s3_prefix("test-bucket", "p/", client=client, max_workers=2)
    assert result.total == 2
    assert result.corrupted == 0
    assert len(result.entries) == 2
    for e in result.entries:
        assert e.width == 5 and e.height == 5


def test_write_metadata_jsonl(tmp_path: Path) -> None:
    result = ScanResult(total=2, corrupted=0, total_size_bytes=100)
    result.entries = [
        ImageMeta(key="a.png", size_bytes=50, width=10, height=10),
        ImageMeta(key="b.png", size_bytes=50, width=20, height=20),
    ]
    path = tmp_path / "meta" / "out.jsonl"
    write_metadata_jsonl(result, str(path))
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    import json
    row = json.loads(lines[0])
    assert row["key"] == "a.png" and row["width"] == 10
