"""Tests for S3 client, download, upload."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

# moto must be imported before boto3 for S3 mocking
try:
    from moto import mock_aws
except ImportError:
    mock_aws = None

from prism.storage.s3_client import get_s3_client
from prism.storage.download import list_objects_under_prefix, download_prefix, download_one
from prism.storage.upload import upload_file, upload_bytes


pytestmark = pytest.mark.skipif(mock_aws is None, reason="moto not installed (pip install moto[s3])")


@mock_aws
def test_get_s3_client() -> None:
    client = get_s3_client()
    assert client is not None
    # Can create bucket (moto)
    client.create_bucket(Bucket="test-bucket")


@mock_aws
def test_upload_bytes_and_list() -> None:
    client = get_s3_client()
    client.create_bucket(Bucket="test-bucket")
    upload_bytes(b"hello", "test-bucket", "foo/bar.txt", client=client)
    keys_sizes = list_objects_under_prefix(client, "test-bucket", "foo/")
    assert len(keys_sizes) == 1
    assert keys_sizes[0][0] == "foo/bar.txt"
    assert keys_sizes[0][1] == 5


@mock_aws
def test_upload_file_and_download_one(tmp_path: Path) -> None:
    client = get_s3_client()
    client.create_bucket(Bucket="test-bucket")
    local = tmp_path / "local.txt"
    local.write_text("content")
    upload_file(local, "test-bucket", "data/local.txt", client=client, show_progress=False)
    out = tmp_path / "out.txt"
    download_one(client, "test-bucket", "data/local.txt", out, show_progress=False)
    assert out.read_text() == "content"


@mock_aws
def test_download_prefix(tmp_path: Path) -> None:
    client = get_s3_client()
    client.create_bucket(Bucket="test-bucket")
    for i in range(3):
        client.put_object(Bucket="test-bucket", Key=f"prefix/f{i}.txt", Body=f"data{i}".encode())
    local_dir = tmp_path / "cache"
    paths = download_prefix(
        "test-bucket", "prefix/", local_dir, client=client, max_workers=2, show_progress=False
    )
    assert len(paths) == 3
    for i in range(3):
        p = local_dir / "prefix" / f"f{i}.txt"
        assert p.exists()
        assert p.read_text() == f"data{i}"
