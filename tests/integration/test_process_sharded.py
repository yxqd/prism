"""Integration test: run_workflow with sharded S3 source (moto-backed)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import webdataset as wds
from PIL import Image

try:
    from moto import mock_aws
except ImportError:
    mock_aws = None

from prism.processing.pipeline import Pipeline
from prism.processing.workflow import run_workflow
from prism.storage.download import list_objects_under_prefix
from prism.storage.s3_client import get_s3_client

pytestmark = pytest.mark.skipif(mock_aws is None, reason="moto not installed (pip install moto[s3])")


def _make_tiny_jpeg_bytes() -> bytes:
    """Return valid JPEG bytes (PIL can open it)."""
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :, 0] = 64
    arr[:, :, 1] = 128
    arr[:, :, 2] = 192
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

BUCKET = "prism-processed"
PREFIX = "sharded/tiny-imagenet-200/train/"


def _make_tar_shard_with_two_samples() -> bytes:
    """Build a WebDataset .tar in memory with 2 image samples."""
    jpg_bytes = _make_tiny_jpeg_bytes()
    buf = io.BytesIO()
    with wds.TarWriter(fileobj=buf) as sink:
        sink.write({"__key__": "img0", "jpg": jpg_bytes, "cls": "n02106662"})
        sink.write({"__key__": "img1", "jpg": jpg_bytes, "cls": "n01930112"})
    return buf.getvalue()


def _make_metadata_parquet_bytes() -> bytes:
    """Build metadata Parquet for 2 samples in one shard (train_00000.tar)."""
    table = pa.table({
        "shard": ["train_00000.tar", "train_00000.tar"],
        "shard_idx": [0, 0],
        "sample_idx": [0, 1],
        "__key__": ["img0", "img1"],
        "cls": ["n02106662", "n01930112"],
        "source": ["train/n02106662/img0.jpg", "train/n01930112/img1.jpg"],
    })
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@mock_aws
def test_run_workflow_sharded_s3(tmp_path: Path) -> None:
    """Run workflow on sharded S3 source: metadata + .tar, then assert processed outputs uploaded."""
    client = get_s3_client()
    client.create_bucket(Bucket=BUCKET)

    # Upload one shard
    tar_bytes = _make_tar_shard_with_two_samples()
    client.put_object(Bucket=BUCKET, Key=PREFIX + "train_00000.tar", Body=tar_bytes)

    # Upload metadata
    meta_bytes = _make_metadata_parquet_bytes()
    client.put_object(Bucket=BUCKET, Key=PREFIX + "metadata/part.0.parquet", Body=meta_bytes)

    pipeline = Pipeline().resize((16, 16)).normalize()
    timings = run_workflow(
        source_bucket=BUCKET,
        source_prefix=PREFIX,
        pipeline=pipeline,
        cache_dir=tmp_path,
        dest_bucket=BUCKET,
        dest_prefix=PREFIX,
        client=client,
        use_dask=False,
        show_progress=False,
        source_is_sharded=True,
    )

    assert "metadata_load" in timings
    assert "download" in timings
    assert "process" in timings

    # Sharded output: one .tar per shard under prefix (overwrites or adds output shard)
    all_keys = list_objects_under_prefix(client, BUCKET, PREFIX, extensions=None)
    tar_keys = [k for k, _ in all_keys if k.endswith(".tar")]
    assert len(tar_keys) >= 1, f"Expected at least one .tar under {PREFIX}, got {all_keys}"
    assert any("train_00000.tar" in k for k in tar_keys)
    # Verify output tar contains processed samples (img0, img1)
    out_shard_key = PREFIX + "train_00000.tar"
    resp = client.get_object(Bucket=BUCKET, Key=out_shard_key)
    out_tar = tmp_path / "out.tar"
    out_tar.write_bytes(resp["Body"].read())
    keys = [s["__key__"] for s in wds.WebDataset(str(out_tar))]
    assert set(keys) == {"img0", "img1"}

    # Checkpoint has processed_keys
    job_id = f"{BUCKET}_{PREFIX.replace('/', '_').strip('_')}"
    checkpoint_path = tmp_path / "workflow" / job_id / "workflow_checkpoint.json"
    assert checkpoint_path.is_file()
    import json
    ck = json.loads(checkpoint_path.read_text())
    assert "processed_keys" in ck
    assert set(ck["processed_keys"]) == {"img0", "img1"}


@mock_aws
def test_run_workflow_sharded_s3_with_dask(tmp_path: Path) -> None:
    """Run sharded workflow with Dask: one task per shard, workers download and process in parallel."""
    from dask.distributed import Client, LocalCluster

    client = get_s3_client()
    client.create_bucket(Bucket=BUCKET)

    tar_bytes = _make_tar_shard_with_two_samples()
    client.put_object(Bucket=BUCKET, Key=PREFIX + "train_00000.tar", Body=tar_bytes)
    meta_bytes = _make_metadata_parquet_bytes()
    client.put_object(Bucket=BUCKET, Key=PREFIX + "metadata/part.0.parquet", Body=meta_bytes)

    pipeline = Pipeline().resize((16, 16)).normalize()
    # processes=False so workers run in same process and see moto's S3 mock
    with LocalCluster(processes=False, n_workers=2) as cluster:
        with Client(cluster) as dask_client:
            timings = run_workflow(
                source_bucket=BUCKET,
                source_prefix=PREFIX,
                pipeline=pipeline,
                cache_dir=tmp_path,
                dest_bucket=BUCKET,
                dest_prefix=PREFIX,
                client=client,
                use_dask=True,
                dask_client=dask_client,
                show_progress=False,
                source_is_sharded=True,
            )

    assert "metadata_load" in timings
    assert "process" in timings

    # Sharded output: one .tar per shard
    all_keys = list_objects_under_prefix(client, BUCKET, PREFIX, extensions=None)
    tar_keys = [k for k, _ in all_keys if k.endswith(".tar")]
    assert len(tar_keys) >= 1
    assert any("train_00000.tar" in k for k in tar_keys)
    resp = client.get_object(Bucket=BUCKET, Key=PREFIX + "train_00000.tar")
    out_tar = tmp_path / "out_dask.tar"
    out_tar.write_bytes(resp["Body"].read())
    keys = [s["__key__"] for s in wds.WebDataset(str(out_tar))]
    assert set(keys) == {"img0", "img1"}

    job_id = f"{BUCKET}_{PREFIX.replace('/', '_').strip('_')}"
    checkpoint_path = tmp_path / "workflow" / job_id / "workflow_checkpoint.json"
    assert checkpoint_path.is_file()
    import json
    ck = json.loads(checkpoint_path.read_text())
    assert set(ck["processed_keys"]) == {"img0", "img1"}
