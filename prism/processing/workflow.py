"""S3 pull -> process -> push workflow with checkpointing and optional Dask."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from prism import config
from prism.storage.download import (
    download_one,
    list_objects_under_prefix,
)
from prism.storage.s3_client import get_s3_client
from prism.storage.upload import upload_file

CHECKPOINT_FILENAME = "workflow_checkpoint.json"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


def _timed(phase_name: str, timings: Dict[str, float]):
    """Decorator that records duration of a function into timings[phase_name]."""

    def decorator(fn: Callable[..., Any]):
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            timings[phase_name] = time.perf_counter() - t0
            return out

        return wrapper

    return decorator


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.is_file():
        return {"downloaded_keys": [], "processed_keys": []}
    try:
        with open(checkpoint_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"downloaded_keys": [], "processed_keys": []}


def _save_checkpoint(checkpoint_path: Path, data: Dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


from prism.processing.workflow_sharded import run_workflow_sharded, source_is_sharded as detect_source_sharded


def run_workflow(
    source_bucket: str,
    source_prefix: str,
    pipeline: Any,
    cache_dir: Optional[Path] = None,
    dest_bucket: Optional[str] = None,
    dest_prefix: Optional[str] = None,
    client: Optional[Any] = None,
    use_dask: bool = False,
    dask_client: Optional[Any] = None,
    max_download_workers: int = 8,
    show_progress: bool = True,
    source_is_sharded: Optional[bool] = None,
    progress_chunk_size: int = 100,
) -> Dict[str, float]:
    """Pull from S3, run pipeline on each image, push to S3. Returns timing dict.

    If source_is_sharded is True (or auto-detected: metadata/*.parquet and *.tar under prefix),
    loads the Parquet manifest and processes samples from sharded .tar files. Checkpoint tracks
    processed sample __key__s. Otherwise treats prefix as flat image keys.

    Checkpoint is written under cache_dir so that on crash you can re-run and resume.
    """
    cache_dir = cache_dir or config.CACHE_DIR
    dest_bucket = dest_bucket or config.BUCKET_PROCESSED
    dest_prefix = dest_prefix or source_prefix
    client = client or get_s3_client()

    job_id = f"{source_bucket}_{source_prefix.replace('/', '_').strip('_')}"
    work_dir = Path(cache_dir) / "workflow" / job_id
    download_dir = work_dir / "download"
    processed_dir = work_dir / "processed"
    checkpoint_path = work_dir / CHECKPOINT_FILENAME
    timings: Dict[str, float] = {}

    use_sharded = source_is_sharded if source_is_sharded is not None else detect_source_sharded(client, source_bucket, source_prefix)
    if use_sharded:
        return run_workflow_sharded(
            source_bucket=source_bucket,
            source_prefix=source_prefix,
            pipeline=pipeline,
            work_dir=work_dir,
            checkpoint_path=checkpoint_path,
            dest_bucket=dest_bucket,
            dest_prefix=dest_prefix,
            client=client,
            use_dask=use_dask,
            dask_client=dask_client,
            timings=timings,
            show_progress=show_progress,
            progress_chunk_size=progress_chunk_size,
        )

    # List keys (image extensions only)
    keys_sizes = list_objects_under_prefix(
        client,
        source_bucket,
        source_prefix,
        extensions={e.lower() for e in IMAGE_EXT},
    )
    all_keys = [k for k, _ in keys_sizes]
    if not all_keys:
        return {"download": 0.0, "process": 0.0, "upload": 0.0}

    ck = _load_checkpoint(checkpoint_path)
    downloaded_keys: List[str] = list(ck.get("downloaded_keys", []))
    processed_keys: List[str] = list(ck.get("processed_keys", []))

    # Phase 1: download missing
    to_download = [k for k in all_keys if k not in downloaded_keys]

    @_timed("download", timings)
    def do_download() -> None:
        download_dir.mkdir(parents=True, exist_ok=True)
        for key in to_download:
            local_path = download_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            download_one(client, source_bucket, key, local_path, show_progress=False)
        if to_download:
            downloaded_keys.extend(to_download)
            _save_checkpoint(checkpoint_path, {"downloaded_keys": downloaded_keys, "processed_keys": processed_keys})

    do_download()

    # Local paths for all keys (we have them all after download)
    key_to_path = {k: download_dir / k for k in all_keys}
    to_process = [k for k in all_keys if k not in processed_keys]

    def process_one(key: str) -> Tuple[str, Path]:
        local_path = key_to_path[key]
        img = np.array(Image.open(local_path).convert("RGB"))
        out_arr = pipeline.apply(img)
        if out_arr.dtype in (np.float32, np.float64):
            out_arr = (np.clip(out_arr, 0, 1) * 255).astype(np.uint8)
        else:
            out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_arr)
        out_path = processed_dir / key
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, quality=95)
        return key, out_path

    @_timed("process", timings)
    def do_process() -> None:
        if use_dask and dask_client is not None:
            from dask import delayed

            delayed_tasks = [delayed(process_one)(k) for k in to_process]
            futures = dask_client.compute(delayed_tasks)
            results = dask_client.gather(futures)
            for key, _ in results:
                processed_keys.append(key)
        else:
            for key in to_process:
                _, out_path = process_one(key)
                processed_keys.append(key)
        if to_process:
            _save_checkpoint(checkpoint_path, {"downloaded_keys": downloaded_keys, "processed_keys": processed_keys})

    do_process()

    # Phase 3: upload processed (only the ones we just processed)
    to_upload = [(k, processed_dir / k) for k in to_process if (processed_dir / k).is_file()]

    @_timed("upload", timings)
    def do_upload() -> None:
        for key, local_path in to_upload:
            dest_key = (dest_prefix.rstrip("/") + "/" + key) if dest_prefix else key
            upload_file(local_path, dest_bucket, dest_key, client=client, show_progress=False)

    do_upload()

    return timings


def print_timing_summary(timings: Dict[str, float]) -> None:
    """Print a one-line summary of phase timings."""
    parts = [f"{k} {v:.1f}s" for k, v in timings.items()]
    print("Timing: " + ", ".join(parts))
