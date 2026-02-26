"""Sharded S3 workflow: metadata + .tar shards, optional Dask. Output is one .tar per shard."""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import webdataset as wds

from prism.storage.download import download_one, list_objects_under_prefix
from prism.storage.s3_client import get_s3_client
from prism.storage.upload import upload_file

from prism.processing.workflow import _load_checkpoint, _save_checkpoint, _timed
from prism.sharding.writer import write_one_shard

METADATA_SUBPATH = "metadata/"


def _load_metadata_from_s3(client: Any, bucket: str, prefix: str) -> Any:
    """Load the shard metadata Parquet from S3 (prefix/metadata/part.0.parquet or first .parquet)."""
    list_prefix = prefix.rstrip("/") + "/" + METADATA_SUBPATH
    keys_sizes = list_objects_under_prefix(client, bucket, list_prefix, extensions={".parquet"})
    if not keys_sizes:
        raise FileNotFoundError(f"No metadata Parquet under s3://{bucket}/{list_prefix}")
    key = keys_sizes[0][0]
    resp = client.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    return pq.read_table(buf)


def source_is_sharded(client: Any, bucket: str, prefix: str) -> bool:
    """Return True if prefix contains metadata/*.parquet and at least one .tar shard."""
    keys_sizes = list_objects_under_prefix(client, bucket, prefix, extensions=None)
    all_keys = [k for k, _ in keys_sizes]
    has_meta = any(METADATA_SUBPATH in k and k.endswith(".parquet") for k in all_keys)
    has_tar = any(k.endswith(".tar") for k in all_keys)
    return bool(has_meta and has_tar)


def process_one_shard(
    shard_name: str,
    rows_in_shard: List[Tuple[int, str, Optional[str], str]],
    source_bucket: str,
    source_prefix: str,
    dest_bucket: str,
    dest_prefix: str,
    work_dir_str: str,
    pipeline: Any,
    progress_variable_name: Optional[str] = None,
    progress_chunk_size: int = 100,
) -> List[str]:
    """Process one shard on a worker: download shard, process images, write local tar, upload tar. Returns list of __key__s.

    Module-level and picklable for Dask submit. Creates its own S3 client inside.
    Saves processed images locally, packs one .tar per shard, uploads that tar (one PUT per shard).
    """
    work_dir = Path(work_dir_str)
    download_dir = work_dir / "download"
    processed_dir = work_dir / "processed"
    out_tars_dir = work_dir / "out_tars"
    client = get_s3_client()

    local_tar = download_dir / shard_name
    local_tar.parent.mkdir(parents=True, exist_ok=True)
    if not local_tar.is_file():
        shard_key = (source_prefix.rstrip("/") + "/" + shard_name) if source_prefix else shard_name
        download_one(client, source_bucket, shard_key, local_tar, show_progress=False)

    processed_dir.mkdir(parents=True, exist_ok=True)
    needed_idxs = {r[0] for r in rows_in_shard}
    idx_to_row = {r[0]: r for r in rows_in_shard}
    done: List[str] = []

    var = None
    pending = 0
    if progress_variable_name:
        try:
            from dask.distributed import Variable, get_client

            var = Variable(name=progress_variable_name, client=get_client())
        except Exception:
            var = None

    for idx, sample in enumerate(wds.WebDataset(str(local_tar))):
        if idx not in needed_idxs:
            continue
        _, out_key, cls, _ = idx_to_row[idx]
        if "jpg" not in sample:
            continue
        img = np.array(Image.open(io.BytesIO(sample["jpg"])).convert("RGB"))
        out_arr = pipeline.apply(img)
        if out_arr.dtype in (np.float32, np.float64):
            out_arr = (np.clip(out_arr, 0, 1) * 255).astype(np.uint8)
        else:
            out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_arr)
        out_path = processed_dir / (out_key + ".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, quality=95)
        done.append(out_key)

        if var is not None and progress_chunk_size > 0:
            pending += 1
            if pending >= progress_chunk_size:
                try:
                    current = var.get()
                    var.set(current + pending)
                except Exception:
                    pass
                pending = 0

    if var is not None and pending > 0:
        try:
            current = var.get()
            var.set(current + pending)
        except Exception:
            pass

    if not done:
        return done

    # Build one .tar from processed images and upload it (one PUT per shard)
    key_to_cls = {r[1]: r[2] for r in rows_in_shard}
    samples = [(processed_dir / (k + ".jpg"), key_to_cls.get(k)) for k in done]
    out_tars_dir.mkdir(parents=True, exist_ok=True)
    local_out_tar = out_tars_dir / shard_name
    write_one_shard(samples, local_out_tar)
    dest_key = (dest_prefix.rstrip("/") + "/" + shard_name) if dest_prefix else shard_name
    upload_file(local_out_tar, dest_bucket, dest_key, client=client, show_progress=False)
    return done


def run_workflow_sharded(
    source_bucket: str,
    source_prefix: str,
    pipeline: Any,
    work_dir: Path,
    checkpoint_path: Path,
    dest_bucket: str,
    dest_prefix: str,
    client: Any,
    use_dask: bool,
    dask_client: Optional[Any],
    timings: Dict[str, float],
    show_progress: bool,
    progress_chunk_size: int = 100,
) -> Dict[str, float]:
    """Run workflow for a sharded S3 source. Output is one .tar per shard under dest_prefix."""
    from tqdm import tqdm

    download_dir = work_dir / "download"
    processed_dir = work_dir / "processed"

    @_timed("metadata_load", timings)
    def load_meta() -> Any:
        return _load_metadata_from_s3(client, source_bucket, source_prefix)

    table = load_meta()
    rows = []
    for i in range(table.num_rows):
        rows.append((
            table.column("shard")[i].as_py(),
            table.column("sample_idx")[i].as_py(),
            table.column("__key__")[i].as_py(),
            table.column("cls")[i].as_py() if table.column("cls")[i].as_py() is not None else None,
            table.column("source")[i].as_py(),
        ))
    if not rows:
        return timings

    ck = _load_checkpoint(checkpoint_path)
    processed_keys: List[str] = list(ck.get("processed_keys", []))
    processed_set = set(processed_keys)
    to_process = [r for r in rows if r[2] not in processed_set]
    if not to_process:
        return timings

    shard_to_rows: Dict[str, List[Tuple[int, str, Optional[str], str]]] = {}
    for shard, sample_idx, key, cls, source in to_process:
        shard_to_rows.setdefault(shard, []).append((sample_idx, key, cls, source))
    needed_shards = sorted(shard_to_rows.keys())

    def process_sample(img_arr: np.ndarray, out_key: str) -> Path:
        out_arr = pipeline.apply(img_arr)
        if out_arr.dtype in (np.float32, np.float64):
            out_arr = (np.clip(out_arr, 0, 1) * 255).astype(np.uint8)
        else:
            out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_arr)
        out_path = processed_dir / (out_key + ".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, quality=95)
        return out_path

    def do_shard(
        shard_name: str,
        rows_in_shard: List[Tuple[int, str, Optional[str], str]],
        pbar: Optional[Any] = None,
    ) -> List[str]:
        local_tar = download_dir / shard_name
        if not local_tar.is_file():
            shard_key = (source_prefix.rstrip("/") + "/" + shard_name) if source_prefix else shard_name
            download_one(client, source_bucket, shard_key, local_tar, show_progress=False)
        needed_idxs = {r[0] for r in rows_in_shard}
        idx_to_row = {r[0]: r for r in rows_in_shard}
        done = []
        for idx, sample in enumerate(wds.WebDataset(str(local_tar))):
            if idx not in needed_idxs:
                continue
            _, out_key, cls, _ = idx_to_row[idx]
            if "jpg" not in sample:
                continue
            img = np.array(Image.open(io.BytesIO(sample["jpg"])).convert("RGB"))
            process_sample(img, out_key)
            done.append((out_key, cls))
            if pbar is not None:
                pbar.update(1)
        if not done:
            return []
        out_tars_dir = work_dir / "out_tars"
        out_tars_dir.mkdir(parents=True, exist_ok=True)
        local_out_tar = out_tars_dir / shard_name
        samples = [(processed_dir / (k + ".jpg"), c) for k, c in done]
        write_one_shard(samples, local_out_tar)
        dest_key = (dest_prefix.rstrip("/") + "/" + shard_name) if dest_prefix else shard_name
        upload_file(local_out_tar, dest_bucket, dest_key, client=client, show_progress=False)
        return [k for k, _ in done]

    if use_dask and dask_client is not None:
        from dask.distributed import Variable, as_completed

        @_timed("process", timings)
        def phase_process_dask() -> None:
            total_images = sum(len(rows) for rows in shard_to_rows.values())
            progress_var: Any = None
            if show_progress:
                progress_var = Variable(
                    name=f"workflow-progress-{work_dir.name}",
                    client=dask_client,
                )
                progress_var.set(0)
            futures = [
                dask_client.submit(
                    process_one_shard,
                    shard_name,
                    shard_to_rows[shard_name],
                    source_bucket,
                    source_prefix,
                    dest_bucket,
                    dest_prefix,
                    str(work_dir),
                    pipeline,
                    progress_var.name if progress_var else None,
                    progress_chunk_size,
                )
                for shard_name in needed_shards
            ]
            stop_event = threading.Event()

            def update_progress(pbar: Any, var: Variable, total: int) -> None:
                while not stop_event.wait(timeout=0.25):
                    try:
                        n = min(var.get(timeout=1), total)
                        delta = n - pbar.n
                        if delta > 0:
                            pbar.update(delta)
                    except Exception:
                        pass

            with tqdm(total=total_images, desc="Images", unit="img", disable=not show_progress) as pbar:
                if progress_var is not None:
                    progress_thread = threading.Thread(
                        target=update_progress,
                        args=(pbar, progress_var, total_images),
                        daemon=True,
                    )
                    progress_thread.start()
                try:
                    for future in as_completed(futures):
                        done_keys = future.result()
                        processed_keys.extend(done_keys)
                        _save_checkpoint(checkpoint_path, {"downloaded_keys": [], "processed_keys": processed_keys})
                        if progress_var is None:
                            pbar.update(len(done_keys))
                finally:
                    if progress_var is not None:
                        stop_event.set()
                        progress_thread.join(timeout=2.0)
                        # Variable can undercount due to get/set races; sync bar to 100% when done
                        if pbar.n < total_images:
                            pbar.update(total_images - pbar.n)

        phase_process_dask()
        return timings

    @_timed("download", timings)
    def phase_download() -> None:
        download_dir.mkdir(parents=True, exist_ok=True)
        for shard_name in needed_shards:
            local_tar = download_dir / shard_name
            if local_tar.is_file():
                continue
            shard_key = (source_prefix.rstrip("/") + "/" + shard_name) if source_prefix else shard_name
            download_one(client, source_bucket, shard_key, local_tar, show_progress=False)

    phase_download()

    @_timed("process", timings)
    def phase_process() -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        total_images = sum(len(rows) for rows in shard_to_rows.values())
        with tqdm(total=total_images, desc="Images", unit="img", disable=not show_progress) as pbar:
            for shard_name in needed_shards:
                rows_in_shard = shard_to_rows[shard_name]
                done_keys = do_shard(shard_name, rows_in_shard, pbar=pbar)
                processed_keys.extend(done_keys)
                _save_checkpoint(checkpoint_path, {"downloaded_keys": [], "processed_keys": processed_keys})

    phase_process()
    return timings
