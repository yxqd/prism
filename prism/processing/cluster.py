"""Dask local cluster with configurable workers for distributed image processing."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple

from distributed import Client, LocalCluster


@contextmanager
def create_local_cluster(
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    dashboard_address: Optional[str] = None,
) -> Generator[Tuple[Client, LocalCluster], None, None]:
    """Create a local Dask cluster and yield (client, cluster). Shuts down on exit.

    Use PRSIM_DASK_WORKER_COUNT env var to set n_workers if not passed, default to 4.
    Use PRSIM_DASK_THREADS_PER_WORKER env var to set threads_per_worker if not passed, default to 2.
    Use PRSIM_DASK_DASHBOARD_ADDRESS env var to set dashboard_address if not passed, default to :8787.
    Dashboard is available at http://localhost:8787 (or the port in dashboard_address).
    """
    if n_workers is None:
        n_workers = int(os.environ.get("PRSIM_DASK_WORKER_COUNT", "4"))
    if threads_per_worker is None:
        threads_per_worker = int(os.environ.get("PRSIM_DASK_THREADS_PER_WORKER", "2"))
    if dashboard_address is None:
        dashboard_address = os.environ.get("PRSIM_DASK_DASHBOARD_ADDRESS", "localhost:8787")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        dashboard_address=dashboard_address,
    )
    client = Client(cluster)
    print(f"Dask cluster created with {n_workers} workers and {threads_per_worker} threads per worker")
    print(f"Dask dashboard available at http://{dashboard_address}")
    try:
        yield client, cluster
    finally:
        client.close()
        cluster.close()
