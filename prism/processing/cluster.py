"""Dask local cluster with configurable workers for distributed image processing."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple

from distributed import Client, LocalCluster


@contextmanager
def create_local_cluster(
    n_workers: Optional[int] = None,
    threads_per_worker: int = 2,
    dashboard_address: str = ":8787",
) -> Generator[Tuple[Client, LocalCluster], None, None]:
    """Create a local Dask cluster and yield (client, cluster). Shuts down on exit.

    Use PRISM_DASK_WORKERS env var to set n_workers if not passed.
    Dashboard is available at http://localhost:8787 (or the port in dashboard_address).
    """
    if n_workers is None:
        n_workers = int(os.environ.get("PRISM_DASK_WORKERS", "4"))
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        dashboard_address=dashboard_address,
    )
    client = Client(cluster)
    try:
        yield client, cluster
    finally:
        client.close()
        cluster.close()
