"""S3 metadata registry: record where each scan's JSONL was written."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from prism.storage.s3_client import get_s3_client

REGISTRY_KEY = "metadata/_registry.json"


def get_registry(
    bucket: str,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Read registry from S3. Returns {"entries": [...]}; empty list if missing or invalid."""
    client = client or get_s3_client()
    try:
        resp = client.get_object(Bucket=bucket, Key=REGISTRY_KEY)
        body = resp["Body"].read().decode("utf-8")
        data = json.loads(body)
        if isinstance(data, dict) and "entries" in data:
            return data
        return {"entries": []}
    except (client.exceptions.NoSuchKey, KeyError, json.JSONDecodeError):
        return {"entries": []}


def append_scan(
    bucket: str,
    entry: Dict[str, Any],
    client: Optional[Any] = None,
    max_entries: int = 50,
) -> None:
    """Append a scan entry to the registry and PUT back. Trims to last max_entries."""
    client = client or get_s3_client()
    registry = get_registry(bucket, client=client)
    registry["entries"].append(entry)
    registry["entries"] = registry["entries"][-max_entries:]
    body = json.dumps(registry, indent=2).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=REGISTRY_KEY,
        Body=body,
        ContentType="application/json",
    )
