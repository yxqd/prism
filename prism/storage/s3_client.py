"""S3 client wrapper with retries."""

from __future__ import annotations

from typing import Any, Optional

import boto3
from botocore.config import Config

from prism import config


def get_s3_client(
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    max_attempts: int = 3,
) -> Any:
    """Return a boto3 S3 client with retries.

    Uses PRISM_REGION and AWS_PROFILE from config if not passed.
    Set endpoint_url for LocalStack (e.g. http://localhost:4566).
    """
    session = boto3.Session(
        profile_name=profile_name or config.AWS_PROFILE,
        region_name=region_name or config.REGION,
    )
    cfg = Config(
        retries={"mode": "standard", "max_attempts": max_attempts},
        signature_version="s3v4",
    )
    return session.client("s3", config=cfg, endpoint_url=endpoint_url)
