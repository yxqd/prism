"""Collect image paths from a local directory or S3 prefix for sharding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Tuple

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _normalize_extensions(extensions: Optional[Iterable[str]]) -> Set[str]:
    raw = extensions or DEFAULT_EXTENSIONS
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in raw}


def list_image_paths(
    root: Path,
    *,
    extensions: Optional[Iterable[str]] = None,
    split: Optional[str] = None,
    with_labels: bool = True,
) -> List[Tuple[Path, Optional[str]]]:
    """List image paths under root, optionally filtered by split and with class labels.

    Args:
        root: Directory containing images, possibly with subdirs (e.g. train/val, or class subdirs).
        extensions: Allowed suffixes (e.g. {".jpg", ".jpeg", ".png"}). Default DEFAULT_EXTENSIONS.
        split: If set, only descend into root/split (e.g. "train" or "val"). Otherwise scan root recursively.
        with_labels: If True, derive label from parent directory name (first non-split parent). Else label is None.

    Returns:
        Ordered list of (path, label). Label is the directory name (e.g. class id) or None.
    """
    ext = _normalize_extensions(extensions)
    base = root / split if split else root
    if not base.is_dir():
        return []

    out: List[Tuple[Path, Optional[str]]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ext:
            continue
        # Exclude common non-image sidecars
        if path.name.endswith("_boxes.txt") or path.name == "val_annotations.txt":
            continue
        label: Optional[str] = None
        if with_labels:
            # Parent of file is class dir when layout is .../train/class_id/img.jpg
            rel = path.relative_to(base)
            parts = rel.parts
            if len(parts) >= 2:
                label = parts[0]  # class dir
            elif len(parts) == 1:
                label = path.stem.split("_")[0] if "_" in path.stem else path.stem
        out.append((path, label))
    return out


def list_image_keys_from_s3(
    client: Any,
    bucket: str,
    prefix: str,
    *,
    extensions: Optional[Iterable[str]] = None,
    split: Optional[str] = None,
    with_labels: bool = True,
) -> List[Tuple[str, Optional[str]]]:
    """List image object keys under an S3 prefix, with optional split and labels.

    Args:
        client: boto3 S3 client.
        bucket: S3 bucket name.
        prefix: S3 prefix (e.g. "tiny-imagenet-200/" or "tiny-imagenet-200/train/").
        extensions: Allowed suffixes (e.g. {".jpg", ".jpeg", ".png"}). Default DEFAULT_EXTENSIONS.
        split: If set, only list keys under prefix/split/ (e.g. "train" or "val").
        with_labels: If True, derive label from key path (first path component under prefix[/split]).

    Returns:
        Ordered list of (key, label). Label is the first path segment under the listed prefix, or None.
    """
    from prism.storage.download import list_objects_under_prefix

    ext = _normalize_extensions(extensions)
    # S3 keys use /; we filter by suffix. list_objects_under_prefix expects extensions with leading dot.
    ext_for_list = {e for e in ext}
    prefix_clean = prefix.rstrip("/") + "/" if prefix else ""
    list_prefix = prefix_clean + split + "/" if split else prefix_clean
    keys_sizes = list_objects_under_prefix(client, bucket, list_prefix, extensions=ext_for_list)
    out: List[Tuple[str, Optional[str]]] = []
    for key, _ in sorted(keys_sizes, key=lambda x: x[0]):
        if key.endswith("_boxes.txt") or "val_annotations.txt" in key:
            continue
        label: Optional[str] = None
        if with_labels:
            # Key is e.g. "tiny-imagenet-200/train/n02106662/img.JPG"; relative to list_prefix
            rel = key[len(list_prefix) :] if key.startswith(list_prefix) else key
            parts = [p for p in rel.split("/") if p]
            if len(parts) >= 2:
                label = parts[0]  # class dir
            elif len(parts) == 1:
                stem = parts[0].rsplit(".", 1)[0] if "." in parts[0] else parts[0]
                label = stem.split("_")[0] if "_" in stem else stem
        out.append((key, label))
    return out
