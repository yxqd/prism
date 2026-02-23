#!/usr/bin/env python3
"""Download ImageNet-style images: Tiny-ImageNet zip or from a URL list.

Usage:
  python scripts/download_imagenet.py tiny-imagenet --out-dir ./data/tiny [--limit-classes 2] [--limit-per-class 100]
  python scripts/download_imagenet.py from-urls --url-list urls.txt --out-dir ./data/images [--max 100] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Add project root so prism is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
DEFAULT_USER_AGENT = "prism-download-imagenet/1.0"
RETRIES = 3


# --- Tiny-ImageNet ---


def _download_zip(url: str, dest: Path, show_progress: bool = True) -> Path:
    """Download zip from url to dest. Returns path to zip file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024
        read_so_far = 0
        with open(dest, "wb") as f:
            with tqdm(
                total=total if total else None,
                unit="B",
                unit_scale=True,
                desc="Downloading zip",
                disable=not show_progress,
            ) as pbar:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    read_so_far += len(chunk)
                    pbar.update(len(chunk))
    return dest


def _tiny_imagenet_class_order(names: List[str]) -> List[str]:
    """Return ordered list of class ids (train subdirs). Uses wnids.txt if present, else sorted from paths."""
    # Names are like "tiny-imagenet-200/train/n01443537/images/..." or "TinyImageNetPath/train/..."
    train_prefix = "train/"
    class_ids = set()
    for n in names:
        if "/train/" in n:
            parts = n.split("/train/", 1)[1].split("/")
            if parts:
                class_ids.add(parts[0])
    if not class_ids:
        return []
    # Prefer wnids.txt order if present
    wnids_name = None
    for n in names:
        if n.endswith("wnids.txt"):
            wnids_name = n
            break
    if wnids_name:
        # We don't have the zip open here; caller can pass content or we sort
        pass
    return sorted(class_ids)


def extract_tiny_imagenet(
    zip_path: Path,
    out_dir: Path,
    limit_classes: Optional[int] = None,
    limit_per_class: Optional[int] = None,
    show_progress: bool = True,
) -> int:
    """Extract zip to out_dir, optionally limiting classes and/or images per class. Returns count of extracted files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    class_order = _tiny_imagenet_class_order(names)
    class_counts: dict = {}
    to_extract = []
    for name in names:
        if name.endswith("/"):
            continue
        # Always include metadata files
        if "wnids.txt" in name or "words.txt" in name or "val_annotations.txt" in name or "boxes.txt" in name:
            to_extract.append(name)
            continue
        if limit_classes is not None and "/train/" in name:
            class_id = name.split("/train/", 1)[1].split("/")[0]
            if class_id not in class_order[:limit_classes]:
                continue
        if limit_per_class is not None and ("/train/" in name or "/val/" in name or "/test/" in name):
            if any(name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                if "/train/" in name:
                    class_id = name.split("/train/", 1)[1].split("/")[0]
                    key = ("train", class_id)
                elif "/val/" in name:
                    key = ("val", "val")
                else:
                    key = ("test", "test")
                if class_counts.get(key, 0) >= limit_per_class:
                    continue
                class_counts[key] = class_counts.get(key, 0) + 1
        to_extract.append(name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in tqdm(to_extract, desc="Extracting", unit="file", disable=not show_progress):
            zf.extract(name, out_dir)
    return len(to_extract)


def cmd_tiny_imagenet(
    out_dir: Path,
    limit_classes: Optional[int] = None,
    limit_per_class: Optional[int] = None,
    url: str = TINY_IMAGENET_URL,
    keep_zip: bool = False,
    show_progress: bool = True,
) -> int:
    """Download Tiny-ImageNet zip and extract to out_dir. Returns number of files extracted."""
    out_dir = Path(out_dir)
    zip_path = out_dir / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        _download_zip(url, zip_path, show_progress=show_progress)
    n = extract_tiny_imagenet(
        zip_path,
        out_dir,
        limit_classes=limit_classes,
        limit_per_class=limit_per_class,
        show_progress=show_progress,
    )
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    return n


# --- From URL list ---


def _parse_url_list(path: Path, max_urls: Optional[int]) -> List[str]:
    """Read URLs from file: one per line, or CSV with 'url' column. Skip empty and # lines."""
    urls = []
    path = Path(path)
    with open(path) as f:
        try:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            reader = csv.DictReader(f, dialect=dialect)
            if "url" in (reader.fieldnames or []):
                for row in reader:
                    u = (row.get("url") or "").strip()
                    if u and not u.startswith("#"):
                        urls.append(u)
                    if max_urls is not None and len(urls) >= max_urls:
                        break
                return urls
        except csv.Error:
            pass
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
            if max_urls is not None and len(urls) >= max_urls:
                break
    return urls


def _extension_from_content_type(ct: Optional[str]) -> str:
    """Map Content-Type to file extension."""
    if not ct:
        return ".jpg"
    ct = ct.split(";")[0].strip().lower()
    m = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tiff"}
    return m.get(ct, ".jpg")


def _extension_from_url(url: str) -> str:
    """Guess image extension from URL path."""
    path = url.split("?")[0]
    for ext in IMAGE_EXTENSIONS:
        if path.lower().endswith(ext):
            return ext
    return ".jpg"


def _download_one_url(
    item: tuple,
    out_dir: Path,
) -> Optional[Path]:
    """Download a single URL to out_dir. Returns local path or None on failure."""
    index, url = item
    for attempt in range(RETRIES):
        try:
            req = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                ct = resp.headers.get("Content-Type")
                ext = _extension_from_content_type(ct) or _extension_from_url(url)
                safe_name = f"{index:05d}{ext}"
                path = out_dir / safe_name
                path.write_bytes(resp.read())
            return path
        except (HTTPError, URLError, OSError):
            if attempt == RETRIES - 1:
                return None
    return None


def cmd_from_urls(
    url_list: Path,
    out_dir: Path,
    max_urls: Optional[int] = None,
    workers: int = 4,
    show_progress: bool = True,
) -> int:
    """Download images from URL list to out_dir. Returns number of files downloaded."""
    urls = _parse_url_list(Path(url_list), max_urls=max_urls)
    if not urls:
        return 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = list(enumerate(urls))
    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one_url, item, out_dir): item for item in items}
        with tqdm(total=len(items), desc="Downloading", unit="file", disable=not show_progress) as pbar:
            for fut in as_completed(futures):
                path = fut.result()
                if path is not None:
                    downloaded += 1
                pbar.update(1)
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ImageNet-style images: Tiny-ImageNet or from a URL list.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # tiny-imagenet
    p_tiny = sub.add_parser("tiny-imagenet", help="Download and extract Tiny-ImageNet-200 zip")
    p_tiny.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p_tiny.add_argument("--limit-classes", type=int, default=None, help="Keep only first N classes")
    p_tiny.add_argument("--limit-per-class", type=int, default=None, help="Keep only first M images per class")
    p_tiny.add_argument("--url", default=TINY_IMAGENET_URL, help="Zip URL")
    p_tiny.add_argument("--keep-zip", action="store_true", help="Do not delete zip after extract")
    p_tiny.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    # from-urls
    p_urls = sub.add_parser("from-urls", help="Download images from a URL list file")
    p_urls.add_argument("--url-list", type=Path, required=True, help="File with one URL per line or CSV with 'url' column")
    p_urls.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p_urls.add_argument("--max", type=int, default=None, help="Maximum number of URLs to download")
    p_urls.add_argument("--workers", type=int, default=4, help="Concurrent downloads")
    p_urls.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    args = parser.parse_args()
    show_progress = not getattr(args, "no_progress", False)

    if args.command == "tiny-imagenet":
        n = cmd_tiny_imagenet(
            args.out_dir,
            limit_classes=args.limit_classes,
            limit_per_class=args.limit_per_class,
            url=args.url,
            keep_zip=args.keep_zip,
            show_progress=show_progress,
        )
        print(f"Extracted {n} files to {args.out_dir}")
    elif args.command == "from-urls":
        n = cmd_from_urls(
            args.url_list,
            args.out_dir,
            max_urls=args.max,
            workers=args.workers,
            show_progress=show_progress,
        )
        print(f"Downloaded {n} images to {args.out_dir}")
    else:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
