#!/usr/bin/env python3
"""Minimal PyTorch baseline training script for DALI/Ray comparison.

Uses ImageFolder + CPU transforms and a small ResNet. Run for a few epochs
and report images/sec and loss so you can compare later with a DALI-backed loader.

Usage:
  uv run python scripts/train_demo.py --data-dir data/tiny-imagenet-200
  uv run python scripts/train_demo.py --data-dir data/tiny-imagenet-200 --epochs 2 --batch-size 64 --max-steps 500
  uv run python scripts/train_demo.py --data-dir data/tiny-imagenet-200 --epochs 2 --batch-size 64 --max-steps 500 --gpu-metrics-csv logs/gpu_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pynvml  # type: ignore[import-untyped]
except ImportError:
    pynvml = None  # type: ignore[assignment]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from prism.torch.dali_pipeline import create_webdataset_dali_iterator


# ImageNet normalization (standard for pretrained ResNet and similar)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _gpu_observability_init(device: torch.device):
    """Initialize NVML and return (handle, device_index) or (None, None) if unavailable."""
    if not device.type == "cuda" or pynvml is None:
        return None, None
    try:
        pynvml.nvmlInit()
        idx = device.index if device.index is not None else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        return handle, idx
    except Exception:
        return None, None


def _sample_gpu(handle, device: torch.device) -> dict | None:
    """After torch.cuda.synchronize(), return dict with gpu_util_pct, mem_used_mb, mem_total_mb, torch_allocated_mb or None."""
    if handle is None or pynvml is None:
        return None
    try:
        torch.cuda.synchronize(device)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        torch_mb = torch.cuda.memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
        return {
            "gpu_util_pct": util.gpu,
            "mem_used_mb": mem.used / (1024 * 1024),
            "mem_total_mb": mem.total / (1024 * 1024),
            "torch_allocated_mb": torch_mb,
        }
    except Exception:
        return None


def get_train_transforms(image_size: int = 224):
    """CPU training transforms: resize, random crop, flip, normalize."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal PyTorch baseline (ImageFolder + ResNet) for throughput comparison.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tiny-imagenet-200"),
        help="Root of Tiny-ImageNet (or directory containing 'train' with class subdirs)",
    )
    parser.add_argument(
        "--train-subdir",
        default="train",
        help="Subdir under --data-dir for training (default: train)",
    )
    parser.add_argument(
        "--loader",
        choices=["pytorch", "dali-webdataset"],
        default="pytorch",
        help="Which data pipeline to use (default: pytorch ImageFolder).",
    )
    parser.add_argument(
        "--webdataset-dir",
        type=Path,
        default=Path("data/sharded/train"),
        help="Root directory of WebDataset shards (*.tar) for DALI loader.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=200,
        help="Number of classes (Tiny-ImageNet has 200). Used when --loader=dali-webdataset.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many steps (default: run full epochs)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print loss every N steps",
    )
    parser.add_argument(
        "--gpu-sample-every",
        type=int,
        default=25,
        help="Record GPU utilization/memory every N steps (0=disable, only when cuda)",
    )
    parser.add_argument(
        "--gpu-metrics-csv",
        type=Path,
        default=None,
        help="Write GPU metrics time series to this CSV path",
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}, loader: {args.loader}")

    device = torch.device(args.device)
    if args.loader == "dali-webdataset" and device.type != "cuda":
        print("DALI WebDataset loader requires a CUDA device. Use --device cuda.", file=sys.stderr)
        return 1

    if args.loader == "pytorch":
        # Resolve dataset root: data_dir/train if present, else data_dir
        data_dir = args.data_dir.resolve()
        train_root = data_dir / args.train_subdir
        if train_root.is_dir():
            root = train_root
        elif data_dir.is_dir():
            root = data_dir
        else:
            print(f"Error: not a directory: {data_dir}", file=sys.stderr)
            return 1

        transform = get_train_transforms()
        dataset = ImageFolder(root=str(root), transform=transform)
        if len(dataset) == 0:
            print(f"Error: no images found under {root}", file=sys.stderr)
            return 1

        num_classes = len(dataset.classes)
        data_iter = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
    else:
        # DALI WebDataset over sharded .tar files.
        num_classes = args.num_classes
        data_iter = create_webdataset_dali_iterator(
            shards_root=args.webdataset_dir,
            batch_size=args.batch_size,
            device_id=device.index or 0,
            num_threads=args.num_workers or 4,
        )

    model = resnet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    gpu_handle, _ = _gpu_observability_init(device)
    gpu_samples: list[dict] = []  # [{step, gpu_util_pct, mem_used_mb, ...}, ...]

    model.train()
    total_images = 0
    total_step_time = 0.0
    step_count = 0
    start_wall = time.perf_counter()

    for epoch in range(args.epochs):
        if args.loader == "pytorch":
            epoch_iter = enumerate(data_iter)
        else:
            # DALI iterator yields a list of outputs per pipeline; use index 0.
            epoch_iter = enumerate(data_iter)

        for batch_idx, batch in epoch_iter:
            if args.max_steps is not None and step_count >= args.max_steps:
                break

            if args.loader == "pytorch":
                images, targets = batch
            else:
                # DALIClassificationIterator returns [ {'data': tensor, 'label': tensor}, ... ]
                out = batch[0] if isinstance(batch, list) else batch
                images = out["data"]
                # Labels are numeric class indices (int64) from DALI; squeeze to (N,)
                targets = out["label"].squeeze().long()

            step_start = time.perf_counter()
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_step_time += time.perf_counter() - step_start

            n = images.size(0)
            total_images += n
            step_count += 1

            if gpu_handle is not None and args.gpu_sample_every > 0 and step_count % args.gpu_sample_every == 0:
                sample = _sample_gpu(gpu_handle, device)
                if sample is not None:
                    gpu_samples.append({"step": step_count, **sample})

            if (batch_idx + 1) % args.print_every == 0:
                print(f"Epoch {epoch + 1} step {step_count} loss={loss.item():.4f}")

            if args.max_steps is not None and step_count >= args.max_steps:
                break

        if args.loader == "dali-webdataset":
            # Required to start a new epoch with DALIClassificationIterator.
            data_iter.reset()

    elapsed = time.perf_counter() - start_wall
    images_per_sec = total_images / elapsed if elapsed > 0 else 0.0
    step_time_ms = (total_step_time / step_count * 1000) if step_count else 0

    if gpu_handle is not None and pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    print("\n--- Summary ---")
    print(f"  Total images: {total_images}")
    print(f"  Steps:        {step_count}")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Images/sec:   {images_per_sec:.1f}")
    print(f"  Step (fwd+bwd) avg: {step_time_ms:.1f} ms")

    if gpu_samples:
        util_pcts = [s["gpu_util_pct"] for s in gpu_samples]
        mem_mbs = [s["mem_used_mb"] for s in gpu_samples]
        torch_mbs = [s["torch_allocated_mb"] for s in gpu_samples]
        print(f"  GPU util %:   mean={sum(util_pcts)/len(util_pcts):.1f} max={max(util_pcts)}")
        print(f"  GPU mem (MB): mean={sum(mem_mbs)/len(mem_mbs):.1f} max={max(mem_mbs):.1f}")
        print(f"  Torch alloc (MB): mean={sum(torch_mbs)/len(torch_mbs):.1f} max={max(torch_mbs):.1f}")
        if gpu_samples[0].get("mem_total_mb") is not None:
            print(f"  GPU mem total: {gpu_samples[0]['mem_total_mb']:.0f} MB")
        if args.gpu_metrics_csv:
            args.gpu_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.gpu_metrics_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step", "gpu_util_pct", "mem_used_mb", "mem_total_mb", "torch_allocated_mb"])
                w.writeheader()
                w.writerows(gpu_samples)
            print(f"  GPU metrics:  written to {args.gpu_metrics_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
