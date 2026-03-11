#!/usr/bin/env python3
"""Minimal PyTorch baseline training script for DALI/Ray comparison.

Uses ImageFolder + CPU transforms and a small ResNet. Run for a few epochs
and report images/sec and loss so you can compare later with a DALI-backed loader.

Usage:
  python scripts/train_demo.py --data-dir data/tiny-imagenet-200
  python scripts/train_demo.py --data-dir data/tiny-imagenet-200 --epochs 2 --batch-size 64 --max-steps 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18


# ImageNet normalization (standard for pretrained ResNet and similar)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    args = parser.parse_args()

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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    device = torch.device(args.device)
    model = resnet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    total_images = 0
    total_step_time = 0.0
    step_count = 0
    start_wall = time.perf_counter()

    for epoch in range(args.epochs):
        for batch_idx, (images, targets) in enumerate(loader):
            if args.max_steps is not None and step_count >= args.max_steps:
                break

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

            if (batch_idx + 1) % args.print_every == 0:
                print(f"Epoch {epoch + 1} step {step_count} loss={loss.item():.4f}")

            if args.max_steps is not None and step_count >= args.max_steps:
                break

    elapsed = time.perf_counter() - start_wall
    images_per_sec = total_images / elapsed if elapsed > 0 else 0.0
    step_time_ms = (total_step_time / step_count * 1000) if step_count else 0

    print("\n--- Summary ---")
    print(f"  Total images: {total_images}")
    print(f"  Steps:        {step_count}")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Images/sec:   {images_per_sec:.1f}")
    print(f"  Step (fwd+bwd) avg: {step_time_ms:.1f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
