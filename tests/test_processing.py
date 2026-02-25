"""Tests for processing: tasks, pipeline, workflow checkpoint, throughput."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from prism.processing.tasks import basic_augment, normalize, resize_image
from prism.processing.pipeline import Pipeline, apply_pipeline, camera_noise, lens_distortion, vignetting
from prism.processing.pipeline_config import pipeline_from_yaml
from prism.processing.workflow import _load_checkpoint, _save_checkpoint


def test_resize_image() -> None:
    arr = np.ones((100, 200, 3), dtype=np.uint8) * 128
    out = resize_image(arr, (50, 50), maintain_aspect=True, pad=True)
    assert out.shape == (50, 50, 3)
    assert out.dtype == np.uint8

    out2 = resize_image(arr.astype(np.float32) / 255.0, (25, 25), maintain_aspect=True, pad=False)
    assert out2.shape == (12, 25, 3)  # aspect preserved: scale 0.125 -> 100*0.125=12, 200*0.125=25
    assert out2.dtype == np.float32


def test_normalize() -> None:
    arr = np.ones((10, 10, 3), dtype=np.uint8) * 255
    out = normalize(arr)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)

    arr2 = np.zeros((10, 10, 3), dtype=np.uint8)
    out2 = normalize(arr2, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert out2.dtype == np.float32
    assert out2.shape == (10, 10, 3)


def test_basic_augment() -> None:
    arr = np.arange(12).reshape(2, 2, 3).astype(np.uint8)
    out = basic_augment(arr, flip_h=True, flip_v=False, rng=np.random.default_rng(42))
    assert out.shape == arr.shape
    assert np.array_equal(out[:, ::-1], arr)  # flip_h so out reversed on columns equals arr

    out2 = basic_augment(arr, flip_h=False, flip_v=True)
    assert np.array_equal(out2[::-1], arr)


def test_pipeline_apply() -> None:
    arr = np.ones((64, 64, 3), dtype=np.uint8) * 128
    pipeline = Pipeline().resize((32, 32)).normalize()
    out = pipeline.apply(arr)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1.0


def test_apply_pipeline() -> None:
    arr = np.ones((10, 10, 3), dtype=np.uint8)
    steps = [lambda x: resize_image(x, (5, 5)), lambda x: normalize(x)]
    out = apply_pipeline(arr, steps)
    assert out.shape == (5, 5, 3)
    assert out.dtype == np.float32


def test_camera_noise() -> None:
    arr = np.ones((20, 20, 3), dtype=np.uint8) * 128
    out = camera_noise(arr, gaussian_std=0.02, poisson_scale=None, rng=np.random.default_rng(0))
    assert out.shape == arr.shape
    # out is uint8 when input is uint8; values should be near 128
    assert np.allclose(out.astype(np.float64), 128.0, atol=25.0)


def test_lens_distortion() -> None:
    arr = np.ones((32, 32, 3), dtype=np.uint8) * 200
    out = lens_distortion(arr, k=-0.2)
    assert out.shape == arr.shape
    assert out.dtype == np.uint8


def test_vignetting() -> None:
    arr = np.ones((32, 32, 3), dtype=np.uint8) * 255
    out = vignetting(arr, strength=0.5)
    assert out.shape == arr.shape
    assert out.dtype == np.uint8
    assert out[0, 0].mean() < out[16, 16].mean()  # corners darker


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    cp = tmp_path / "workflow_checkpoint.json"
    data = {"downloaded_keys": ["a.jpg", "b.png"], "processed_keys": ["a.jpg"]}
    _save_checkpoint(cp, data)
    assert cp.is_file()
    loaded = _load_checkpoint(cp)
    assert loaded["downloaded_keys"] == data["downloaded_keys"]
    assert loaded["processed_keys"] == data["processed_keys"]

    empty = _load_checkpoint(tmp_path / "nonexistent.json")
    assert empty["downloaded_keys"] == []
    assert empty["processed_keys"] == []


def test_pipeline_from_yaml(tmp_path: Path) -> None:
    """Build pipeline from YAML dict and file; apply produces expected shape."""
    config = {
        "steps": [
            {"name": "resize", "target_size": [32, 32], "maintain_aspect": True, "pad": True},
            {"name": "normalize"},
        ]
    }
    pipeline = pipeline_from_yaml(config)
    arr = np.ones((64, 64, 3), dtype=np.uint8) * 128
    out = pipeline.apply(arr)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32

    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        "steps:\n"
        "  - name: resize\n"
        "    target_size: [16, 16]\n"
        "  - name: normalize\n"
    )
    pipeline2 = pipeline_from_yaml(str(yaml_path))
    out2 = pipeline2.apply(arr)
    assert out2.shape == (16, 16, 3)


def test_throughput_1000_images() -> None:
    """Process 1,000 small images and log throughput (images/sec)."""
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(1000)]
    pipeline = Pipeline().resize((32, 32)).normalize()

    t0 = time.perf_counter()
    for img in images:
        pipeline.apply(img)
    elapsed = time.perf_counter() - t0
    throughput = len(images) / elapsed
    assert throughput > 10, f"Throughput {throughput:.1f} img/s below 10"
    print(f"Throughput: {throughput:.1f} images/sec")
