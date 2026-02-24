"""Image processing tasks: resize, normalize, augment. Pure functions over numpy arrays (HWC)."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image


def resize_image(
    arr: np.ndarray,
    target_size: Tuple[int, int],
    maintain_aspect: bool = True,
    pad: bool = True,
) -> np.ndarray:
    """Resize image to fit within target_size (H, W), optionally pad to exact size.

    Input arr: HWC, uint8 or float in [0,1]. Returns same dtype.
    """
    h, w = arr.shape[:2]
    th, tw = target_size
    if maintain_aspect:
        scale = min(th / h, tw / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
    else:
        new_h, new_w = th, tw

    is_float = arr.dtype in (np.float32, np.float64)
    if is_float:
        arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    else:
        arr_uint8 = arr

    pil = Image.fromarray(arr_uint8)
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    out = np.array(pil)
    if pad and (new_h != th or new_w != tw):
        pad_h, pad_w = th - new_h, tw - new_w
        # Center pad
        top = pad_h // 2
        left = pad_w // 2
        padded = np.zeros((th, tw, out.shape[2]), dtype=out.dtype)
        padded[top : top + new_h, left : left + new_w] = out
        out = padded

    if is_float:
        out = out.astype(arr.dtype) / 255.0
    return out


def normalize(
    arr: np.ndarray,
    mean: Optional[Union[Tuple[float, ...], np.ndarray]] = None,
    std: Optional[Union[Tuple[float, ...], np.ndarray]] = None,
    to_float: bool = True,
) -> np.ndarray:
    """Convert to float [0,1] or standardize with optional mean/std (e.g. ImageNet).

    If mean/std are None, just scale to [0,1]. Otherwise apply (x - mean) / std.
    Returns float array.
    """
    out = arr.astype(np.float64)
    if out.max() > 1.0 or out.min() < 0.0:
        out = np.clip(out, 0, 255) / 255.0
    if not to_float:
        return out
    if mean is not None and std is not None:
        mean = np.asarray(mean, dtype=out.dtype)
        std = np.asarray(std, dtype=out.dtype)
        if mean.size == 3 and out.shape[-1] == 3:
            out = (out - mean) / np.maximum(std, 1e-7)
        else:
            out = (out - mean) / np.maximum(std, 1e-7)
    return out.astype(np.float32)


def basic_augment(
    arr: np.ndarray,
    flip_h: bool = False,
    flip_v: bool = False,
    rotate_deg: float = 0,
    color_jitter: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply flip, rotate, and optional color jitter. Deterministic if rng is None or fixed seed.

    color_jitter: dict with optional keys brightness, contrast, saturation (each 0..1 factor range).
    Returns same shape; dtype preserved (uint8 or float).
    """
    out = arr.copy()
    if flip_h:
        out = np.ascontiguousarray(out[:, ::-1])
    if flip_v:
        out = np.ascontiguousarray(out[::-1])
    if rotate_deg != 0:
        pil = Image.fromarray(out if out.dtype == np.uint8 else (np.clip(out, 0, 1) * 255).astype(np.uint8))
        pil = pil.rotate(-rotate_deg, expand=False, resample=Image.Resampling.BILINEAR)
        out = np.array(pil)
        if arr.dtype in (np.float32, np.float64):
            out = out.astype(arr.dtype) / 255.0

    if color_jitter and out.shape[-1] >= 3:
        rng = rng or np.random.default_rng()
        brightness = color_jitter.get("brightness", 1.0)
        contrast = color_jitter.get("contrast", 1.0)
        saturation = color_jitter.get("saturation", 1.0)
        if isinstance(brightness, (list, tuple)) and len(brightness) == 2:
            brightness = float(rng.uniform(brightness[0], brightness[1]))
        if isinstance(contrast, (list, tuple)) and len(contrast) == 2:
            contrast = float(rng.uniform(contrast[0], contrast[1]))
        if isinstance(saturation, (list, tuple)) and len(saturation) == 2:
            saturation = float(rng.uniform(saturation[0], saturation[1]))
        work = out.astype(np.float64)
        if work.max() > 1.0:
            work = work / 255.0
        work = np.clip(work, 0, 1)
        work = work * brightness
        mean = work.mean()
        work = (work - mean) * contrast + mean
        if work.shape[-1] == 3:
            gray = work.mean(axis=-1, keepdims=True)
            work = work * saturation + (1 - saturation) * gray
        work = np.clip(work, 0, 1)
        out = (work * 255).astype(np.uint8) if arr.dtype == np.uint8 else work.astype(arr.dtype)

    return out
