"""Augmentation pipeline: chain resize, normalize, augment, and camera effects."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from prism.processing import tasks


def _ensure_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return np.clip(img.astype(np.float64) / 255.0, 0, 1)
    return np.clip(img.astype(np.float64), 0, 1)


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def camera_noise(
    img: np.ndarray,
    gaussian_std: float = 0.02,
    poisson_scale: Optional[float] = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian (read) and/or Poisson (shot) noise; clip to valid range."""
    rng = rng or np.random.default_rng()
    out = _ensure_float(img)
    if gaussian_std > 0:
        out = out + rng.normal(0, gaussian_std, out.shape).astype(np.float64)
    if poisson_scale is not None and poisson_scale > 0:
        lam = out * (1.0 / max(poisson_scale, 1e-6))
        out = rng.poisson(np.clip(lam, 0, 1e10)).astype(np.float64) * poisson_scale
    out = np.clip(out, 0, 1)
    return out.astype(np.float32) if img.dtype in (np.float32, np.float64) else _ensure_uint8(out)


def lens_distortion(
    img: np.ndarray,
    k: float = -0.2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Radial distortion (barrel if k < 0, pincushion if k > 0). Simple polynomial model."""
    h, w = img.shape[:2]
    work = img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img.astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yy, xx = yy - cy, xx - cx
    r = np.sqrt(xx * xx + yy * yy) / (max(h, w) * 0.5 + 1e-6)
    r2 = r * r
    factor = 1.0 + k * r2
    x2 = cx + xx * factor
    y2 = cy + yy * factor
    coords = np.stack([y2, x2], axis=0)
    out = np.zeros_like(work)
    for c in range(work.shape[-1]):
        out[..., c] = ndimage.map_coordinates(work[..., c], coords, order=1, mode="reflect")
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8) if img.dtype == np.uint8 else out.astype(img.dtype)


def vignetting(
    img: np.ndarray,
    strength: float = 0.4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Darken toward corners with radial falloff."""
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = np.sqrt(cx * cx + cy * cy) + 1e-6
    r_norm = np.clip(r / r_max, 0, 1)
    # Darken toward corners: center mask=1, corners mask=1-strength
    mask = 1.0 - strength * (r_norm ** 2)
    work = img.astype(np.float64)
    if img.dtype == np.uint8:
        work = work / 255.0
    out = work * mask[..., np.newaxis]
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8) if img.dtype == np.uint8 else out.astype(img.dtype)


class Pipeline:
    """Chain processing steps; apply(image) runs them in order. Composable for Dask."""

    def __init__(self) -> None:
        self._steps: List[Tuple[str, Callable[..., np.ndarray]]] = []

    def resize(
        self,
        target_size: Tuple[int, int],
        maintain_aspect: bool = True,
        pad: bool = True,
    ) -> "Pipeline":
        self._steps.append(
            ("resize", lambda img: tasks.resize_image(img, target_size, maintain_aspect, pad))
        )
        return self

    def normalize(
        self,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
    ) -> "Pipeline":
        self._steps.append(("normalize", lambda img: tasks.normalize(img, mean=mean, std=std)))
        return self

    def augment(
        self,
        flip_h: bool = False,
        flip_v: bool = False,
        rotate_deg: float = 0,
        color_jitter: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> "Pipeline":
        self._steps.append(
            (
                "augment",
                lambda img: tasks.basic_augment(img, flip_h=flip_h, flip_v=flip_v, rotate_deg=rotate_deg, color_jitter=color_jitter, rng=rng),
            )
        )
        return self

    def camera_noise(
        self,
        gaussian_std: float = 0.02,
        poisson_scale: Optional[float] = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> "Pipeline":
        self._steps.append(
            ("camera_noise", lambda img: camera_noise(img, gaussian_std, poisson_scale, rng))
        )
        return self

    def lens_distortion(self, k: float = -0.2, rng: Optional[np.random.Generator] = None) -> "Pipeline":
        self._steps.append(("lens_distortion", lambda img: lens_distortion(img, k=k, rng=rng)))
        return self

    def vignetting(self, strength: float = 0.4, rng: Optional[np.random.Generator] = None) -> "Pipeline":
        self._steps.append(("vignetting", lambda img: vignetting(img, strength=strength, rng=rng)))
        return self

    def apply(self, image: np.ndarray) -> np.ndarray:
        out = image
        for _name, step in self._steps:
            out = step(out)
        return out


def apply_pipeline(image: np.ndarray, steps: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Run a list of callables in order. Alternative to Pipeline().apply()."""
    out = image
    for step in steps:
        out = step(out)
    return out
