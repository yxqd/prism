"""Load a processing pipeline from a YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from prism.processing.pipeline import Pipeline


def _to_tuple(x: Any) -> Optional[tuple]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


def pipeline_from_yaml(path_or_config: Union[str, Path, Dict[str, Any]]) -> Pipeline:
    """Build a Pipeline from a YAML file or a config dict.

    YAML format:
      steps:
        - name: resize
          target_size: [224, 224]
          maintain_aspect: true
          pad: true
        - name: normalize
          mean: [0.485, 0.456, 0.406]   # optional
          std: [0.229, 0.224, 0.225]    # optional
        - name: augment
          flip_h: false
          flip_v: false
          rotate_deg: 0
          color_jitter: { brightness: [0.9, 1.1], contrast: [0.9, 1.1] }
          seed: 42
        - name: camera_noise
          gaussian_std: 0.02
          poisson_scale: 0.1
          seed: 42
        - name: lens_distortion
          k: -0.2
          seed: 42
        - name: vignetting
          strength: 0.4
          seed: 42

    Use seed in a step for reproducible randomness (augment, camera_noise, etc.).
    """
    if isinstance(path_or_config, (str, Path)):
        path = Path(path_or_config)
        if not path.is_file():
            raise FileNotFoundError(f"Pipeline config not found: {path}")
        with open(path) as f:
            config = yaml.safe_load(f)
    else:
        config = path_or_config

    if not isinstance(config, dict):
        raise ValueError("Pipeline config must be a dict with a 'steps' key")
    steps_config = config.get("steps")
    if not isinstance(steps_config, list):
        raise ValueError("Pipeline config must have 'steps' as a list")

    pipeline = Pipeline()
    for i, step in enumerate(steps_config):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i} must be a dict")
        name = step.get("name")
        if not name:
            raise ValueError(f"Step {i} must have a 'name' key")
        name = str(name).strip().lower()

        seed = step.get("seed")
        rng = np.random.default_rng(int(seed)) if seed is not None else None

        # Build kwargs from step, excluding 'name' and 'seed'
        kwargs: Dict[str, Any] = {}
        for k, v in step.items():
            if k in ("name", "seed"):
                continue
            if k == "target_size":
                kwargs[k] = tuple(v) if isinstance(v, (list, tuple)) else (v, v)
            elif k in ("mean", "std"):
                kwargs[k] = _to_tuple(v)
            elif k == "color_jitter" and isinstance(v, dict):
                kwargs[k] = v
            else:
                kwargs[k] = v
        if name == "resize":
            pipeline.resize(
                target_size=kwargs.get("target_size", (224, 224)),
                maintain_aspect=kwargs.get("maintain_aspect", True),
                pad=kwargs.get("pad", True),
            )
        elif name == "normalize":
            pipeline.normalize(
                mean=kwargs.get("mean"),
                std=kwargs.get("std"),
            )
        elif name == "augment":
            if rng is not None:
                kwargs["rng"] = rng
            pipeline.augment(
                flip_h=kwargs.get("flip_h", False),
                flip_v=kwargs.get("flip_v", False),
                rotate_deg=kwargs.get("rotate_deg", 0),
                color_jitter=kwargs.get("color_jitter"),
                rng=kwargs.get("rng"),
            )
        elif name == "camera_noise":
            if rng is not None:
                kwargs["rng"] = rng
            pipeline.camera_noise(
                gaussian_std=kwargs.get("gaussian_std", 0.02),
                poisson_scale=kwargs.get("poisson_scale", 0.1),
                rng=kwargs.get("rng"),
            )
        elif name == "lens_distortion":
            if rng is not None:
                kwargs["rng"] = rng
            pipeline.lens_distortion(
                k=kwargs.get("k", -0.2),
                rng=kwargs.get("rng"),
            )
        elif name == "vignetting":
            if rng is not None:
                kwargs["rng"] = rng
            pipeline.vignetting(
                strength=kwargs.get("strength", 0.4),
                rng=kwargs.get("rng"),
            )
        else:
            raise ValueError(f"Unknown pipeline step: {name}")

    return pipeline
