from __future__ import annotations

from typing import Any

import imkit as imk
import numpy as np
from PIL import Image

DEFAULT_UPSCALE_FACTOR = 1
SUPPORTED_UPSCALE_FACTORS = (1, 2, 4)


def normalize_upscale_factor(value: Any) -> int:
    try:
        factor = int(value)
    except (TypeError, ValueError):
        return DEFAULT_UPSCALE_FACTOR

    if factor in SUPPORTED_UPSCALE_FACTORS:
        return factor
    return DEFAULT_UPSCALE_FACTOR


def get_upscale_factor_from_export_settings(export_settings: dict | None) -> int:
    if not isinstance(export_settings, dict):
        return DEFAULT_UPSCALE_FACTOR
    return normalize_upscale_factor(export_settings.get("image_upscale_factor", DEFAULT_UPSCALE_FACTOR))


def upscale_image(image: np.ndarray, factor: int) -> np.ndarray:
    normalized_factor = normalize_upscale_factor(factor)
    if normalized_factor <= 1 or image is None or image.size == 0:
        return image

    height, width = image.shape[:2]
    new_width = max(1, int(width * normalized_factor))
    new_height = max(1, int(height * normalized_factor))
    return imk.resize(image, (new_width, new_height), mode=Image.Resampling.LANCZOS)
