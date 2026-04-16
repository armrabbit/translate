from __future__ import annotations

import logging
import os
import threading
from typing import Any

import imkit as imk
import numpy as np
from PIL import Image
from modules.utils.download import notify_download_event
from modules.utils.download_file import download_url_to_file
from modules.utils.paths import get_user_data_dir

DEFAULT_UPSCALE_FACTOR = 1
SUPPORTED_UPSCALE_FACTORS = (1, 2, 4)
REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "RealESRGAN_x4plus_anime_6B.pth"
)
REALESRGAN_MODEL_PATH = os.path.join(
    get_user_data_dir(),
    "models",
    "upscaler",
    "realesrgan",
    "RealESRGAN_x4plus_anime_6B.pth",
)

logger = logging.getLogger(__name__)
_realesrgan_lock = threading.Lock()
_realesrgan_upsampler = None


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


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


def _lanczos_upscale(image: np.ndarray, factor: int) -> np.ndarray:
    normalized_factor = normalize_upscale_factor(factor)
    if normalized_factor <= 1:
        return image

    height, width = image.shape[:2]
    new_width = max(1, int(width * normalized_factor))
    new_height = max(1, int(height * normalized_factor))
    return imk.resize(image, (new_width, new_height), mode=Image.Resampling.LANCZOS)


def _ensure_realesrgan_model() -> str:
    if os.path.exists(REALESRGAN_MODEL_PATH):
        return REALESRGAN_MODEL_PATH

    with _realesrgan_lock:
        if os.path.exists(REALESRGAN_MODEL_PATH):
            return REALESRGAN_MODEL_PATH

        os.makedirs(os.path.dirname(REALESRGAN_MODEL_PATH), exist_ok=True)
        model_name = os.path.basename(REALESRGAN_MODEL_PATH)
        notify_download_event("start", model_name)
        try:
            download_url_to_file(
                REALESRGAN_MODEL_URL,
                REALESRGAN_MODEL_PATH,
                progress=True,
            )
        finally:
            notify_download_event("end", model_name)

        return REALESRGAN_MODEL_PATH


def _build_realesrgan_upsampler():
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as exc:
        raise RuntimeError(
            "Real-ESRGAN dependencies are missing. Please install: "
            "`pip install torch realesrgan basicsr`"
        ) from exc

    model_path = _ensure_realesrgan_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device == "cuda"
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=4,
    )
    return RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device,
    )


def _get_realesrgan_upsampler():
    global _realesrgan_upsampler
    if _realesrgan_upsampler is not None:
        return _realesrgan_upsampler

    with _realesrgan_lock:
        if _realesrgan_upsampler is None:
            _realesrgan_upsampler = _build_realesrgan_upsampler()
    return _realesrgan_upsampler


def _realesrgan_upscale(image: np.ndarray, factor: int) -> np.ndarray:
    upsampler = _get_realesrgan_upsampler()
    image = _ensure_uint8(image)
    image_bgr = image[:, :, ::-1]
    output_bgr, _ = upsampler.enhance(image_bgr, outscale=float(factor))
    if output_bgr is None:
        raise RuntimeError("Real-ESRGAN failed to produce an output image.")
    return output_bgr[:, :, ::-1].astype(np.uint8)


def upscale_image(image: np.ndarray, factor: int, strict_ai: bool = False) -> np.ndarray:
    normalized_factor = normalize_upscale_factor(factor)
    if normalized_factor <= 1 or image is None or image.size == 0:
        return image

    try:
        return _realesrgan_upscale(image, normalized_factor)
    except Exception as exc:
        if strict_ai:
            raise
        logger.warning(
            "Real-ESRGAN unavailable; falling back to Lanczos resize: %s",
            exc,
        )
        return _lanczos_upscale(image, normalized_factor)
