from __future__ import annotations

import importlib
import logging
import os
import site
import subprocess
import sys
import threading
import types
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
_realesrgan_lock = threading.RLock()
_realesrgan_upsampler = None
_deps_install_attempted = False
_deps_install_error: str | None = None
_fallback_warning_emitted = False


def _is_auto_install_enabled() -> bool:
    raw = os.getenv("COMICTRANSLATE_AUTO_INSTALL_UPSCALER_DEPS", "1")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _run_pip_install(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--user", *args]
    logger.info("Installing Real-ESRGAN dependency: %s", " ".join(args))
    subprocess.check_call(cmd)


def _ensure_user_site_on_path() -> None:
    """Ensure user-site packages are importable in the current process."""
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    if user_site and user_site not in sys.path and os.path.isdir(user_site):
        sys.path.append(user_site)
        logger.info("Added user site-packages to sys.path: %s", user_site)


def _ensure_torchvision_compat_shim() -> None:
    """
    Provide `torchvision.transforms.functional_tensor` for newer torchvision.

    `basicsr` imports this legacy module, but some newer torchvision builds
    removed it. We expose a lightweight shim that forwards attributes to
    `torchvision.transforms.functional`.
    """
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except Exception:
        pass

    try:
        from torchvision.transforms import functional as tv_functional
    except Exception:
        return

    shim = types.ModuleType(module_name)

    def _fallback_getattr(name: str):
        return getattr(tv_functional, name)

    shim.__getattr__ = _fallback_getattr  # type: ignore[attr-defined]
    if hasattr(tv_functional, "__all__"):
        shim.__all__ = list(tv_functional.__all__)  # type: ignore[attr-defined]

    # Ensure direct access for the most common symbol used by basicsr.
    if hasattr(tv_functional, "rgb_to_grayscale"):
        shim.rgb_to_grayscale = tv_functional.rgb_to_grayscale

    sys.modules[module_name] = shim
    logger.info("Applied torchvision compatibility shim: %s", module_name)


def _import_realesrgan_dependencies():
    _ensure_torchvision_compat_shim()
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    return torch, RRDBNet, RealESRGANer


def _auto_install_realesrgan_dependencies() -> None:
    global _deps_install_attempted, _deps_install_error
    _ensure_user_site_on_path()

    with _realesrgan_lock:
        if _deps_install_attempted:
            if _deps_install_error:
                raise RuntimeError(_deps_install_error)
            return
        _deps_install_attempted = True

    try:
        try:
            import torch  # noqa: F401
        except Exception:
            try:
                _run_pip_install(
                    [
                        "torch",
                        "torchvision",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                    ]
                )
            except Exception:
                _run_pip_install(["torch", "torchvision"])

        _run_pip_install(["realesrgan", "basicsr"])
        _ensure_user_site_on_path()
        importlib.invalidate_caches()
        _import_realesrgan_dependencies()
        _deps_install_error = None
        logger.info("Real-ESRGAN dependencies installed successfully.")
    except Exception as exc:
        logger.exception("Automatic install for Real-ESRGAN dependencies failed: %s", exc)
        reason = str(exc).strip()
        if reason:
            reason = f" ({reason})"
        _deps_install_error = (
            "Automatic install for Real-ESRGAN dependencies failed. "
            "Please install manually: `pip install --user torch realesrgan basicsr`"
            f"{reason}"
        )
        raise RuntimeError(_deps_install_error) from exc


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
    _ensure_user_site_on_path()
    try:
        torch, RRDBNet, RealESRGANer = _import_realesrgan_dependencies()
    except Exception as exc:
        if _is_auto_install_enabled():
            logger.info("Real-ESRGAN dependencies missing, attempting automatic install.")
            _auto_install_realesrgan_dependencies()
            torch, RRDBNet, RealESRGANer = _import_realesrgan_dependencies()
        else:
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
    global _fallback_warning_emitted
    normalized_factor = normalize_upscale_factor(factor)
    if normalized_factor <= 1 or image is None or image.size == 0:
        return image

    try:
        return _realesrgan_upscale(image, normalized_factor)
    except Exception as exc:
        if strict_ai:
            raise
        if not _fallback_warning_emitted:
            logger.warning(
                "Real-ESRGAN unavailable; falling back to Lanczos resize: %s",
                exc,
            )
            _fallback_warning_emitted = True
        else:
            logger.debug(
                "Real-ESRGAN unavailable; using Lanczos fallback: %s",
                exc,
            )
        return _lanczos_upscale(image, normalized_factor)
