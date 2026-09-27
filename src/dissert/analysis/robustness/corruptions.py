"""
robustness/corruptions.py — spec §13's PHOTOMETRIC and ACQUISITION
corruption families: Gaussian noise, speckle, blur, JPEG compression,
brightness/contrast shift, gamma (photometric); resolution change,
resampling (acquisition). Severity levels 1-5, declared once here
(``SEVERITY_LEVELS``) and shared — robustness/geometric.py's own
severities reuse this same 1-5 scale, not a fresh one.

Every function operates on one ``(H, W, 3)`` uint8 RGB image (the
convention datasets/augment.py's albumentations pipeline already uses) and
is meant to run at test time only, after resize, never during training —
nothing here is wired into the training pipeline.
"""
from __future__ import annotations

from typing import Callable, Dict

import albumentations as A
import cv2
import numpy as np

SEVERITY_LEVELS = (1, 2, 3, 4, 5)


def _check_severity(severity: int) -> None:
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"severity must be one of {SEVERITY_LEVELS}, got {severity}")


def gaussian_noise(image: np.ndarray, severity: int) -> np.ndarray:
    _check_severity(severity)
    var_limit = {1: (5, 10), 2: (10, 20), 3: (20, 40), 4: (40, 70), 5: (70, 110)}[severity]
    return A.GaussNoise(var_limit=var_limit, p=1.0)(image=image)["image"]


def speckle_noise(image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    """Multiplicative noise (``x + x*n``, ``n ~ N(0, sigma^2)``) — the
    ImageNet-C sense of "speckle", distinct from additive Gaussian noise
    above. No built-in albumentations transform for this, implemented
    directly.
    """
    _check_severity(severity)
    sigma = {1: 0.06, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.28}[severity]
    rng = np.random.default_rng(seed)
    img_f = image.astype(np.float32) / 255.0
    noisy = img_f + img_f * rng.normal(0.0, sigma, img_f.shape).astype(np.float32)
    return (np.clip(noisy, 0.0, 1.0) * 255).astype(np.uint8)


def blur(image: np.ndarray, severity: int) -> np.ndarray:
    _check_severity(severity)
    blur_limit = {1: (3, 3), 2: (3, 5), 3: (5, 7), 4: (7, 9), 5: (9, 11)}[severity]
    return A.GaussianBlur(blur_limit=blur_limit, p=1.0)(image=image)["image"]


def jpeg_compression(image: np.ndarray, severity: int) -> np.ndarray:
    _check_severity(severity)
    quality = {1: 80, 2: 65, 3: 50, 4: 30, 5: 15}[severity]
    return A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0)(image=image)["image"]


def brightness_contrast_shift(image: np.ndarray, severity: int) -> np.ndarray:
    _check_severity(severity)
    limit = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}[severity]
    return A.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=limit, p=1.0)(image=image)["image"]


def gamma_shift(image: np.ndarray, severity: int) -> np.ndarray:
    _check_severity(severity)
    gamma_limit = {1: (90, 110), 2: (80, 120), 3: (70, 140), 4: (60, 160), 5: (50, 180)}[severity]
    return A.RandomGamma(gamma_limit=gamma_limit, p=1.0)(image=image)["image"]


def resolution_change(image: np.ndarray, severity: int) -> np.ndarray:
    """Downscale (area-averaging, an anti-aliased/smooth reduction) then
    upscale back to the original resolution via bilinear interpolation —
    a genuine information-loss "lower sensor resolution" simulation,
    distinct from ``resampling``'s aliasing artefact below.
    """
    _check_severity(severity)
    scale = {1: 0.9, 2: 0.75, 3: 0.6, 4: 0.45, 5: 0.3}[severity]
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def resampling(image: np.ndarray, severity: int) -> np.ndarray:
    """Downscale/upscale round-trip via nearest-neighbour interpolation —
    a blocky aliasing artefact, mechanically distinct from
    ``resolution_change``'s smooth information loss (same severity->scale
    schedule, different interpolation kernel throughout).
    """
    _check_severity(severity)
    scale = {1: 0.9, 2: 0.75, 3: 0.6, 4: 0.45, 5: 0.3}[severity]
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


CORRUPTIONS: Dict[str, Callable[..., np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "speckle_noise": speckle_noise,
    "blur": blur,
    "jpeg_compression": jpeg_compression,
    "brightness_contrast": brightness_contrast_shift,
    "gamma": gamma_shift,
    "resolution_change": resolution_change,
    "resampling": resampling,
}

PHOTOMETRIC = ("gaussian_noise", "speckle_noise", "blur", "jpeg_compression", "brightness_contrast", "gamma")
ACQUISITION = ("resolution_change", "resampling")
