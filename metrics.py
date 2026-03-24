from __future__ import annotations

import cv2
import numpy as np


def _as_bool_mask(mask: np.ndarray | None, image_shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(image_shape[:2], dtype=bool)
    return np.asarray(mask, dtype=bool)


def _masked_gray(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    mask_array = _as_bool_mask(mask, image.shape)
    if image.ndim == 2:
        gray = np.asarray(image, dtype=np.float32)
    else:
        gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
    return gray[mask_array]


def compute_sharpness(image, mask=None):
    gray_values = _masked_gray(image, mask)
    if gray_values.size == 0:
        return 0.0
    if image.ndim == 2:
        gray_image = np.asarray(image, dtype=np.float32)
    else:
        gray_image = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
    laplacian = cv2.Laplacian(gray_image, cv2.CV_32F)
    mask_array = _as_bool_mask(mask, image.shape)
    return float(np.var(laplacian[mask_array]))


def compute_noise(image, mask=None):
    gray_values = _masked_gray(image, mask)
    if gray_values.size == 0:
        return 0.0
    return float(np.std(gray_values))


def compute_exposure(image, mask=None):
    mask_array = _as_bool_mask(mask, image.shape)
    if np.count_nonzero(mask_array) == 0:
        return 0.0
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        v_channel = image_array.astype(np.float32)
    else:
        hsv = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_BGR2HSV)
        v_channel = hsv[..., 2].astype(np.float32)
    return float(v_channel[mask_array].mean())


def normalize_sharpness(value):
    return float(np.clip((value / 300.0) * 100.0, 0.0, 100.0))


def normalize_noise(value):
    return float(np.clip(100.0 - (value / 40.0) * 100.0, 0.0, 100.0))


def normalize_exposure(value):
    return float(np.clip(100.0 - (abs(value - 128.0) / 128.0) * 100.0, 0.0, 100.0))


def compute_region_metrics(image, mask=None):
    mask_array = _as_bool_mask(mask, np.asarray(image).shape)
    total_pixels = int(mask_array.size)
    masked_pixels = int(np.count_nonzero(mask_array))
    coverage_ratio = float(masked_pixels / total_pixels) if total_pixels else 0.0

    if masked_pixels == 0:
        return {
            "sharpness": 0.0,
            "noise": 0.0,
            "exposure": 0.0,
            "score": 0.0,
            "coverage_ratio": 0.0,
            "valid": False,
        }

    sharpness = compute_sharpness(image, mask_array)
    noise = compute_noise(image, mask_array)
    exposure = compute_exposure(image, mask_array)
    valid = masked_pixels >= 64 and coverage_ratio >= 0.005

    if not valid:
        return {
            "sharpness": 0.0,
            "noise": 0.0,
            "exposure": 0.0,
            "score": 0.0,
            "coverage_ratio": coverage_ratio,
            "valid": False,
        }

    sharpness_score = normalize_sharpness(sharpness)
    noise_score = normalize_noise(noise)
    exposure_score = normalize_exposure(exposure)
    score = 0.4 * sharpness_score + 0.3 * noise_score + 0.3 * exposure_score

    return {
        "sharpness": float(sharpness),
        "noise": float(noise),
        "exposure": float(exposure),
        "score": float(score),
        "coverage_ratio": coverage_ratio,
        "valid": True,
    }
