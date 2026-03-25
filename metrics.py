from __future__ import annotations

import cv2
import numpy as np


def _as_bool_mask(mask: np.ndarray | None, image_shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(image_shape[:2], dtype=bool)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2 or mask_array.shape != tuple(image_shape[:2]):
        raise ValueError("mask must be 2-D and match image height/width")
    return mask_array


def _masked_gray(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    mask_array = _as_bool_mask(mask, image.shape)
    if image.ndim == 2:
        gray = np.asarray(image, dtype=np.float32)
    else:
        gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
    return gray[mask_array]


def compute_laplacian_clarity(image, mask=None):
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


def compute_edgegrad_mean(image, mask=None):
    mask_array = _as_bool_mask(mask, image.shape)
    if not np.count_nonzero(mask_array):
        return 0.0
    if image.ndim == 2:
        gray_image = np.asarray(image, dtype=np.float32)
    else:
        gray_image = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    return float(np.mean(gradient[mask_array]))


def compute_sigma_l(image, mask=None):
    gray_values = _masked_gray(image, mask)
    if gray_values.size == 0:
        return 0.0
    return float(np.std(gray_values))


def compute_sigma_c(image, mask=None):
    mask_array = _as_bool_mask(mask, image.shape)
    if np.count_nonzero(mask_array) == 0:
        return 0.0
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return 0.0
    ycrcb = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cb = ycrcb[..., 1]
    cr = ycrcb[..., 2]
    sigma_cb = float(np.std(cb[mask_array]))
    sigma_cr = float(np.std(cr[mask_array]))
    return float(np.sqrt((sigma_cb ** 2 + sigma_cr ** 2) / 2.0))


def normalize_laplacian_clarity(value):
    return float(np.clip((value / 300.0) * 100.0, 0.0, 100.0))


def normalize_edgegrad_mean(value):
    return float(np.clip((value / 160.0) * 100.0, 0.0, 100.0))


def normalize_sigma_l(value):
    return float(np.clip(100.0 - (value / 40.0) * 100.0, 0.0, 100.0))


def normalize_sigma_c(value):
    return float(np.clip(100.0 - (value / 40.0) * 100.0, 0.0, 100.0))


def compute_region_metrics(image, mask=None):
    mask_array = _as_bool_mask(mask, np.asarray(image).shape)
    total_pixels = int(mask_array.size)
    masked_pixels = int(np.count_nonzero(mask_array))
    coverage_ratio = float(masked_pixels / total_pixels) if total_pixels else 0.0

    if masked_pixels == 0:
        return {
            "Laplacian_Clarity": 0.0,
            "EdgeGrad_mean": 0.0,
            "sigma_L": 0.0,
            "sigma_C": 0.0,
            "score": 0.0,
            "coverage_ratio": 0.0,
            "valid": False,
        }

    laplacian_clarity = compute_laplacian_clarity(image, mask_array)
    edgegrad_mean = compute_edgegrad_mean(image, mask_array)
    sigma_l = compute_sigma_l(image, mask_array)
    sigma_c = compute_sigma_c(image, mask_array)
    valid = masked_pixels >= 64 and coverage_ratio >= 0.005

    if not valid:
        return {
            "Laplacian_Clarity": 0.0,
            "EdgeGrad_mean": 0.0,
            "sigma_L": 0.0,
            "sigma_C": 0.0,
            "score": 0.0,
            "coverage_ratio": coverage_ratio,
            "valid": False,
        }

    laplacian_clarity_score = normalize_laplacian_clarity(laplacian_clarity)
    edgegrad_mean_score = normalize_edgegrad_mean(edgegrad_mean)
    sigma_l_score = normalize_sigma_l(sigma_l)
    sigma_c_score = normalize_sigma_c(sigma_c)
    score = (
        0.35 * laplacian_clarity_score
        + 0.25 * edgegrad_mean_score
        + 0.2 * sigma_l_score
        + 0.2 * sigma_c_score
    )

    return {
        "Laplacian_Clarity": float(laplacian_clarity),
        "EdgeGrad_mean": float(edgegrad_mean),
        "sigma_L": float(sigma_l),
        "sigma_C": float(sigma_c),
        "score": float(score),
        "coverage_ratio": coverage_ratio,
        "valid": True,
    }
