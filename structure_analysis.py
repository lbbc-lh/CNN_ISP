from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

STRUCTURE_REGIONS = ("flat", "edge", "texture")
STRUCTURE_COLORS = {
    "flat": (214, 167, 57),
    "edge": (53, 106, 255),
    "texture": (46, 166, 92),
}


@dataclass
class StructureAnalysisResult:
    masks: dict[str, np.ndarray]
    overlay: np.ndarray
    mask_visualizations: dict[str, np.ndarray]
    gradient_map: np.ndarray
    variance_map: np.ndarray


def _normalize_map(values: np.ndarray) -> np.ndarray:
    max_value = float(values.max()) if values.size else 0.0
    if max_value <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return values.astype(np.float32) / max_value


def build_structure_masks(
    image: np.ndarray,
    gradient_threshold: float = 0.22,
    flat_gradient_threshold: float = 0.10,
    flat_variance_threshold: float = 0.08,
    variance_kernel_size: int = 9,
) -> dict[str, np.ndarray]:
    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient_n = _normalize_map(gradient)

    local_mean = cv2.blur(gray, (variance_kernel_size, variance_kernel_size))
    local_sq_mean = cv2.blur(gray * gray, (variance_kernel_size, variance_kernel_size))
    local_variance = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    local_variance_n = _normalize_map(local_variance)

    edge_mask = gradient_n >= gradient_threshold
    flat_mask = (gradient_n <= flat_gradient_threshold) & (local_variance_n <= flat_variance_threshold)
    texture_mask = ~(edge_mask | flat_mask)

    return {
        "flat": flat_mask,
        "edge": edge_mask,
        "texture": texture_mask,
    }


def compute_structure_maps(
    image: np.ndarray,
    variance_kernel_size: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    local_mean = cv2.blur(gray, (variance_kernel_size, variance_kernel_size))
    local_sq_mean = cv2.blur(gray * gray, (variance_kernel_size, variance_kernel_size))
    local_variance = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    return gradient, local_variance


def compute_flat_noise(image: np.ndarray, mask: np.ndarray) -> float:
    if not np.count_nonzero(mask):
        return 0.0
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    return float(np.std(gray[mask]))


def compute_edge_sharpness(gradient_map: np.ndarray, mask: np.ndarray) -> float:
    if not np.count_nonzero(mask):
        return 0.0
    return float(np.mean(gradient_map[mask]))


def compute_texture_clarity(variance_map: np.ndarray, mask: np.ndarray) -> float:
    if not np.count_nonzero(mask):
        return 0.0
    return float(np.mean(variance_map[mask]))


def compute_semantic_structure_metrics(
    image: np.ndarray,
    semantic_masks: dict[str, np.ndarray],
    structure_masks: dict[str, np.ndarray],
    gradient_map: np.ndarray,
    variance_map: np.ndarray,
) -> dict[str, dict[str, float | bool]]:
    total_pixels = image.shape[0] * image.shape[1]
    results: dict[str, dict[str, float | bool]] = {}

    for semantic_name, semantic_mask in semantic_masks.items():
        semantic_pixels = int(np.count_nonzero(semantic_mask))
        flat_mask = semantic_mask & structure_masks["flat"]
        edge_mask = semantic_mask & structure_masks["edge"]
        texture_mask = semantic_mask & structure_masks["texture"]

        flat_pixels = int(np.count_nonzero(flat_mask))
        edge_pixels = int(np.count_nonzero(edge_mask))
        texture_pixels = int(np.count_nonzero(texture_mask))

        results[semantic_name] = {
            "valid": semantic_pixels > 0,
            "coverage_ratio": float(semantic_pixels / total_pixels) if total_pixels else 0.0,
            "flat_coverage_ratio": float(flat_pixels / semantic_pixels) if semantic_pixels else 0.0,
            "edge_coverage_ratio": float(edge_pixels / semantic_pixels) if semantic_pixels else 0.0,
            "texture_coverage_ratio": float(texture_pixels / semantic_pixels) if semantic_pixels else 0.0,
            "flat_noise": compute_flat_noise(image, flat_mask),
            "edge_sharpness": compute_edge_sharpness(gradient_map, edge_mask),
            "texture_clarity": compute_texture_clarity(variance_map, texture_mask),
        }

    return results


def create_structure_overlay(image: np.ndarray, masks: dict[str, np.ndarray], alpha: float = 0.40) -> np.ndarray:
    overlay_layer = np.zeros_like(image, dtype=np.uint8)
    for region, color in STRUCTURE_COLORS.items():
        overlay_layer[masks[region]] = color
    return cv2.addWeighted(image, 1.0 - alpha, overlay_layer, alpha, 0.0)


def create_structure_mask_visualizations(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    visuals: dict[str, np.ndarray] = {}
    for region in STRUCTURE_REGIONS:
        mask = masks[region].astype(np.uint8) * 255
        visuals[region] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return visuals


def analyze_structure(image: np.ndarray) -> StructureAnalysisResult:
    gradient_map, variance_map = compute_structure_maps(image)
    masks = build_structure_masks(image)
    return StructureAnalysisResult(
        masks=masks,
        overlay=create_structure_overlay(image, masks),
        mask_visualizations=create_structure_mask_visualizations(masks),
        gradient_map=gradient_map,
        variance_map=variance_map,
    )
