import numpy as np

from structure_analysis import analyze_structure
from structure_analysis import build_structure_masks
from structure_analysis import compute_semantic_structure_metrics


def test_build_structure_masks_returns_three_regions():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    masks = build_structure_masks(image)
    assert set(masks.keys()) == {"flat", "edge", "texture"}


def test_flat_image_produces_only_flat_region():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    masks = build_structure_masks(image)
    assert masks["flat"].all()
    assert not masks["edge"].any()
    assert not masks["texture"].any()


def test_edge_image_marks_some_pixels_as_edge():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    masks = build_structure_masks(image)
    assert masks["edge"].any()


def test_analyze_structure_returns_overlay_and_mask_visualizations():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    result = analyze_structure(image)
    assert result.overlay.shape == image.shape
    assert set(result.mask_visualizations.keys()) == {"flat", "edge", "texture"}


def test_compute_semantic_structure_metrics_returns_three_isp_metrics():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    structure = analyze_structure(image)
    semantic_masks = {
        "person": np.ones((32, 32), dtype=bool),
        "sky": np.zeros((32, 32), dtype=bool),
    }
    metrics = compute_semantic_structure_metrics(
        image,
        semantic_masks,
        structure.masks,
        structure.gradient_map,
        structure.variance_map,
    )
    assert set(metrics["person"].keys()) == {
        "valid",
        "coverage_ratio",
        "flat_coverage_ratio",
        "edge_coverage_ratio",
        "texture_coverage_ratio",
        "flat_noise",
        "edge_sharpness",
        "texture_clarity",
    }
    assert metrics["person"]["edge_sharpness"] >= 0.0
    assert metrics["sky"]["valid"] is False
