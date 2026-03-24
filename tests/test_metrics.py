import numpy as np
import pytest

from metrics import compute_exposure
from metrics import compute_noise
from metrics import compute_region_metrics
from metrics import compute_sharpness


def test_compute_exposure_returns_masked_v_mean():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:] = [250, 250, 250]
    image[:2, :2] = [10, 40, 200]
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    assert compute_exposure(image, mask) == pytest.approx(200.0)


def test_compute_noise_uses_grayscale_std_inside_mask():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2, :2] = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    assert compute_noise(image, mask) == pytest.approx(85.3185213657655)


def test_compute_sharpness_detects_edge_content():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:] = 255
    assert compute_sharpness(image) > 0.0


def test_compute_region_metrics_with_none_mask_uses_full_image():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    result = compute_region_metrics(image)
    assert result["valid"] is True
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["exposure"] == pytest.approx(128.0)


def test_compute_region_metrics_marks_empty_mask_invalid():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    result = compute_region_metrics(image, mask)
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert result["coverage_ratio"] == 0.0


def test_compute_region_metrics_preserves_coverage_for_tiny_invalid_mask():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[0:2, 0:2] = True
    result = compute_region_metrics(image, mask)
    assert result["valid"] is False
    assert result["sharpness"] == 0.0
    assert result["noise"] == 0.0
    assert result["exposure"] == 0.0
    assert result["score"] == 0.0
    assert result["coverage_ratio"] == pytest.approx(4 / 256)


def test_compute_region_metrics_returns_full_result_for_valid_mask():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    image[:, 4:] = 255
    mask = np.ones((8, 8), dtype=bool)
    result = compute_region_metrics(image, mask)
    assert set(result) == {
        "sharpness",
        "noise",
        "exposure",
        "score",
        "coverage_ratio",
        "valid",
    }
    assert result["valid"] is True
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["sharpness"] > 0.0
    assert 0.0 <= result["score"] <= 100.0


def test_compute_region_metrics_rejects_invalid_mask_shape():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8, 1), dtype=bool)
    with pytest.raises(ValueError, match="mask must be 2-D and match image height/width"):
        compute_region_metrics(image, mask)
