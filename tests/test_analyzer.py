import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from analyzer import ImageQualityAnalyzer
from analyzer import decode_base64_image


class FakeSegmentationResult:
    def __init__(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        person = np.zeros((h, w), dtype=bool)
        person[: h // 2, : w // 2] = True
        sky = np.zeros((h, w), dtype=bool)
        sky[: h // 2, w // 2 :] = True
        vegetation = np.zeros((h, w), dtype=bool)
        vegetation[h // 2 :, : w // 2] = True
        background = ~(person | sky | vegetation)

        self.masks = {
            "person": person,
            "sky": sky,
            "vegetation": vegetation,
            "background": background,
        }
        self.overlay = image.copy()
        self.mask_visualizations = {
            region: cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for region, mask in self.masks.items()
        }


class FakeEngine:
    def segment(self, image: np.ndarray) -> FakeSegmentationResult:
        return FakeSegmentationResult(image)


@pytest.fixture
def sample_base64() -> str:
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def test_decode_image_accepts_data_uri_prefix(sample_base64):
    payload = f"data:image/png;base64,{sample_base64}"
    image = decode_base64_image(payload)
    assert image.shape == (16, 16, 3)


def test_analyze_image_returns_expected_top_level_keys(sample_base64, tmp_path: Path):
    analyzer = ImageQualityAnalyzer(segmentation_engine=FakeEngine(), output_dir=tmp_path / "outputs")
    result = analyzer.analyze_base64(sample_base64)

    assert "request_id" in result
    assert "global_metrics" in result
    assert "region_metrics" in result
    assert "semantic_structure_metrics" in result
    assert "final_score" in result
    assert "effective_weights" in result
    assert "score_breakdown" in result
    assert "missing_regions" in result
    assert "detected_regions" in result
    assert "suggestions" in result
    assert "visualizations" in result
    assert "structure_visualizations" in result
    assert set(result["region_metrics"].keys()) == {"person", "sky", "vegetation", "background"}
    assert set(result["visualizations"]["mask_base64"].keys()) == {"person", "sky", "vegetation", "background"}
    assert set(result["semantic_structure_metrics"].keys()) == {"person", "sky", "vegetation", "background"}
    assert set(result["structure_visualizations"]["mask_base64"].keys()) == {"flat", "edge", "texture"}


def test_final_score_renormalizes_only_across_valid_regions(sample_base64, tmp_path: Path):
    analyzer = ImageQualityAnalyzer(segmentation_engine=FakeEngine(), output_dir=tmp_path / "outputs")
    result = analyzer.analyze_base64(sample_base64)
    active = {
        "person": result["region_metrics"]["person"]["valid"],
        "sky": result["region_metrics"]["sky"]["valid"],
        "vegetation": result["region_metrics"]["vegetation"]["valid"],
        "global": result["global_metrics"]["valid"],
    }
    total_weight = sum(
        base_weight
        for name, base_weight in {
            "person": 0.5,
            "sky": 0.2,
            "vegetation": 0.2,
            "global": 0.1,
        }.items()
        if active[name]
    )
    expected = (
        result["effective_weights"]["person"] * result["region_metrics"]["person"]["score"]
        + result["effective_weights"]["sky"] * result["region_metrics"]["sky"]["score"]
        + result["effective_weights"]["vegetation"] * result["region_metrics"]["vegetation"]["score"]
        + result["effective_weights"]["global"] * result["global_metrics"]["score"]
    )
    assert result["final_score"] == pytest.approx(expected)
    assert pytest.approx(sum(result["effective_weights"].values())) == 1.0
    assert result["effective_weights"]["background"] == 0.0 if "background" in result["effective_weights"] else True
    assert total_weight > 0.0


def test_missing_regions_are_reported_when_region_is_invalid(sample_base64, tmp_path: Path):
    analyzer = ImageQualityAnalyzer(segmentation_engine=FakeEngine(), output_dir=tmp_path / "outputs")
    result = analyzer.analyze_base64(sample_base64)
    assert set(result["missing_regions"]).issubset({"person", "sky", "vegetation"})


def test_region_metrics_use_new_isp_metric_names(sample_base64, tmp_path: Path):
    analyzer = ImageQualityAnalyzer(segmentation_engine=FakeEngine(), output_dir=tmp_path / "outputs")
    result = analyzer.analyze_base64(sample_base64)
    assert set(result["global_metrics"].keys()) == {
        "Laplacian_Clarity",
        "EdgeGrad_mean",
        "sigma_L",
        "sigma_C",
        "score",
        "coverage_ratio",
        "valid",
    }


def test_compare_base64_returns_reference_test_and_delta(sample_base64, tmp_path: Path):
    analyzer = ImageQualityAnalyzer(segmentation_engine=FakeEngine(), output_dir=tmp_path / "outputs")
    result = analyzer.compare_base64(sample_base64, sample_base64)
    assert set(result.keys()) == {"reference", "test", "delta"}
    assert result["delta"]["final_score_gap"] == pytest.approx(0.0)
    assert "global_metrics_gap" in result["delta"]
    assert "region_score_gap" in result["delta"]
