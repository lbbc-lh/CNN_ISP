from pathlib import Path

from fastapi.testclient import TestClient

import api


class StubAnalyzer:
    def analyze_base64(self, payload: str) -> dict:
        if payload == "not-base64":
            raise ValueError("invalid base64 image payload")
        return {
            "request_id": "req-1",
            "global_metrics": {"Laplacian_Clarity": 1.0, "EdgeGrad_mean": 2.0, "sigma_L": 3.0, "sigma_C": 4.0, "score": 5.0, "coverage_ratio": 1.0, "valid": True},
            "region_metrics": {
                "person": {"Laplacian_Clarity": 1.0, "EdgeGrad_mean": 2.0, "sigma_L": 3.0, "sigma_C": 4.0, "score": 5.0, "coverage_ratio": 0.2, "valid": True},
                "sky": {"Laplacian_Clarity": 1.0, "EdgeGrad_mean": 2.0, "sigma_L": 3.0, "sigma_C": 4.0, "score": 5.0, "coverage_ratio": 0.2, "valid": True},
                "vegetation": {"Laplacian_Clarity": 1.0, "EdgeGrad_mean": 2.0, "sigma_L": 3.0, "sigma_C": 4.0, "score": 5.0, "coverage_ratio": 0.2, "valid": True},
                "background": {"Laplacian_Clarity": 1.0, "EdgeGrad_mean": 2.0, "sigma_L": 3.0, "sigma_C": 4.0, "score": 5.0, "coverage_ratio": 0.4, "valid": True},
            },
            "semantic_structure_metrics": {
                "person": {
                    "valid": True,
                    "coverage_ratio": 0.2,
                    "flat_coverage_ratio": 0.2,
                    "edge_coverage_ratio": 0.5,
                    "texture_coverage_ratio": 0.3,
                    "flat_noise": 2.0,
                    "edge_sharpness": 10.0,
                    "texture_clarity": 11.0,
                },
                "sky": {
                    "valid": False,
                    "coverage_ratio": 0.0,
                    "flat_coverage_ratio": 0.0,
                    "edge_coverage_ratio": 0.0,
                    "texture_coverage_ratio": 0.0,
                    "flat_noise": 0.0,
                    "edge_sharpness": 0.0,
                    "texture_clarity": 0.0,
                },
                "vegetation": {
                    "valid": False,
                    "coverage_ratio": 0.0,
                    "flat_coverage_ratio": 0.0,
                    "edge_coverage_ratio": 0.0,
                    "texture_coverage_ratio": 0.0,
                    "flat_noise": 0.0,
                    "edge_sharpness": 0.0,
                    "texture_clarity": 0.0,
                },
                "background": {
                    "valid": True,
                    "coverage_ratio": 0.8,
                    "flat_coverage_ratio": 0.4,
                    "edge_coverage_ratio": 0.2,
                    "texture_coverage_ratio": 0.4,
                    "flat_noise": 3.0,
                    "edge_sharpness": 7.0,
                    "texture_clarity": 5.0,
                },
            },
            "final_score": 5.0,
            "effective_weights": {"person": 0.7, "sky": 0.0, "vegetation": 0.0, "global": 0.3},
            "score_breakdown": {"person": 3.5, "sky": 0.0, "vegetation": 0.0, "global": 1.5},
            "missing_regions": ["sky", "vegetation"],
            "detected_regions": [
                {"label": "person", "coverage_ratio": 0.2, "weight": 0.7, "metrics": {"Laplacian_Clarity": 1.0, "EdgeGrad_mean": 2.0, "sigma_L": 3.0, "sigma_C": 4.0, "score": 5.0, "coverage_ratio": 0.2, "valid": True}}
            ],
            "suggestions": ["增加锐化"],
            "segmentation_debug": {
                "top_labels": [
                    {"label_id": 3, "label_name": "sky", "pixel_count": 64, "coverage_ratio": 0.25}
                ]
            },
            "visualizations": {
                "overlay_base64": "abc",
                "mask_base64": {
                    "person": "a",
                    "sky": "b",
                    "vegetation": "c",
                    "background": "d",
                },
                "saved_files": {
                    "overlay": "outputs/req-1_overlay.png",
                    "person": "outputs/req-1_person_mask.png",
                    "sky": "outputs/req-1_sky_mask.png",
                    "vegetation": "outputs/req-1_vegetation_mask.png",
                    "background": "outputs/req-1_background_mask.png",
                },
            },
        }

    def compare_base64(self, reference_payload: str, test_payload: str) -> dict:
        if reference_payload == "not-base64" or test_payload == "not-base64":
            raise ValueError("invalid base64 image payload")
        analyzed = self.analyze_base64("ok")
        return {
            "reference": analyzed,
            "test": analyzed,
            "delta": {
                "final_score_gap": 0.0,
                "global_metrics_gap": {
                    "Laplacian_Clarity": 0.0,
                    "EdgeGrad_mean": 0.0,
                    "sigma_L": 0.0,
                    "sigma_C": 0.0,
                },
                "region_score_gap": {
                    "person": 0.0,
                    "sky": 0.0,
                    "vegetation": 0.0,
                    "background": 0.0,
                },
            },
        }


def test_analyze_endpoint_returns_200(monkeypatch):
    monkeypatch.setattr(api, "analyzer", StubAnalyzer())
    client = TestClient(api.app)

    response = client.post("/analyze", json={"image": "ok"})

    assert response.status_code == 200
    body = response.json()
    assert "request_id" in body
    assert "final_score" in body
    assert "effective_weights" in body
    assert "missing_regions" in body


def test_compare_endpoint_returns_200(monkeypatch):
    monkeypatch.setattr(api, "analyzer", StubAnalyzer())
    client = TestClient(api.app)

    response = client.post("/compare", json={"reference_image": "ok", "test_image": "ok"})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"reference", "test", "delta"}
    assert body["delta"]["final_score_gap"] == 0.0


def test_homepage_returns_html():
    client = TestClient(api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CNN 图像质量分析看板" in response.text
    assert "开始对比" in response.text


def test_analyze_endpoint_rejects_invalid_payload(monkeypatch):
    monkeypatch.setattr(api, "analyzer", StubAnalyzer())
    client = TestClient(api.app)

    response = client.post("/analyze", json={"image": "not-base64"})

    assert response.status_code == 400


def test_compare_endpoint_rejects_invalid_payload(monkeypatch):
    monkeypatch.setattr(api, "analyzer", StubAnalyzer())
    client = TestClient(api.app)

    response = client.post("/compare", json={"reference_image": "not-base64", "test_image": "ok"})

    assert response.status_code == 400


def test_analyze_endpoint_returns_500_for_unexpected_failure(monkeypatch):
    class BrokenAnalyzer:
        def analyze_base64(self, payload: str) -> dict:
            raise RuntimeError("boom")

    monkeypatch.setattr(api, "analyzer", BrokenAnalyzer())
    client = TestClient(api.app)

    response = client.post("/analyze", json={"image": "ok"})

    assert response.status_code == 500
