from pathlib import Path

from fastapi.testclient import TestClient

import api


class StubAnalyzer:
    def analyze_base64(self, payload: str) -> dict:
        if payload == "not-base64":
            raise ValueError("invalid base64 image payload")
        return {
            "request_id": "req-1",
            "global_metrics": {"sharpness": 1.0, "noise": 2.0, "exposure": 3.0, "score": 4.0, "coverage_ratio": 1.0, "valid": True},
            "region_metrics": {
                "person": {"sharpness": 1.0, "noise": 2.0, "exposure": 3.0, "score": 4.0, "coverage_ratio": 0.2, "valid": True},
                "sky": {"sharpness": 1.0, "noise": 2.0, "exposure": 3.0, "score": 4.0, "coverage_ratio": 0.2, "valid": True},
                "vegetation": {"sharpness": 1.0, "noise": 2.0, "exposure": 3.0, "score": 4.0, "coverage_ratio": 0.2, "valid": True},
                "background": {"sharpness": 1.0, "noise": 2.0, "exposure": 3.0, "score": 4.0, "coverage_ratio": 0.4, "valid": True},
            },
            "final_score": 4.0,
            "effective_weights": {"person": 0.7, "sky": 0.0, "vegetation": 0.0, "global": 0.3},
            "score_breakdown": {"person": 2.8, "sky": 0.0, "vegetation": 0.0, "global": 1.2},
            "missing_regions": ["sky", "vegetation"],
            "detected_regions": [
                {"label": "person", "coverage_ratio": 0.2, "weight": 0.7, "metrics": {"sharpness": 1.0, "noise": 2.0, "exposure": 3.0, "score": 4.0, "coverage_ratio": 0.2, "valid": True}}
            ],
            "suggestions": ["increase sharpening"],
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


def test_homepage_returns_html():
    client = TestClient(api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CNN Image Quality Analyzer" in response.text


def test_analyze_endpoint_rejects_invalid_payload(monkeypatch):
    monkeypatch.setattr(api, "analyzer", StubAnalyzer())
    client = TestClient(api.app)

    response = client.post("/analyze", json={"image": "not-base64"})

    assert response.status_code == 400


def test_analyze_endpoint_returns_500_for_unexpected_failure(monkeypatch):
    class BrokenAnalyzer:
        def analyze_base64(self, payload: str) -> dict:
            raise RuntimeError("boom")

    monkeypatch.setattr(api, "analyzer", BrokenAnalyzer())
    client = TestClient(api.app)

    response = client.post("/analyze", json={"image": "ok"})

    assert response.status_code == 500
