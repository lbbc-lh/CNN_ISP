from __future__ import annotations

import base64
import binascii
import logging
import time
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from metrics import compute_region_metrics
from segmentation import SegmentationEngine
from segmentation import TARGET_REGIONS
from structure_analysis import analyze_structure
from structure_analysis import compute_semantic_structure_metrics

LOGGER = logging.getLogger(__name__)

FINAL_SCORE_WEIGHTS = {
    "person": 0.5,
    "sky": 0.2,
    "vegetation": 0.2,
    "global": 0.1,
}


def decode_base64_image(payload: str) -> np.ndarray:
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]

    try:
        binary = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 image payload") from exc

    buffer = np.frombuffer(binary, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        pil_image = Image.open(BytesIO(binary)).convert("RGB")
    except Exception as exc:
        raise ValueError("unable to decode image payload") from exc

    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_image_to_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("unable to encode image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def build_suggestions(global_metrics: dict, region_metrics: dict[str, dict]) -> list[str]:
    suggestions: list[str] = []

    def add(value: str) -> None:
        if value not in suggestions and len(suggestions) < 6:
            suggestions.append(value)

    ordered_scopes = [
        ("person", region_metrics.get("person", {})),
        ("global", global_metrics),
        ("sky", region_metrics.get("sky", {})),
        ("vegetation", region_metrics.get("vegetation", {})),
    ]

    for scope, metrics in ordered_scopes:
        if scope != "global" and not metrics.get("valid", False):
            continue

        laplacian_clarity = metrics.get("Laplacian_Clarity", 0.0)
        edgegrad_mean = metrics.get("EdgeGrad_mean", 0.0)
        sigma_l = metrics.get("sigma_L", 0.0)
        sigma_c = metrics.get("sigma_C", 0.0)

        if laplacian_clarity < 60 or edgegrad_mean < 35:
            add("增强人物细节" if scope == "person" else "增加锐化")
        if sigma_l > 18:
            add("启用亮度降噪")
        if sigma_c > 18:
            add("降低色度噪声")
        if scope == "sky" and sigma_l > 18:
            add("降低平坦区域的亮度/色度噪声")
        if scope == "vegetation" and laplacian_clarity < 60 and edgegrad_mean > 80:
            add("减轻纹理区域的锐化光晕")

    return suggestions


class ImageQualityAnalyzer:
    def __init__(self, segmentation_engine: SegmentationEngine | None = None, output_dir: str | Path = "outputs") -> None:
        self.segmentation_engine = segmentation_engine or SegmentationEngine()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_visualizations(
        self,
        request_id: str,
        overlay: np.ndarray,
        mask_visualizations: dict[str, np.ndarray],
        prefix: str = "",
    ) -> dict[str, str]:
        saved_files: dict[str, str] = {}
        stem_prefix = f"{prefix}_" if prefix else ""
        overlay_path = self.output_dir / f"{request_id}_{stem_prefix}overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        saved_files["overlay"] = str(overlay_path.relative_to(self.output_dir.parent))

        for region, mask_image in mask_visualizations.items():
            path = self.output_dir / f"{request_id}_{stem_prefix}{region}_mask.png"
            cv2.imwrite(str(path), mask_image)
            saved_files[region] = str(path.relative_to(self.output_dir.parent))

        LOGGER.info("Saved visualizations for request %s", request_id)
        return saved_files

    def _compute_final_score(
        self,
        global_metrics: dict,
        region_metrics: dict[str, dict],
    ) -> tuple[float, dict[str, float], dict[str, float], list[str]]:
        available_scores = {
            "global": global_metrics["score"],
            "person": region_metrics["person"]["score"],
            "sky": region_metrics["sky"]["score"],
            "vegetation": region_metrics["vegetation"]["score"],
        }
        validity = {
            "global": global_metrics.get("valid", False),
            "person": region_metrics["person"].get("valid", False),
            "sky": region_metrics["sky"].get("valid", False),
            "vegetation": region_metrics["vegetation"].get("valid", False),
        }

        active_components = [name for name, valid in validity.items() if valid]
        missing_regions = [region for region in ("person", "sky", "vegetation") if not region_metrics[region]["valid"]]

        total_active_weight = sum(FINAL_SCORE_WEIGHTS[name] for name in active_components)
        if total_active_weight == 0:
            effective_weights = {name: 0.0 for name in FINAL_SCORE_WEIGHTS}
            score_breakdown = {name: 0.0 for name in FINAL_SCORE_WEIGHTS}
            return 0.0, effective_weights, score_breakdown, missing_regions

        effective_weights = {
            name: (FINAL_SCORE_WEIGHTS[name] / total_active_weight if name in active_components else 0.0)
            for name in FINAL_SCORE_WEIGHTS
        }
        score_breakdown = {
            name: effective_weights[name] * available_scores[name]
            for name in FINAL_SCORE_WEIGHTS
        }
        final_score = float(sum(score_breakdown.values()))
        return final_score, effective_weights, score_breakdown, missing_regions

    def _build_detected_regions(self, region_metrics: dict[str, dict], effective_weights: dict[str, float]) -> list[dict]:
        detected_regions: list[dict] = []
        for region in TARGET_REGIONS:
            metrics = region_metrics[region]
            if not metrics.get("valid", False):
                continue
            detected_regions.append(
                {
                    "label": region,
                    "coverage_ratio": metrics["coverage_ratio"],
                    "weight": effective_weights.get(region, 0.0),
                    "metrics": metrics,
                }
            )
        return detected_regions

    def analyze_base64(self, payload: str) -> dict:
        start_time = time.perf_counter()
        request_id = f"{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

        LOGGER.info("Starting image analysis for request %s", request_id)
        image = decode_base64_image(payload)
        segmentation_result = self.segmentation_engine.segment(image)
        structure_result = analyze_structure(image)

        global_metrics = compute_region_metrics(image)
        region_metrics = {
            region: compute_region_metrics(image, segmentation_result.masks[region])
            for region in TARGET_REGIONS
        }
        semantic_structure_metrics = compute_semantic_structure_metrics(
            image,
            segmentation_result.masks,
            structure_result.masks,
            structure_result.gradient_map,
            structure_result.variance_map,
        )

        final_score, effective_weights, score_breakdown, missing_regions = self._compute_final_score(
            global_metrics,
            region_metrics,
        )
        suggestions = build_suggestions(global_metrics, region_metrics)
        saved_files = self._save_visualizations(
            request_id,
            segmentation_result.overlay,
            segmentation_result.mask_visualizations,
        )
        structure_saved_files = self._save_visualizations(
            request_id,
            structure_result.overlay,
            structure_result.mask_visualizations,
            prefix="structure",
        )

        result = {
            "request_id": request_id,
            "global_metrics": global_metrics,
            "region_metrics": region_metrics,
            "semantic_structure_metrics": semantic_structure_metrics,
            "final_score": final_score,
            "effective_weights": effective_weights,
            "score_breakdown": score_breakdown,
            "missing_regions": missing_regions,
            "detected_regions": self._build_detected_regions(region_metrics, effective_weights),
            "suggestions": suggestions,
            "segmentation_debug": {
                "top_labels": segmentation_result.label_summary,
            },
            "visualizations": {
                "overlay_base64": encode_image_to_base64(segmentation_result.overlay),
                "mask_base64": {
                    region: encode_image_to_base64(segmentation_result.mask_visualizations[region])
                    for region in TARGET_REGIONS
                },
                "saved_files": saved_files,
            },
            "structure_visualizations": {
                "overlay_base64": encode_image_to_base64(structure_result.overlay),
                "mask_base64": {
                    region: encode_image_to_base64(structure_result.mask_visualizations[region])
                    for region in structure_result.mask_visualizations
                },
                "saved_files": structure_saved_files,
            },
        }

        LOGGER.info(
            "Completed image analysis for request %s in %.3fs",
            request_id,
            time.perf_counter() - start_time,
        )
        return result

    def compare_base64(self, reference_payload: str, test_payload: str) -> dict:
        reference_result = self.analyze_base64(reference_payload)
        test_result = self.analyze_base64(test_payload)

        metric_names = ("Laplacian_Clarity", "EdgeGrad_mean", "sigma_L", "sigma_C")
        global_metrics_gap = {
            name: float(test_result["global_metrics"].get(name, 0.0) - reference_result["global_metrics"].get(name, 0.0))
            for name in metric_names
        }
        region_score_gap = {
            region: float(
                test_result["region_metrics"].get(region, {}).get("score", 0.0)
                - reference_result["region_metrics"].get(region, {}).get("score", 0.0)
            )
            for region in TARGET_REGIONS
        }

        return {
            "reference": reference_result,
            "test": test_result,
            "delta": {
                "final_score_gap": float(test_result["final_score"] - reference_result["final_score"]),
                "global_metrics_gap": global_metrics_gap,
                "region_score_gap": region_score_gap,
            },
        }
