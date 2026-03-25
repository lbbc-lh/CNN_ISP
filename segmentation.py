from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

LOGGER = logging.getLogger(__name__)

SEGFORMER_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"
TARGET_REGIONS = ("person", "sky", "vegetation", "background")
ADE20K_REGION_MAP = {
    "person": {12},
    "sky": {2},
    "vegetation": {4, 9, 17, 66},
}
OVERLAY_COLORS = {
    "person": (60, 20, 220),
    "sky": (235, 206, 135),
    "vegetation": (34, 139, 34),
    "background": (128, 128, 128),
}


@dataclass
class SegmentationResult:
    label_map: np.ndarray
    masks: dict[str, np.ndarray]
    overlay: np.ndarray
    mask_visualizations: dict[str, np.ndarray]
    label_summary: list[dict[str, int | float | str]]


def summarize_label_map(
    label_map: np.ndarray,
    limit: int = 10,
    id_to_label: dict[int, str] | None = None,
) -> list[dict[str, int | float | str]]:
    label_map = np.asarray(label_map, dtype=np.int64)
    total_pixels = int(label_map.size)
    if total_pixels == 0:
        return []

    unique_ids, counts = np.unique(label_map, return_counts=True)
    ranked = sorted(zip(unique_ids.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)
    summary: list[dict[str, int | float | str]] = []
    for label_id, count in ranked[:limit]:
        summary.append(
            {
                "label_id": int(label_id),
                "label_name": (id_to_label or {}).get(int(label_id), f"class_{int(label_id)}"),
                "pixel_count": int(count),
                "coverage_ratio": float(count / total_pixels),
            }
        )
    return summary


def build_region_masks(label_map: np.ndarray) -> dict[str, np.ndarray]:
    label_map = np.asarray(label_map, dtype=np.int64)
    if label_map.ndim != 2:
        raise ValueError("label_map must be 2-D")

    masks: dict[str, np.ndarray] = {}
    foreground = np.zeros(label_map.shape, dtype=bool)

    for region, label_ids in ADE20K_REGION_MAP.items():
        mask = np.isin(label_map, list(label_ids))
        masks[region] = mask
        foreground |= mask

    masks["background"] = ~foreground
    return masks


def create_mask_visualizations(label_map_or_masks: np.ndarray | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(label_map_or_masks, dict):
        masks = label_map_or_masks
    else:
        masks = build_region_masks(label_map_or_masks)

    visuals: dict[str, np.ndarray] = {}
    for region in TARGET_REGIONS:
        mask = masks[region].astype(np.uint8) * 255
        visuals[region] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return visuals


def create_overlay(image: np.ndarray, masks: dict[str, np.ndarray], alpha: float = 0.45) -> np.ndarray:
    overlay_layer = np.zeros_like(image, dtype=np.uint8)
    for region, color in OVERLAY_COLORS.items():
        overlay_layer[masks[region]] = color
    return cv2.addWeighted(image, 1.0 - alpha, overlay_layer, alpha, 0.0)


class SegmentationEngine:
    def __init__(
        self,
        checkpoint: str = SEGFORMER_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: Any | None = None
        self._model: Any | None = None
        self._id_to_label: dict[int, str] | None = None

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        from transformers import AutoImageProcessor
        from transformers import SegformerForSemanticSegmentation

        LOGGER.info("Loading segmentation model from %s", self.checkpoint)
        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self._model = SegformerForSemanticSegmentation.from_pretrained(self.checkpoint)
        self._model.to(self.device)
        self._model.eval()
        self._id_to_label = {int(key): value for key, value in self._model.config.id2label.items()}

    def predict_label_map(self, image: np.ndarray) -> np.ndarray:
        self._load()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        upsampled = F.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)

    def segment(self, image: np.ndarray) -> SegmentationResult:
        label_map = self.predict_label_map(image)
        masks = build_region_masks(label_map)
        overlay = create_overlay(image, masks)
        mask_visualizations = create_mask_visualizations(masks)
        return SegmentationResult(
            label_map=label_map,
            masks=masks,
            overlay=overlay,
            mask_visualizations=mask_visualizations,
            label_summary=summarize_label_map(label_map, id_to_label=self._id_to_label),
        )
