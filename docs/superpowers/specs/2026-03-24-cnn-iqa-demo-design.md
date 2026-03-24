# CNN-Enhanced Image Quality Assessment System Design

## Summary

Build a runnable Python demo for ISP debugging that accepts a base64 image, performs semantic segmentation with a pretrained SegFormer model, computes image quality metrics globally and per semantic region, produces a weighted final score, generates rule-based tuning suggestions, and exposes the workflow through a FastAPI endpoint.

The project will be a new standalone repository at `~/Desktop/cnn_iqa_demo/` and will prioritize clear module boundaries, reproducible local execution, and practical visual debugging outputs over platform-heavy architecture.

The chosen segmentation checkpoint is `nvidia/segformer-b2-finetuned-ade-512-512`.

## Goals

- Accept a base64-encoded image payload through an API.
- Decode the image into a NumPy/OpenCV representation.
- Run pretrained semantic segmentation and map classes into:
  - `person`
  - `sky`
  - `vegetation`
  - `background`
- Compute image quality metrics:
  - sharpness via Laplacian variance
  - noise via masked grayscale standard deviation
  - exposure via HSV V-channel mean
- Return both global and region-level metrics.
- Compute a weighted final score using:
  - `person`: `0.5`
  - `sky`: `0.2`
  - `vegetation`: `0.2`
  - `global`: `0.1`
- Generate rule-based ISP tuning suggestions.
- Produce required visualizations:
  - segmentation overlay
  - one binary mask image per region
- Save visualizations to disk and also return them as base64 in the API response.

## Non-Goals

- No model training or fine-tuning.
- No frontend UI beyond FastAPI service.
- No persistent database or user management.
- No GPU-specific optimization work beyond using PyTorch if available.
- No advanced subjective IQA model; this demo remains rules-plus-classical-metrics based.

## Project Structure

```text
cnn_iqa_demo/
├── analyzer.py
├── api.py
├── main.py
├── metrics.py
├── segmentation.py
├── requirements.txt
├── README.md
├── outputs/
├── tests/
│   ├── test_analyzer.py
│   ├── test_api.py
│   ├── test_metrics.py
│   └── test_segmentation.py
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-03-24-cnn-iqa-demo-design.md
        └── plans/
```

## Architecture

### 1. `segmentation.py`

Responsibilities:

- Lazily load Hugging Face SegFormer processor and model.
- Convert OpenCV BGR input into RGB/PIL-compatible model input.
- Run semantic segmentation inference.
- Upsample logits back to image resolution.
- Map model labels into the four target classes.
- Return:
  - semantic class map
  - boolean masks per target region
  - overlay image
  - binary mask visualizations

Mapping strategy:

- `person`: map from person/human labels.
- `sky`: map from sky label.
- `vegetation`: map from tree/grass/plant/flower-related labels that exist in the chosen SegFormer label set.
- `background`: everything not claimed by the three target foreground classes.

Exact label mapping for `nvidia/segformer-b2-finetuned-ade-512-512`:

- `person`: ADE20K label `12` (`person`)
- `sky`: ADE20K label `3` (`sky`)
- `vegetation`: ADE20K labels
  - `4` (`tree`)
  - `9` (`grass`)
  - `17` (`plant`)
  - `66` (`flower`)
- `background`: all remaining label IDs

`background` is the exact complement of the union of the `person`, `sky`, and `vegetation` masks after label mapping.

The implementation should avoid hard-coding model internals outside one mapping table so model replacement remains localized.

### 2. `metrics.py`

Responsibilities:

- Provide reusable metric functions for full-image and masked-region analysis.
- Validate masks and handle tiny/empty regions gracefully.
- Normalize raw metrics into bounded quality sub-scores for score fusion.

Functions:

- `compute_sharpness(image, mask=None) -> float`
- `compute_noise(image, mask=None) -> float`
- `compute_exposure(image, mask=None) -> float`
- `compute_region_metrics(image, mask=None) -> dict`
- helper normalization functions for score fusion

Metric definitions:

- Sharpness: variance of Laplacian over grayscale pixels inside the mask.
- Noise: standard deviation of grayscale values inside the mask.
- Exposure: mean of HSV `V` values inside the mask.

Edge cases:

- Empty mask: return zeros plus a `valid: false` marker.
- Very small mask: treat the region as invalid if either condition is true:
  - masked pixel count `< 64`
  - coverage ratio `< 0.005`

When invalid:

- return `sharpness = 0`
- return `noise = 0`
- return `exposure = 0`
- return `score = 0`
- return `valid = false`
- keep `coverage_ratio` in the response schema at all times
- set `coverage_ratio = 0.0` only for empty masks
- if the region has pixels but remains invalid due to minimum-size thresholds, preserve the computed `coverage_ratio`

### 3. `analyzer.py`

Responsibilities:

- Decode request payloads.
- Call segmentation and metric modules.
- Aggregate metrics and scores.
- Build suggestions.
- Save outputs under `outputs/`.
- Return a single structured response object for the API layer.

Primary workflow:

1. Decode base64 image.
2. Generate segmentation masks and visualizations.
3. Compute global metrics.
4. Compute region metrics for `person`, `sky`, `vegetation`, `background`.
5. Normalize metrics into per-scope quality scores.
6. Compute weighted final score using requested weights.
7. Build suggestions from region and global thresholds.
8. Save overlay and mask files with timestamp/UUID-based filenames.
9. Encode visual outputs to base64 for API response.

### 4. `api.py`

Responsibilities:

- Define FastAPI app and schemas.
- Validate request payload.
- Call analyzer service.
- Return JSON response.
- Log request lifecycle and error states.

Endpoint:

- `POST /analyze`

Request:

```json
{
  "image": "base64_string"
}
```

Response shape:

```json
{
  "request_id": "20260324_abcd1234",
  "global_metrics": {
    "sharpness": 0.0,
    "noise": 0.0,
    "exposure": 0.0,
    "score": 0.0
  },
  "region_metrics": {
    "person": {
      "sharpness": 0.0,
      "noise": 0.0,
      "exposure": 0.0,
      "score": 0.0,
      "coverage_ratio": 0.0,
      "valid": true
    },
    "sky": {
      "sharpness": 0.0,
      "noise": 0.0,
      "exposure": 0.0,
      "score": 0.0,
      "coverage_ratio": 0.0,
      "valid": true
    },
    "vegetation": {
      "sharpness": 0.0,
      "noise": 0.0,
      "exposure": 0.0,
      "score": 0.0,
      "coverage_ratio": 0.0,
      "valid": true
    },
    "background": {
      "sharpness": 0.0,
      "noise": 0.0,
      "exposure": 0.0,
      "score": 0.0,
      "coverage_ratio": 0.0,
      "valid": true
    }
  },
  "final_score": 0.0,
  "suggestions": [
    "increase sharpening"
  ],
  "visualizations": {
    "overlay_base64": "....",
    "mask_base64": {
      "person": "....",
      "sky": "....",
      "vegetation": "....",
      "background": "...."
    },
    "saved_files": {
      "overlay": "outputs/20260324_xxx_overlay.png",
      "person": "outputs/20260324_xxx_person_mask.png",
      "sky": "outputs/20260324_xxx_sky_mask.png",
      "vegetation": "outputs/20260324_xxx_vegetation_mask.png",
      "background": "outputs/20260324_xxx_background_mask.png"
    }
  }
}
```

### 5. `main.py`

Responsibilities:

- Provide a simple local startup entrypoint.
- Import the FastAPI app from `api.py`.
- Support `python main.py` and document `uvicorn api:app --reload`.

## Scoring Design

Each scope gets raw metrics plus a derived score. The derived score should be normalized to a stable 0-100 range so weighted fusion is interpretable.

Suggested normalization:

- sharpness score:
  - `sharpness_score = clip((laplacian_var / 300.0) * 100, 0, 100)`
- noise score:
  - `noise_score = clip(100 - (gray_std / 40.0) * 100, 0, 100)`
- exposure score:
  - compute `v_mean` on `0-255`
  - `exposure_score = clip(100 - (abs(v_mean - 128.0) / 128.0) * 100, 0, 100)`

Per-scope score:

```text
scope_score = 0.4 * sharpness_score
            + 0.3 * noise_score
            + 0.3 * exposure_score
```

Final score:

```text
final_score = 0.5 * person_score
            + 0.2 * sky_score
            + 0.2 * vegetation_score
            + 0.1 * global_score
```

`background` is intentionally excluded from weighted final score fusion. It is reported for diagnostics only.

If a region is absent or invalid, the system should:

- still include the region in output,
- mark it as low-confidence or invalid,
- set `valid: false`
- set raw metrics and score to `0`
- still keep the response schema stable

This keeps the result deterministic for a demo API.

## Suggestion Rules

Rule-based suggestions are derived from region and global metrics. Suggestions should be deduplicated and kept concise.

Trigger thresholds:

- low sharpness: `sharpness < 80`
- high noise: `noise > 18`
- underexposure: `exposure < 90`
- overexposure: `exposure > 185`

Priority rules:

- evaluate `person`, then `global`, then `sky`, then `vegetation`
- if the same suggestion is triggered by multiple scopes, keep one copy
- cap the output at 6 suggestions
- ignore invalid regions when generating suggestions

Examples:

- low sharpness:
  - `increase sharpening`
- high noise:
  - `apply denoising`
- low exposure:
  - `increase exposure`
- high exposure:
  - `reduce exposure or highlight clipping`

Region-aware examples:

- person sharpness low:
  - `increase face/person detail enhancement`
- sky noise high:
  - `reduce chroma/luma noise in flat regions`
- vegetation oversharpened/noisy:
  - `reduce sharpening halos in textured regions`

The demo should remain rule-based rather than trying to synthesize complex natural-language diagnostics.

## Visualization Design

Required outputs:

- one segmentation overlay image
- one binary mask image per target region

Overlay behavior:

- color each semantic region distinctly
- blend overlay with original image for context
- include a simple legend only if it can be done without introducing heavy plotting dependencies

Storage behavior:

- create `outputs/` automatically if missing
- write all generated images per request with a shared request ID
- return relative saved paths in the API response

## Logging

Use Python `logging` with module-level loggers.

Log:

- model loading
- request start/end
- image decode failures
- segmentation inference timing
- output save paths
- analyzer warnings for missing/empty masks

Avoid logging raw base64 payloads.

## Error Handling

Expected failures:

- invalid base64 payload
- unsupported/empty image decode
- model loading failure
- runtime inference failure

Base64 input handling:

- accept both raw base64 strings and `data:image/...;base64,...` prefixed payloads
- strip the optional data URI prefix before decode

API behavior:

- return HTTP 400 for invalid input image payloads
- return HTTP 500 for internal processing failures
- include concise error messages without stack traces in the API response

## Testing Strategy

The implementation should be developed test-first for units that do not depend on live model downloads.

Test categories:

- metrics unit tests using synthetic arrays and masks
- analyzer tests with a stubbed segmentation engine
- API tests using FastAPI `TestClient`
- segmentation mapping tests that mock model outputs and validate region aggregation

The plan should avoid requiring real model downloads in the main test suite. Integration with a real SegFormer model can remain manual validation documented in the README.

## Dependencies

Expected runtime dependencies:

- `fastapi`
- `uvicorn`
- `numpy`
- `opencv-python`
- `torch`
- `torchvision` (optional helper dependency if used in transforms/utilities)
- `transformers`
- `Pillow`
- `python-multipart` if needed by future extensions
- `pytest`

## Open Questions Resolved

- Project location: new standalone project on desktop.
- Segmentation model: SegFormer.
- Visualization delivery: return base64 and also save to `outputs/`.

## Implementation Readiness

This design is ready to move into a concrete implementation plan. The plan should lock down:
- test-first task ordering
- exact file creation order
- how tests stub segmentation inference to avoid network/model dependency
- when manual smoke validation happens relative to automated tests
