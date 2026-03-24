# CNN IQA Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable FastAPI demo that accepts base64 images, performs SegFormer semantic segmentation, computes global and region-level IQA metrics, generates visual outputs, and returns ISP tuning suggestions.

**Architecture:** The system is split into four focused Python modules. `segmentation.py` owns model loading, label mapping, and region masks; `metrics.py` owns masked IQA computations and normalization; `analyzer.py` orchestrates decoding, scoring, visualization, and persistence; `api.py` exposes the workflow through FastAPI with structured schemas and error handling.

**Tech Stack:** Python, FastAPI, PyTorch, Hugging Face Transformers, OpenCV, NumPy, Pillow, pytest

---

## File Structure

- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/segmentation.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/metrics.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/analyzer.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/api.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/main.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/requirements.txt`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/README.md`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/.gitignore`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_bootstrap.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_metrics.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_segmentation.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_analyzer.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_api.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_readme.py`

### Task 1: Bootstrap Project Skeleton

**Files:**
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/.gitignore`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/requirements.txt`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/__init__.py`

- [ ] **Step 1: Write the failing bootstrap test**

```python
def test_project_has_expected_dependency_entries():
    required = {
        "fastapi",
        "uvicorn",
        "numpy",
        "opencv-python",
        "torch",
        "transformers",
        "pillow",
        "pytest",
    }
    contents = Path("requirements.txt").read_text().lower()
    for item in required:
        assert item in contents
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_bootstrap.py -v`
Expected: FAIL because files do not exist yet

- [ ] **Step 3: Write minimal implementation**

Create `.gitignore`, `requirements.txt`, and `tests/__init__.py` with the required dependencies and ignored paths:

```text
__pycache__/
.pytest_cache/
outputs/
*.pyc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_bootstrap.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/hedyliang/Desktop/cnn_iqa_demo add .gitignore requirements.txt tests/__init__.py tests/test_bootstrap.py
git -C /Users/hedyliang/Desktop/cnn_iqa_demo commit -m "chore: bootstrap project files"
```

### Task 2: Build Metric Primitives

**Files:**
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/metrics.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_exposure_returns_masked_v_mean():
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    assert compute_exposure(image, mask) == pytest.approx(128.0)


def test_compute_noise_uses_grayscale_std_inside_mask():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2, :2] = 100
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    assert compute_noise(image, mask) == pytest.approx(0.0)


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
    assert result["coverage_ratio"] == pytest.approx(4 / 256)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_metrics.py -v`
Expected: FAIL with import or missing function errors

- [ ] **Step 3: Write minimal implementation**

Implement:

```python
def compute_sharpness(image, mask=None): ...
def compute_noise(image, mask=None): ...
def compute_exposure(image, mask=None): ...
def normalize_sharpness(value): ...
def normalize_noise(value): ...
def normalize_exposure(value): ...
def compute_region_metrics(image, mask=None): ...
```

Return stable dicts containing:

```python
{
    "sharpness": float,
    "noise": float,
    "exposure": float,
    "score": float,
    "coverage_ratio": float,
    "valid": bool,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/hedyliang/Desktop/cnn_iqa_demo add metrics.py tests/test_metrics.py
git -C /Users/hedyliang/Desktop/cnn_iqa_demo commit -m "feat: add metric primitives"
```

### Task 3: Build Segmentation Mapping Layer

**Files:**
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/segmentation.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_map_labels_groups_ade20k_ids_into_expected_regions():
    label_map = np.array([
        [12, 3, 4, 0],
        [9, 17, 66, 1],
    ], dtype=np.int64)

    masks = build_region_masks(label_map)

    assert masks["person"][0, 0]
    assert masks["sky"][0, 1]
    assert masks["vegetation"][0, 2]
    assert masks["vegetation"][1, 0]
    assert masks["background"][0, 3]


def test_create_mask_visualizations_returns_all_target_regions():
    label_map = np.zeros((2, 2), dtype=np.int64)
    result = create_mask_visualizations(label_map)
    assert set(result.keys()) == {"person", "sky", "vegetation", "background"}


def test_background_mask_is_complement_of_foreground_masks():
    label_map = np.array([[12, 3], [4, 0]], dtype=np.int64)
    masks = build_region_masks(label_map)
    foreground = masks["person"] | masks["sky"] | masks["vegetation"]
    assert np.array_equal(masks["background"], ~foreground)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_segmentation.py -v`
Expected: FAIL with missing module or missing function errors

- [ ] **Step 3: Write minimal implementation**

Implement:

```python
SEGFORMER_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"
ADE20K_REGION_MAP = {
    "person": {12},
    "sky": {3},
    "vegetation": {4, 9, 17, 66},
}

def build_region_masks(label_map): ...
def create_overlay(image, masks): ...
def create_mask_visualizations(label_map_or_masks): ...
class SegmentationEngine:
    def __init__(self): ...
    def segment(self, image): ...
```

The tests should mock the model-facing path; do not require real weight downloads.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_segmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/hedyliang/Desktop/cnn_iqa_demo add segmentation.py tests/test_segmentation.py
git -C /Users/hedyliang/Desktop/cnn_iqa_demo commit -m "feat: add segmentation mapping layer"
```

### Task 4: Build Analyzer Orchestration

**Files:**
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/analyzer.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_analyzer.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_decode_image_accepts_data_uri_prefix():
    payload = "data:image/png;base64," + VALID_BASE64
    image = decode_base64_image(payload)
    assert image.shape == (2, 2, 3)


def test_analyze_image_returns_expected_top_level_keys(fake_engine, sample_base64):
    analyzer = ImageQualityAnalyzer(segmentation_engine=fake_engine, output_dir="outputs")
    result = analyzer.analyze_base64(sample_base64)
    assert "request_id" in result
    assert "global_metrics" in result
    assert "region_metrics" in result
    assert "final_score" in result
    assert "suggestions" in result
    assert "visualizations" in result
    assert set(result["region_metrics"].keys()) == {"person", "sky", "vegetation", "background"}
    assert set(result["visualizations"]["mask_base64"].keys()) == {"person", "sky", "vegetation", "background"}


def test_final_score_uses_spec_weights_and_excludes_background(fake_engine, sample_base64):
    analyzer = ImageQualityAnalyzer(segmentation_engine=fake_engine, output_dir="outputs")
    result = analyzer.analyze_base64(sample_base64)
    expected = (
        0.5 * result["region_metrics"]["person"]["score"]
        + 0.2 * result["region_metrics"]["sky"]["score"]
        + 0.2 * result["region_metrics"]["vegetation"]["score"]
        + 0.1 * result["global_metrics"]["score"]
    )
    assert result["final_score"] == pytest.approx(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_analyzer.py -v`
Expected: FAIL with import or missing implementation errors

- [ ] **Step 3: Write minimal implementation**

Implement:

```python
def decode_base64_image(payload): ...
def encode_image_to_base64(image): ...
def build_suggestions(global_metrics, region_metrics): ...
class ImageQualityAnalyzer:
    def __init__(self, segmentation_engine=None, output_dir="outputs"): ...
    def analyze_base64(self, payload): ...
```

The analyzer should:

- decode image
- call segmentation engine
- compute metrics for global plus all regions
- compute weighted final score
- exclude `background` from weighted final score
- save overlay and masks
- include `request_id`
- return response-ready dictionaries

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_analyzer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/hedyliang/Desktop/cnn_iqa_demo add analyzer.py tests/test_analyzer.py
git -C /Users/hedyliang/Desktop/cnn_iqa_demo commit -m "feat: add analysis pipeline"
```

### Task 5: Expose FastAPI Interface

**Files:**
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/api.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/main.py`
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_analyze_endpoint_returns_200(client, sample_base64):
    response = client.post("/analyze", json={"image": sample_base64})
    assert response.status_code == 200
    body = response.json()
    assert "request_id" in body
    assert "final_score" in body


def test_analyze_endpoint_rejects_invalid_payload(client):
    response = client.post("/analyze", json={"image": "not-base64"})
    assert response.status_code == 400


def test_analyze_endpoint_returns_500_for_unexpected_failure(client, monkeypatch):
    monkeypatch.setattr("api.analyzer.analyze_base64", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    response = client.post("/analyze", json={"image": "ZmFrZQ=="})
    assert response.status_code == 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_api.py -v`
Expected: FAIL with missing app or route errors

- [ ] **Step 3: Write minimal implementation**

Implement Pydantic schemas and FastAPI route:

```python
class AnalyzeRequest(BaseModel):
    image: str


app = FastAPI(title="CNN IQA Demo")

@app.post("/analyze")
def analyze(request: AnalyzeRequest): ...
```

`main.py` should expose `app` and allow `python main.py` to run `uvicorn`.

Include module-level logging and verify request start/end plus error paths are logged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/hedyliang/Desktop/cnn_iqa_demo add api.py main.py tests/test_api.py
git -C /Users/hedyliang/Desktop/cnn_iqa_demo commit -m "feat: add fastapi interface"
```

### Task 6: Write Documentation

**Files:**
- Create: `/Users/hedyliang/Desktop/cnn_iqa_demo/README.md`

- [ ] **Step 1: Write the failing documentation test**

```python
def test_readme_mentions_run_steps_and_api():
    content = Path("README.md").read_text()
    assert "pip install -r requirements.txt" in content
    assert "uvicorn api:app --reload" in content
    assert "POST /analyze" in content
    assert "```mermaid" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_readme.py -v`
Expected: FAIL because README does not exist yet

- [ ] **Step 3: Write minimal implementation**

Document:

- project structure
- install steps
- run steps
- sample request/response
- model choice and region mapping
- Mermaid architecture diagram
- notes about first-run model download

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_readme.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/hedyliang/Desktop/cnn_iqa_demo add README.md tests/test_readme.py
git -C /Users/hedyliang/Desktop/cnn_iqa_demo commit -m "docs: add usage and architecture readme"
```

### Task 7: End-to-End Verification

**Files:**
- Modify: `/Users/hedyliang/Desktop/cnn_iqa_demo/*`

- [ ] **Step 1: Run focused unit tests**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_metrics.py /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_segmentation.py /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_analyzer.py /Users/hedyliang/Desktop/cnn_iqa_demo/tests/test_api.py -v`
Expected: PASS

- [ ] **Step 2: Run full test suite**

Run: `pytest /Users/hedyliang/Desktop/cnn_iqa_demo/tests -v`
Expected: PASS

- [ ] **Step 3: Start the service manually**

Run: `python /Users/hedyliang/Desktop/cnn_iqa_demo/main.py`
Expected: Uvicorn startup log without import errors

- [ ] **Step 4: Perform one manual API smoke test**

Run:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d @sample-request.json
```

Before running, create `sample-request.json` containing one valid base64-encoded image payload.

Expected: JSON with `request_id`, `global_metrics`, `region_metrics`, `final_score`, `suggestions`, and `visualizations`

- [ ] **Step 5: Commit**

```bash
git -C /Users/hedyliang/Desktop/cnn_iqa_demo add .
git -C /Users/hedyliang/Desktop/cnn_iqa_demo commit -m "feat: complete cnn iqa demo"
```
