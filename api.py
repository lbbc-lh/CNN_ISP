from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from analyzer import ImageQualityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="CNN IQA Demo", version="0.1.0")
analyzer = ImageQualityAnalyzer()
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class AnalyzeRequest(BaseModel):
    image: str


class CompareRequest(BaseModel):
    reference_image: str
    test_image: str


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    LOGGER.info("Received /analyze request")
    try:
        return analyzer.analyze_base64(request.image)
    except ValueError as exc:
        LOGGER.warning("Bad /analyze request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Unexpected /analyze failure")
        raise HTTPException(status_code=500, detail="internal analysis error") from exc


@app.post("/compare")
def compare(request: CompareRequest) -> dict:
    LOGGER.info("Received /compare request")
    try:
        return analyzer.compare_base64(request.reference_image, request.test_image)
    except ValueError as exc:
        LOGGER.warning("Bad /compare request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Unexpected /compare failure")
        raise HTTPException(status_code=500, detail="internal compare error") from exc
