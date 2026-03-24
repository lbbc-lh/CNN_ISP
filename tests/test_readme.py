from pathlib import Path


def test_readme_mentions_run_steps_and_api():
    content = Path("README.md").read_text()
    assert "pip install -r requirements.txt" in content
    assert "uvicorn api:app --reload" in content
    assert "POST /analyze" in content
    assert "```mermaid" in content
