from pathlib import Path


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
