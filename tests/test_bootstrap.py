from pathlib import Path


def _read_non_comment_lines(path: str) -> set[str]:
    return {
        line.strip().lower()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


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
    entries = _read_non_comment_lines("requirements.txt")
    for item in required:
        assert item in entries


def test_project_has_expected_gitignore_entries():
    required = {
        "__pycache__/",
        ".pytest_cache/",
        "outputs/",
        "*.pyc",
    }
    entries = _read_non_comment_lines(".gitignore")
    for item in required:
        assert item in entries


def test_tests_package_marker_exists():
    assert Path("tests/__init__.py").exists()
