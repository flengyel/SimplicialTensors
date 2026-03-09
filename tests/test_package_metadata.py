"""Smoke tests for package metadata and documentation coverage."""

import importlib
import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_package_importable() -> None:
    pkg = importlib.import_module("simplicial_tensors")
    assert set(pkg.__all__) == {"tensor_ops", "symbolic_tensor_ops"}


def test_sagemath_module_importable() -> None:
    mod = importlib.import_module("simplicial_tensors.sagemath_compatible_tensor_ops")
    assert hasattr(mod, "SymbolicTensor")


def test_readme_reflects_no_examples_directory() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Operations-Tensorielles-Simpliciales" not in readme
    assert "examples/" not in readme
    assert "python examples/" not in readme


def _git_ls_files(prefix: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("git is required to validate documentation coverage")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _tracked_experiment_files_by_ext(*extensions: str) -> set[str]:
    tracked = _git_ls_files("experiments")
    if not extensions:
        return set(tracked)
    return {p for p in tracked if any(p.endswith(ext) for ext in extensions)}


def _catalog_documented_experiment_files() -> set[str]:
    catalog_path = REPO_ROOT / "docs" / "experiments_catalog.md"
    assert catalog_path.is_file(), "Missing docs/experiments_catalog.md"
    text = catalog_path.read_text(encoding="utf-8")
    return set(re.findall(r"^- `([^`]+)`", text, flags=re.MULTILINE))


def _doc_basenames() -> set[str]:
    return {p.stem.lower() for p in (REPO_ROOT / "docs").glob("*.md")}


def test_experiments_catalog_covers_tracked_files() -> None:
    tracked = _tracked_experiment_files_by_ext()
    documented = _catalog_documented_experiment_files()

    missing = sorted(tracked - documented)
    extra = sorted(documented - tracked)

    assert not missing, f"Experiments missing from docs/experiments_catalog.md: {missing}"
    assert not extra, f"Catalog entries not present in experiments/: {extra}"


def test_each_experiment_script_has_dedicated_doc_page() -> None:
    scripts = _tracked_experiment_files_by_ext(".py", ".sage")
    expected_doc_stems = {pathlib.Path(script).stem.lower() for script in scripts}
    docs = _doc_basenames()

    missing_docs = sorted(stem for stem in expected_doc_stems if stem not in docs)
    assert not missing_docs, f"Missing docs/*.md for experiment scripts: {missing_docs}"
