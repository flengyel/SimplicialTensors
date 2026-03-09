"""Smoke tests for package metadata, examples, and documentation coverage."""

import importlib
import pathlib
import re
import runpy
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_package_importable() -> None:
    pkg = importlib.import_module("simplicial_tensors")
    assert set(pkg.__all__) == {"tensor_ops", "symbolic_tensor_ops"}


def test_sagemath_module_importable() -> None:
    mod = importlib.import_module("simplicial_tensors.sagemath_compatible_tensor_ops")
    assert hasattr(mod, "SymbolicTensor")


def test_readme_example_commands_point_to_existing_files() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Operations-Tensorielles-Simpliciales" not in readme
    scripts = re.findall(r"python (examples/[A-Za-z0-9_.-]+\.py)", readme)
    assert scripts, "README should include at least one runnable examples/*.py command."
    for rel_script in scripts:
        assert (REPO_ROOT / rel_script).is_file(), f"README points to missing script: {rel_script}"


def test_ab_example_main_runs() -> None:
    namespace = runpy.run_path(str(REPO_ROOT / "examples" / "ab_mlp_demo.py"))
    namespace["main"]()


def _git_tracked_experiment_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "experiments"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("git is required to validate experiments catalog coverage")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _catalog_documented_experiment_files() -> set[str]:
    catalog_path = REPO_ROOT / "docs" / "experiments_catalog.md"
    assert catalog_path.is_file(), "Missing docs/experiments_catalog.md"
    text = catalog_path.read_text(encoding="utf-8")
    return set(re.findall(r"^- `([^`]+)`", text, flags=re.MULTILINE))


def test_experiments_catalog_covers_tracked_files() -> None:
    tracked = _git_tracked_experiment_files()
    documented = _catalog_documented_experiment_files()

    missing = sorted(tracked - documented)
    extra = sorted(documented - tracked)

    assert not missing, f"Experiments missing from docs/experiments_catalog.md: {missing}"
    assert not extra, f"Catalog entries not present in experiments/: {extra}"
