"""Smoke tests for package metadata and examples."""

import importlib
import pathlib
import re
import runpy

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
