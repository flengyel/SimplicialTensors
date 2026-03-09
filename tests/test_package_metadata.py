"""Smoke tests for the SimplicialTensors package."""

import importlib


def test_package_importable() -> None:
    pkg = importlib.import_module("simplicial_tensors")
    assert set(pkg.__all__) == {"tensor_ops", "symbolic_tensor_ops"}

def test_sagemath_module_importable() -> None:
    mod = importlib.import_module("simplicial_tensors.sagemath_compatible_tensor_ops")
    assert hasattr(mod, "SymbolicTensor")