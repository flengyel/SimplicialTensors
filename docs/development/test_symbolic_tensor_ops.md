# Symbolic tensor tests

[tests/test_symbolic_tensor_ops.py](../../tests/test_symbolic_tensor_ops.py) tests [simplicial_tensors.symbolic_tensor_ops](../../src/simplicial_tensors/symbolic_tensor_ops.py) with NumPy object arrays and SymPy expressions. Pytest skips the module if either dependency cannot be imported.

The tests cover entry names, face and degeneracy shapes, boundary construction, agreement between included horn faces and a filler, the five families of simplicial identities, and boundary squared equal to zero. Two tests check comparison control flow. A `(2, 2)` tensor has no inner horns, so the default comparison returns `True` vacuously. A monkeypatch changes an earlier inner-horn filler while preserving a later one and checks that the discrepancy is reported.

[tests/test_tensor_coefficients.py](../../tests/test_tensor_coefficients.py) checks additive degeneracy membership in both symbolic backends, zero in degree zero, and a zero tensor whose selected filler is itself despite a nonzero horn kernel. It also checks correction rank on linearly dependent expressions. Differences `x`, `2*x`, `y`, and `3*y` have coefficient rank two.

Run the SymPy tests and coefficient regressions with

```sh
python -m pytest -q tests/test_symbolic_tensor_ops.py tests/test_tensor_coefficients.py
```

SageMath-specific tests are in [tests/test_sagemath_compatible_tensor_ops.py](../../tests/test_sagemath_compatible_tensor_ops.py) and require SageMath. The coefficient regressions exercise the compatible module's selected backend, including its SymPy fallback.

```sh
python -m pytest -q
```

The last command runs the repository suite.
