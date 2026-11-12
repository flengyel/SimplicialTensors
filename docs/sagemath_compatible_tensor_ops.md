# SageMath-compatible symbolic operations

[experiments/sagemath_compatible_tensor_ops.py](../experiments/sagemath_compatible_tensor_ops.py) runs the example in [simplicial_tensors.sagemath_compatible_tensor_ops](../src/simplicial_tensors/sagemath_compatible_tensor_ops.py). The package uses SageMath when `sage.all` imports successfully and SymPy otherwise. `HAVE_SAGE` records the backend.

## Interface

`SymbolicTensor(shape, tensor=None, init_type="range")` stores a NumPy object array. Initialization modes are `"range"`, `"zeros"`, and `"ones"`. Range entries have names such as `x_0_1`. Methods include addition, subtraction, `face`, `degen`, `bdry`, `horn`, `filler`, `is_degen`, `simplify`, and `subs`.

Faces, degeneracies, boundaries, horns, and fillers call the core tensor implementation. A face deletes the same active index on every axis. A degeneracy duplicates that index on every axis, preserving native Sage rational coefficients exactly, including in vectors. The boundary is the alternating sum of faces. A horn is a list of faces with a zero tensor in the omitted position.

`is_degen` tests membership in the sum of the degeneracy images through the Moore projection. Its meaning agrees with the [numeric and SymPy interfaces](is_degen.md), including zero in degree zero.

`correction_rank(original, filler)` forms the coefficient matrix of the entry differences over the rationals, with one column per monomial, including the constant monomial. Its rank counts linearly independent polynomial expressions. For homogeneous linear entries in independent input symbols, this is the rank of the represented correction map. For example, differences `x`, `2*x`, `y`, and `3*y` have rank two. Nonpolynomial entries or coefficients outside the rationals, including irrational constant entries, raise `ValueError`.

`n_hypergroupoid_comparison` validates the selected Moore fillers and applies the full-module horn-kernel criterion to decide uniqueness. It raises `SimplicialException` for a degenerate boundary unless `allow_degen=True`. With no selected horns it returns `True` vacuously. The helper `test_symbolic_n_hypergroupoid` includes outer horns.

## Running the example

From an installed checkout, run

```sh
python experiments/sagemath_compatible_tensor_ops.py
```

For SageMath, install the package dependencies in its Python environment and run

```sh
sage -python experiments/sagemath_compatible_tensor_ops.py
```

The fixed `(3, 3)` example prints horn comparisons, the original tensor, a filler, and missing-symbol checks. With SageMath or the SymPy fallback, the filler reproduces the included faces and `correction_rank` is two. The script writes no files.

Run the native Sage backend tests with

```sh
sage -python -m pytest -o addopts= -q tests/test_sagemath_compatible_tensor_ops.py
```

These tests require `HAVE_SAGE=True` and native Sage symbolic entries. They check the mixed identities $d_i s_j=s_{j-1}d_i$ for $i<j$, $d_j s_j=d_{j+1}s_j=1$, and $d_i s_j=s_jd_{i-1}$ for $i>j+1$. Native symbols and rational coefficients also test correction rank, degeneracy-span membership, horn uniqueness, and exact vector and rectangular tensor operations above $2^{53}$. Backend comparisons convert rational numerators and denominators exactly.

## CoCalc notebook adapter

The retained [CoCalc module](../src/simplicial_tensors/notebooks/cocalc_compatible_tensor_ops.py) imports the shared implementation. Its `SymbolicTensor` adapter accepts the legacy `tensor_data` keyword, retains `order()` and two-dimensional `to_latex()`, and preserves the adapter type under simplicial operations. `decompose_degen()` returns a normalized tensor $u$ and pairs $(a_i,i)$ satisfying $T=u+\sum_i s_i a_i$. This is an additive decomposition in the degeneracy span.

The [CoCalc notebook](../src/simplicial_tensors/notebooks/cocalc_compatible_tensor_ops.ipynb) imports this adapter and checks a generic horn, the correction rank, a zero tensor, and a sum of degeneracies. Its shape-comparison helper accepts additional shapes explicitly. All default examples are small and use the SymPy fallback when SageMath is absent. Run the module with `python -m simplicial_tensors.notebooks.cocalc_compatible_tensor_ops`.
