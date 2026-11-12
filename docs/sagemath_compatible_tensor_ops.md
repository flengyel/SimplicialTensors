# SageMath-compatible symbolic operations

[experiments/sagemath_compatible_tensor_ops.py](../experiments/sagemath_compatible_tensor_ops.py) calls the demonstration in [simplicial_tensors.sagemath_compatible_tensor_ops](../src/simplicial_tensors/sagemath_compatible_tensor_ops.py). The package uses SageMath when importing `sage.all` succeeds and falls back to SymPy otherwise. `HAVE_SAGE` records the selected backend.

## Interface

`SymbolicTensor(shape, tensor=None, init_type="range")` stores a NumPy object array. Initialization modes are "range", "zeros", and "ones". Range entries have names such as x_0_1. Methods include addition, subtraction, face, degen, bdry, horn, filler, `is_degen`, simplify, and subs.

A face deletes the same active index on every axis. A degeneracy duplicates it on every axis. The boundary is the alternating sum of faces. The horn method returns all faces with a zero tensor in the omitted position. The filler algorithm corrects the included faces in sequence.

The `is_degen` method tests equality with a single expression of the form `s_i d_i T`. This differs from membership in the span of all degeneracy images. The `correction_rank` function counts distinct string representations of simplified nonzero entry differences. It does not compute a matrix rank or prove independence of corrections.

The `n_hypergroupoid_comparison` method compares the input with its selected filler for each tested horn. It raises `SimplicialException` for a degenerate boundary unless `allow_degen=True`. It tests only inner horns by default. With no inner horns, its loop returns `True` without testing a horn. The helper `test_symbolic_n_hypergroupoid` explicitly includes outer horns.

## Running the example

From an installed checkout:

~~~sh
python experiments/sagemath_compatible_tensor_ops.py
~~~

For SageMath, make the package and NumPy available to its Python environment, then run:

~~~sh
sage -python experiments/sagemath_compatible_tensor_ops.py
~~~

The fixed (3, 3) example prints the horn comparisons, original tensor, filler, and missing-symbol checks. With the SymPy fallback, the filler reproduces the included faces and `correction_rank` is 2. The script writes no files. SageMath backend behavior requires a separate run under SageMath.
