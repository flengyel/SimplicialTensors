# Membership in the degenerate submodule

[`tensor_ops.py`](../src/simplicial_tensors/tensor_ops.py) defines `is_degen(a)` as membership in the sum of the degeneracy images. For a tensor of degree `n = min(a.shape) - 1`, these are the images of `s_0, ..., s_(n-1)`.

`split_normalized(a)` returns a pair `(u, terms)` satisfying

$$
a=u+\sum_{i=0}^{n-1}s_i(\mathrm{terms}[i]),
\qquad d_i u=0\quad(1\leq i\leq n).
$$

The routine applies `1 - s_(i-1) d_i` in decreasing order of `i`. The tensor belongs to the degenerate submodule exactly when `u` is zero. This gives the decomposition without constructing a dense linear system.

`decompose_degen(a)` returns the `n` lower-degree tensors in `terms` if `u` is zero, and `None` otherwise. The decomposition need not be unique. Membership in the sum does not imply membership in one image.

```python
import numpy as np
from simplicial_tensors.tensor_ops import degen, is_degen

a = np.array([[0, 1], [0, 0]])
assert not is_degen(a)
assert is_degen(degen(a, 0))

b = np.array([[1, 1, 0], [1, 2, 1], [0, 1, 1]])
assert is_degen(b)
```

The matrix `b` is a sum of two degeneracies and belongs to neither individual degeneracy image.

## Coefficients and degree zero

`decompose_degen` converts integer and rational entries to exact SymPy rationals. Finite floating-point entries are interpreted as their exact binary rational values. Boolean arrays use the field with two elements. Unsupported entries return `None`. The symbolic tensor classes use the same Moore projection, with symbolic simplification for the zero test.

In degree zero the degenerate submodule is zero. The zero tensor has the empty decomposition `[]`, and a nonzero tensor has no decomposition. Empty arrays and scalars are outside the domain. `is_generator_numeric` excludes zero in every degree.

`split_normalized` uses the arithmetic of its input array. Integer object arrays avoid overflow. The optional `decompose_degen_numpy` routine instead solves a floating-point least-squares system and accepts reconstruction with `numpy.allclose`.

See [coordinate-tensor counts](count_standard_basis_tensors.md) for small examples.
