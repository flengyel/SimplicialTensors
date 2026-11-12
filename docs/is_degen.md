# Membership in the degenerate submodule

[`tensor_ops.py`](../src/simplicial_tensors/tensor_ops.py) defines `is_degen(a)` as `decompose_degen(a) is not None`. For degree `n = min(a.shape) - 1 > 0`, it tests membership in the sum of the images of `s_0, ..., s_(n-1)`.

`decompose_degen` solves a system whose columns are flattened degeneracies of coordinate tensors. A successful return is a list of `n` arrays `b_i` with every axis shorter by one, satisfying `a = sum(degen(b_i, i) for i in range(n))`. Free parameters are set to zero. The decomposition need not be unique, and membership in this sum does not imply membership in one image.

```python
import numpy as np
from simplicial_tensors.tensor_ops import degen, is_degen

a = np.array([[0, 1], [0, 0]])
assert not is_degen(a)
assert is_degen(degen(a, 0))
```

The SymPy routine converts input entries with `Rational`, but builds its system initially with floating-point zeros and ones. `decompose_degen_numpy` uses `numpy.linalg.lstsq` and accepts reconstruction with `numpy.allclose`. Neither interface specifies a finite coefficient field.

For degree zero or below, `decompose_degen` returns `None`. Thus `is_degen` returns `False` even on the degree-zero zero array. `is_generator_numeric` treats zero separately. Unconvertible entries can also return `None`. `SymbolicTensor` has a separate degeneracy implementation.

See [coordinate-tensor counts](count_standard_basis_tensors.md) for small examples.
