# Adjoint boundary implementation

The adjoint boundary API is implemented in
[adjoint_ops.py](../../src/simplicial_tensors/adjoint_ops.py).
Its tests are in [test_adjoint_ops.py](../../tests/test_adjoint_ops.py).
This page records the implemented interface.

For a diagonal face that deletes index $i$ on every axis, the
Frobenius adjoint inserts the smaller array into the coordinates
avoiding $i$ and sets the other entries to zero. The boundary
adjoint sums these insertions with alternating signs.

## API

- `face_adjoint(y, original_shape, i)` returns the adjoint face image.
- `bdry_adjoint(y, original_shape)` returns the adjoint boundary image.
- `boundary_homeostasis_gradient(w, target_boundary=None)` returns
  the gradient of half the squared boundary residual norm.
- `boundary_homeostasis_feedback(w, target_boundary=None, alpha=1e-3, eps=1e-12)`
  returns a negative-gradient perturbation with norm scaled relative
  to `w`.

The target defaults to zero. The adjoint functions require
`y.shape == tuple(s - 1 for s in original_shape)`.
They reject empty shapes, nonpositive dimensions, and face indices
outside `0 <= i < min(original_shape)`.

The feedback perturbation is an increment to add to a weight array.
Its normalization means it is a scaled gradient direction. The
chosen scale controls whether a finite update reduces the energy.

## Verification

```sh
python -m pip install -e .
python -m pip install pytest
python -m pytest tests/test_adjoint_ops.py -q
```

The tests compare the face and boundary adjoints with Frobenius inner
products for several rectangular and higher-axis shapes. They also
compare the spectral operators with dense matrix calculations and
test a small energy-decreasing gradient step.

See [boundary_homeostasis.md](../api/boundary_homeostasis.md) for the formulas
and examples, and [AGENTS.md](../../AGENTS.md) for the distinction between
energy regularization and controller-driven boundary synthesis.
