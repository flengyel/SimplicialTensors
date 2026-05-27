# Codex task: implement adjoint boundary feedback

Read `AGENTS.md` first. Implement only the adjoint boundary foundation. Do not modify neural-network training experiments in this task.

## Goal

Add a principled return map from boundary tensors back to original weight tensors using the adjoint of the diagonal boundary operator.

For a tensor `W` with shape `original_shape`, the diagonal face map `d_i` deletes index `i` along every axis. Its adjoint `d_i^*` must insert a smaller tensor back into the positions avoiding index `i`, filling the deleted hyperplanes with zeros.

The adjoint boundary is:

```python
bdry_adjoint(Y, original_shape) = sum_i (-1)**i * face_adjoint(Y, original_shape, i)
```

where `i` ranges over `0 <= i < min(original_shape)`.

## Files to create or modify

Create:

```text
src/simplicial_tensors/adjoint_ops.py
tests/test_adjoint_ops.py
docs/boundary_homeostasis.md
```

Do not duplicate existing boundary or face implementations. Import `bdry` and `face` from `simplicial_tensors.tensor_ops`.

## Required API

Implement:

```python
def face_adjoint(y: np.ndarray, original_shape: tuple[int, ...], i: int) -> np.ndarray:
    """Adjoint of the diagonal face map d_i."""

def bdry_adjoint(y: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Adjoint of the diagonal boundary map."""

def boundary_homeostasis_gradient(
    W: np.ndarray,
    target_boundary: np.ndarray | None = None,
) -> np.ndarray:
    """Return bdry_adjoint(bdry(W) - target_boundary, W.shape)."""

def boundary_homeostasis_feedback(
    W: np.ndarray,
    target_boundary: np.ndarray | None = None,
    alpha: float = 1e-3,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return a same-shape negative-feedback perturbation scaled to alpha * ||W||_F."""
```

Use Frobenius norms.

## Shape validation

- `face_adjoint(y, original_shape, i)` must require `y.shape == tuple(s - 1 for s in original_shape)`.
- Reject empty shapes.
- Reject any dimension of `original_shape` less than 1.
- Reject `i < 0` or `i >= min(original_shape)`.
- `bdry_adjoint(y, original_shape)` must use the same shape validation.

## Tests

Add tests proving the adjoint identities numerically:

```python
<face(W, i), Y> == <W, face_adjoint(Y, W.shape, i)>
<bdry(W), Y> == <W, bdry_adjoint(Y, W.shape)>
```

Use `np.vdot` or `np.tensordot` consistently.

Test shapes:

```python
(3, 3)
(4, 5)
(3, 3, 3)
(4, 3, 3, 3)
```

Add a monotonic energy test. Define:

```python
E(W) = 0.5 * np.linalg.norm(bdry(W)) ** 2
G = boundary_homeostasis_gradient(W)
```

Use a small step, with backtracking if necessary, and verify:

```python
E(W - step * G) < E(W)
```

Skip the strict inequality only when `G` is numerically zero.

## Documentation

Create `docs/boundary_homeostasis.md` with:

- Definition of the boundary signal `bdry(W)`.
- Statement that `bdry(bdry(W)) == 0` terminates boundary-level introspection.
- Definition of the adjoint return map `bdry_adjoint`.
- Definition of the energy `0.5 * ||bdry(W) - H||_F^2`.
- Explanation that the feedback update is negative gradient descent on this energy.
- A short NumPy example.

Do not make empirical claims about improved accuracy.
Do not add Weights & Biases integration in this task.
Run `pytest -q` before finishing.
