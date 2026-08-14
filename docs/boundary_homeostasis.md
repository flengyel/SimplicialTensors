# Boundary homeostasis, exact decoding, and its limits

Let (W) be a tensor and let the diagonal face (d_iW) delete index (i)
along every tensor axis.  The diagonal boundary signal is

\[
\partial W=\sum_{i=0}^{p}(-1)^i d_iW,
\qquad p=\min(\operatorname{shape}W)-1.
\]

Consecutive degree-lowering boundaries satisfy

\[
\partial_{p-1}\partial_p=0.
\]

This terminates the **graded feature chain**
(W\mapsto\partial W\mapsto0).  It does not make a returned same-shape
feedback update nilpotent.  With the Frobenius adjoint, the returned operator
is (L=\partial^*\partial), which is positive semidefinite and generally
satisfies (L^2W\ne0).

## Frobenius adjoint and energy

The adjoint face (d_i^*) inserts a smaller tensor into all positions that
avoid (i) and fills the deleted hyperplanes with zero.  Therefore

\[
\partial^*Y=\sum_{i=0}^{p}(-1)^i d_i^*Y,
\qquad
\langle\partial W,Y\rangle_F
=\langle W,\partial^*Y\rangle_F.
\]

For a target boundary (H), define

\[
E_H(W)=\frac12\|\partial W-H\|_F^2.
\]

Its Euclidean gradient is

\[
\nabla E_H(W)=\partial^*(\partial W-H).
\]

The ordinary homeostatic update

\[
W\leftarrow W-\eta\lambda\partial^*(\partial W-H)
\]

is thus gradient descent on a generalized Tikhonov penalty.  It is an
iterative diffusion, not a consequence of (partial^2=0).

## Exact finite alternative

For tensor order (k) and degree (p), the nonzero spectrum of
(L=\partial^*\partial) is

\[
\{2,3,\ldots,\min(k,p)+1\}.
\]

Consequently the polynomial

\[
P_{\ker\partial}
=\prod_{\lambda=2}^{\min(k,p)+1}
\left(I-\frac{L}{\lambda}\right)
\]

is the exact Frobenius-orthogonal projector onto (ker\partial).  This gives
a genuinely finite correction, requiring at most (min(k,p)) boundary and
adjoint-boundary passes.

For a realizable target (H), the exact closest tensor is

\[
W_{\mathrm{new}}
=W-\partial^\dagger(\partial W-H).
\]

The implementation evaluates (partial^\dagger) as another low-degree
polynomial in (L), without forming a dense matrix or computing an SVD.  If
(H) is not realizable, it is automatically projected onto
(\operatorname{range}\partial).

## NumPy example

```python
import numpy as np

from simplicial_tensors.adjoint_ops import (
    boundary_homeostasis_gradient,
    exact_cycle_projection,
    project_to_boundary,
)
from simplicial_tensors.tensor_ops import bdry

rng = np.random.default_rng(7)
W = rng.normal(size=(4, 5))

# Iterative negative gradient feedback.
gradient = boundary_homeostasis_gradient(W)
W_step = W - 0.1 * gradient
assert np.linalg.norm(bdry(W_step)) < np.linalg.norm(bdry(W))

# Exact finite orthogonal projection onto the cycle space.
W_cycle = exact_cycle_projection(W)
assert np.linalg.norm(bdry(W_cycle)) < 1e-12

# Exact closest solution for a realizable nonzero target.
H = bdry(rng.normal(size=W.shape))
W_target = project_to_boundary(W, H)
assert np.allclose(bdry(W_target), H)
```

## Neural-network scope warning

The diagonal boundary identifies label (i) on every tensor axis.  Generic
dense-layer row and column indices are different neuron populations and may
be independently relabeled.  Therefore this boundary is not intrinsic to a
generic dense weight matrix.  It should be used only when the axes have a
declared common ordered simplicial index, or as a deliberately
coordinate-dependent prior.  For ordinary networks, use the typed
architecture/path operations in `simplicial_tensors.architecture_ops` or
another permutation- and gauge-aware construction.
