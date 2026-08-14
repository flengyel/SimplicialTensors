# Adjoint boundary regularization and exact boundary synthesis

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

The adjoint is one optional return map.  Choosing it inside a negative-gradient
update produces a regularizer; that choice is not implied by the original
introspection proposal.  Exact pseudoinverse synthesis supplies a separate
return mechanism without an implicit scalar penalty.

## Adjoint-return regularizer (baseline)

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

## Exact analysis--controller--synthesis

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

is the exact Frobenius-orthogonal projector onto (ker\partial).  With
(B=\partial), (P_0=P_{\ker B}), and the exact Moore--Penrose decoder
(B^\dagger), every tensor has the lossless decomposition

\[
W=P_0W+B^\dagger BW.
\]

A controller can replace the syndrome (s=BW) by a proposed target without
minimizing a boundary norm:

\[
\widetilde h=\Phi(s,\text{context}),\qquad
h=BB^\dagger\widetilde h,\qquad
W^+=P_0W+B^\dagger h.
\]

Exactly (P_0W^+=P_0W) and (BW^+=h).  For a fixed target the equivalent closest
tensor

\[
W^+=W-B^\dagger(BW-h)
\]

is reached in one exact pass and repeating the same update changes nothing.
An arbitrary controller recomputed after each update need not be idempotent or
nilpotent.  The implementation evaluates (B^\dagger) as a low-degree
polynomial in (L), requiring at most (min(k,p)) boundary/adjoint passes and no
dense matrix or SVD.

## NumPy example

```python
import numpy as np

from simplicial_tensors.adjoint_ops import (
    boundary_homeostasis_gradient,
    exact_cycle_projection,
    project_to_boundary,
)
from simplicial_tensors.introspection_ops import (
    boundary_analyze,
    boundary_controller_feedback,
    boundary_synthesize,
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

# Lossless observation and identity-controller synthesis.
analysis = boundary_analyze(W)
assert np.allclose(
    boundary_synthesize(analysis.cycle, analysis.syndrome), W
)
W_identity = boundary_controller_feedback(W, lambda syndrome: syndrome)
assert np.allclose(W_identity, W)
```

## Neural-network scope warning

The diagonal boundary identifies label (i) on every tensor axis.  Generic
dense-layer row and column indices are different neuron populations and may
be independently relabeled.  Therefore this boundary is not intrinsic to the
function represented by a generic dense weight matrix.  That prevents a
function-level claim, but it does not forbid inspection of a deliberately
labeled parameter/optimizer state.  A tied recurrent or residual map
(W:H\to H) fixes the row/column type mismatch, though neuron ordering remains
a robustness ablation.  For function-intrinsic work, use the typed
architecture/path operations in `simplicial_tensors.architecture_ops` or
another permutation- and gauge-aware construction.
