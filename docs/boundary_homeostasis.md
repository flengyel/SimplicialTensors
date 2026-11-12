# Boundary regularization and exact synthesis

The diagonal boundary deletes a common index on every tensor axis and takes the alternating sum. It lowers the degree by one and satisfies $\partial^2=0$. The adjoint, projector, and pseudoinverse operations use the entrywise Euclidean or Frobenius inner product.

## Adjoint regularization

The adjoint face inserts a smaller tensor in the positions avoiding the deleted index and fills the other hyperplanes with zero. Thus
$$
\partial^*Y=\sum_i(-1)^i d_i^*Y,\qquad
\langle\partial W,Y\rangle_F=\langle W,\partial^*Y\rangle_F.
$$
For a boundary-shaped target $H$, the energy
$$
E_H(W)=\tfrac12\|\partial W-H\|_F^2
$$
has gradient $\partial^*(\partial W-H)$. The routine `boundary_homeostasis_gradient` returns that gradient. `boundary_homeostasis_feedback` returns a perturbation in its negative direction, normalized to Frobenius norm `alpha * norm(W)` when the gradient is nonzero. A zero gradient gives a zero perturbation.

This update is a quadratic regularization method. Repeated same-shape application involves $\partial^*\partial$, which is positive semidefinite and generally has nonzero powers.

## Analysis and synthesis

Write $B=\partial$ and let $P_0$ be the orthogonal projector onto $\ker B$. The [spectral formula](exact_boundary_spectrum.md) gives $P_0$ as a polynomial in $L=B^*B$ and the Moore–Penrose inverse as $B^\dagger=g(L)B^*$ for a polynomial $g$. Then
$$
W=P_0W+B^\dagger BW.
$$

The module [introspection_ops.py](../src/simplicial_tensors/introspection_ops.py) implements the following operations.

| Function | Operation |
|---|---|
| `boundary_analyze(W)` | Returns `BoundaryAnalysis(cycle=P0W, syndrome=bdry(W))`. |
| `project_boundary_signal(H, original_shape)` | Projects $H$ onto $\operatorname{im}B$. |
| `boundary_synthesize(cycle, syndrome, original_shape=None)` | Adds the cycle to the decoded syndrome. It checks that the cycle is finite and numerically lies in $\ker B$. |
| `boundary_target_update(W, H)` | Replaces the observed boundary by the projection of $H$ while preserving $P_0W$. |
| `boundary_controller_feedback(W, controller)` | Passes a copy of `bdry(W)` to a callable, projects its output to the boundary range, and synthesizes the new tensor. |
| `boundary_projector_feedback(W, gates)` | Retains the cycle and selects nonzero spectral bands using binary gates. |

A fixed-target update is
$$
W^+=P_0W+B^\dagger BB^\dagger H.
$$
It preserves $P_0W$ and attains the realizable component of $H$. Repeating the same fixed-target update is idempotent. Binary spectral gates also give an idempotent projector. An adaptive controller can produce different targets on later calls.

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
W_step = W - 0.1 * boundary_homeostasis_gradient(W)
assert np.linalg.norm(bdry(W_step)) < np.linalg.norm(bdry(W))

cycle = exact_cycle_projection(W)
assert np.linalg.norm(bdry(cycle)) < 1e-12

H = bdry(rng.normal(size=W.shape))
assert np.allclose(bdry(project_to_boundary(W, H)), H)

analysis = boundary_analyze(W)
assert np.allclose(boundary_synthesize(*analysis), W)
assert np.allclose(
    boundary_controller_feedback(W, lambda syndrome: syndrome), W
)
```

The routines evaluate finite identities with floating-point arithmetic and test residuals up to numerical tolerance. `tests/test_adjoint_ops.py` and `tests/test_introspection_ops.py` compare the filters with dense small-matrix references.

## Coordinate and experimental scope

The DSTM boundary identifies index $i$ on every axis. Dense-layer row and column indices usually label different neuron sets, and their independent relabeling changes this operation. A tied square map supplies the same source and target space, while the chosen ordering still affects its boundary.

The [neural study](neural_boundary_study.md) tests quadratic penalties and gradient smoothing. The [observer study](boundary_observer_study.md) tests passive boundary information on a tied map. Its stored full run did not pass the criterion for adding a learned controller. The separate `architecture_ops` module supplies layer-incidence and path-product diagnostics with explicitly typed neural layers.
