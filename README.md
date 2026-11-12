# SimplicialTensors

SimplicialTensors implements simplicial operations on matrices and
hypermatrices using NumPy and SymPy. Its research scripts study horn
fillers, ordered graph boundaries, and boundary-based neural diagnostics.

For a hypermatrix of shape $(n_1,\ldots,n_k)$, the numeric module defines
its order as $k$ and its simplicial degree as $\min_a n_a-1$.
The face `face(t, i)` deletes index `i` on every axis. The degeneracy
`degen(t, i)` duplicates that index on every axis. For a matrix these
operations delete or duplicate the same row and column.

## Installation

Python 3.10 or later is required. The package dependencies are NumPy,
SymPy, and SciPy.

```sh
git clone https://github.com/flengyel/SimplicialTensors.git
cd SimplicialTensors
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1`.

Additional dependencies belong to individual experiments. Graph scripts
use NetworkX. Plotting and statistical scripts may require Matplotlib,
pandas, seaborn, or scikit-learn. Distributed and neural scripts may use
Dask, tqdm, or PyTorch. The Sage notebook and scripts require SageMath.
Each experiment page lists its dependencies and entry point.

## Numeric operations

Run the following after installation.

```python
import numpy as np
from simplicial_tensors.tensor_ops import bdry, degen, face, filler, horn, kan_condition

t = np.arange(9).reshape(3, 3)
print(face(t, 1))
print(degen(t, 1).shape)
print(bdry(t).shape)

h = horn(t, 0)
assert kan_condition(h, 0)
u = filler(h, 0)
print(all(np.array_equal(face(u, i), h[i]) for i in (1, 2)))
```

The face is `[[0, 2], [6, 8]]`, the degeneracy has shape `(4, 4)`,
the boundary has shape `(2, 2)`, and the final comparison is `True`.
The horn array contains a zero placeholder at the omitted face. That
placeholder can promote integer inputs to floating point.
`filler` constructs one filler and checks the omitted-face index.
Call `kan_condition(h, k)` to check compatibility before filling.

The boundary is the alternating sum of diagonal faces. `bdry` preserves
the input dtype, so fixed-width integer arithmetic can overflow.
`random_tensor` returns `int16` values. Use an appropriate coefficient
representation for the calculation.

The current numeric implementation has two relevant limitations.

- `face` and `bdry` raise `IndexError` on one-axis arrays because the
  internal face routine accesses a second axis.
- At degree zero, `bdry` returns an array with each axis shortened by
  one. The manuscript uses an unaugmented chain complex with zero
  outgoing differential in degree zero. Code that constructs that
  complex must apply the degree-zero convention explicitly.

`is_degen` tests membership in the sum of degeneracy images.
`decompose_degen` uses SymPy to solve a system whose columns are first
built as floating-point arrays. `decompose_degen_numpy` uses least
squares and a numerical reconstruction check.

The legacy name `n_hypergroupoid_conjecture(shape)` evaluates
`len(shape) < min(shape) - 1`. `n_hypergroupoid_comparison` compares an
input with selected horn fillers and may reject degenerate boundaries.
Its return value alone does not establish uniqueness of every filler.
See [the comparison experiment](docs/n_hypergroupoid_conjecture.md).

## Library modules

| Module | Implemented operations |
|---|---|
| `tensor_ops` | Tensor constructors, diagonal faces and degeneracies, boundaries, horns and fillers, degeneracy decomposition, and axis permutations. |
| `symbolic_tensor_ops` | SymPy-backed `SymbolicTensor` with face, degeneracy, boundary, horn, and filler methods. |
| `sagemath_compatible_tensor_ops` | A symbolic variant using SageMath when available and SymPy otherwise. |
| `cyclic_tensor_ops` | Face, degeneracy, and cyclic operations on a specified axis. |
| `adjoint_ops` | Frobenius adjoints, boundary energy gradients, cycle and range projections, a boundary pseudoinverse, and spectral filters. |
| `introspection_ops` | Boundary analysis, target projection, synthesis, and controller or projector feedback. |
| `architecture_ops` | Hidden-node balance diagnostics, ReLU equi-normalization, path products, and path-diamond energy. |

Import these modules through `simplicial_tensors`, for example
`from simplicial_tensors.adjoint_ops import face_adjoint`.
The package root lazily exports `adjoint_ops`, `architecture_ops`,
`introspection_ops`, `tensor_ops`, and `symbolic_tensor_ops`.
The other modules are available through their full submodule paths.

Importing `tensor_ops` resets Python's and NumPy's global random seeds
to 123 and creates a shared generator. Supplying `seed` to
`random_tensor` or `random_real_tensor` creates a generator for that
call. Calls with `seed=None` advance the shared generator.

## Manuscript and computations

The current manuscript is *Computations with simplicial tensor modules*.
Its driver is [tex/dstm_short.tex](tex/dstm_short.tex).
The [manuscript instructions](tex/README.md) describe the build and
arXiv source package.

The full simplicial module specified by a shape is contractible.
A submodule generated by a single hypermatrix can have nonzero homology.
For example, the off-diagonal matrix unit $E_{01}$ generates a submodule
with one-dimensional first homology, although it is a boundary in the
full square-matrix complex.

The scripts in [tex/anc](tex/anc) reproduce that example, a matrix horn,
and the Petersen graph incidence calculation. Their
[instructions](tex/anc/README.txt) pin the software revisions they
require. The pinned revisions include fixes and modules absent from the current
default-branch package. Follow the ancillary instructions to reproduce
these calculations.

## Experiments

[The experiment catalog](docs/experiments_catalog.md) links every
tracked script and result artifact. Experiment pages give the source,
dependencies, command, output, and applicable qualifications. Several
scripts run substantial computations at import time.

Useful starting points include the
[range-tensor examples](docs/examples_range_tensors.md),
[matrix-horn computation](docs/filler_combinatorics.md),
[Petersen rank distribution](docs/petersen_brd_right_cosets.md), and
[boundary adjoints](docs/boundary_homeostasis.md).

The neural studies distinguish adjoint energy regularization from
boundary analysis followed by controller-driven synthesis. The latter
retains the cycle component and projects the controller's target into
the realizable boundary range. The spectral formulas in these modules
are evaluated with floating-point arithmetic.
Read the [research report](docs/neural_boundary_research_report.md) and
[observer results](experiments/boundary_observer_report.md) for their
protocols, measured results, and scope.

## Tests and development

```sh
python -m pip install pytest
python -m pytest -q
```

The tests cover the maintained numeric, symbolic, adjoint, architecture,
and feedback APIs, along with package metadata and experiment
documentation coverage. Sage-specific tests are skipped when SageMath
is unavailable.

GitHub Actions runs Ruff, selected Mypy checks, Bandit, and a pytest
matrix for Python 3.10, 3.11, and 3.12. The exact commands are in
[.github/workflows/ci.yml](.github/workflows/ci.yml).
Repository conventions are in [GUIDELINES.md](GUIDELINES.md).
Reusable implementations belong in `src/simplicial_tensors`, experiments
in `experiments`, documentation in `docs`, and manuscript sources in `tex`.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
