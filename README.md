# SimplicialTensors

SimplicialTensors implements simplicial operations on matrices and
hypermatrices using NumPy and SymPy. It computes horn fillers,
normalized homology of generated submodules, and graph homology through
sparse tensor representations. Research scripts also study ordered
graph boundaries and boundary-based neural diagnostics.

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
The horn array contains a zero placeholder at the omitted face and
preserves the input dtype.
`filler` constructs one filler and checks the omitted-face index.
Call `kan_condition(h, k)` to check compatibility before filling.

The boundary is the alternating sum of faces. `face` and `bdry` support
one-axis arrays. By default, arithmetic uses the input dtype. For
integer arrays, `bdry(t, exact=True)` uses Python integers to avoid
overflow. Boolean arrays use arithmetic over $\mathbb F_2$.
`random_tensor` samples signed 64-bit integers. At degree zero, the raw
boundary has each axis shortened by one and therefore has no entries.
The homology routines use the unaugmented convention $\partial_0=0$.

`split_normalized(t)` returns the Moore projection and the components
of a degeneracy decomposition. `is_degen` tests whether that projection
vanishes. `decompose_degen` uses exact rational arithmetic for integer
and rational inputs, interprets finite floats as binary rationals, and
uses $\mathbb F_2$ for Boolean inputs. The zero tensor in degree zero is
the empty sum of degeneracies. `decompose_degen_numpy` supplies a
separate least-squares calculation with a numerical reconstruction check.

The legacy name `n_hypergroupoid_conjecture(shape)` evaluates
`len(shape) < min(shape) - 1`, the horn-uniqueness threshold in the
current degree. `n_hypergroupoid_comparison` compares an
input with selected horn fillers and checks the full-module horn kernel.
Its default precondition rejects degenerate boundaries. The empty
family of inner horns in degree at most one is a vacuous comparison.
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
| `homology` | Exact normalized chain dimensions, boundary ranks, and homology dimensions of tensor-generated submodules and relative horn quotients over $\mathbb Q$ or a prime field. |
| `rank_shifted` | Sparse rational tensors in a declared simplicial degree, graph representations, and a circle generator. |
| `graph_ops` | Graph incidence homology from tensor faces, deletion-chain homology over $\mathbb F_2$, and ordered boundary-rank distributions. |

Import these modules through `simplicial_tensors`, for example
`from simplicial_tensors.adjoint_ops import face_adjoint`.
The package root lazily exports the numeric, symbolic, adjoint,
architecture, introspection, homology, graph, and rank-shifted modules.
The other modules are available through their full submodule paths.

`tensor_ops` has a private generator initialized with seed 123.
Importing it leaves Python's and NumPy's global random states unchanged.
Supplying `seed` to
`random_tensor` or `random_real_tensor` creates a generator for that
call. Calls with `seed=None` advance the shared generator.

## Manuscript and computations

The current manuscript is *Computations with simplicial tensor modules*.
Its driver is [tex/dstm_short.tex](tex/dstm_short.tex).
The compiled [ten-page manuscript](tex/dstm_short.pdf) accompanies the source.
The [manuscript instructions](tex/README.md) describe the build and
arXiv source package.

The full simplicial module specified by a shape is contractible.
A submodule generated by a single hypermatrix can have nonzero homology.
For example, the off-diagonal matrix unit $E_{01}$ generates a submodule
with one-dimensional first homology, although it is a boundary in the
full square-matrix complex.

```python
from simplicial_tensors.homology import generated_homology
from simplicial_tensors.graph_ops import graph_chain_homology
from simplicial_tensors.rank_shifted import graph_embedding

circle = np.array([[0, 1], [0, 0]])
assert generated_homology(circle).homology_dimensions == (0, 1)

triangle = graph_embedding(3, [(1, 2), (2, 3), (3, 1)])
assert graph_chain_homology(triangle).homology_dimensions == (1, 1)
```

Graph vertices have labels $1,\ldots,v$. The graph representation gives
each edge an independent normalized tensor generator whose boundary is
target minus source. This realizes the incidence complex of the graph.
Deleting matching rows and columns of its adjacency matrix gives a
different complex.

The exact homology functions return dimensions over $\mathbb Q$ by
default. A prime `modulus=p` selects $\mathbb F_p$. They retain linear
relations among tensor entries and subtract both incoming and outgoing
boundary ranks. They do not compute integral torsion or homology bases.
See [the homology examples](docs/homology_basis.md) and
[sparse tensor representations](docs/rank_shifted.md).

The scripts in [tex/anc](tex/anc) reproduce that example, a matrix horn,
and the Petersen graph incidence calculation. Their
[instructions](tex/anc/README.txt) give the commands for an installed
checkout.

## Experiments

[The experiment catalog](docs/experiments_catalog.md) links every
tracked script and result artifact. Experiment pages give the source,
dependencies, command, output, and interpretation.

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
feedback, exact homology, and sparse graph APIs, along with experiment
entry points and documentation coverage. Sage-specific and optional
experiment tests are skipped when their dependencies are unavailable.

GitHub Actions runs Ruff, selected Mypy checks, Bandit, and a pytest
matrix for Python 3.10, 3.11, and 3.12. The exact commands are in
[.github/workflows/ci.yml](.github/workflows/ci.yml).
Repository conventions are in [GUIDELINES.md](GUIDELINES.md).
Reusable implementations belong in `src/simplicial_tensors`, experiments
in `experiments`, documentation in `docs`, and manuscript sources in `tex`.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
