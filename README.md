# SimplicialTensors

Simplicial operations on matrices and higher-order tensors, with research scripts for diagonal simplicial tensor modules (DSTM), horn/filler behavior, and related combinatorics.

## What This Repository Contains

This repository has two main layers:

1. Reusable library code in `src/simplicial_tensors`.
2. Research scripts and computational studies in `experiments`.

The core package implements simplicial operators on NumPy arrays (and symbolic analogues), including:

- Face maps `d_i`
- Degeneracy maps `s_i`
- Boundary/coboundary operators
- Horn construction and filler computation
- Utilities for testing the n-hypergroupoid uniqueness criterion on shapes/tensors

The TeX sources in `tex` develop the mathematical framework around diagonal simplicial tensor modules and strict algebraic `n`-hypergroupoids.

## Mathematical Scope

The implementation is aligned with standard simplicial identities and supports computational checks of statements used throughout the project, including:

- `d_i d_j = d_{j-1} d_i` for `i < j`
- `s_i s_j = s_{j+1} s_i` for `i <= j`
- Mixed `d_i s_j` relations
- `d^2 = 0` (boundary squared is zero)

For this codebase, the shape-dependent heuristic/conjecture function is:

- `n_hypergroupoid_conjecture(shape)` returns whether `order(shape) < dimen(shape)`

and the computational check:

- `n_hypergroupoid_comparison(tensor, ...)` compares reconstructed horns/fillers and can raise `SimplicialException` when assumptions fail (for example degenerate boundaries unless explicitly allowed).

## Package Modules

### `simplicial_tensors.tensor_ops`

Primary numeric API (NumPy/SymPy-based), including:

- Tensor constructors: `random_tensor`, `random_real_tensor`, `range_tensor`
- Shape/index helpers: `get_index`, `dimen`, `order`
- Simplicial maps: `face`, `hface`, `vface`, `degen`, `hdegen`, `vdegen`
- Chain-level maps: `bdry`, `hbdry`, `vbdry`, `cobdry`, `bdry_n`, `bdry_mod1`
- Horn/filler pipeline: `horn`, `kan_condition`, `filler`
- Degeneracy analysis: `is_degen`, `decompose_degen`, `decompose_degen_numpy`
- Conjecture/comparison helpers: `n_hypergroupoid_conjecture`, `n_hypergroupoid_comparison`
- Tensor permutation/cycle helpers: `permute_tensor`, `random_axis_permutation`, `cyclic`, `cyclic_signed`

### `simplicial_tensors.adjoint_ops`

Frobenius adjoints and exact finite boundary filters:

- `face_adjoint`, `bdry_adjoint`
- `boundary_homeostasis_gradient`, `boundary_homeostasis_feedback`
- `lower_hodge_laplacian`, `exact_cycle_projection`
- `boundary_pseudoinverse`, `project_to_boundary`
- `boundary_sobolev_filter`

### `simplicial_tensors.introspection_ops`

Exact boundary analysis--controller--synthesis without an implicit scalar
penalty:

- `boundary_analyze`, `boundary_synthesize`
- `project_boundary_signal`, `boundary_target_update`
- `boundary_controller_feedback`, `boundary_projector_feedback`

The analysis retains the cycle channel and the boundary signal, so it is
lossless. Controller targets are projected into the realizable boundary range
and returned to weight space with the exact Moore--Penrose decoder.
Individual controllers can still impose hard constraints; for example, a zero
target projects onto the cycle space.

### `simplicial_tensors.architecture_ops`

Typed neural-architecture operations that avoid identifying unrelated weight
axes:

- hidden-node incidence/balance residuals, energies, and gradients
- function-preserving ReLU equi-normalization
- two-edge path products and a path-diamond diagnostic

### `simplicial_tensors.symbolic_tensor_ops`

SymPy-backed symbolic tensor class:

- `SymbolicTensor` with symbolic `face`, `degen`, `bdry`, `horn`, `filler`
- Symbolic uniqueness checks and diagnostics

### `simplicial_tensors.sagemath_compatible_tensor_ops`

SageMath-friendly symbolic variant:

- Uses Sage when available
- Falls back to SymPy so imports still work outside Sage

### `simplicial_tensors.cyclic_tensor_ops`

Axis-specific cyclic/face/degeneracy helpers:

- `face_axis`, `degen_axis`, `cyclic`, `cyclic_signed`

## Installation

### Requirements

- Python 3.10+
- Core dependencies (declared in `pyproject.toml`):
  - `numpy`
  - `sympy`
  - `scipy`

### Editable install

```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

## Quick Start

```python
import numpy as np
from simplicial_tensors.tensor_ops import range_tensor, face, degen, bdry, horn, filler

t = range_tensor((4, 4, 4))
f1 = face(t, 1)
g1 = degen(f1, 1)
b = bdry(t)

h = horn(t, 1)
t_prime = filler(h, 1)

print("t:", t.shape)
print("face:", f1.shape)
print("degen(face):", g1.shape)
print("bdry:", b.shape)
print("horn length:", len(h))
print("filler matches input:", np.array_equal(t, t_prime))
```

## Reproducibility and Random Seeds

`tensor_ops` exposes `___SEED___ = 123` and uses a module-level RNG stream.

- `random_tensor(..., seed=None)` and `random_real_tensor(..., seed=None)` use the shared stream.
- Passing an explicit seed creates a deterministic per-call generator.

So repeated calls with the same explicit seed are reproducible, while calls without a seed advance the shared stream.

## Tests

Run all tests:

```bash
pytest -q
```

Test coverage includes:

- Simplicial identities for numeric and symbolic tensors
- Equivariance under tensor-axis permutations
- Cyclic operator behavior
- Graph-related equivariance checks
- Package metadata and documentation coverage checks

Sage-specific tests are present and skipped automatically unless Sage is available.

To run Sage-specific tests explicitly:

```bash
sage -python -m pytest tests/test_sagemath_compatible_tensor_ops.py -q
```

## CI Quality Gates

GitHub Actions (`.github/workflows/ci.yml`) runs:

1. `quality` job (Python 3.12):
   - Ruff fatal checks
   - Mypy on selected core modules
   - Bandit high-severity scan
2. `tests` matrix job (Python 3.10, 3.11, 3.12), dependent on `quality`

Local equivalents:

```bash
ruff check src tests --exclude src/simplicial_tensors/notebooks --select E9,F821,F822,F823
mypy --ignore-missing-imports --follow-imports=skip src/simplicial_tensors/__init__.py src/simplicial_tensors/tensor_ops.py src/simplicial_tensors/cyclic_tensor_ops.py
bandit -q -r src/simplicial_tensors -x src/simplicial_tensors/notebooks -lll
pytest -q
```

## Experiments and Documentation

The `experiments` directory contains standalone scripts and artifacts used for research exploration.

The boundary/AI investigation, literature synthesis, exact spectrum result,
legacy regularization findings, and corrected observer--feedback framing are
collected in
[`docs/neural_boundary_research_report.md`](docs/neural_boundary_research_report.md).
The subsequent passive observer test and its negative go/no-go decision are
reported in
[`experiments/boundary_observer_report.md`](experiments/boundary_observer_report.md).

Current documentation policy in this repository:

- Each tracked `experiments/*.py` and `experiments/*.sage` script has a corresponding `docs/<script_stem>.md` page.
- `docs/experiments_catalog.md` inventories tracked files in `experiments`.

Some experiment scripts depend on additional packages not declared in core package metadata (for example `networkx`, `matplotlib`, `pandas`, `seaborn`, `torch`, `dask`, `tqdm`, or SageMath). Install these as needed for the specific script you run.

A few scripts also assume local helper modules or a particular working directory; consult each script header and its paired page in `docs` before running.

### Interpreting experiment output

Experiment scripts are research diagnostics, not all-or-nothing package tests. Some scripts print labels such as:

- `CHECKED`: a local identity, invariant, or sanity check was verified on the selected finite data.
- `CLASSIFIES`: the script classified a construction, often showing that it works only in a restricted or trivial sense.
- `FLAGS`: the script found that a proposed verification is weak, tautological, or insufficient for the stronger claim being examined.
- `REFUTES`: the script found a finite counterexample or obstruction to the stronger claim being examined.
- `CONFIRMS`: the script confirmed a local finite fact, without implying a broader theorem.

A script may exit successfully when it finds an expected counterexample. In that case, success means the diagnostic behaved as intended; it does not mean every mathematical claim under examination was confirmed.

If an experiment cannot import `simplicial_tensors`, either run `pip install -e .` from the repository root or set `PYTHONPATH=src` before invoking the script.

## TeX Manuscript Assets

The `tex` directory contains the paper sources and supporting files, including:

- Main file: `tex/dstm.tex`
- Section files such as `introduction.tex`, `horns.tex`, `normalization.tex`, `combinatorics.tex`, and `dichotomy.tex`
- Bibliography: `tex/dstm.bib`
- Generated PDFs/artifacts used in drafting and submission

## Repository Layout

- `src/simplicial_tensors`: reusable library code
- `tests`: pytest suites
- `experiments`: runnable research scripts and generated artifacts
- `docs`: documentation for experiments and project notes
- `tex`: manuscript source and paper artifacts
- `plots`, `logs`: generated outputs
- `Cleanup`: maintenance scripts

## Development Notes

Repository conventions are documented in `GUIDELINES.md`.

Key expectations:

- Reusable logic lives in `src/simplicial_tensors`.
- Experiment-specific workflows live in `experiments`.
- Avoid duplicating library logic across scripts.

## License

GPL-3.0-or-later. See `LICENSE`.
