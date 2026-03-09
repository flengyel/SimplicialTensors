# SimplicialTensors

Simplicial operations on matrices and higher-order tensors.

## Overview

`SimplicialTensors` provides NumPy/SymPy-based implementations of core simplicial operators on diagonal tensor objects:

- face maps `d_i`
- degeneracy maps `s_i`
- boundary operators and related constructions
- horn/filler utilities and symbolic variants

The package focuses on algebraic experimentation and verification workflows for simplicial tensor constructions.

## Mathematical Identities

The implementation targets the standard simplicial identities:

$$
\begin{aligned}
d_i d_j &= d_{j-1} d_i, && \text{if } i < j; \\
s_i s_j &= s_j s_{i-1}, && \text{if } i > j; \\
d_i s_j &=
\begin{cases}
s_{j-1} d_i, & \text{if } i < j; \\
1, & \text{if } i \in \{j, j+1\}; \\
s_j d_{i-1}, & \text{if } i > j+1.
\end{cases}
\end{aligned}
$$

## Installation

1. Clone the repository.
2. Change into the project directory:

   ```bash
   cd SimplicialTensors
   ```

3. Install in editable mode:

   ```bash
   pip install -e .
   ```

Project metadata currently declares:

- Python `>=3.10`
- dependencies: `numpy`, `sympy`, `scipy`

## Quick Start

```python
from simplicial_tensors.tensor_ops import face, degen, bdry, range_tensor

t = range_tensor((4, 4, 4))
f = face(t, 1)
s = degen(f, 1)
b = bdry(t)

print(t.shape, f.shape, s.shape, b.shape)
```

## Run Tests

```bash
pytest -q
```

## Repository Layout

- `src/simplicial_tensors/`: package source modules
- `tests/`: automated test suite
- `experiments/`: research and exploratory scripts/artifacts
- `docs/`: project and experiment documentation
- `logs/`, `plots/`, `tex/`: generated artifacts and paper materials

## Experiments Documentation

- Every git-tracked `experiments/*.py` and `experiments/*.sage` script has a dedicated page at `docs/<script_stem>.md`.
- A complete inventory of all git-tracked files under `experiments/` is maintained in [docs/experiments_catalog.md](docs/experiments_catalog.md).
- Notebook and artifact files are documented in the catalog and linked to dedicated docs when applicable.

## Notes

- `experiments/` contains exploratory code and artifacts; interfaces there are less stable than the package API under `src/`.
- For SageMath-focused workflows, see `src/simplicial_tensors/sagemath_compatible_tensor_ops.py` and related docs.

## License

This repository is distributed under the GPLv3 (see [LICENSE](LICENSE)).

