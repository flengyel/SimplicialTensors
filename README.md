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
import numpy as np
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
- `examples/`: runnable small demos
- `experiments/`: research and exploratory scripts/artifacts
- `docs/`: project and experiment documentation

## Experiments Documentation

Each git-tracked file under `experiments/` is documented in:

- [docs/experiments_catalog.md](docs/experiments_catalog.md)

Several experiments also have dedicated deep-dive docs in `docs/` (for example `horn_map_reduce.md`, `degenerate_counterexample.md`, `verify_injective_face_map.md`).

## Example Script

Run the current lightweight example:

```bash
python examples/ab_mlp_demo.py
```

## Notes

- `experiments/` contains exploratory code and artifacts; interfaces there are less stable than the package API under `src/`.
- For SageMath-focused workflows, see `src/simplicial_tensors/sagemath_compatible_tensor_ops.py` and related docs.

## License

This repository is distributed under the GPLv3 (see [LICENSE](LICENSE)).
