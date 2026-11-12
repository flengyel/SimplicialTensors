# Coordinate-counting prototype

[experiments/homology_basis.py](../experiments/homology_basis.py) counts coordinate tensors that fail the package's degeneracy test. Its printed `homology_dim` label does not describe a homology calculation.

For a supplied shape, it sets `n = min(shape) - 1` and enumerates the coordinate units of the cube with n + 1 labels on every axis. This discards rectangular offsets. It applies `tensor_ops.is_degen` to each unit, which solves a degeneracy-span linear system.

The parameter `j` is unused by `get_normalized_basis`. For n ≥ 1, every enumerated tuple satisfies the condition in `get_missing_indices`, so `compute_homology_dimension` returns the count of units accepted by the earlier filter. Selecting units outside the degeneracy submodule does not construct a normalized basis. The script constructs no differential or quotient by boundaries.

Run it from an installed checkout:

~~~sh
python experiments/homology_basis.py
~~~

The current fixed examples produce these counts on each reported axis.

| Shape | Count |
|---|---:|
| (3, 3) | 7 |
| (4, 4) | 0 |
| (5, 5) | 0 |
| (3, 4, 5) | 25 |

The degeneracy solver is called separately for each unit, so the examples can take time.

For a verified generated-submodule example, follow [the pinned ancillary instructions](../tex/anc/README.txt). The finite normalized-chain construction is given in §6 of [the current manuscript source](../tex/generated_section.tex).
