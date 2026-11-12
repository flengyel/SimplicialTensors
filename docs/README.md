# Documentation

Use the API references for library behavior and coefficient domains, and the
experiment guides for reproducible commands and their interpretation. Shell
commands run from the repository root unless a page specifies another directory.

## API references

- [Degeneracy membership and Moore projection](api/is_degen.md)
- [Exact generated and relative homology](api/homology.md)
- [Sparse shifted tensors and graph chains](api/rank_shifted.md)
- [Boundary adjoints, regularization, and synthesis](api/boundary_homeostasis.md)
- [SageMath-compatible operations and the CoCalc adapter](api/sagemath_compatible_tensor_ops.md)

## Experiments and notebooks

[The experiment catalog](experiments_catalog.md) lists scripts, notebook documents,
and stored results. Each linked guide describes inputs, dependencies, commands,
outputs, and interpretation. Useful starting points are the
[homology examples](experiments/homology_basis.md),
[Petersen calculation](experiments/petersen_coset_enumeration.md), and
[tensor-boundary notebooks](experiments/tensor_boundary_notebooks.md).

## Research notes

- [Spectrum of the diagonal tensor boundary](research/exact_boundary_spectrum.md)
- [Boundary operators in neural experiments](research/neural_boundary_research_report.md)

The [manuscript guide](../tex/README.md) identifies the drivers, build commands,
ancillary computations, and submission packages.

## Development

- [Shell setup and repository commands](development/no_terminal_troubleshooting.md)
- [Publishing repository changes](development/pushing_from_codex_workspace.md)
- [Symbolic backend tests](development/test_symbolic_tensor_ops.md)
- [Adjoint implementation notes](development/CODEX_ADJOINT_BOUNDARY_TASK.md)
- [Adjoint implementation task specification](development/Codex_issue.txt)
- [SagePod verification handoff, 2026-11-12](development/CODEX_SAGEPOD_HANDOFF_2026-11-12.md)
- [SagePod validation, 2026-11-12](development/SAGEPOD_VALIDATION_2026-11-12.md)
