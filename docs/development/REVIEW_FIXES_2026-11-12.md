# Code repairs and validation, 2026-11-12

The changes correct graph-basis coordinates and finite-field reduction,
numerical reconstruction near a zero cycle, unsigned and integer arithmetic,
and omitted-face validation. The affected API guides describe the resulting
coefficient, coordinate, and tolerance conventions.

## Provenance

| State | Identifier |
|---|---|
| Reviewed baseline | `4e6cfa4f61ac50ccd2f217235aa3d333518f8311` |
| Local implementation commit | `5b966df3894c51eb8ec6fd7552612b1efdb6c11c` |
| Implementation Git tree | `e854f7e1d38088a495cca2cc0733a6211cc48570` |
| Repository-API implementation commit | `0dd8dce18b87d435d80851c961d7113c3fd6f6e6` |

The local and repository-API commit objects share the tree shown above.
This record and the documentation index are separate from the implementation
commit. The additional graph-guide clarification describes the already
implemented behavior of manually modified tensor models.

## Corrected behavior

| Finding | Repair and regression coverage |
|---|---|
| Graph coordinates (B1) | `graph_incidence_matrix` solves `V X = B` in the supplied vertex basis over the selected field. It checks vertex independence and boundary containment. Tests include the non-coordinate basis that previously gave Betti numbers `(2,1)` instead of `(1,0)`, over the rationals and characteristics three and five. |
| Rational prime-field coefficients (B2) | The graph routines reduce rational coefficients through the shared exact-field arithmetic. All input denominators must be invertible, including coefficients on marker coordinates with zero boundary. Tests distinguish rank loss modulo a prime from a denominator that vanishes. |
| Near-zero numerical cycles (B3) | Analysis refines its projected cycle once. Synthesis scales its cycle-residual tolerance by the larger of the cycle norm and decoded-syndrome norm, with no absolute floor. Scaled norms avoid spurious overflow and underflow from squaring. Tests cover the supplied failure, identity and zero-target feedback, dense references, and real/complex scales from `1e-200` to `1e200`. |
| Unsigned numerical arithmetic (B4) | Numerical adjoint and introspection inputs are promoted before signed boundary operations. Boolean, integer, floating, and complex dtypes use at least `float64` or `complex128`; complex components and wider inexact dtypes are retained. Object and string arrays are rejected. The exact/dtype conventions of `tensor_ops` remain separate. |
| Architecture gradients and products (C1) | Weights and biases share a promoted dtype before squared norms, products, and gradient accumulation. Tests verify the analytic gradients `-6,12`, integer overflow controls, a complex-bias directional derivative, complex-bias preservation, and preservation of the real ReLU network function. |
| Omitted-face indices and return descriptions (C2) | Dense and based-chain relative-horn APIs consistently reject Boolean face indices and accept supported exact integer indices. The two homology docstrings now identify their dimension/rank summaries as the return values. |

Factory-generated graph embeddings retain ordinary incidence homology. For a
manually modified model, each supplied edge tensor represents a formal
degree-one generator, and the fallback computes the tensor-incidence complex
in the chosen vertex basis. That complex can have field-dependent rank and
need not recover the
ordinary homology of the graph metadata. See the
[graph API](../api/rank_shifted.md) and
[numerical API](../api/boundary_homeostasis.md) for the public contracts.

## Validation

| Check | Result |
|---|---|
| Full suite in SageMath 10.10 | 744 passed, 10 skipped: eight optional Dask/PyTorch cases and two Git-metadata cases in the container snapshot. |
| Git-metadata follow-up in the owner checkout with SageMath 10.7 | Both skipped metadata checks passed. |
| Full ordinary-Python suite | 627 passed, 26 skipped with Python 3.13.5, NumPy 2.5.4, SymPy 1.14, SciPy 1.18.1, and pytest 9.2.0. The skips cover unavailable SageMath, Dask, or PyTorch dependencies. |
| Affected tests in SageMath 10.7 | 232 passed. |
| Independent native Sage graph reference | 14 passed; these tests are also included in the full SageMath suite. |
| Static checks | Ruff `E9,F821,F822,F823`, Mypy on the eight CI-selected source files, and Bandit at severity `-lll` passed. |
| Isolated installed wheel | `python -I tests/wheel_smoke.py` passed in a fresh environment; all 24 packaged Python source files matched the implementation source byte for byte. |
| Affected API examples | The Python code blocks in both affected API guides executed successfully. |
| Original independent numerical probes | All 15 numeric comparisons matched. The two supplied floating-point reconstructions had errors approximately `8.83e-16` and `1.67e-15`; the integer gradient control returned `-6,12`. |
| Coefficient controls | Exact rational tensor operations passed unchanged. `complex64`/`complex128` adjoint, decoder, analysis, and complex-target controls passed; the recorded Hermitian-adjoint residual was approximately `4.44e-16`. |

The wheel SHA-256 is
`972457078e2d555f909811ca78186fb8fbee85c0ea0c8946ae2830e21ff0de4b`.

Representative Bash commands, run from the repository root with the required
packages installed, are:

```sh
PYTHONPATH=src python -m pytest -o addopts= -q -rs
ruff check src tests --select E9,F821,F822,F823
mypy --ignore-missing-imports --follow-imports=skip src/simplicial_tensors/{__init__,tensor_ops,cyclic_tensor_ops,chain_complex,hypergraph_homology,tensor_chain_complex,chain_maps,kernel_sequences}.py
bandit -q -r src/simplicial_tensors -lll
python -I tests/wheel_smoke.py
```

The last command uses the Python interpreter from the isolated wheel
installation. The full SageMath run used the existing SagePod container:

```sh
podman exec -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/home/sage/notebooks/SimplicialTensors/src \
  -w /home/sage/notebooks/SimplicialTensors sagepod \
  /sage/sage -python -m pytest -q -p no:cacheprovider
```

The explicit `PYTHONPATH` also reaches experiment subprocesses. An earlier
invocation without it had ten subprocess import failures; the corrected
invocation passed without an import-related source change.

The retained logs and reproduction scripts are outside the repository at
`/home/flengyel/Jupyter/simplicial-review-fixes-2026-11-12`. Principal records
are `python-full.log`, `numeric-full-suite-pythonpath.log`,
`sage-10.7-compatibility.log`, `sage-10.7-metadata.log`,
`native-graph-reference.log`, `quality-final.log`, `wheel-smoke.log`, and
`artifact-and-docs.log`. The independent controls are in
`numeric-probes-fixed.log` and `numeric-contracts-fixed.log`. The numerical
reproductions and architecture
before/after logs are in the same directory.

## Review scope and limits

The preceding review read all 14 manuscript pages, the three active TeX inputs,
and their bibliography. Collectively, the code reviewers read all 24 shipped
Python files, totaling 4,878 source lines at the reviewed baseline. No fatal or
major mathematical defect was confirmed in that manuscript. This finding does
not establish completeness, correctness, or novelty beyond the stated review.
The original reports are retained outside the repository at
`/home/flengyel/Jupyter/simplicial-full-review-2026-11-12`.

The manuscript wording proposals remain unapplied: declare the tail vector
$\mathbf c=(r,r,r)$ explicitly, type and name the full tree-horn restriction
$\rho_F$, and use $S^q(K)$ consistently for the one-term complex. The requested
code repairs did not change TeX sources or the manuscript PDF.

The test totals are bounded computational evidence, not proofs or an exhaustive
floating-point stability analysis for arbitrarily high tensor order. Optional
Dask/PyTorch paths skipped above were not executed in these validation
environments. The native references and regression tests verify the displayed
counterexamples and the recorded configurations; they do not certify every
possible tensor or manually modified graph model.
