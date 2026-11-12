# SagePod validation, 2026-11-12

Tested repair revision: `b4ef508e68e865869a88bee9a4292026b6701543`.
This report was first published in
`c375540c66f3abe2f2bd4349cc57c001fd4afa7e`.
The repair revision's tree `f91067f362c33d2a0e5f92db85f01e42d1aeec64` matches the locally
tested commit `188173fc2acbc54a8ddc97f18c37a736a08988fe` exactly.
Publication used the authenticated GitHub app after shell HTTPS and SSH
authentication proved unavailable. Every uploaded blob and the complete tree
were checked against their local Git hashes before advancing `main`.
Fetched `origin/main` at `792b50bc0fe3de814457d075d092c11ce912d748`;
the implementation baseline `59947bc06a5bcdd9bb1c8246347af1a5377baed7`
remains an ancestor. No force push or reset was used.

## Environment

The existing WSL Debian installation and running `sagemath` container were reused.
SagePod checkout: `4bc1bc364589777e8baa27bfc687d158310d71a5`;
its unrelated untracked `txt` file was preserved. Container image:
`localhost/sagequeue-sagemath:10.7-pycryptosat`, image ID
`e10ecad61f8fe381eeb9d68d7003247b48d13da0903cdd75b7888f94e917393d`.
Host `~/Jupyter` is mounted at `/home/sage/notebooks`; the test checkout is
`~/Jupyter/SimplicialTensors`. The container was not rebuilt, recreated, or stopped.

| Interpreter | Python | NumPy | SymPy | SciPy | Pytest |
|---|---|---|---|---|---|
| Sage 10.7, `/sage/sage -python` | 3.12.5 | 2.2.4 | 1.13.2 | 1.15.2 | 8.3.2 |
| Isolated ordinary Python | 3.13.5 | 2.5.4 | 1.14.0 | 1.18.1 | 9.2.0 |

The native gate verified `HAVE_SAGE=True`, native Sage `Expression` entries,
and backend source `/home/sage/notebooks/SimplicialTensors/src/simplicial_tensors/sagemath_compatible_tensor_ops.py`.
Sage's supplied numerical packages were retained. The container lacked Git;
Git 2.30.2 and its three required Debian packages were added without upgrades.
Stale security-package URLs failed; the available Debian main versions installed
successfully. The image ID above identifies the underlying image, before these
container-layer additions.

## Demonstrated defects and repairs

| Anchor | Original evidence | Repair |
|---|---|---|
| `tests/test_sagemath_compatible_tensor_ops.py::test_simplicial_identities` | Original dedicated run: 6 passed, 1 failed. For `arange(9).reshape(3,3)`, `(i,j)=(0,2),(2,0)` pass the correct identities and fail the old comparisons. | Use `T.face(i).degen(j-1)` for `i<j`, and `T.face(i-1).degen(j)` for `i>j+1`. |
| `tensor_ops.degen` | A native Sage vector entry `(2**60+1)/3` became `384307168202282273` through NumPy scalar insertion. | Duplicate integer indices with `np.take`, preserving the exact scalar objects on every axis. |
| `symbolic_tensor_ops._polynomial_coefficient_rank` | Constant `sqrt(2)` bypassed the documented rational coefficient restriction. | Apply exact `QQ` conversion even when there are no variables; irrational constants raise `ValueError`. |
| `petersen_coset_enumeration.sage::gap_perm_to_list` | `g.Image(i)` raised GAP's general-mapping error. Direct image lists then covered only 12,264 distinct matrices. | Use `libgap.ListPerm(g**-1, size)` for the script's reorder-list convention. Check distinct matrices before accepting the histograms. |
| `test_import_does_not_reset_global_random_generators` | An isolated Sage probe showed that importing SymPy alone changes Python RNG state; importing `tensor_ops` afterward preserves both Python and NumPy RNG state. | Initialize the existing SymPy dependency before seeding and taking the package-import snapshot. |

The Petersen script now imports the shared Boolean `bdry`; its companion notebook
runs the same entry point. The Python right-coset implementation is unchanged.
Native regression identities use seed 42. The isolated RNG test uses seeds
17 and 23. Rank references and graph examples use fixed explicit inputs.

## Checks and results

| Check on the repaired source | Result |
|---|---|
| Dedicated native Sage backend module | 24 passed; all seven original tests execute |
| Independent Sage homology references | 27 passed over `QQ`, `GF(2)`, and `GF(3)` |
| Bounded Sage/GAP Petersen regressions | 3 passed |
| Full Sage suite, `pytest -o addopts= -q -ra` | 368 passed, 10 skipped, 82.93 s |
| Two skipped Git metadata checks, rerun as checkout owner | 2 passed, 3 deselected, 0.62 s |
| Full ordinary Python suite, `pytest -o addopts= -q -ra` | 316 passed, 11 skipped, 19.60 s |
| CI Ruff, Mypy, Bandit | All passed |
| Eight handoff example commands | All exited 0 under native Sage |
| CoCalc notebook | All four code cells completed in one shared Sage Python namespace |
| Petersen script with Sage preparser | Completed with all coverage and rank assertions |
| Petersen companion notebook | Its code cell completed with the same results |

The Sage suite's skips comprise one Dask case, seven PyTorch cases, and two
Git metadata checks. Git rejected the rootless mounted checkout's ownership.
Automatic approval review rejected adding a persistent global Git trust entry.
The two read-only checks instead ran as the checkout's container owner, using an
isolated temporary home and disabling pytest's cache writer. Git trust settings
were not changed. Thus all 370 applicable Sage test cases executed successfully
across the full run and this targeted rerun; eight optional cases remain skipped.
Ordinary Python additionally skips the three Sage-only modules.

An earlier integrated run under concurrent load had three subprocess import/help
timeouts. All three passed in the final run without changing their time limits.
There are no unresolved test failures.

The independent homology references form the quotient by degeneracies and included
horn faces directly from all monotone pullbacks, then use native Sage matrix ranks.
They compare chain dimensions, both boundary ranks entering each Betti number,
and the zero differential out of degree zero. Native Sage rationals are converted
to `Fraction` by numerator and denominator, including coefficients above `2**53`.
The public homology input contract was not extended to arbitrary Sage scalars.

Petersen's shifted graph complex has chain dimensions `(10,15)`, boundary ranks
`(0,9)`, and Betti numbers `(1,6)` over all three tested fields. The `E01`-generated
module has Betti numbers `(0,1)`. The full order-two ambient module has normalized
dimensions `(1,3,2,0)` and ranks `(0,1,2,0)`, giving zero homology.

The exhaustive Petersen adjacency calculation gives group order 120 and 30,240
distinct relabeled matrices. These deletion-boundary ranks concern a different
map from the ordinary graph incidence differential:

| Rank | Distinct matrices | Weighted permutations |
|---:|---:|---:|
| 4 | 376 | 45,120 |
| 6 | 8,496 | 1,019,520 |
| 8 | 21,368 | 2,564,160 |
| Total | 30,240 | 3,628,800 |

## Reproduction

In the mounted checkout, the Sage Python command prefix was:

```sh
podman exec -i --workdir /home/sage/notebooks/SimplicialTensors \
  --env PYTHONPATH=/home/sage/notebooks/SimplicialTensors/src \
  sagemath /sage/sage -python
```

The backend gate, original seven-test collection and failure run, dedicated tests,
full suite, and all commands in the handoff were executed. The eight example
arguments, in order, were:

```text
experiments/sagemath_compatible_tensor_ops.py
-m simplicial_tensors.notebooks.cocalc_compatible_tensor_ops
experiments/homology_basis.py
experiments/homology_basis.py --modulus 2
experiments/normalized_moore_complex_homology.py --modulus 3
experiments/sagemath_normalized_tensor_homology.py --modulus 3
tex/anc/dstm_examples.py
tex/anc/rank_shifted_homology.py
```

The Petersen script used the same workdir and `PYTHONPATH`, with `/sage/sage`
instead of `/sage/sage -python`. Ordinary Python used an editable installation
in `~/Jupyter/simplicial-validation-2026-11-12/python-env`. CI tools were
Ruff 0.17.4, Mypy 2.4.1, and Bandit 1.9.4, with these exact checks:

```sh
ruff check src tests --exclude src/simplicial_tensors/notebooks --select E9,F821,F822,F823
mypy --ignore-missing-imports --follow-imports=skip src/simplicial_tensors/__init__.py src/simplicial_tensors/tensor_ops.py src/simplicial_tensors/cyclic_tensor_ops.py
bandit -q -r src/simplicial_tensors -x src/simplicial_tensors/notebooks -lll
```

Raw logs, original failures, and independent probes are outside the tracked tree
at `~/Jupyter/simplicial-validation-2026-11-12/`. They include
`backend-original-tests.log`, `backend-native-before-repair.log`,
`backend-final-gate.log`, `sage-full-initial.log`, `sage-full-final.log`,
`sage-metadata-owner.log`, `python-full.log`, `homology-sage-reference.log`,
`petersen-original.log`, `petersen-action-probe.log`, `petersen-final.log`,
`petersen-notebook.log`, `example-1.log` through `example-8.log`, and `notebook.log`.
No Jupyter tokens were recorded.

## Scoped mathematical and documentation audit

### (A) Kill-shots / hard blockers

The demonstrated scalar and permutation defects above were repaired and verified.
No confirmed blocker remains in the tested scope.

### (B) Major weaknesses

No remaining major issue was found. For GAP's right action, a right coset is
`Aut(P)g`; inversion and automorphism invariance justify the reorder-list action.
Distinct-matrix coverage was checked independently of the weighted total.

### (C) Minor issues

The affected documentation was reviewed using Section 1 of Knuth, Larrabee, and
Roberts's [Mathematical Writing](https://www-cs-faculty.stanford.edu/~knuth/papers/cs1193.pdf).
The backend page proceeds from interface and coefficient domain to commands and
the notebook adapter. The Petersen page proceeds from construction to commands,
counts, and the permutation convention. No reorganization was warranted.
Exact coefficients, native execution, and the corrected action are stated at
their points of use. No mathematical manuscript was edited.

### (D) Cross-section consistency failures

None found in the affected pages and implementations. The independent rank tests
retain the distinction between a generated submodule and the full ambient module,
and between adjacency deletion boundaries and ordinary graph incidence.

### (E) Acceptance conditions, non-negotiable

Native backend execution, exact scalar regressions, independent homology ranks,
full Petersen coverage and histograms, both notebooks, examples, ordinary Python
tests, and CI quality gates are complete. The report records optional skips and
the separate ownership-compatible metadata run explicitly.

### (F) Not checked / uncertainty log

PyTorch and Dask execution was unavailable. This was a scoped code and affected-
documentation review, not a whole-manuscript review, novelty assessment, or
mathematical certification. The manuscript and its committed PDF were unchanged.
Strongest confirmed remaining blocker: none. Strongest suspected blocker: none
identified. Most important unperformed check: optional PyTorch/Dask execution.
The operational handoff is complete; a journal-level recommendation is unknown
and outside this review's scope.
