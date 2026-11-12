# Exact homology improvements and validation, 2026-11-12

Base revision: `0c21eafedecbd42fdcd07c24b112e26c5edfbd5a`.
Locally tested implementation revision: `e98078cd6002a8e3a4eb01a059744f11917f38e0`.
GitHub implementation revision: `299fc9a71218314cc2f6be40ef9e876c5090a22f`.
Both implementation revisions have tree
`aa2334e1b75056e899ed1cce720a00fb30e0a1c8`.
The checks below concern that implementation tree.

## Changes

- Added [finite exact chain complexes](../api/chain_complex.md), with immutable
  differential matrices, cycle and boundary bases, homology representatives,
  and chosen ambient lifts over the rationals or a declared prime field.
- Added based complexes for dense and sparse tensor generators and relative
  horn quotients. Sparse tensors retain their declared degree and axis shifts;
  construction uses finite supports without allocating the dense ambient array.
- Added [embedded hypergraph homology](../api/hypergraph_homology.md), using the
  infimum complex and joint constraints on missing faces. Singleton hyperedges
  are supplied explicitly. The summary calculation uses immediate facets and
  exact ranks; constructing the full downward closure is a separate operation.
- Reused normalized restrictions in the existing relative homology routine,
  removed zero and duplicate rank columns, and constructed sparse field matrices
  directly. Existing dimension-summary interfaces retain their behavior.
- Used connected-component counts for graph embeddings whose sparse vertex and
  edge tensors pass the incidence checks. The fallback computes exact
  tensor-incidence ranks, retaining their coefficient-field dependence.
- Unified shape validation in both symbolic constructors. A declared shape such
  as `(2, 2)` with data `[[1]]` previously constructed an inconsistent object;
  it now raises `ValueError` at construction. Negative and nonintegral dimensions
  are rejected; exact Sage and NumPy index integers and zero-length output axes
  remain supported.
- Added hypergraph examples, checked finite dendroidal representables, API guides,
  and [a dendroidal research note](../research/dendroidal_homology.md). The active
  manuscript sources and PDF were not changed. `FutureDirections.md` gained one
  reader link to the research note.
- Extended selected Mypy coverage and the isolated installed-wheel smoke test.
  Git-dependent metadata tests now skip explicitly when the executable is absent.

## Environments and checks

The primary native run reused the running `sagepod` container without restart,
rebuild, package installation, or Git trust changes. It used SageMath 10.10,
Python 3.13.5, NumPy 2.4.4, SymPy 1.14.0, SciPy 1.17.1, and Pytest 9.0.2.
The native backend gate passed. The image was
`localhost/sagepod-sagemath:10.10-pycryptosat`, image ID
`5268b48bd5b27df1c415b5578da10884f588894c3982f0bd7c02873c48ce7caf`.
The SagePod checkout was at
`4bc1bc364589777e8baa27bfc687d158310d71a5`; its unrelated untracked file was
preserved.

Ordinary Python used Python 3.13.5, NumPy 2.5.4, SymPy 1.14.0, SciPy 1.18.1,
and Pytest 9.2.0. A compatibility subset also ran in the existing `sagemath`
container with SageMath 10.7, Python 3.12.5, NumPy 2.2.4, SymPy 1.13.2,
SciPy 1.15.2, and Pytest 8.3.2.

| Check | Result |
|---|---|
| Ordinary Python full suite | 430 passed, 14 skipped; 36.88 s |
| Native SageMath 10.10 full suite | 489 passed, 10 skipped; 77.46 s |
| Git metadata checks in the SageMath 10.7 container, as checkout owner | 2 passed, 4 deselected; 0.58 s |
| SageMath 10.7 chain/hypergraph/tensor compatibility subset | 61 passed; 22.09 s |
| Independent native Sage hypergraph comparison | 381 cases passed: all 127 nonempty hypergraphs on three labels over each of three fields |
| Independent native Sage tensor comparison | 84 absolute/relative cases passed; seed 20261112 |
| Two new example CLIs over the rationals, characteristic two, and characteristic three | Completed |
| CoCalc notebook | All four cells completed in one native Sage namespace |
| Ruff, selected Mypy (six source files), Bandit | Passed |
| Three new API examples and 24 local links in the reviewed API/experiment pages | Passed |
| Fresh wheel installation and isolated smoke check | Passed; all 22 Python source files match the checkout byte for byte |

The primary Sage run skipped one Dask case, seven PyTorch cases, and two Git
metadata cases because Git is absent from that container. The metadata cases
ran in the ordinary suite and passed separately in the existing SageMath 10.7
container as the mounted checkout owner, with temporary home and pytest cache
writing disabled. Combining the primary full suite with those two metadata
checks gives 491 passed applicable cases; eight optional-dependency cases
remained skipped. The 61-case compatibility subset is recorded separately.
The ordinary run also skipped
three Sage-only modules and three native dendroidal reference cases; those
cases ran in the primary native suite.

The independent native comparisons use Sage matrices and separately assembled
infimum constraints and tensor boundaries. They cover tensor pullbacks before
normalization, quotient relations, exact large rational coefficients, and lifted
representatives over the rationals, `GF(2)`, and `GF(3)`. Separate tests compare
infimum homology with independently assembled supremum complexes for the same
127 hypergraphs over each field, using manual Gaussian elimination. Other tests
cover cancellation between hyperedges, all 24 relabelings of a four-vertex
fixture, a field-dependent six-vertex complex, and sparse allocation guards.

The wheel smoke used a fresh virtual environment and `python -I` from `/tmp`,
with a source `PYTHONPATH` deliberately ignored. Wheel SHA-256:

```text
956ae4d98de8cd3dd5813b54512b8096a329c67845af20b1eef229026d9d4f90
```

## Bounded performance measurements

The existing algorithms were compared with base revision `0c21eaf` in ordinary
Python. Each row below uses the median of three interleaved runs, with exact
results checked for equality. The relative-horn input was a `12 x 12` integer
array with entries from -3 through 3, NumPy seed 147, and omitted face zero.
The graph input was a 120-vertex cycle.

| Calculation | Base seconds | Revised seconds | Base / revised |
|---|---:|---:|---:|
| Relative homology over the rationals | 0.991 | 0.098 | 10.1 |
| Relative homology over `GF(2)` | 0.774 | 0.054 | 14.3 |
| Relative homology over `GF(3)` | 0.738 | 0.083 | 8.9 |
| Graph homology over the rationals | 0.168 | 0.020 | 8.5 |

Review of the new hypergraph implementation also removed an unnecessary
downward-closure expansion. A single 16-vertex hyperedge had caused 65,535
subsets to be generated; the repaired summary uses its immediate facets.
In one bounded SageMath 10.10 measurement, runtime changed from 2.310 s to 0.0126 s, with the
same zero homology dimensions. For the boundary of all three-element subsets
on 12 labels, the `66 x 220` matrix has 660 nonzero entries. In ordinary Python, sparse field
conversion reduced processed scalar entries from 14,520 to 660 and traced peak
allocation from 1,195,118 to 346,005 bytes in one measured run; the rank stayed
55. These measurements describe the stated fixtures and environments, not
general speed or memory bounds.

Face enumeration and exact basis calculations can still grow with tensor
degree and support. Explicit downward closure has exponential output size for
a single large hyperedge. The based-complex APIs perform more work than a
dimension-only summary.

## Documentation review

The Mathematical Writing workflow consulted Section 1 of Knuth, Larrabee, and
Roberts's [Mathematical Writing](https://www-cs-faculty.stanford.edu/~knuth/papers/cs1193.pdf),
especially definitions at introduction, consistent notation, and unambiguous
sentences (points 11, 14, and 17). The API pages introduce coefficient fields,
degrees, matrix directions, and coordinate conventions before operations and
examples. They distinguish chain coordinates from chosen ambient representatives
and explain that a lift of a quotient cycle can have boundary in the discarded
subcomplex. The hypergraph guide defines the infimum complex before its rank
formulas. The dendroidal note separates its two source constructions before
giving finite matrices. Numerical claims were retained only with corresponding
calculations or cited results.

## Scoped mathematical review

### (A) Kill-shots / hard blockers

No remaining blocker was confirmed in the changed definitions, formulas, or
finite calculations. This verdict applies to these changes and examples, not
to all claims in the manuscript.

### (B) Major weaknesses

The array presheaf's full dendroidal homology remains uncomputed. The
[research note](../research/dendroidal_homology.md) distinguishes the planar,
tree-indexed normalization of Gutiérrez–Lukacs–Weiss from the stable
dendroidal-set homology of Bašić–Nikolaus. The binary-corolla input interchange
fixes the zero array, so the underlying dendroidal set is not normal. Moreover,
free chains on set elements do not inherit the arrays' additive relations.
A comparison theorem and the appropriate normal replacement would be needed
for a claimed identification with that stable homology. A tensor shape alone
does not determine its tree or dendroidal degree.

### (C) Minor issues

The implemented tree matrices cover a binary corolla and a unary-above-binary
tree, with their relative boundary quotients. Their absolute Betti numbers are
`(2, 0)` and `(2, 0, 0)` over all three tested fields. Their relative quotient
homology has one generator in degree one and degree two, respectively, and is
zero in the other degrees. These are representable dendroidal sets,
not the array presheaf. No general tree enumeration or normalization engine was
added.

### (D) Cross-section consistency failures

No remaining inconsistency was found in the reviewed changes. The existing
summary APIs and new representative APIs are named separately. Sparse tensor
degree and shifts are distinguished from array shape. Missing singleton
hyperedges are not silently supplied. Quotient representatives carry their
ambient-boundary qualification.

### (E) Acceptance conditions, non-negotiable

The following endpoints were met for the scoped claims:

1. Constructor contracts and existing summary interfaces pass the recorded
   ordinary and native suites.
2. Chain identities, representatives, and lifts agree with 381 native infimum
   cases and the separate supremum-reference comparisons.
3. Tensor and relative-quotient results agree with 84 independent native cases,
   including boundary membership of lifted quotient representatives.
4. Finite dendroidal matrices satisfy the chain identity and agree with the
   representable and relative-cell results of Bašić–Nikolaus, Corollaries
   5.12–5.13. Source conventions are given in the research note.

Failure of an endpoint would invalidate its corresponding API, homology, or
finite-fixture claim. These endpoints assert no full array-presheaf comparison.

### (F) Not checked / uncertainty log

The implemented changes are supported by the scoped tests and source review.
They do not compute integral torsion, persistence, a general hypergraph-to-tensor
realization, or the array presheaf's full dendroidal homology. No novelty claim
is made.

The strongest confirmed implementation defect was the constructor shape/storage
mismatch, now repaired. No unconfirmed blocker was identified in the checked
implementation. The most important unperformed work is a normal replacement
and comparison for the full array presheaf; an identification would currently
be unsupported. The scoped assessment uses available code, logs, and primary
sources. The recommendation for a full array-presheaf identification is unknown.

## Evidence

Logs and independent audit scripts are outside the repository at
`~/Jupyter/simplicial-homology-validation-2026-11-12/`. Key records are
`python-full.log`, `quality-final.log`, `sage-10.10-gate.log`, `sage-10.10-full.log`,
`sage-10.7-metadata.log`, `sage-10.7-chain-compatibility.log`,
`sage-cli-and-notebook.log`, `independent-new-homology-audit-final.log`,
`benchmark_core-subsets.log`, the `hypergraph-closure-*` and `hypergraph-sparse-*`
logs, `wheel-smoke.log`, and `mathematical-writing-review.md`.
