# Kernel sequences and lifting validation, 2026-11-12

Base revision: `fe984d95d9ece08659e23b574c9721266f6e7c94`.
Locally tested implementation revision: `187e5b04aadf690590eef57278c6bd4839a36886`.
GitHub implementation revision: `d03139e4b597088b1e1fe6960b9dc1f188d51c64`.
Both implementation revisions have tree
`3314ba8115dea43c8d6aee3e2998a6cedf06046b`.
This record concerns that implementation tree; the report and its navigation
link form a subsequent documentation commit.

## Manuscript

The current manuscript, [Computations with simplicial tensor modules](../../tex/dstm_short.pdf),
now has fourteen pages. Its [generated-submodule section](../../tex/generated_section.tex)
contains two propositions with proofs:

- For a field $K$ and fixed $n\geq 0$, every specified chain isomorphism
  $C/L\to C/M$ between quotient complexes of $C=N(K[\Delta^n])$ lifts to a
  chain automorphism of $C$. The proof uses projectivity of the standard simplex complex,
  a Fitting decomposition, and cancellation of finite vector-space complexes.
- For the tensor module $X=X(c;K)$ with nonnegative shifts $c$, a subcomplex
  $L\subseteq C$ is realized as a tensor's kernel exactly when $C/L$ embeds in
  $N(X)$. The equivalent inequalities compare both ordinary
  cycle dimensions and boundary dimensions in every degree. The proof constructs
  an injection of the boundary--cycle flags and lifts the remaining basis vectors
  through the differentials.

The text specifies the action on realizable kernels by precomposition. It
distinguishes ordinary degree-zero cycles from the reduced coordinate module
used in the ambient disk decomposition. Worked examples give the homology jump
of $T(t)$ at $t=1$, its varying kernel planes, and sparse generators whose
generated complexes have one-dimensional homology in degree $q\geq 1$, using
$q+1$ tensor axes.
The abstract, introduction, further-work note, and manuscript instructions agree
with these statements. The previous arXiv-v2 driver and section files were not
edited.

The final driver and inputs compiled without unresolved references or citations,
duplicate labels, or overfull or underfull box warnings. All fourteen pages were
rendered and inspected in a contact sheet. Pages 7–9, containing the principal
new proofs and examples, were also inspected individually at the larger rendered
size. Final PDF SHA-256:

```text
0872449c878ca24861daa42b511b4c9e79177311ee85c272cf587736a7c6ba65
```

A source-only archive was prepared outside the checkout. All eight archived
files match the corresponding manuscript source bytes. Its SHA-256 is
`16816a9ca9afd2ecc37e1fae80f5c0fc9e348164a3984ee56eae257a2f5b6480`.

## Software and examples

The [kernel-sequence API](../api/kernel_sequences.md) retains the full normalized
standard simplex source, ordered face labels, realization matrices, kernel
bases and differentials, quotient maps, and the quotient-to-image comparison.
It retains all degrees through the generator degree, including zero realization
maps above the tensor order. This preserves the distinguished-generator data
that an image basis alone discards.

`realize_kernel` constructs a sparse tensor from an admissible kernel inclusion.
It uses only selected coordinate cycles, their boundary sections, and required
tail classes. The [chain-map implementation](../../src/simplicial_tensors/chain_maps.py)
provides checked immutable maps, embeddings, isomorphisms, explicit capacity
errors, and automorphism lifts of specified quotient maps. Singular initial
lifts are corrected by a finite Fitting construction.

The [example program](../experiments/kernel_sequences.md) covers the parameter
family, distinct marked kernels with isomorphic quotients, higher homology,
kernel realization, quotient-isomorphism lifting, and failed cycle or boundary
capacities. It prints JSON and creates no files. The
[dendroidal program](../experiments/dendroidal_representable_chains.md) also
computes the two boundary subcomplexes. The two-vertex boundary has Betti numbers
$(2,1)$; its class $[u+v-w]$ is the connecting obstruction to lifting the relative
top class to an absolute cycle.

Three exhibited implementation problems were repaired: a supplied invalid
backward lift was ignored when the forward lift was already invertible; an
initial tail iterator eagerly pooled large ranges; and native Sage exact
integers/rationals were rejected by the shared field and integer validators.
Tail selection now uses bounded ordinal decoding. Exact Sage inputs use integer
and rational protocols; floating-point and Sage real-field coefficients remain
rejected. The existing `SparseTensor` input contract was not broadened.

## Environments and results

The primary tests reused the running `sagepod` container: SageMath 10.10,
Python 3.13.5, NumPy 2.4.4, SymPy 1.14.0, SciPy 1.17.1, and Pytest 9.0.2.
Ordinary Python used Python 3.13.5, NumPy 2.5.4, SymPy 1.14.0, SciPy 1.18.1,
and Pytest 9.2.0. Compatibility checks reused `sagemath`, with SageMath 10.7,
Python 3.12.5, NumPy 2.2.4, SymPy 1.13.2, SciPy 1.15.2, and Pytest 8.3.2.
No container was restarted or rebuilt, and no packages or Git trust settings
were changed in either container.

| Check | Result |
|---|---|
| Ordinary full suite | 528 passed, 24 skipped; 85.06 s |
| Native SageMath 10.10 full suite | 630 passed, 10 skipped; 135.84 s |
| Git metadata checks in the SageMath 10.7 container as checkout owner | 2 passed, 4 deselected; 0.40 s |
| SageMath 10.7 kernel/lifting/reference subset | 128 passed; 31.33 s |
| Independent native kernel and lift reference module | 34 passed; included in the full native suite |
| Focused sparse/lifting tests under a 1 GiB virtual-memory limit, with single-threaded BLAS | 108 passed, 12 skipped; 19.32 s |
| Kernel CLI over the rationals and characteristics two, three, and five | Completed |
| Three Python examples in the new API guide | Executed |
| Documentation links and package metadata | 7 passed |
| Ruff, selected Mypy on eight source files, Bandit | Passed |
| Fresh wheel installation and isolated smoke test | Passed; all 24 Python source files match the checkout byte for byte |

The primary native suite skipped one Dask case, seven PyTorch cases, and two
Git metadata cases because Git is absent from that container. The latter two
passed separately as the mounted checkout owner, using a temporary home and
disabled pytest cache writing. Thus 632 applicable full-suite cases passed
across those two runs; eight optional-dependency cases remain skipped. The
128-case compatibility subset is separate and is not added to that total.
Ordinary Python also skips four Sage-only modules and twelve individual native
Sage cases; these execute in the primary native suite.

The independent Sage reference constructs normalized source columns from
nondecreasing maps and assembles tensor pullbacks separately. It checks eighteen
tensor/field configurations, twelve reconstructed-tensor configurations, and
three specified quotient-isomorphism lifts, plus its source-index convention.
Other tests cover singular lift correction, nonprojective-source rejection,
positive-tail boundary capacity, exact large rational coefficients, and sparse
shapes with tail lengths $10^8$. The matrix family was checked over the rationals
and three prime fields. Example inputs are fixed; random reference tensors use
NumPy seed 20261112.

The wheel smoke ran in a fresh virtual environment with `python -I` from `/tmp`,
deliberately ignoring a source `PYTHONPATH`. Final wheel SHA-256:

```text
e06055fa85f1e75e44e2716e84d97337326c17332a51586510ed7da1c8d16d99
```

## Documentation and scoped mathematical review

The Mathematical Writing workflow consulted Section 1 of Knuth, Larrabee, and
Roberts's [Mathematical Writing](https://www-cs-faculty.stanford.edu/~knuth/papers/cs1193.pdf).
The manuscript introduces the chain category and ordinary cycle/boundary
notation before the lifting statements. Proposition statements retain their
field, finite-dimensional, and presentation hypotheses. API pages define source
coordinates, ambient representatives, coefficient fields, and composition order
before examples. Editorial reports remain outside the mathematical manuscript.

### (A) Kill-shots / hard blockers

No remaining confirmed blocker was found in the two propositions, their worked
examples, or the corresponding checked implementations.

### (B) Major weaknesses

The propositions concern vector spaces over a field. Projectivity as a chain
object is required for quotient-isomorphism lifting; degreewise freeness alone
does not suffice. Over general rings an invertible lift can fail. The software
implements the rationals and declared prime fields.

### (C) Minor issues

The inverse tensor construction and automorphism lift depend on basis choices.
They guarantee exact kernel equality and the specified quotient-map identity;
neither construction is unique.

### (D) Cross-section consistency failures

No remaining inconsistency was found in the reviewed changes. Precomposition
preserves the realizable-kernel locus. Ordinary $Z_0$ includes the tail summand;
both capacity inequalities are checked. Quotient representatives and absolute
cycles remain distinct. The dendroidal calculations concern the specified
representables and their boundary subpresheaves.

### (E) Acceptance conditions, non-negotiable

The recorded checks met the scoped endpoints: full marked kernel subspaces,
exact realization after reconstruction, injective chain embeddings, invertible
specified quotient lifts, explicit capacity failures, native Sage agreement,
isolated installation, and a compiled and inspected manuscript. Failure of one
would invalidate the corresponding computational or manuscript claim.

### (F) Not checked / uncertainty log

The strongest confirmed implementation defects were the repaired eager tail
allocation and ignored invalid backward lift. No unresolved suspected blocker
was identified in the changed scope. The scoped mathematical audit did not assess
whole-manuscript correctness or novelty; a recommendation on the complete
manuscript remains unknown. Full array-presheaf dendroidal homology, integral
torsion, and general coefficient-ring lifting remain outside this implementation.

## Evidence

Logs and the source archive are outside the checkout at
`~/Jupyter/simplicial-kernel-validation-2026-11-12/`. Key records are
`python-full.log`, `sage-10.10-full.log`, `sage-10.7-metadata.log`,
`sage-10.7-compatibility.log`, `native-kernel-reference.log`,
`focused-python-memory.log`, `sage-compat-native.log`, `quality-final.log`,
`examples-kernel-*.json`, `wheel-smoke.log`, and
`mathematical-writing-review.md`. The first focused chain-map audit is also
recorded in `~/Jupyter/simplicial-lifting-validation-2026-11-12/`.
