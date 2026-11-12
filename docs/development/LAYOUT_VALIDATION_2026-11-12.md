# Repository layout validation, 2026-11-12

Base revision: `c375540c66f3abe2f2bd4349cc57c001fd4afa7e`.
Locally tested cleanup revision: `8f70b67e03d4e24932a22f42eb87922b2a27e311`.
Published cleanup revision: `10610447ba18068bbdab06f68cfe707d3246ee9b`.
Both cleanup revisions have tree `a35e39323ce4b9a9d370de50cb8ea03bea66dc7e`.
This record reports checks on that tree.

## Changes

- Grouped documentation under `api/`, `experiments/`, `research/`, and
  `development/`, with [navigation](../README.md) and the existing
  [experiment catalog](../experiments_catalog.md).
- Moved five notebook documents to `experiments/notebooks/`. The six importable
  Python modules retain their `simplicial_tensors.notebooks` paths and now ship
  in the wheel.
- Consolidated result files by study. Twenty-five moved artifact payloads
  match their original SHA-256 hashes. Three duplicate plot copies were removed
  after equality checks; all distinct plots remain.
- Directed new output defaults to ignored `experiments/results/<study>/runs/`
  directories. Explicit CLI destinations remain supported. Packaged analysis
  loggers use the caller's working directory rather than the installation prefix.
- Grouped submission files in `tex/submissions/`, placed the package-layout
  utility in `scripts/maintenance/`, and removed tracked Windows folder metadata.
  The maintenance utility previews changes by default and refuses collisions.
- Added recursive documentation coverage, local-link checks, and an isolated
  wheel-install CI job. Ruff and Bandit now cover every shipped source directory.

## Validation

Ordinary Python 3.13.5 used NumPy 2.5.4, SymPy 1.14.0, SciPy 1.18.1, and
Pytest 9.2.0. The existing SagePod container used Sage 10.7, Python 3.12.5,
NumPy 2.2.4, SymPy 1.13.2, SciPy 1.15.2, and Pytest 8.3.2. Its native Sage
backend gate passed. The container was reused without restart or rebuild.

| Check | Result |
|---|---|
| Ordinary Python suite | 318 passed, 11 skipped; 50.41 s |
| Native Sage full suite | 370 passed, 10 skipped; 135.49 s |
| Sage Git metadata checks, run as checkout owner | 2 passed, 4 deselected; 0.80 s |
| Ruff, selected Mypy, Bandit, CI workflow parse | Passed |
| Isolated wheel installation | Passed; 18 Python source files, no notebook documents |
| Installed CoCalc and two homology module commands | Completed from a temporary working directory |
| Relocated CoCalc notebook | All four cells completed in one native Sage namespace |
| Output paths and CLI overrides | Bounded smoke and parser checks passed |
| Markdown and notebook Markdown local links | Passed |
| Submission/result payload preservation | All 25 unchanged payload hashes matched |

The Sage full run skipped one Dask case, seven PyTorch cases, and two Git metadata
checks because of mounted-checkout ownership. The two metadata checks passed in
a separate read-only run as the checkout owner, with a temporary home and pytest
cache writing disabled. Thus 372 applicable Sage cases passed across both runs;
eight optional cases remain skipped. Git trust settings were not changed.
Ordinary Python additionally skips the three Sage-only modules.

The wheel check used a fresh virtual environment and `python -I` from `/tmp`,
with a source `PYTHONPATH` deliberately ignored. All six module locations were
inside the installed environment. Every one of the wheel's 18 Python source
files matches the final checkout byte for byte. Wheel SHA-256:

```text
956d26c91f0b7b5783f74a7585180f4406ab29aa2d99dcb537962ba7460a4fa8
```

The five relocated notebooks also retain their original bytes. The output smoke
used small loss, permutation, and discrepancy calculations and temporary paths;
the three neural-study defaults were checked through their parsers. No training
run was repeated. The report renderer was checked separately against stored
summary data, including its link to the relocated guide.

Logs and artifact mappings are outside the source tree at
`~/Jupyter/simplicial-layout-validation-2026-11-12/`. Key logs are
`python-full.log`, `sage-full.log`, `sage-metadata-owner.log`,
`cocalc-notebook.log`, `wheel-final-build.log`, `wheel-final-install.log`,
`wheel-final-smoke.log`, `output-smoke.log`, and
`study-output-defaults-smoke.log`.

## Documentation review

The Mathematical Writing review consulted Section 1 of Knuth, Larrabee, and
Roberts's [Mathematical Writing](https://www-cs-faculty.stanford.edu/~knuth/papers/cs1193.pdf).
It applied definitions at first use, consistent notation, and unambiguous
sentences (Section 1, points 11, 14, and 17). The directory grouping is a
repository-design choice. API pages proceed from objects and coefficient conventions to operations and
examples; experiment guides give inputs, commands, outputs, and interpretation.
Research notes retain their mathematical organization. Navigation and directory
labels state purpose. Definitions and formulas in relocated API and research
pages retain their order and content.

The substantive correction is in [Further work](../../tex/FutureDirections.md):
the three homology scripts already compute exact normalized field homology
dimensions. The author approved the exact replacement after automatic approval
review initially treated it as beyond the layout task. The revised text matches
the shared implementation and distinguishes tensor-generated submodules from
the full ambient module. The active manuscript sources and PDF retain their
original contents.

## Scoped mathematical review

### (A) Kill-shots / hard blockers

No remaining blocker was confirmed in the changed mathematics.

### (B) Major weaknesses

The claim in the base revision's “Exact computations for generated submodules”
subsection of [Further work](../../tex/FutureDirections.md), that the scripts
“do not provide that calculation,” was corrected with the approved wording. The source constructs
normalized face spans with their ambient linear relations and computes exact
field ranks, chain dimensions, and homology dimensions.

### (C) Minor issues

The notebook guide now identifies the alternating principal-deletion boundary
in its syzygy worksheet. The `cobdry` description specifies diagonal degeneracies.

### (D) Cross-section consistency failures

No remaining inconsistency was found in the reviewed changes. Coefficient fields,
generated submodules, relative horn quotients, and graph incidence calculations
retain their separate definitions. The notebook guide describes worksheet
contents without claiming execution of every worksheet.

### (E) Acceptance conditions, non-negotiable

Packaging, import compatibility, output destinations, content preservation,
documentation links, integrated tests, native Sage execution, and the isolated
wheel commands have been checked. The approved mathematical correction has
received a separate evidence-bound review.

### (F) Not checked / uncertainty log

The four interactive worksheets were read statically; their polynomial-ring and
symbolic coercions were not re-executed. Dask analysis computations and PyTorch
runs were unavailable. Dask imports were stubbed only for logger-path checks.
Optional loss plotting was unavailable in Sage because pandas was absent.
The observer report renderer was checked without importing its training module,
whose optional scikit-learn dependency was unavailable in the ordinary environment.
The maintenance utility received syntax and embedded-Python checks without being
executed. No whole-manuscript proof or novelty assessment was performed.

Strongest confirmed remaining blocker: none. No unconfirmed blocker was
established; worksheet coefficient coercions remain unverified. The most
important unchecked item is execution of those four worksheets. A
whole-manuscript recommendation is unknown and outside this cleanup's scope.
