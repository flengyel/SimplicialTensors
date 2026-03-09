# Normalized Moore Complex Homology

## Source

```text
experiments/normalized_moore_complex_homology.py
```

## Type

Python script

## Research Goal

Computes or validates normalized Moore-complex homology data and compares observed dimensions with theoretical predictions.

## Theoretical Anchors

- tex/normalization.tex (normalized complex N_*(X) and Moore filler machinery).
- tex/combinatorics.tex (rank formulas for Z_p and N_p).
- tex/dichotomy.tex (subcomplex and quotient-homology interpretation).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/normalized_moore_complex_homology.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Cycle/boundary ranks, basis representatives, and homology summaries by degree.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

