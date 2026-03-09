# Random Zero One Mask Cocycle Test

## Source

```text
experiments/random_zero_one_mask_cocycle_test.py
```

## Type

Python script

## Research Goal

Supports exploratory validation of diagonal simplicial tensor module behavior in concrete computational regimes.

## Theoretical Anchors

- tex/preliminaries.tex (DSTM construction and indexing).
- tex/horns.tex (horn kernels and missing indices).
- tex/combinatorics.tex (classification and rank formulas).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/random_zero_one_mask_cocycle_test.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Script-defined diagnostics and optional saved artifacts.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

