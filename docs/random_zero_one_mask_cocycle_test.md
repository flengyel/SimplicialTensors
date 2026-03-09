# Random Zero One Mask Cocycle Test

## Source

```text
experiments/random_zero_one_mask_cocycle_test.py
```

## Type

Python script

## Research Goal

Supports exploratory validation of diagonal simplicial tensor module behavior in concrete computational regimes.

## Relevant Results in TeX Sources

- tex/preliminaries.tex (DSTM construction and indexing).
- tex/horns.tex (horn kernels and missing indices).
- tex/combinatorics.tex (classification and rank formulas).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/random_zero_one_mask_cocycle_test.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Script-defined diagnostics and optional saved artifacts.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




