# Atlas Cospectral Scan

## Source

```text
experiments/atlas_cospectral_scan.py
```

## Type

Python script

## Research Goal

Computes graph-derived simplicial statistics (including boundary-rank style invariants) for cospectral/non-isomorphic comparisons and structural scans.

## Relevant Results in TeX Sources

- tex/horns.tex (kernel-support interpretation of omitted faces).
- tex/combinatorics.tex (rank counting framework and inclusion-exclusion viewpoint).
- tex/generated_subobjects.tex (realization matrices and rank-loss/collision perspective).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/atlas_cospectral_scan.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Enumerations, rank distributions, cospectral pair comparisons, and optional saved summaries.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




