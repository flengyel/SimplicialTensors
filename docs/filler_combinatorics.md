# Filler Combinatorics

## Source

```text
experiments/filler_combinatorics.py
```

## Type

Python script

## Research Goal

Analyzes horn kernels, missing-index structure, and filler reconstruction behavior in concrete computational settings.

## Relevant Results in TeX Sources

- tex/horns.tex (support characterization and missing-index criterion).
- tex/normalization.tex (Moore filler map and split exact sequence).
- tex/nondegeneracy_lemma.tex (R_{p,j} ∩ D_p = {0} and horn decomposition).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/filler_combinatorics.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Horn compatibility diagnostics, filler comparisons, and kernel-support measurements.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




