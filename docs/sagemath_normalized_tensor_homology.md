# Sagemath Normalized Tensor Homology

## Source

```text
experiments/sagemath_normalized_tensor_homology.py
```

## Type

Python script

## Research Goal

Computes or validates normalized Moore-complex homology data and compares observed dimensions with theoretical predictions.

## Relevant Results in TeX Sources

- tex/normalization.tex (normalized complex N_*(X) and Moore filler machinery).
- tex/combinatorics.tex (rank formulas for Z_p and N_p).
- tex/dichotomy.tex (subcomplex and quotient-homology interpretation).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/sagemath_normalized_tensor_homology.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Cycle/boundary ranks, basis representatives, and homology summaries by degree.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




