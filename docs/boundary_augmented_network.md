# Boundary Augmented Network

## Source

```text
experiments/boundary_augmented_network.py
```

## Type

Python script

## Research Goal

Explores ML-style training behavior under simplicial boundary/degeneracy-inspired transformations and compares optimization behavior across configurations.

## Relevant Results in TeX Sources

- tex/normalization.tex (decomposition into normalized and degenerate components).
- tex/combinatorics.tex (rank behavior and boundary-related structure in degrees).
- tex/equivariant_homotopy.tex (contractibility perspective motivating robustness checks).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/boundary_augmented_network.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Training/validation statistics, loss curves, and optional plot artifacts for comparative runs.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




