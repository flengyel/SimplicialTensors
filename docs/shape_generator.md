# Shape Generator

## Source

```text
experiments/shape_generator.py
```

## Type

Python script

## Research Goal

Generates shape families (n_1,...,n_k) for systematic testing across simplicial dimension/order regimes.

## Relevant Results in TeX Sources

- tex/preliminaries.tex (shape vector and simplicial dimension n = min(n_a)-1).
- tex/combinatorics.tex (classification behavior as k and n vary).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/shape_generator.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Enumerated shape tuples used as inputs to downstream computational experiments.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




