# Sagemath Compatible Tensor Ops

## Source

```text
experiments/sagemath_compatible_tensor_ops.py
```

## Type

Python script

## Research Goal

Provides symbolic/Sage-based computation paths mirroring core DSTM operations to validate exact identities over symbolic coefficient domains.

## Relevant Results in TeX Sources

- tex/preliminaries.tex (definition of X_p(s;A) via index sets).
- tex/normalization.tex (normalization theorem and Moore filler algorithm).
- tex/combinatorics.tex (rank/cycle formulas used for symbolic checks).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/sagemath_compatible_tensor_ops.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Symbolic identities, exact kernel computations, and reproducible Sage session outputs.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




