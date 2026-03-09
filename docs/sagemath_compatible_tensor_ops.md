# Sagemath Compatible Tensor Ops

## Source

```text
experiments/sagemath_compatible_tensor_ops.py
```

## Type

Python script

## Research Goal

Provides symbolic/Sage-based computation paths mirroring core DSTM operations to validate exact identities over symbolic coefficient domains.

## Theoretical Anchors

- tex/preliminaries.tex (definition of X_p(s;A) via index sets).
- tex/normalization.tex (normalization theorem and Moore filler algorithm).
- tex/combinatorics.tex (rank/cycle formulas used for symbolic checks).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/sagemath_compatible_tensor_ops.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Symbolic identities, exact kernel computations, and reproducible Sage session outputs.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

