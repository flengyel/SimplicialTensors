# Shape Generator

## Source

```text
experiments/shape_generator.py
```

## Type

Python script

## Research Goal

Generates shape families (n_1,...,n_k) for systematic testing across simplicial dimension/order regimes.

## Theoretical Anchors

- tex/preliminaries.tex (shape vector and simplicial dimension n = min(n_a)-1).
- tex/combinatorics.tex (classification behavior as k and n vary).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/shape_generator.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Enumerated shape tuples used as inputs to downstream computational experiments.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

