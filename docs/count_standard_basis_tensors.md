# Count Standard Basis Tensors

## Source

```text
experiments/count_standard_basis_tensors.py
```

## Type

Python script

## Research Goal

Performs targeted computational checks of combinatorial identities and structural invariants on explicit tensor families.

## Theoretical Anchors

- tex/horns.tex (missing-index characterization and face-kernel basis).
- tex/combinatorics.tex (rank and finite-difference formulas).
- tex/normalization.tex (decomposition through Moore filler map).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/count_standard_basis_tensors.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Consistency checks, discrepancy reports, and compact experiment summaries.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

