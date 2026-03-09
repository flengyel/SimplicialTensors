# Homotopy Constraint Verification

## Source

```text
experiments/homotopy_constraint_verification.py
```

## Type

Python script

## Research Goal

Verifies chain-homotopy identities and compatibility constraints used to prove contractibility and filtration compatibility.

## Theoretical Anchors

- tex/equivariant_homotopy.tex (explicit homotopy operator H and identities).
- tex/normalization.tex (Moore filler decomposition and exact horn sequence).
- tex/dichotomy.tex (induced homological consequences of contractibility).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/homotopy_constraint_verification.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Identity checks, mismatch reports, and supporting numeric/symbolic evidence for homotopy constraints.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

