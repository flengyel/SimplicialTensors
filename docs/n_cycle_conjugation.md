# N Cycle Conjugation

## Source

```text
experiments/n_cycle_conjugation.py
```

## Type

Python script

## Research Goal

Studies permutation-action invariants and symmetry effects on simplicial tensor constructions.

## Theoretical Anchors

- tex/preliminaries.tex (Stab(s) action and equivariance under axis permutations).
- tex/equivariant_homotopy.tex (equivariance of homotopy operator).
- tex/combinatorics.tex (rank effects under symmetry-restricted settings).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/n_cycle_conjugation.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Permutation statistics, orbit-level summaries, and invariance checks.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

