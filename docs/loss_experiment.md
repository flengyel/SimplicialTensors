# Loss Experiment

## Source

```text
experiments/loss_experiment.py
```

## Type

Python script

## Research Goal

Explores ML-style training behavior under simplicial boundary/degeneracy-inspired transformations and compares optimization behavior across configurations.

## Theoretical Anchors

- tex/normalization.tex (decomposition into normalized and degenerate components).
- tex/combinatorics.tex (rank behavior and boundary-related structure in degrees).
- tex/equivariant_homotopy.tex (contractibility perspective motivating robustness checks).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/loss_experiment.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Training/validation statistics, loss curves, and optional plot artifacts for comparative runs.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

