# Degenerate Counterexample

## Source

```text
experiments/degenerate_counterexample.py
```

## Type

Python script

## Research Goal

Investigates degenerate-versus-nondegenerate behavior of tensors and fillers, including edge cases and counterexample candidates.

## Theoretical Anchors

- tex/nondegeneracy_lemma.tex (Horn Non-Degeneracy Lemma).
- tex/normalization.tex (normalized/degenerate decomposition X_p = N_p ⊕ D_p).
- tex/dichotomy.tex (impact on filler uniqueness and cycle structure).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/degenerate_counterexample.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Counterexample candidates, decomposition checks, and degeneracy classification logs.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

