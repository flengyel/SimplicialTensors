# Gemini Hallicination

## Source

```text
experiments/Gemini_hallicination.py
```

## Type

Python script

## Research Goal

Critically checks conjectural statements against the proven DSTM framework and identifies mismatches between heuristic claims and manuscript-level results.

## Theoretical Anchors

- tex/introduction.tex (overall DSTM program and classification objective).
- tex/combinatorics.tex (strict algebraic n-hypergroupoid criterion k = n).
- tex/dichotomy.tex (filler uniqueness vs non-uniqueness regimes).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/Gemini_hallicination.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Diagnostic comparisons in console output documenting where claims fail or require proof-level support.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

