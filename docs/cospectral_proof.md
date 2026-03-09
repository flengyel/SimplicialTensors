# Cospectral Proof

## Source

```text
experiments/cospectral_proof.py
```

## Type

Python script

## Research Goal

Computes graph-derived simplicial statistics (including boundary-rank style invariants) for cospectral/non-isomorphic comparisons and structural scans.

## Theoretical Anchors

- tex/horns.tex (kernel-support interpretation of omitted faces).
- tex/combinatorics.tex (rank counting framework and inclusion-exclusion viewpoint).
- tex/generated_subobjects.tex (realization matrices and rank-loss/collision perspective).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/cospectral_proof.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Enumerations, rank distributions, cospectral pair comparisons, and optional saved summaries.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

